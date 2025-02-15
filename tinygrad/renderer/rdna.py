import struct, heapq
from collections import defaultdict, deque
from tinygrad.dtype import DType, PtrDType, dtypes
from tinygrad.ops import Ops, GroupOp, UOp, UPat, PatternMatcher
from tinygrad.device import GracefulSkip
from tinygrad.codegen.rewriter import graph_rewrite, symbolic_simple, get_late_rewrite_patterns, TRANSCENDENTAL, powers_of_two
from tinygrad.codegen.linearize import BasicBlock
from tinygrad.renderer import Renderer, TensorCore
from tinygrad.renderer.cstyle import CStyleLanguage
from tinygrad.renderer.support.rdna_asm import RDNAOp, RDNAOps, RDNAImm, RDNAReg, RDNAValue, SRC9_full, RDNAOPS_VALU, RDNAOPS_LOAD, EXEC_LO, LBL, \
                                               serialize_tinyasm, basewalk, same_reg, DUAL_MAP, DUAL_OP, RDNARegSet, s_waitcnt_simm
from tinygrad.helpers import DEBUG, getenv, round_up

def fuse_acc_chain(assign: UOp, acc: UOp, alu: UOp) -> UOp|None:
  def _inner_recurse(cur: UOp, assigned: bool) -> UOp|None:
    if cur.op is Ops.ASSIGN:
      assert cur.src[0] == acc
      if (new_src1:=_inner_recurse(cur.src[1], assigned=True)) is not None:
        return cur.replace(src=(cur.src[0], new_src1))
      else:
        return None
    if cur.op is Ops.ADD:
      if (new_src0:=_inner_recurse(cur.src[0], assigned=False)) is not None:
        return cur.replace(src=(new_src0, cur.src[1]))
      else:
        return None if assigned else acc.assign(cur)
    if cur.op in {Ops.MULACC, Ops.WMMA}:
      if (new_src2:=_inner_recurse(cur.src[2], assigned=False)) is not None:
        return cur.replace(src=(cur.src[0], cur.src[1], new_src2))
      else:
        return None if assigned else acc.assign(cur)
    assert cur == acc, (cur, assigned)
    return None
  return acc.assign(new_alu) if (new_alu:=_inner_recurse(alu, assigned=True)) is not None else None

def fuse_dot2acc(acc: UOp, a1: UOp, a2: UOp, b1: UOp, b2: UOp):
  a, b = UOp(Ops.VECTORIZE, dtypes.float16.vec(2), (a1, b1)), UOp(Ops.VECTORIZE, dtypes.float16.vec(2), (a2, b2))
  return UOp(Ops.WMMA, dtypes.float, (a, b, acc), arg=('V_DOT2_F32_F16', None, None, None, None, None, None, None))

def rewrite_cast(cast: UOp, val: UOp):
  if dtypes.is_int(cast.dtype) and dtypes.is_int(val.dtype) and cast.dtype.itemsize == val.dtype.itemsize: return val.bitcast(cast.dtype)
  return None

rdna_matcher = PatternMatcher([
  (UPat(Ops.CAST, name="x"), lambda x: x.src[0] if isinstance(x.dtype, PtrDType) else None),
  (UPat(Ops.CAST, dtype=dtypes.bool, name="x"), lambda x: x.src[0].ne(0)),
  (UPat(Ops.CAST, name='cast', src=(UPat.var('val'),)), rewrite_cast),
  (UPat.var('x', dtype=dtypes.bool).ne(UPat.var('y')), lambda x,y: x^y),
  (UPat.var('x', dtype=dtypes.bool)<UPat.var('y'), lambda x,y: (x^True)&y),
  # normal store
  (UPat.index(UPat((Ops.DEFINE_GLOBAL,Ops.DEFINE_LOCAL), name='buf'), UPat.var("idx")).store(UPat.var("val"), name='store'),
    lambda store, buf, idx, val: UOp(Ops.STORE, store.dtype, (buf, idx*buf.dtype.itemsize, val), 'global' if buf.op is Ops.DEFINE_GLOBAL else 'ds')),
  # gated store
  (UPat.index(UPat((Ops.DEFINE_GLOBAL,Ops.DEFINE_LOCAL), name='buf'), UPat.var("idx")).store(UPat.var("val"), UPat(Ops.IF, name='cond'), name='store'), # noqa: E501
    lambda store, buf, idx, val, cond: UOp(Ops.STORE, store.dtype, (buf, idx*buf.dtype.itemsize, val, cond), 'global' if buf.op is Ops.DEFINE_GLOBAL else 'ds')), # noqa: E501
  # normal load
  (UPat.index(UPat((Ops.DEFINE_GLOBAL,Ops.DEFINE_LOCAL), name='buf'), UPat.var("idx")).load(name='load'),
    lambda load, buf, idx: UOp(Ops.LOAD, load.dtype, (buf, idx*buf.dtype.itemsize), 'global' if buf.op is Ops.DEFINE_GLOBAL else 'ds')),
  # gated load
  (UPat.index(UPat((Ops.DEFINE_GLOBAL,Ops.DEFINE_LOCAL), name='buf'), UPat.var("idx")).load(UPat(Ops.IF, name='cond'), name='load'),
    lambda load, buf, idx, cond: UOp(Ops.LOAD, load.dtype, (buf, idx*buf.dtype.itemsize, cond), 'global' if buf.op is Ops.DEFINE_GLOBAL else 'ds')),
  # load with fallback
  (UPat.index(UPat((Ops.DEFINE_GLOBAL,Ops.DEFINE_LOCAL), name='buf'), UPat.var("idx")).load(UPat.var("fallback"), UPat(dtype=dtypes.bool, name='cond'), name='load'), # noqa: E501
    lambda load, buf, idx, fallback, cond: UOp(Ops.LOAD, load.dtype, (buf, idx*buf.dtype.itemsize, fallback, cond), 'global' if buf.op is Ops.DEFINE_GLOBAL else 'ds')), # noqa: E501
  # idiv
  (UPat.var("x", dtypes.ints)//UPat.cvar("c"), lambda x,c: x >> powers_of_two[c.arg] if c.arg in powers_of_two else None),
  # --- perf only --
  # this should be replaced by dot2acc
  # ((UPat.var('a', dtypes.float16)*UPat.var('b', dtypes.float16)).cast(dtypes.float32)+UPat.var('c', dtypes.float32),
  #   lambda a,b,c: a.cast(dtypes.float32)*b.cast(dtypes.float32)+c),
  # dot2acc
  (UPat.var('acc')+(UPat.var('a1', dtypes.float16)*UPat.var('a2', dtypes.float16)).cast(dtypes.float32) + \
                   (UPat.var('b1', dtypes.float16)*UPat.var('b2', dtypes.float16)).cast(dtypes.float32), fuse_dot2acc),
  # fma
  (UPat.var('a')*UPat.var('b')+UPat.var('c'), lambda a,b,c: a.alu(Ops.MULACC, b, c)),
  # gep followed by vectorize simplification for dot2acc with less copy overhead
  (UPat(Ops.VECTORIZE, dtypes.float16.vec(2), (UPat.var('x').gep(0), UPat.var('x').gep(1))),
    lambda x: UOp(Ops.GEP, dtypes.float16.vec(2), src=(x,), arg=(0,))),
  (UPat(Ops.VECTORIZE, dtypes.float16.vec(2), (UPat.var('x').gep(2), UPat.var('x').gep(3))),
    lambda x: UOp(Ops.GEP, dtypes.float16.vec(2), src=(x,), arg=(1,))),
  # fuse acc chains so codegen can do fmac instead of just fma
  (UPat(Ops.ASSIGN, name='assign', src=(UPat(Ops.DEFINE_ACC, name='acc'), UPat.var('alu'))), fuse_acc_chain),
])

def ls_valid(x: UOp): return x.arg in {'global', 'ds'}

rdna_spec = PatternMatcher([
  (UPat(Ops.STORE, src=(UPat((Ops.DEFINE_GLOBAL,Ops.DEFINE_LOCAL)), UPat.var(None), UPat.var(None)), name='x'), ls_valid),
  (UPat(Ops.STORE, src=(UPat((Ops.DEFINE_GLOBAL,Ops.DEFINE_LOCAL)), UPat.var(None), UPat.var(None), UPat(Ops.IF)), name='x'), ls_valid),
  (UPat(Ops.LOAD, src=(UPat((Ops.DEFINE_GLOBAL,Ops.DEFINE_LOCAL)), UPat.var(None)), name='x'), ls_valid),
  (UPat(Ops.LOAD, src=(UPat((Ops.DEFINE_GLOBAL,Ops.DEFINE_LOCAL)), UPat.var(None), UPat(Ops.IF)), name='x'), ls_valid),
  (UPat(Ops.LOAD, src=(UPat((Ops.DEFINE_GLOBAL,Ops.DEFINE_LOCAL)), UPat.var(None), UPat.var(None), UPat.var(None, dtype=dtypes.bool)), name='x'), ls_valid), # noqa: E501
  # (UPat(Ops.GEP, dtypes.float16.vec(2), (UPat.var(None, dtypes.float16.vec(4)),), name="gep"), lambda gep: gep.arg in {(0,),(1,)}),
])

RDNA_COMMENTS = DEBUG>=5

class RDNACodegen:
  def __init__(self):
    self.rops: list[RDNAOp] = []
    self.ref: dict[UOp, RDNAReg|RDNAImm] = {}
    self.constref: dict[UOp, RDNAImm] = {}
    self.kernarg_ptr = 0
    self.local_ptr = 0
    self.accs: dict[UOp, RDNAReg] = {}
    self.args: list[int] = []
    self.next_rangeid = 0
    self.ranges: dict[UOp, int] = {}
    self.requiredMaxThreadsPerBlock = 1

    if RDNA_COMMENTS:
      self.names: dict[UOp, str] = {}
      self.last_name: dict[str, int] = defaultdict(int)

  # Assuming DEFINE_GLOBALs are sorted
  def lower_define_global(self, u: UOp):
    sz = u.dtype.itemsize if not isinstance(u.dtype, PtrDType) else 8
    self.ref[u] = out = RDNAReg('s', sz)
    self.kernarg_ptr = round_up(self.kernarg_ptr, sz)
    self.rops.append(RDNAOp(RDNAOps.S_LOAD_B64, out, RDNAReg.prealloc('s', 8, 0), RDNAImm(dtypes.int32, self.kernarg_ptr)))
    self.args.append(sz)
    self.kernarg_ptr += sz

  def lower_define_local(self, u: UOp):
    assert isinstance(u.dtype, PtrDType)
    size = u.dtype.itemsize * u.dtype.size
    self.ref[u] = RDNAImm(dtypes.uint16, self.local_ptr)
    if RDNA_COMMENTS:
      self.rops.append(RDNAOp(RDNAOps.COMMENT, note=f'Allocated {self.names[u]} at LDS[{self.local_ptr}:{self.local_ptr+size}]'))
    self.local_ptr += size
    assert self.local_ptr <= 2**16, self.local_ptr

  def lower_special(self, u: UOp):
    assert u.dtype is dtypes.int32
    ns, idx = u.arg[0][0], int(u.arg[0][-1])
    assert ns in {'l', 'g'} and idx in {0,1,2}
    if ns == 'l':
      self.ref[u] = RDNAReg('v', 4)
      self.requiredMaxThreadsPerBlock *= u.arg[1]
      self.rops.append(RDNAOp(RDNAOps.V_BFE_U32, self.ref[u], RDNAReg.prealloc('v', 4, 0), RDNAImm(dtypes.int32, idx*10), RDNAImm(dtypes.int32, 10)))
    if ns == 'g':
      self.ref[u] = RDNAReg.prealloc('s', 4, (2+idx)*4)

  def lower_const(self, u: UOp):
    self.constref[u] = self.ref[u] = RDNAImm(u.dtype, u.arg)
    if getenv("PREMATERIALIZE", 0): self.maybe_lower_materialize(u)

  def rop_mov(self, dst: RDNAReg, src: RDNAValue):
    assert dst.size == (src.size if isinstance(src, RDNAReg) else src.dtype.itemsize)
    if dst.size in {2, 4} and (isinstance(src, RDNAReg) or src.dtype.count == 1):
      self.rops.append(RDNAOp(getattr(RDNAOps, f'V_MOV_B{dst.size*8}'), dst, src))
    elif dst.size > 4 and dst.size%4==0 and isinstance(src, RDNAReg):
      for i in range(0, dst.size, 4):
        self.rops.append(RDNAOp(RDNAOps.V_MOV_B32, dst.sub(4, i), src.sub(4, i)))
    else:
      raise AssertionError((dst, src))

  def maybe_lower_materialize(self, u: UOp):
    if isinstance(self.ref[u], RDNAReg): return self.ref[u]
    imm_ref = self.ref[u]
    self.ref[u] = out = RDNAReg('v', u.dtype.itemsize)
    self.rops.append(RDNAOp(getattr(RDNAOps, f'V_MOV_B{u.dtype.itemsize*8}'), out, imm_ref))
    return out

  def lower_gep(self, u: UOp):
    sreg = self.ref[u.src[0]]
    assert isinstance(sreg, RDNAReg)
    self.ref[u] = sreg.sub(u.dtype.itemsize, u.arg[0]*u.dtype.itemsize)

  def lower_vectorize(self, u: UOp):
    self.ref[u] = out = RDNAReg('v', u.dtype.itemsize)
    # happy zero copy assign path
    if all([isinstance(ref:=self.ref[src], RDNAReg) and ref.start is None and ref.base is None for src in u.src]) and len(set(u.src)) == len(u.src):
      if RDNA_COMMENTS: self.rops.append(RDNAOp(RDNAOps.COMMENT, note='Performed zero copy VECTORIZE'))
      for i,src in enumerate(u.src):
        ref = self.ref[src]
        assert isinstance(ref, RDNAReg) and ref.start is None and ref.base is None # redundant check just to be safe
        ref.base, ref.offset = out.base if out.base is not None else out, (i*src.dtype.itemsize) + (out.offset if out.offset is not None else 0)
      return
    # sad slow copy path
    for i,src in enumerate(u.src):
      self.rops.append(RDNAOp(getattr(RDNAOps, f'V_MOV_B{src.dtype.itemsize*8}'), out.sub(src.dtype.itemsize, i*src.dtype.itemsize), self.ref[src]))

  # This is very similar to vectorize
  def lower_assign(self, u: UOp):
    lhs, rhs = self.ref[u.src[0]], self.ref[u.src[1]]
    assert isinstance(lhs, RDNAReg)
    self.ref[u] = lhs
    # happy zero copy assign path
    if isinstance(rhs, RDNAReg) and rhs.start is None and rhs.base is None:
      if RDNA_COMMENTS: self.rops.append(RDNAOp(RDNAOps.COMMENT, note='Performed zero copy ASSIGN'))
      rhs.base, rhs.offset = lhs.base if lhs.base is not None else lhs, rhs.offset if rhs.offset is not None else 0
      return
    # sad slow copy path
    self.rop_mov(lhs, rhs)

  def lower_shl(self, u: UOp):
    self.ref[u] = out = RDNAReg('v', u.dtype.itemsize)
    self.rops.append(RDNAOp(RDNAOps.shl(u.dtype), out, *[self.ref[src] for src in reversed(u.src)]))

  def lower_shr(self, u: UOp):
    self.ref[u] = out = RDNAReg('v', u.dtype.itemsize)
    self.rops.append(RDNAOp(RDNAOps.shr(u.dtype), out, *[self.ref[src] for src in reversed(u.src)]))

  def lower_add(self, u: UOp):
    self.ref[u] = out = RDNAReg('v', u.dtype.itemsize)
    self.rops.append(RDNAOp(RDNAOps.add(u.dtype), out, *[self.ref[src] for src in u.src]))

  def lower_mul(self, u: UOp):
    self.ref[u] = out = RDNAReg('v', u.dtype.itemsize)
    self.rops.append(RDNAOp(RDNAOps.mul(u.dtype), out, *[self.ref[src] for src in u.src]))

  def lower_mulacc(self, u: UOp):
    self.ref[u] = out = RDNAReg('v', u.dtype.itemsize)
    self.rops.append(RDNAOp(RDNAOps.mulacc(u.dtype), out, *[self.ref[src] for src in u.src]))

  def lower_xor(self, u: UOp):
    if u.dtype == dtypes.bool:
      self.ref[u] = out = RDNAReg('s', 4)
      self.rops.append(RDNAOp(RDNAOps.S_XOR_B32, out, *[self.ref[src] for src in u.src]))
    else:
      self.ref[u] = out = RDNAReg('v', u.dtype.itemsize)
      self.rops.append(RDNAOp(getattr(RDNAOps, f'V_XOR_B{u.dtype.itemsize*8}'), out, *[self.ref[src] for src in u.src]))

  def lower_and(self, u: UOp):
    if u.dtype == dtypes.bool:
      # self.ref[u] = out = RDNAReg('s', 4)
      # self.rops.append(RDNAOp(RDNAOps.S_AND_B32, out, *[self.ref[src] for src in u.src]))
      raise AssertionError(u)
    # else:
    self.ref[u] = out = RDNAReg('v', u.dtype.itemsize)
    self.rops.append(RDNAOp(getattr(RDNAOps, f'V_AND_B{u.dtype.itemsize*8}'), out, *[self.ref[src] for src in u.src]))

  def lower_cast(self, u: UOp):
    assert len(u.src) == 1
    self.ref[u] = out = RDNAReg('v', u.dtype.itemsize)
    self.rops.append(RDNAOp(RDNAOps.cast(u.dtype, u.src[0].dtype), out, self.ref[u.src[0]]))

  def lower_bitcast(self, u: UOp):
    assert len(u.src) == 1
    self.ref[u] = self.ref[u.src[0]]

  def lower_max(self, u: UOp):
    self.ref[u] = out = RDNAReg('v', u.dtype.itemsize)
    self.rops.append(RDNAOp(RDNAOps.max(u.dtype), out, *[self.ref[src] for src in u.src]))

  def lower_min(self, u: UOp):
    self.ref[u] = out = RDNAReg('v', u.dtype.itemsize)
    self.rops.append(RDNAOp(RDNAOps.min(u.dtype), out, *[self.ref[src] for src in u.src]))

  def lower_cmplt(self, u: UOp):
    self.ref[u] = out = RDNAReg('s', 4)
    self.rops.append(RDNAOp(RDNAOps.cmplt(u.src[0].dtype), out, *[self.ref[src] for src in u.src]))

  def lower_cmpne(self, u: UOp):
    self.ref[u] = out = RDNAReg('s', 4)
    self.rops.append(RDNAOp(RDNAOps.cmpne(u.src[0].dtype), out, *[self.ref[src] for src in u.src]))

  def lower_where(self, u: UOp):
    self.ref[u] = out = RDNAReg('v', u.dtype.itemsize)
    if u.dtype.itemsize == 2:
      raise AssertionError(u)
    if u.dtype.itemsize == 4:
      self.rops.append(RDNAOp(RDNAOps.V_CNDMASK_B32, out, self.ref[u.src[2]], self.ref[u.src[1]], self.ref[u.src[0]]))
    else:
      raise AssertionError(u)

  def lower_wmma(self, u: UOp):
    self.ref[u] = out = RDNAReg('v', u.dtype.itemsize)
    wmma_name, _, _, _, _, _, _, _ = u.arg
    self.rops.append(RDNAOp(getattr(RDNAOps, wmma_name), out, *[self.ref[src] for src in u.src]))

  def lower_store(self, u: UOp):
    assert len(u.src) == 3 or (len(u.src) == 4 and u.src[3].op is Ops.IF), u
    op = getattr(RDNAOps, f'{u.arg.upper()}_STORE_B{u.src[2].dtype.itemsize*8}')
    self.rops.append(RDNAOp(op, self.maybe_lower_materialize(u.src[1]), self.maybe_lower_materialize(u.src[2]), self.ref[u.src[0]]))

  def lower_load(self, u: UOp):
    assert len(u.src) in {2, 3, 4}
    self.ref[u] = out = RDNAReg('v', u.dtype.itemsize)
    op = getattr(RDNAOps, f'{u.arg.upper()}_LOAD_B{u.dtype.itemsize*8}')
    if len(u.src) == 4:
      self.rop_mov(out, self.ref[u.src[2]])
      self.rops.append(RDNAOp(RDNAOps.S_AND_SAVEEXEC_B32, save:=RDNAReg('s', 4), self.ref[u.src[3]]))
    self.rops.append(RDNAOp(op, out, self.maybe_lower_materialize(u.src[1]), self.ref[u.src[0]]))
    if len(u.src) == 4:
      self.rops.append(RDNAOp(RDNAOps.S_MOV_B32, EXEC_LO, save))

  def lower_define_acc(self, u: UOp):
    self.ref[u] = out = RDNAReg('v', u.dtype.itemsize)
    self.accs[u.src[1]] = out
    self.rop_mov(out, self.ref[u.src[0]])

  def lower_range(self, u: UOp):
    assert u.dtype == dtypes.int
    self.ranges[u] = self.next_rangeid
    self.next_rangeid += 1
    self.ref[u] = out = RDNAReg('s', u.dtype.itemsize)
    self.rops.append(RDNAOp(RDNAOps.S_MOV_B32, out, self.constref[u.src[0]]))
    self.rops.append(RDNAOp(RDNAOps.LABEL, note=f'range{self.ranges[u]}'))
    pass

  def lower_endrange(self, u: UOp):
    rid = self.ref[u.src[0]]
    self.rops.append(RDNAOp(RDNAOps.S_ADD_I32, rid, rid, RDNAImm(dtypes.int32, 1)))
    self.rops.append(RDNAOp(RDNAOps.S_CMP_LT_I32, rid, self.ref[u.src[0].src[1]]))
    self.rops.append(RDNAOp(RDNAOps.S_CBRANCH_SCC1, LBL(f'range{self.ranges[u.src[0]]}')))

  def lower_barrier(self, u: UOp):
    self.rops.append(RDNAOp(RDNAOps.S_BARRIER))

  def lower_if(self, u: UOp):
    self.ref[u] = out = RDNAReg('s', 4)
    self.rops.append(RDNAOp(RDNAOps.S_AND_SAVEEXEC_B32, out, self.ref[u.src[0]]))

  def lower_endif(self, u: UOp):
    self.rops.append(RDNAOp(RDNAOps.S_MOV_B32, EXEC_LO, self.ref[u.src[0]]))

  def make_comment(self, u: UOp):
    if not RDNA_COMMENTS: return
    if u.op is Ops.SPECIAL:
      self.names[u] = u.arg[0]
    else:
      prefix = {Ops.RANGE: "ridx", Ops.DEFINE_GLOBAL: 'buf', Ops.DEFINE_LOCAL: "temp", Ops.VECTORIZE: "vec", Ops.INDEX: "bidx",
                Ops.DEFINE_ACC: "acc"}.get(u.op, u.op.name.lower())
      self.names[u] = f'{prefix}{self.last_name[prefix]}'
      self.last_name[prefix] += 1
    # TODO: cool names like aluX from cstyle
    def _fmt(u: UOp):
      if u.op is Ops.CONST and u not in self.names: return str(u.arg)
      if u.op is Ops.GEP: return f'{_fmt(u.src[0])}[{u.arg[0]}]'
      if u not in self.names: return '--' # define_acc refs range
      return self.names[u]
    self.rops.append(RDNAOp(RDNAOps.COMMENT, note=f"{self.names[u]} := UOp({u.op}, {u.dtype}{(', '+', '.join(map(_fmt, u.src))) if u.src else ''}{f', arg={repr(u.arg)}' if u.arg is not None else ''})")) # noqa: E501

  def lower(self, uops:list[UOp]):
    for i,u in enumerate(uops):
      if u.op not in {Ops.CONST, Ops.GEP}: self.make_comment(u)
      # if not hasattr(self, f'lower_{u.op.name.lower()}'):
      #   if DEBUG>=4: print(f'RDNA BREAK {u}')
      #   break
      getattr(self, f'lower_{u.op.name.lower()}')(u)

  def hazards(self):
    new_rops: list[RDNAOp] = []
    sloads_pending: bool = False
    lgkmqueue: deque[RDNAReg] = deque()
    lgkmqueue_set: set[RDNAReg] = set()
    vmqueue: deque[RDNAReg] = deque()
    vmqueue_set: set[RDNAReg] = set()
    for i,rop in enumerate(self.rops):
      lgkmcnt = len(lgkmqueue)
      vmcnt = len(vmqueue)
      for arg in rop.args:
        if not isinstance(arg, RDNAReg): continue
        arg, _ = basewalk(arg)
        while arg in lgkmqueue_set: lgkmqueue_set.remove(lgkmqueue.popleft())
        while arg in vmqueue_set: vmqueue_set.remove(vmqueue.popleft())
      # sloads return out of order, just nuke it
      if lgkmcnt != len(lgkmqueue) and sloads_pending:
        lgkmqueue, lgkmqueue_set = deque(), set()
      # reset when switching contextes (loops, ifs, etc)
      if rop.op in {RDNAOps.S_BARRIER, RDNAOps.LABEL, RDNAOps.S_CBRANCH_SCC1, RDNAOps.S_AND_SAVEEXEC_B32} or \
         any(arg == EXEC_LO or isinstance(arg, LBL) for arg in rop.args):
        lgkmqueue, lgkmqueue_set = deque(), set()
        vmqueue, vmqueue_set = deque(), set()
      # --
      wait_lgkmcnt = len(lgkmqueue) if len(lgkmqueue) != lgkmcnt else None
      wait_vmcnt = len(vmqueue) if len(vmqueue) != vmcnt else None
      if wait_lgkmcnt is not None or wait_vmcnt is not None:
        new_rops.append(RDNAOp(RDNAOps.S_WAITCNT, RDNAImm(dtypes.uint16, s_waitcnt_simm(wait_lgkmcnt, wait_vmcnt))))
      new_rops.append(rop)
      if rop.op.name.startswith('S_LOAD'):
        assert isinstance(rop.args[0], RDNAReg)
        dst, _ = basewalk(rop.args[0])
        assert dst not in lgkmqueue_set
        lgkmqueue.append(dst)
        lgkmqueue_set.add(dst)
        sloads_pending = True
      if rop.op.name.startswith('GLOBAL_LOAD') or rop.op.name.startswith('DS_LOAD'):
        assert isinstance(rop.args[0], RDNAReg)
        dst, _ = basewalk(rop.args[0])
        assert dst not in vmqueue_set
        vmqueue.append(dst)
        vmqueue_set.add(dst)
    self.rops = new_rops

  # Trash-tier register allocator. Somebody who knows what they're doing can probably write an llvm-comparable one in like 100 lines
  def allocate(self):
    # get info
    prealloc: dict[RDNAReg, None] = {}
    liveness: dict[RDNAReg, tuple[int, int]] = {}
    ctx_stack: list[str|None] = [None]
    ctx2start: dict[str|None, int] = {}
    ctx2end: dict[str|None, int] = {}
    reg2sctx: dict[RDNAReg, str|None] = {}
    reg2ectx: dict[RDNAReg, str|None] = {}
    for i,rop in enumerate(self.rops):
      if rop.op is RDNAOps.LABEL:
        ctx_stack.append(rop.note)
        ctx2start[rop.note] = i
      if rop.op is RDNAOps.S_CBRANCH_SCC1:
        lbl = rop.args[0]
        assert isinstance(lbl, LBL)
        assert ctx_stack.pop() == lbl.dst
        ctx2end[lbl.dst] = i
      read, written = [], []
      for arg,arginfo in zip(rop.args, rop.argsinfo()):
        if not isinstance(arg, RDNAReg): continue
        if arg.ns == 's' and arg.start is not None and arg.start > 105*4: continue # special registers like vcc, null, exec, ...
        if arginfo.read: read.append(arg)
        if arginfo.write: written.append(arg)
      for reg in read:
        reg, _ = basewalk(reg)
        if reg in liveness:
          liveness[reg] = (liveness[reg][0], i)
        else:
          assert reg.start is not None, reg
          prealloc[reg] = None
          reg2sctx[reg] = None
          liveness[reg] = (0, i)
        reg2ectx[reg] = ctx_stack[-1]
      for reg in written:
        reg, _ = basewalk(reg)
        if reg in liveness:
          liveness[reg] = (liveness[reg][0], i)
        else:
          reg2sctx[reg] = ctx_stack[-1]
          liveness[reg] = (i, i)
        reg2ectx[reg] = ctx_stack[-1]
    assert ctx_stack == [None]
    # extend liveness of register that peek into ranges (all accs and sometimes just normal regs do)
    liveness = {reg:(s,e) if reg2sctx[reg] == reg2ectx[reg] else (s,ctx2end[reg2ectx[reg]]) for reg,(s,e) in liveness.items()}
    start2reg: dict[int, list[RDNAReg]] = defaultdict(list)
    end2reg: dict[int, list[RDNAReg]] = defaultdict(list)
    for reg,(start,end) in liveness.items():
      start2reg[start].append(reg)
      end2reg[end].append(reg)
    if DEBUG>=6:
      print(f"RDNA PREALLOC:\n{chr(10).join([f'  {repr(reg)}' for reg in prealloc])}")
      for reg in prealloc: assert reg.render() in {'s[0:1]', 's2', 's3', 's4', 'v0', 'v1'}, reg
      print(f"RDNA LIVENESS:\n{chr(10).join([f'  {repr(reg):<96} | ({start:>6}) => ({end:>6})' for reg,(start,end) in liveness.items()])}")
    # allocate
    last_reg = {'v': 1*4, 's': 5*4}
    freelist: dict[tuple[str,int], list[int]] = defaultdict(list)
    for i in range(len(self.rops)):
      # free registers
      for reg in end2reg[i]:
        assert reg.base is None
        if reg.start is not None and not getenv("REGDEBUG", 0):
          freelist[(reg.ns, reg.size)].append(reg.start)
      for reg in start2reg[i]:
        assert reg.base is None
        if reg.start is not None:
          if not (reg in prealloc and i == 0):
            rop = self.rops[i]
            print(f'UNINITIALIZED READ, THIS IS A BUG, THE CODE IS INVALID: {i} {rop.op} {reg} {reg.base}')
          assert reg in prealloc and i == 0, reg
          continue
        if freelist[(reg.ns, reg.size)]:
          reg.start = freelist[(reg.ns, reg.size)].pop()
        else:
          unpad_last_reg, last_reg[reg.ns] = last_reg[reg.ns], round_up(last_reg[reg.ns], reg.size)
          while last_reg[reg.ns] > unpad_last_reg and not getenv("REGDEBUG", 0):
            for j in [16,8,4,2]:
              if unpad_last_reg % j == 0 and unpad_last_reg+j<=last_reg[reg.ns]:
                freelist[(reg.ns, j)].append(unpad_last_reg)
                unpad_last_reg += j
                break
          reg.start = last_reg[reg.ns]
          last_reg[reg.ns] += reg.size
    nmwaves = (self.requiredMaxThreadsPerBlock+31)//32
    # A wave can have a maximum of 106 SGPRs and 256 VGPRs, but there's a catch:
    # In CU mode all waves in a workgroup must run on the same CU, a single CU has two SIMD32s with each of them having a register file capable of
    # holding up to 1024 VGPRs, so if our workgroup has a ton of waves we might be limited by less than 256 VGPRs.
    # Ignoring this gives you undebuggable gpu hangs that also deadlock amdgpu kmodule after it tries to recover from the hang, fun stuff.
    if last_reg['s']>105*4 or last_reg['v']>min((2048//nmwaves), 255)*4:
      raise GracefulSkip(f"Out of registers: waves={nmwaves}, {', '.join([f'{k}={(v+3)//4}' for k,v in last_reg.items()])}")

  def fuse(self):
    for i,rop in enumerate(self.rops):
      # FMA => FMAC
      if rop.op in {RDNAOps.V_FMA_F32, RDNAOps.V_FMA_F16}:
        dst, a, b, c = rop.args
        assert isinstance(dst, RDNAReg)
        if isinstance(c, RDNAReg) and same_reg(dst, c):
          self.rops[i] = RDNAOp(getattr(RDNAOps, f'V_FMAC_F{dst.size*8}'), dst, a, b)
      # DOT2 => DOT2ACC
      if rop.op is RDNAOps.V_DOT2_F32_F16:
        dst, a, b, c = rop.args
        assert isinstance(dst, RDNAReg)
        if isinstance(a, RDNAReg) and isinstance(b, RDNAReg) and isinstance(c, RDNAReg) and same_reg(dst, c):
          self.rops[i] = RDNAOp(RDNAOps.V_DOT2ACC_F32_F16, dst, a, b)
    new_rops: list[RDNAOp] = []
    issued: set[RDNAOp] = set()
    for i,rop in enumerate(self.rops):
      if rop in issued: continue
      new_rops.append(rop)
      if rop.op is RDNAOps.V_MOV_B16 and i+1<len(self.rops) and (candidate:=self.rops[i+1]).op is RDNAOps.V_MOV_B16 and candidate not in issued:
        a_dst, a_src = rop.args
        b_dst, b_src = candidate.args
        if not (isinstance(a_dst, RDNAReg) and isinstance(a_src, RDNAReg)): continue
        if not (isinstance(b_dst, RDNAReg) and isinstance(b_src, RDNAReg)): continue
        assert a_dst.start is not None and b_dst.start is not None and a_src.start is not None and b_src.start is not None
        if a_dst.start%4!=0 or b_dst.start!=a_dst.start+2 or same_reg(a_dst, b_src): continue
        new_rops[-1] = RDNAOp(RDNAOps.V_PACK_B32_F16, RDNAReg('v', 4, a_dst.start), a_src, b_src)
        issued.add(candidate)
      if rop.op in DUAL_MAP and DUAL_MAP[rop.op] < 16 and (len(rop.args) < 3 or isinstance(rop.args[2], RDNAReg)):
        if DEBUG>=5: print(rop.render_text(), '=>')
        written = RDNARegSet(arg for arg,arginfo in zip(rop.args, rop.argsinfo()) if isinstance(arg, RDNAReg) and arginfo.write)
        read = RDNARegSet()
        for candidate in self.rops[i+1:]:
          if candidate.op in {RDNAOps.LABEL, RDNAOps.S_CBRANCH_SCC1, RDNAOps.S_AND_SAVEEXEC_B32}: break
          if any(arg == EXEC_LO or isinstance(arg, LBL) for arg in candidate.args): break
          if candidate in issued: continue
          can_pass = True
          newly_written: set[RDNAReg] = set()
          newly_read: set[RDNAReg] = set()
          for c_arg,c_arginfo in zip(candidate.args, candidate.argsinfo()):
            if not isinstance(c_arg, RDNAReg): continue
            if c_arginfo.write:
              newly_written.add(c_arg)
              if read.overlaps(c_arg): can_pass = False
            if c_arginfo.read:
              newly_read.add(c_arg)
              if written.overlaps(c_arg): can_pass = False
          written.update(newly_written)
          read.update(newly_read)
          if not can_pass or candidate.op not in DUAL_MAP: continue
          if not (len(rop.args) < 3 or isinstance(rop.args[2], RDNAReg)): continue
          for arga,arginfoa,argb,arginfob in zip(rop.args, rop.argsinfo(), candidate.args, candidate.argsinfo()):
            if isinstance(arga, RDNAImm) and isinstance(argb, RDNAImm):
              can_pass &= (arga.dtype, arga.val) == (argb.dtype, argb.val)
            if isinstance(arga, RDNAReg) and isinstance(argb, RDNAReg):
              can_pass &= arga.bank != argb.bank
              if arginfoa.write or arginfob.write: can_pass &= arga.even != argb.even
          if not can_pass: continue
          if DEBUG>=5: print('  ', candidate.render_text())
          issued.add(candidate)
          new_rops[-1] = RDNAOp(RDNAOps.V_DUAL, DUAL_OP(rop.op), *rop.args, DUAL_OP(candidate.op), *candidate.args)
          break
    self.rops = new_rops

  def hints(self):
    pass

class RDNAScheduler:
  def __init__(self, uops, local_children):
    self.cycle = 0
    self.prevlat = 0
    self.pipeline = defaultdict(list)
    self.in_flight = 0
    self.free = []
    self.priorities: dict = {}
    for u in reversed(uops):
      self.priorities[u] = sum([-10000 if u.op is Ops.LOAD else -500] + [self.priorities[x]//2 for x in local_children[u]])
    # stats
    self.stalls = 0
  def push(self, u):
    if u.op is Ops.SINK:
      heapq.heappush(self.free, (-9999999999999999, u.tuplize, u))
    else:
      self.pipeline[self.cycle+self.prevlat].append(u)
      self.in_flight += 1
  def pick(self):
    self.complete()
    while len(self.free) == 0:
      if DEBUG>=4: print(self.cycle, 'pipeline stall')
      self.cycle += 1 # pipeline stall
      self.stalls += 1
      self.complete()
    _,_,ret = heapq.heappop(self.free)
    if ret.op not in {Ops.SINK, Ops.ASSIGN, Ops.VECTORIZE, Ops.GEP}:
      if DEBUG>=4: print(self.cycle, ret.op)
      self.cycle += 1
      self.prevlat = 10 # todo: actual latencies
    return ret
  def complete(self):
    for u in self.pipeline[self.cycle]:
      self.in_flight -= 1
      if DEBUG>=4: print('PUSH', self.priorities[u], u.op)
      heapq.heappush(self.free, (self.priorities[u], u.tuplize, u))
    del self.pipeline[self.cycle]
  def avail(self):
    return len(self.free) + self.in_flight

class RDNARenderer(Renderer):
  device = "AMD"
  suffix = "RDNA"
  shared_max = 65536
  # https://gpuopen.com/learn/wmma_on_rdna3/
  # tensor_cores = [TensorCore(dims=(16,16,16), threads=32, elements_per_thread=(16,16,8), dtype_in=di, dtype_out=do,
  #   opts=("l0","l0","l0","l0","l1","u1","u1","u1"), swizzle=(((4,9,10,11,0),(1,2,3,5,6,7,8)), ((0,1,2,3,4),(9,10,11,5,6,7,8))))
  #   for di,do in [(dtypes.half,dtypes.float),(dtypes.half,dtypes.half)]]
  extra_matcher = rdna_matcher
  extra_spec = rdna_spec
  custom_scheduler = RDNAScheduler
  # this is used by get_late_rewrite_patterns for simplification
  code_for_op = {k: lambda: None for k in [Ops.AND, Ops.XOR, Ops.OR, Ops.ADD, Ops.SUB, Ops.MUL, Ops.CMPNE, Ops.CMPLT, Ops.SHL, Ops.SHR, Ops.WHERE]} # noqa: E501

  def __init__(self, arch:str, device="AMD"):
    self.arch, self.device = arch, device

  def __reduce__(self):
    return self.__class__, (self.arch, self.device)

  def render(self, name:str, uops:list[UOp]) -> str:
    codegen = RDNACodegen()
    codegen.lower(uops)
    codegen.allocate()
    codegen.fuse()
    codegen.hazards()
    codegen.hints()

    return f'''
.kernel {name}
.args {' '.join([str(sz) for sz in codegen.args])}{(chr(10)+f'.lds {codegen.local_ptr}') if codegen.local_ptr > 0 else ''}
.arch {self.arch}
{serialize_tinyasm(codegen.rops)}
s_nop 0:int
s_sendmsg 3:int
s_endpgm
'''

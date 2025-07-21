from __future__ import annotations
import functools, struct
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from tinygrad.helpers import i2u, unwrap, getbits, flatten
from tinygrad.dtype import DType, dtypes, DTYPES_DICT, INVERSE_DTYPES_DICT
from tinygrad.ops import ConstType

def basewalk(r: RDNAReg) -> tuple[RDNAReg, int]:
  offset = 0
  while r.base is not None: r, offset = r.base, offset+unwrap(r.offset)
  return r, offset

class RDNAReg:
  def __init__(self, ns:str, size:int, start:int|None=None, base:RDNAReg|None=None, offset:int|None=None, mod_neg:bool=False, mod_abs:bool=False):
    self.ns, self.size, self._start = ns, size, start
    self.base, self.offset = base, offset
    self.mod_neg, self.mod_abs = mod_neg, mod_abs
  @functools.cache
  @staticmethod
  def prealloc(*args, **kwargs): return RDNAReg(*args, **kwargs)
  @staticmethod
  def from_text(text: str):
    text = text.strip()
    if text[0] == '-': return RDNAReg.from_text(text[1:]).sub(mod_neg=True)
    if text[0] == '|' and text[-1] == '|': return RDNAReg.from_text(text[1:-1]).sub(mod_abs=True)
    if text.startswith('ttmp'):
      start_offset, ns, text = 108*4, 's', text[4:]
    elif text == 'exec_lo':
      start_offset, ns, text = 0, 's', '126'
    elif text == 'exec_hi':
      start_offset, ns, text = 0, 's', '127'
    elif text[0] in {'v', 's'}:
      start_offset, ns, text = 0, text[0], text[1:]
    else:
      raise AssertionError(text)
    if text[0] == '[' and text[-1] == ']':
      text = text[1:-1]
      str_start, str_end = text.split(':')
      start = int(str_start)*4
      size = (int(str_end)+1)*4 - start
    elif text.isnumeric():
      start = int(text)*4
      size = 4
    elif '.' in text:
      str_start, part = text.split('.')
      start = int(str_start)*4 + {'l': 0, 'h': 2}[part]
      size = 2
    else:
      raise RuntimeError(text)
    return RDNAReg(ns, size, start_offset+start)
  @property
  def start(self) -> int|None:
    if self.base is None:
      assert self.offset is None and not self.mod_neg and not self.mod_abs
      return self._start
    else:
      assert self._start is None and self.offset is not None
      return None if self.base.start is None else (self.base.start+self.offset)
  @start.setter
  def start(self, val:int|None):
    assert self.base is None
    self._start = val
  def sub(self, size:int|None=None, offset:int=0, mod_neg=False, mod_abs=False):
    # NOTE: zero copy assigns/vectorizes and mod_neg/mod_sub
    return RDNAReg(
      ns=self.ns,
      size=size if size is not None else self.size,
      base=self if self.base is None else self.base,
      offset=offset if self.base is None else (self.offset + offset), # type: ignore
      mod_neg=mod_neg if mod_abs else (self.mod_neg != mod_neg),
      mod_abs=self.mod_abs or mod_abs,
    )
  @property
  def even(self):
    assert self.start is not None and self.start%4==0
    return self.start%8==0
  @property
  def bank(self):
    assert self.start is not None and self.start%4==0
    return getbits(self.start//4, 0, 1)
  def render(self):
    assert self.start is not None
    if self.size == 2:
      rdr = f"{self.ns}{self.start//4}.{'l' if self.start%4 == 0 else 'h'}"
    elif self.size == 4:
      rdr = f"{self.ns}{self.start//4}"
    elif self.size > 4 and self.size%4 == 0:
      rdr = f"{self.ns}[{self.start//4}:{(self.start+self.size-4)//4}]"
    else:
      raise RuntimeError()
    if self.mod_abs: rdr = f'|{rdr}|'
    if self.mod_neg: rdr = f'-{rdr}'
    return rdr
  def __repr__(self):
    if self.start is not None: return self.render()
    return f"RDNAReg('{self.ns}', {self.start}, {self.size}, mod_abs={self.mod_abs}, mod_neg={self.mod_neg})"

class RDNARegSet:
  def __init__(self, it=None):
    self.added: dict[str, set[int]] = defaultdict(set)
    if it is not None: self.update(it)
  def add(self, other: RDNAReg):
    assert other.start is not None
    self.added[other.ns].update(range(other.start, other.start+other.size))
  def overlaps(self, other: RDNAReg):
    assert other.start is not None
    return len(set.intersection(set(range(other.start, other.start+other.size)), self.added[other.ns])) > 0
  def update(self, it):
    for i in it: self.add(i)

# mod_neg mod_abs don't count
def same_reg(a, b):
  return (a.size == b.size) and (a.start == b.start if a.start is not None else basewalk(a) == basewalk(b))

EXEC_LO = RDNAReg.prealloc('s', 4, 126*4)

class RDNAImm:
  def __init__(self, dtype: DType, val: ConstType):
    self.dtype, self.val = dtype, val
  @staticmethod
  def from_text(text):
    val, dt = text.split(':')
    dtype = DTYPES_DICT[dt]
    if val.startswith('0x'):
      base, val = 16, val[2:]
    else:
      base = 10
    return RDNAImm(dtype, float(val) if dtypes.is_float(dtype) else (int(val, base) if dtype != dtypes.bool else val=='True'))
  def encode_src9(self) -> tuple[int, bytes]:
    if dtypes.is_int(self.dtype) or dtypes.is_unsigned(self.dtype):
      assert isinstance(self.val, int)
      if self.val in range(0, 64+1):
        return 128+self.val, b''
      if self.val in range(-1, -16-1, -1):
        return 192-self.val, b''
    if self.dtype == dtypes.bool:
      assert self.val in {True, False}
      return 193 if self.val else 128, b''
    assert self.dtype.itemsize <= 4
    return 255, struct.pack(f'<{self.dtype.fmt}', self.val).ljust(4, b'\x00')
  def __repr__(self):
    return f'{repr(self.val)}:{INVERSE_DTYPES_DICT[self.dtype.name]}'

class LBL(RDNAImm):
  def __init__(self, name:str):
    self.dtype, self.val = dtypes.void, False
    self.dst = name
  def __repr__(self):
    return self.dst

class DUAL_OP(RDNAImm):
  def __init__(self, op:RDNAOps):
    self.dtype, self.val = dtypes.void, False
    self.op = op
  def __repr__(self):
    return repr(self.op)

def s_waitcnt_simm(lgkmcnt:int|None=None, vmcnt:int|None=None):
  if lgkmcnt is None: lgkmcnt = 0x3f
  else: assert lgkmcnt < 0x3f, lgkmcnt
  if vmcnt is None: vmcnt = 0x3f
  else: assert vmcnt < 0x3f, vmcnt
  return (vmcnt << 10 | lgkmcnt << 4 | 0x7)

RDNAValue = RDNAReg|RDNAImm

@dataclass(frozen=True)
class ArgInfo:
  read: bool
  write: bool
  dtype: DType|None
  def matches(self, val:RDNAValue):
    if self.dtype is None: return True
    return val.size == self.dtype.itemsize if isinstance(val, RDNAReg) else val.dtype == self.dtype

class ArgEncoding:
  info: ArgInfo
  def accepts(self, val:RDNAValue):
    raise RuntimeError('override')

class SIMM16(ArgEncoding):
  def __init__(self, dtype:DType|None):
    self.info = ArgInfo(True, False, dtype)
  def accepts(self, imm:RDNAValue):
    return isinstance(imm, RDNAImm) and self.info.matches(imm) and isinstance(imm.val, int) and 0<=imm.val<2**16
  def encode(self, imm:RDNAImm):
    assert isinstance(imm, RDNAImm) and self.info.matches(imm) and isinstance(imm.val, int) and 0<=imm.val<2**16
    return imm.val

class SIMM13(ArgEncoding):
  def __init__(self):
    self.info = ArgInfo(True, False, None)
  def accepts(self, imm:RDNAValue):
    return isinstance(imm, RDNAImm) and isinstance(imm.val, int) and -2**12<=imm.val<2**12
  def encode(self, imm:RDNAImm):
    assert isinstance(imm, RDNAImm) and isinstance(imm.val, int) and -2**12<=imm.val<2**12
    return imm.val

class BDST(SIMM16):
  def __init__(self): super().__init__(dtypes.int16)
  def accepts(self, imm:RDNAValue):
    return isinstance(imm, LBL)
  def encode(self, imm:RDNAImm):
    assert isinstance(imm, LBL)
    if isinstance(imm.val, bool): return 0
    assert isinstance(imm.val, int) and imm.val%4==0
    return i2u(16, imm.val//4-1)

class xREG(ArgEncoding):
  def __init__(self, ns:str, sz:int, info:ArgInfo):
    self.ns, self.sz, self.info = ns, sz, info
  def accepts(self, value:RDNAValue):
    if not isinstance(value, RDNAReg) or value.start is None: return False
    if self.sz == 2 and value.start//4 > 127: return False
    return value.ns == self.ns and value.size == self.sz
  def encode(self, value:RDNAReg) -> int:
    assert value.ns == self.ns and value.start is not None and value.size == self.sz
    if self.sz != 2: assert value.start%4 == 0
    return (value.start//4) + (128 if value.start%4!=0 else 0)

class xREG_opsel(ArgEncoding):
  def __init__(self, ns:str, sz:int, info:ArgInfo):
    self.ns, self.sz, self.info = ns, sz, info
  def accepts(self, value:RDNAValue):
    return isinstance(value, RDNAReg) and value.ns == self.ns and value.size == self.sz
  def encode(self, value:RDNAReg) -> tuple[int, bool]:
    assert value.ns == self.ns and value.start is not None and value.size == self.sz
    if self.sz != 2: assert value.start%4 == 0
    return value.start//4, value.start%4!=0

class SSRC8(xREG):
  def __init__(self, sz:int, dtype:DType|None):
    super().__init__('s', sz, ArgInfo(True, False, dtype))
  def accepts(self, value: RDNAValue):
    if isinstance(value, RDNAReg): return super().accepts(value)
    if isinstance(value, RDNAImm): return self.info.dtype == value.dtype if self.info.dtype is not None else self.sz == value.dtype.itemsize
    raise RuntimeError()
  def encode(self, value: RDNAValue):
    if isinstance(value, RDNAReg): return super().encode(value), b''
    if isinstance(value, RDNAImm): return value.encode_src9()

class SRC9(ArgEncoding):
  def __init__(self, sz:int, dtype:DType|None):
    self.sz, self.info = sz, ArgInfo(True, False, dtype)
  def accepts(self, value:RDNAValue):
    if isinstance(value, RDNAReg):
      assert value.start is not None
      if self.sz < 4 and value.start//4 > 127: return False
      if value.mod_neg or value.mod_abs: return False
      return self.sz == value.size
    if isinstance(value, RDNAImm):
      return self.sz == value.dtype.itemsize
    raise RuntimeError()
  def encode(self, value:RDNAValue) -> tuple[int, bytes]:
    assert self.accepts(value)
    if isinstance(value, RDNAImm):
      return value.encode_src9()
    if isinstance(value, RDNAReg):
      base = {'v': 256, 's': 0}[value.ns]
      assert value.start is not None
      return base + (value.start//4) + (128 if value.start%4!=0 else 0), b''

class SRC9_full(ArgEncoding):
  def __init__(self, sz:int, dtype:DType|None):
    self.sz, self.info = sz, ArgInfo(True, False, dtype)
  def accepts(self, value:RDNAValue):
    if isinstance(value, RDNAReg):
      return self.sz == value.size
    if isinstance(value, RDNAImm):
      return self.sz == value.dtype.itemsize
    raise RuntimeError()
  def encode(self, value:RDNAValue) -> tuple[int, bytes, bool, bool, bool]:
    if isinstance(value, RDNAImm):
      return value.encode_src9() + (False, False, False)
    if isinstance(value, RDNAReg):
      base = {'v': 256, 's': 0}[value.ns]
      assert value.start is not None
      return base + value.start//4, b'', value.start%4!=0, value.mod_neg, value.mod_abs
    raise RuntimeError()

class OpEncoding:
  def __init__(self, **kwargs):
    self.proto = dict(kwargs)
  def encode(self, *args) -> bytes|None:
    argl = list(args)
    kwargs = {}
    for key,enc in self.proto.items():
      if isinstance(enc, ArgEncoding):
        if not enc.accepts(argl[0]): return None
        if len(argl) == 0: return None
        kwargs[f'{key}_t'] = (argl.pop(0), enc)
      else:
        kwargs[key] = enc
    if len(argl) != 0: return None
    return self._encode(**kwargs)
  def argsinfo(self, *args) -> list[ArgInfo]:
    return [x.info for x in self.proto.values() if isinstance(x, ArgEncoding)]
  @staticmethod
  def _encode(*args, **kwargs):
    raise RuntimeError('override')

class FakeOp(OpEncoding):
  def __init__(self): pass
  def encode(self, *args): return b''
  @staticmethod
  def _encode(*args, **kwargs): return b''
  def argsinfo(self, *args) -> list[ArgInfo]: return []

class MultiOpEncoding(OpEncoding):
  def __init__(self, *encs:OpEncoding):
    self.encs = list(encs)
  def encode(self, *args) -> bytes|None:
    for enc in self.encs:
      if (encoded:=enc.encode(*args)) is not None: return encoded
    return None
  def argsinfo(self, *args) -> list[ArgInfo]:
    infos = [enc.argsinfo() for enc in self.encs]
    # assert all_same(infos), infos
    return sorted(infos, key=lambda x: len(x))[-1]

class SOPP(OpEncoding):
  @staticmethod
  def c(opcode:int, dtype:DType|None):
    return SOPP(opcode=opcode, simm16=SIMM16(dtype))
  @staticmethod
  def _encode(opcode:int, simm16_t:tuple[RDNAImm, SIMM16]|None=None):
    e_simm16 = simm16_t[1].encode(simm16_t[0]) if simm16_t else 0
    return struct.pack('<I', 0b101111111 << 23 | opcode << 16 | e_simm16)

class SOPK(OpEncoding):
  @staticmethod
  def c(opcode: int, sz: int):
    return SOPK(opcode=opcode, sdst=xREG('s', sz, ArgInfo(False, True, None)), simm16=SIMM16(None))
  @staticmethod
  def _encode(opcode:int, sdst_t:tuple[RDNAReg, xREG], simm16_t:tuple[RDNAImm, SIMM16]|None=None):
    e_sdst = sdst_t[1].encode(sdst_t[0])
    e_simm16 = simm16_t[1].encode(simm16_t[0]) if simm16_t else 0
    return struct.pack('<I', 0b1011 << 28 | opcode << 23 | e_sdst << 16 | e_simm16)

class SOP1(OpEncoding):
  @staticmethod
  def c(opcode: int, sz: int):
    return SOP1(opcode=opcode, sdst=xREG('s', sz, ArgInfo(False, True, None)), src0=SSRC8(sz, None))
  @staticmethod
  def _encode(opcode:int, sdst_t:tuple[RDNAReg, xREG]|None=None, src0_t:tuple[RDNAValue, SSRC8]|None=None):
    e_sdst = sdst_t[1].encode(sdst_t[0]) if sdst_t is not None else 0
    e_src0, e_src0_lit = src0_t[1].encode(src0_t[0]) if src0_t is not None else (0, b'')
    return struct.pack('<I', 0b101111101 << 23 | e_sdst << 16 | opcode << 8 | e_src0) + e_src0_lit

class SOP1i(OpEncoding):
  @staticmethod
  def c(opcode: int, sz: int):
    return SOP1i(opcode=opcode, sdst=xREG('s', sz, ArgInfo(False, True, None)), src0=SIMM16(None))
  @staticmethod
  def _encode(opcode:int, sdst_t:tuple[RDNAReg, xREG]|None=None, src0_t:tuple[RDNAImm, SIMM16]|None=None):
    e_sdst = sdst_t[1].encode(sdst_t[0]) if sdst_t is not None else 0
    e_src0 = src0_t[1].encode(src0_t[0]) if src0_t is not None else 0
    return struct.pack('<I', 0b101111101 << 23 | e_sdst << 16 | opcode << 8 | e_src0)

class SOP2(OpEncoding):
  @staticmethod
  def c(opcode: int, sz: int, dtype:DType|None=None):
    return SOP2(opcode=opcode, sdst=xREG('s', sz, ArgInfo(False, True, None)), src0=SSRC8(sz, dtype), src1=SSRC8(sz, dtype))
  @staticmethod
  def _encode(opcode:int, sdst_t:tuple[RDNAReg, xREG], src0_t:tuple[RDNAValue, SSRC8], src1_t:tuple[RDNAValue, SSRC8]):
    e_sdst = sdst_t[1].encode(sdst_t[0])
    e_src0, e_src0_lit = src0_t[1].encode(src0_t[0])
    e_src1, e_src1_lit = src1_t[1].encode(src1_t[0])
    lit = b''.join([e_src0_lit, e_src1_lit])
    assert len(lit) in {0, 4}
    return struct.pack('<I', 0b10 << 30 | opcode << 23 | e_sdst << 16 | e_src1 << 8 | e_src0) + lit

class SOPC(OpEncoding):
  @staticmethod
  def c(opcode: int, sz: int, dtype:DType|None=None):
    return SOPC(opcode=opcode, src0=SSRC8(sz, dtype), src1=SSRC8(sz, dtype))
  @staticmethod
  def _encode(opcode:int, src0_t:tuple[RDNAValue, SSRC8], src1_t:tuple[RDNAValue, SSRC8]):
    e_src0, e_src0_lit = src0_t[1].encode(src0_t[0])
    e_src1, e_src1_lit = src1_t[1].encode(src1_t[0])
    lit = b''.join([e_src0_lit, e_src1_lit])
    assert len(lit) in {0, 4}
    return struct.pack('<I', 0b101111110 << 23 | opcode << 16 | e_src1 << 8 | e_src0) + lit

class SMEM(OpEncoding):
  @staticmethod
  def c(opcode:int, sz:int):
    return SMEM(opcode=opcode, sdata=xREG('s', sz, ArgInfo(False, True, None)), sbase=xREG('s', 8, ArgInfo(True, False, None)), offset=SRC9(4, None))
  @staticmethod
  def _encode(opcode:int, sdata_t:tuple[RDNAReg, xREG], sbase_t:tuple[RDNAReg, xREG], offset_t:tuple[RDNAValue, SRC9]):
    sdata, _ = sdata_t
    sbase, _ = sbase_t
    offset, _ = offset_t
    assert sdata.start is not None and sbase.start is not None
    assert sbase.start%4 == 0 and sbase.start%8 == 0
    e_sdata = sdata.start//4
    e_sbase = sbase.start//8
    if isinstance(offset, RDNAImm):
      assert isinstance(offset.val, int)
      e_soffset = 124
      e_offset = i2u(21, offset.val)
    if isinstance(offset, RDNAReg):
      assert offset.start is not None and offset.ns == 's' and offset.size == 4
      e_soffset = offset.start//4
      e_offset = 0
    return struct.pack('<I', 0b111101 << 26 | opcode << 18 | e_sdata << 6 | e_sbase) + \
           struct.pack('<I', e_soffset << 25 | e_offset)

class VOP3(OpEncoding):
  @staticmethod
  def c(opcode:int, sz:int, nargs:int=3):
    assert nargs in {1,2,3}
    kwargs = {}
    if nargs >= 2: kwargs['src1'] = SRC9_full(sz, None)
    if nargs >= 3: kwargs['src2'] = SRC9_full(sz, None)
    return VOP3(opcode=opcode, vdst=xREG_opsel('v', sz, ArgInfo(False, True, None)), src0=SRC9_full(sz, None), **kwargs)
  @staticmethod
  def _encode(opcode:int, vdst_t:tuple[RDNAReg, xREG_opsel], src0_t:tuple[RDNAValue, SRC9_full], src1_t:tuple[RDNAValue, SRC9_full]|None=None,
              src2_t:tuple[RDNAValue, SRC9_full]|None=None):
    e_vdst, e_opsel_vdst = vdst_t[1].encode(vdst_t[0])
    nil = 0, b'', False, False, False
    e_src0, e_src0_lit, e_opsel_src0, e_neg_src0, e_abs_src0 = src0_t[1].encode(src0_t[0])
    e_src1, e_src1_lit, e_opsel_src1, e_neg_src1, e_abs_src1 = src1_t[1].encode(src1_t[0]) if src1_t is not None else nil
    e_src2, e_src2_lit, e_opsel_src2, e_neg_src2, e_abs_src2 = src2_t[1].encode(src2_t[0]) if src2_t is not None else nil
    lit = b''.join([e_src0_lit, e_src1_lit, e_src2_lit])
    assert len(lit) in {0, 4}
    e_opsel = e_opsel_vdst << 3 | e_opsel_src2 << 2 | e_opsel_src1 << 1 | e_opsel_src0
    e_neg = e_neg_src2 << 2 | e_neg_src1 << 1 | e_neg_src0
    e_abs = e_abs_src2 << 2 | e_abs_src1 << 1 | e_abs_src0
    return struct.pack('<I', 0b110101 << 26 | opcode << 16 | e_opsel << 11 | e_abs << 8 | e_vdst) + \
           struct.pack('<I', e_neg << 29 | e_src2 << 18 | e_src1 << 9 | e_src0) + lit

class VOP3P(OpEncoding):
  @staticmethod
  def c(opcode:int, sz:int, nargs:int=3):
    assert nargs in {1,2,3}
    kwargs = {}
    if nargs >= 2: kwargs['src1'] = SRC9_full(sz, None)
    if nargs >= 3: kwargs['src2'] = SRC9_full(sz, None)
    return VOP3P(opcode=opcode, vdst=xREG_opsel('v', sz, ArgInfo(False, True, None)), src0=SRC9_full(sz, None), **kwargs)
  @staticmethod
  def _encode(opcode:int, vdst_t:tuple[RDNAReg, xREG_opsel], src0_t:tuple[RDNAValue, SRC9_full], src1_t:tuple[RDNAValue, SRC9_full]|None=None,
              src2_t:tuple[RDNAValue, SRC9_full]|None=None):
    e_vdst, _ = vdst_t[1].encode(vdst_t[0])
    nil = 0, b'', False, False, False
    e_src0, e_src0_lit, _, _, _ = src0_t[1].encode(src0_t[0])
    e_src1, e_src1_lit, _, _, _ = src1_t[1].encode(src1_t[0]) if src1_t is not None else nil
    e_src2, e_src2_lit, _, _, _ = src2_t[1].encode(src2_t[0]) if src2_t is not None else nil
    lit = b''.join([e_src0_lit, e_src1_lit, e_src2_lit])
    assert len(lit) in {0, 4}
    return struct.pack('<I', 0b110011 << 26 | opcode << 16 | 0b1 << 14 | 0b000 << 11 | e_vdst) + \
           struct.pack('<I', 0b11 << 27 | e_src2 << 18 | e_src1 << 9 | e_src0) + lit

class VOP3SD(OpEncoding):
  @staticmethod
  def c(opcode:int, sz:int, ssz:int=4, nargs:int=3):
    assert nargs in {1,2,3}
    kwargs = {}
    if nargs >= 2: kwargs['src1'] = SRC9(sz, None)
    if nargs >= 3: kwargs['src2'] = SRC9(sz, None)
    return VOP3SD(opcode=opcode, vdst=xREG('v', sz, ArgInfo(False, True, None)), sdst=xREG('s', ssz, ArgInfo(False, True, None)), src0=SRC9(sz, None), **kwargs) # noqa: E501
  @staticmethod
  def _encode(opcode:int, vdst_t:tuple[RDNAReg, xREG], sdst_t:tuple[RDNAReg, xREG],
              src0_t:tuple[RDNAValue, SRC9], src1_t:tuple[RDNAValue, SRC9]|None=None, src2_t:tuple[RDNAValue, SRC9]|None=None):
    e_vdst = vdst_t[1].encode(vdst_t[0])
    e_sdst = sdst_t[1].encode(sdst_t[0])
    e_src0, e_src0_lit = src0_t[1].encode(src0_t[0])
    e_src1, e_src1_lit = src1_t[1].encode(src1_t[0]) if src1_t is not None else (0, b'')
    e_src2, e_src2_lit = src2_t[1].encode(src2_t[0]) if src2_t is not None else (0, b'')
    lit = b''.join([e_src0_lit, e_src1_lit, e_src2_lit])
    assert len(lit) in {0, 4}
    return struct.pack('<I', 0b110101 << 26 | opcode << 16 | e_sdst << 8 | e_vdst) + \
           struct.pack('<I', e_src2 << 18 | e_src1 << 9 | e_src0) + lit

class VOP1(OpEncoding):
  @staticmethod
  def c(opcode:int, sz:int, src0sz:int|None=None):
    if src0sz is None: src0sz = sz
    return MultiOpEncoding(VOP1(opcode=opcode,       vdst=xREG('v', sz, ArgInfo(False, True, None)),       src0=SRC9(src0sz, None)),
                           VOP3(opcode=opcode+0x180, vdst=xREG_opsel('v', sz, ArgInfo(False, True, None)), src0=SRC9_full(src0sz, None)))
  @staticmethod
  def _encode(opcode:int, vdst_t:tuple[RDNAReg, xREG], src0_t:tuple[RDNAValue, SRC9]):
    e_vdst = vdst_t[1].encode(vdst_t[0])
    e_src0, e_src0_lit = src0_t[1].encode(src0_t[0])
    return struct.pack('<I', 0b0111111 << 25 | e_vdst << 17 | opcode << 9 | e_src0) + e_src0_lit

class VOP2(OpEncoding):
  @staticmethod
  def c(opcode:int, sz:int, dstr:bool=False):
    return MultiOpEncoding(VOP2(opcode=opcode,       vdst=xREG('v', sz, ArgInfo(dstr, True, None)),       src0=SRC9(sz, None),      src1=xREG('v', sz, ArgInfo(True, False, None))), # noqa: E501
                           VOP3(opcode=opcode+0x100, vdst=xREG_opsel('v', sz, ArgInfo(dstr, True, None)), src0=SRC9_full(sz, None), src1=SRC9_full(sz, None))) # noqa: E501
  @staticmethod
  def _encode(opcode:int, vdst_t:tuple[RDNAReg, xREG], src0_t:tuple[RDNAValue, SRC9], src1_t:tuple[RDNAReg, xREG]):
    e_vdst = vdst_t[1].encode(vdst_t[0])
    e_src0, e_src0_lit = src0_t[1].encode(src0_t[0])
    e_src1 = src1_t[1].encode(src1_t[0])
    return struct.pack('<I', opcode << 25 | e_vdst << 17 | e_src1 << 9 | e_src0) + e_src0_lit

class DUAL(OpEncoding):
  def __init__(self): pass
  def parse(self, *_args) -> tuple[tuple[RDNAOps, list[ArgInfo], list[RDNAValue]], tuple[RDNAOps, list[ArgInfo], list[RDNAValue]]]:
    args = list(reversed(_args))
    # --
    opx = args.pop()
    argsx = []
    assert isinstance(opx, DUAL_OP), opx
    for _ in opx.op.value.argsinfo():
      argsx.append(args.pop())
    # ---
    opy = args.pop()
    argsy = []
    assert isinstance(opy, DUAL_OP), opy
    for _ in opy.op.value.argsinfo():
      argsy.append(args.pop())
    assert len(args) == 0, str(_args)
    return (opx.op, opx.op.value.argsinfo(), argsx), (opy.op, opy.op.value.argsinfo(), argsy)
  def render_text(self, *args) -> str:
    (a_op, _, a_args), (b_op, _, b_args) = self.parse(*args)
    return RDNAOp(a_op, *a_args).render_text() + ' :: ' + RDNAOp(b_op, *b_args).render_text()
  def encode(self, *args) -> bytes|None:
    (x_op, _, x_args), (y_op, _, y_args) = self.parse(*args)
    assert isinstance(x_args[0], RDNAReg) and isinstance(y_args[0], RDNAReg)
    e_dstx, e_dsty = xREG('v', x_args[0].size, ArgInfo(True, True, None)).encode(x_args[0]), xREG('v', y_args[0].size, ArgInfo(True, True, None)).encode(y_args[0]) # noqa: E501
    e_srcx0, e_srcx0_lit = SRC9(x_args[1].size if isinstance(x_args[1], RDNAReg) else x_args[1].dtype.itemsize, None).encode(x_args[1])
    e_srcy0, e_srcy0_lit = SRC9(y_args[1].size if isinstance(y_args[1], RDNAReg) else y_args[1].dtype.itemsize, None).encode(y_args[1])
    if len(x_args)>2:
      assert len(x_args) == 3 and isinstance(x_args[2], RDNAReg), x_args
      e_srcx1 = xREG('v', x_args[2].size, ArgInfo(True, False, None)).encode(x_args[2])
    else:
      e_srcx1 = 0
    if len(y_args)>2:
      assert len(y_args) == 3 and isinstance(y_args[2], RDNAReg), y_args
      e_srcy1 = xREG('v', y_args[2].size, ArgInfo(True, False, None)).encode(y_args[2])
    else:
      e_srcy1 = 0
    assert (e_dsty & 1) != (e_dstx & 1)
    e_dsty = e_dsty >> 1
    lit = b''
    if e_srcx0_lit != b'':
      assert e_srcy0_lit in {e_srcx0_lit, b''}
      lit = e_srcx0_lit
    if e_srcy0_lit != b'':
      assert e_srcx0_lit in {e_srcy0_lit, b''}
      lit = e_srcy0_lit
    return struct.pack('<I', 0b110010 << 26 | DUAL_MAP[x_op] << 22 | DUAL_MAP[y_op] << 17 | e_srcx1 << 9 | e_srcx0) + \
           struct.pack('<I', e_dstx << 24 | e_dsty << 17 | e_srcy1 << 9 | e_srcy0) + lit
  def argsinfo(self, *args) -> list[ArgInfo]:
    (_, a_info, _), (_, b_info, _) = self.parse(*args)
    return [ArgInfo(True, False, None)] + a_info + [ArgInfo(True, False, None)] + b_info

class VOPC(OpEncoding):
  @staticmethod
  def c(opcode:int, sz:int):
    return VOP3(opcode=opcode, vdst=xREG_opsel('s', 4, ArgInfo(False, True, None)), src0=SRC9_full(sz, None), src1=SRC9_full(sz, None))

class VMEMSeg(Enum):
  FLAT = 0
  SCRATCH = 1
  GLOBAL = 2

class VMEM(OpEncoding):
  @staticmethod
  def store(opcode:int, sz:int, seg:VMEMSeg):
    return MultiOpEncoding(VMEM(opcode=opcode, seg=seg.value, vaddr=xREG('v', 8, ArgInfo(True, False, None)), vdata=xREG_opsel('v', sz, ArgInfo(True, False, None))), # noqa: E501
                           VMEM(opcode=opcode, seg=seg.value, vaddr=xREG('v', 4, ArgInfo(True, False, None)), vdata=xREG_opsel('v', sz, ArgInfo(True, False, None)), saddr=xREG('s', 8, ArgInfo(True, False, None)))) # noqa: E501
  @staticmethod
  def load(opcode:int, sz:int, seg:VMEMSeg):
    return MultiOpEncoding(VMEM(opcode=opcode, seg=seg.value, vdst=xREG_opsel('v', sz, ArgInfo(False, True, None)), vaddr=xREG('v', 8, ArgInfo(True, False, None))), # noqa: E501
                           VMEM(opcode=opcode, seg=seg.value, vdst=xREG_opsel('v', sz, ArgInfo(False, True, None)), vaddr=xREG('v', 4, ArgInfo(True, False, None)), saddr=xREG('s', 8, ArgInfo(True, False, None)))) # noqa: E501
  @staticmethod
  def atomic(opcode:int, sz:int, seg:VMEMSeg):
    return VMEM(opcode=opcode, seg=seg.value, vdst=xREG_opsel('v', sz, ArgInfo(False, True, None)), vaddr=xREG('v', 4, ArgInfo(True, False, None)),
                vdata=xREG_opsel('v', sz, ArgInfo(True, False, None)), saddr=xREG('s', 8, ArgInfo(True, False, None)), glc=SIMM16(dtypes.bool))
  @staticmethod
  def _encode(opcode:int, seg:int, saddr_t:tuple[RDNAReg, xREG]|None=None, vaddr_t:tuple[RDNAReg, xREG]|None=None,
              vdst_t:tuple[RDNAReg, xREG_opsel]|None=None, vdata_t:tuple[RDNAReg, xREG_opsel]|None=None, glc_t:tuple[RDNAImm, SIMM16]|None=None):
    offset = 0
    e_saddr = saddr_t[1].encode(saddr_t[0]) if saddr_t is not None else 124
    e_vaddr = vaddr_t[1].encode(vaddr_t[0]) if vaddr_t is not None else 0
    e_vdst, _ = vdst_t[1].encode(vdst_t[0]) if vdst_t is not None else (0, False)
    e_vdata, _ = vdata_t[1].encode(vdata_t[0]) if vdata_t is not None else (0, False)
    e_glc = int(glc_t[0].val) if glc_t is not None else 0
    return struct.pack('<I', 0b110111 << 26 | opcode << 18 | seg << 16 | e_glc << 14 | offset) + \
           struct.pack('<I', e_vdst << 24 | e_saddr << 16 | e_vdata << 8 | e_vaddr)

class SDWA(OpEncoding):
  def __init__(self, lo:OpEncoding, hi:OpEncoding):
    self.lo, self.hi = lo, hi
  def encode(self, *args) -> bytes|None:
    for arg in args:
      if isinstance(arg, RDNAReg) and arg.size == 2 and arg.start is not None:
        return self.lo.encode(*args) if arg.start%4==0 else self.hi.encode(*args)
    raise RuntimeError()
  def argsinfo(self, *args) -> list[ArgInfo]:
    return self.lo.argsinfo()

class LDS(OpEncoding):
  @staticmethod
  def store(opcode:int, sz:int):
    return LDS(opcode=opcode, vaddr=xREG('v', 4, ArgInfo(True, False, None)), vdata=xREG('v', sz, ArgInfo(True, False, None)), offset=SIMM16(dtypes.uint16)) # noqa: E501
  @staticmethod
  def load(opcode:int, sz:int):
    return LDS(opcode=opcode, vdst=xREG('v', sz, ArgInfo(False, True, None)), vaddr=xREG('v', 4, ArgInfo(True, False, None)), offset=SIMM16(dtypes.uint16)) # noqa: E501
  @staticmethod
  def _encode(opcode:int, vaddr_t:tuple[RDNAReg, xREG]|None=None, vdst_t:tuple[RDNAReg, xREG]|None=None, vdata_t:tuple[RDNAReg, xREG]|None=None,
              offset_t: tuple[RDNAImm, SIMM16]|None=None):
    offset = offset_t[1].encode(offset_t[0]) if offset_t is not None else 0
    e_vaddr = vaddr_t[1].encode(vaddr_t[0]) if vaddr_t is not None else 0
    e_vdst = vdst_t[1].encode(vdst_t[0]) if vdst_t is not None else 0
    e_vdata = vdata_t[1].encode(vdata_t[0]) if vdata_t is not None else 0
    return struct.pack('<I', 0b110110 << 26 | opcode << 18 | offset) + \
           struct.pack('<I', e_vdst << 24 | e_vdata << 8 | e_vaddr)

def dt2rdt(dtype):
  if dtypes.is_float(dtype): return f'F{dtype.itemsize*8}'
  if dtypes.is_unsigned(dtype): return f'U{dtype.itemsize*8}'
  if dtypes.is_int(dtype): return f'I{dtype.itemsize*8}'
  raise RuntimeError(dtype)

class RDNAOps(Enum):
  @staticmethod
  def add(dtype: DType):
    return getattr(RDNAOps, f"V_ADD{'_NC' if dtypes.is_int(dtype) or dtypes.is_unsigned(dtype) else ''}_{dt2rdt(dtype)}")
  @staticmethod
  def mul(dtype: DType):
    if dtype in {dtypes.int32, dtypes.uint32}: return RDNAOps.V_MUL_LO_U32
    return getattr(RDNAOps, f"V_MUL_{dt2rdt(dtype)}")
  @staticmethod
  def mulacc(dtype: DType):
    if dtype == dtypes.int32: return RDNAOps.V_MAD_I32_I24
    if dtype == dtypes.uint32: return RDNAOps.V_MAD_U32_U24
    return getattr(RDNAOps, f"V_FMA_{dt2rdt(dtype)}")
  @staticmethod
  def cast(dst: DType, src: DType):
    return getattr(RDNAOps, f"V_CVT_{dt2rdt(dst)}_{dt2rdt(src)}")
  @staticmethod
  def min(dtype: DType):
    return getattr(RDNAOps, f"V_MIN_{dt2rdt(dtype)}")
  @staticmethod
  def max(dtype: DType):
    return getattr(RDNAOps, f"V_MAX_{dt2rdt(dtype)}")
  @staticmethod
  def cmplt(dtype: DType):
    return getattr(RDNAOps, f"V_CMP_LT_{dt2rdt(dtype)}")
  @staticmethod
  def cmpne(dtype: DType):
    return getattr(RDNAOps, f"V_CMP_NE_{dt2rdt(dtype)}")
  @staticmethod
  def shl(dtype: DType):
    assert dtypes.is_int(dtype) or dtypes.is_unsigned(dtype)
    return getattr(RDNAOps, f'V_LSHLREV_B{dtype.itemsize*8}')
  @staticmethod
  def shr(dtype: DType):
    if dtypes.is_unsigned(dtype):
      return getattr(RDNAOps, f'V_LSHRREV_B{dtype.itemsize*8}')
    if dtypes.is_int(dtype):
      return getattr(RDNAOps, f'V_ASHRREV_I{dtype.itemsize*8}')
    raise RuntimeError(dtype)
  # mostly real ops
  S_NOP = SOPP.c(0, None)
  S_CODE_END = SOPP(opcode=31)
  S_ENDPGM = SOPP(opcode=48)
  S_TRAP = SOPP.c(16, None)
  S_SENDMSG = SOPP.c(54, None)
  S_WAITCNT = SOPP.c(9, dtypes.uint16)
  S_CLAUSE = SOPP.c(5, dtypes.uint16)
  S_DELAY_ALU = SOPP.c(7, dtypes.uint16)
  S_GETREG_B32 = SOPK.c(17, 4)
  S_LOAD_B64 = SMEM.c(1, 8)
  S_MOV_B32 = SOP1.c(0, 4)
  S_ADD_U32 = SOP2.c(0, 4, dtypes.uint32)
  S_SUB_U32 = SOP2.c(1, 4, dtypes.uint32)
  S_ADD_I32 = SOP2.c(2, 4, dtypes.int32)
  S_SUB_I32 = SOP2.c(3, 4, dtypes.int32)
  S_ADDC_U32 = SOP2.c(4, 4, dtypes.uint32)
  S_SUBB_U32 = SOP2.c(5, 4, dtypes.uint32)
  S_AND_B32 = SOP2.c(22, 4)
  S_LSHR_B32 = SOP2.c(10, 4)
  S_RFE_B64 = SOP1(opcode=74, src0=SSRC8(8, None))
  S_SENDMSG_RTN_B32 = SOP1i.c(76, 4)
  S_SENDMSG_RTN_B64 = SOP1i.c(77, 8)
  S_XOR_B32 = SOP2.c(26, 4, dtypes.bool)
  S_AND_SAVEEXEC_B32 = SOP1.c(32, 4)
  S_CMP_LT_I32 = SOPC.c(4, 4, dtypes.int32)
  S_CMP_EQ_U32 = SOPC.c(6, 4, dtypes.uint32)
  # PC = PC + (SIMM16 * 4) + 4
  S_BRANCH = SOPP(opcode=32, simm16=BDST())
  S_CBRANCH_SCC1 = SOPP(opcode=34, simm16=BDST())
  S_BARRIER = SOPP(opcode=61)
  # --
  V_MOV_B16 = VOP1.c(28, 2)
  V_MOV_B32 = VOP1.c(1, 4)
  V_PACK_B32_F16 = VOP3(opcode=785, vdst=xREG_opsel('v', 4, ArgInfo(False, True, None)), src0=SRC9_full(2, None), src1=SRC9_full(2, None))
  # bitwise
  V_AND_B32 = VOP2.c(27, 4)
  V_XOR_B32 = VOP2.c(29, 4)
  V_LSHLREV_B32 = VOP2.c(24, 4)
  V_LSHRREV_B32 = VOP2.c(25, 4)
  V_ASHRREV_I32 = VOP2.c(26, 4)
  V_LSHLREV_B64 = VOP3(opcode=828, vdst=xREG_opsel('v', 8, ArgInfo(False, True, None)), src0=SRC9_full(4, None), src1=SRC9_full(8, None))
  # int add
  V_ADD_NC_U32 = VOP2.c(37, 4)
  V_ADD_NC_I32 = VOP3.c(806, 4, 2)
  V_ADD_CO_U32 = VOP3SD.c(768, 4, nargs=2)
  V_ADD_CO_CI_U32 = VOP3SD.c(288, 4)
  # int mul
  V_MUL_LO_U32 = VOP3.c(812, 4, 2)
  # int mad
  V_MAD_I32_I24 = VOP3.c(522, 4)
  V_MAD_U32_U24 = VOP3.c(523, 4)
  # int min
  V_MIN_I32 = VOP2.c(17, 4)
  # int max
  V_MAX_I32 = VOP2.c(18, 4)
  # fp add
  V_ADD_F16 = VOP2.c(50, 2)
  V_ADD_F32 = VOP2.c(3, 4)
  V_ADD_F64 = VOP3.c(807, 8, 2)
  # fp sub
  V_SUB_F32 = VOP2.c(4, 4)
  # fp mul
  V_MUL_F16 = VOP2.c(53, 2)
  V_MUL_F32 = VOP2.c(8, 4)
  # fp min
  V_MIN_F32 = VOP2.c(15, 4)
  # fp max
  V_MAX_F32 = VOP2.c(16, 4)
  # --
  V_BFE_U32 = VOP3.c(528, 4)
  # fp fma
  V_FMA_F16 = VOP3.c(584, 2)
  V_FMAC_F16 = VOP2.c(54, 2, dstr=True)
  V_FMA_F32 = VOP3.c(531, 4)
  V_FMAC_F32 = VOP2.c(43, 4, dstr=True)
  V_FMA_F64 = VOP3.c(532, 8)
  # fp wmma
  V_DOT2_F32_F16 = VOP3P.c(19, 4, 3)
  V_DOT2ACC_F32_F16 = VOP2.c(2, 4, dstr=True)
  V_DOT2_F32_BF16 = VOP3P.c(26, 4, 3)
  # fp cvt
  V_CVT_F16_F32 = VOP1.c(10, 2, 4)
  V_CVT_F32_F16 = VOP1.c(11, 4, 2)
  # --
  V_CMP_LT_I32 = VOPC.c(65, 4)
  V_CMP_NE_I32 = VOPC.c(69, 4)
  # --
  V_CNDMASK_B16 = VOP3(opcode=605, vdst=xREG_opsel('v', 2, ArgInfo(False, True, None)), src0=SRC9_full(2, None), src1=SRC9_full(2, None), src2=SRC9_full(4, None)) # noqa: E501
  V_CNDMASK_B32 = VOP3(opcode=257, vdst=xREG_opsel('v', 4, ArgInfo(False, True, None)), src0=SRC9_full(4, None), src1=SRC9_full(4, None), src2=SRC9_full(4, None)) # noqa: E501
  # --
  GLOBAL_LOAD_D16_B16 = VMEM.load(32, 2, VMEMSeg.GLOBAL)
  GLOBAL_LOAD_D16_HI_B16 = VMEM.load(35, 2, VMEMSeg.GLOBAL)
  GLOBAL_LOAD_B16 = SDWA(GLOBAL_LOAD_D16_B16, GLOBAL_LOAD_D16_HI_B16)
  GLOBAL_LOAD_B32 = VMEM.load(20, 4, VMEMSeg.GLOBAL)
  GLOBAL_LOAD_B64 = VMEM.load(21, 8, VMEMSeg.GLOBAL)
  GLOBAL_LOAD_B128 = VMEM.load(23, 16, VMEMSeg.GLOBAL)
  GLOBAL_STORE_D16_B16 = VMEM.store(25, 4, VMEMSeg.GLOBAL)
  GLOBAL_STORE_D16_HI_B16 = VMEM.store(37, 4, VMEMSeg.GLOBAL)
  GLOBAL_STORE_B16 = SDWA(GLOBAL_STORE_D16_B16, GLOBAL_STORE_D16_HI_B16)
  GLOBAL_STORE_B32 = VMEM.store(26, 4, VMEMSeg.GLOBAL)
  GLOBAL_STORE_B64 = VMEM.store(27, 8, VMEMSeg.GLOBAL)
  GLOBAL_STORE_B128 = VMEM.store(29, 16, VMEMSeg.GLOBAL)
  GLOBAL_ATOMIC_INC_U32 = VMEM.atomic(63, 4, VMEMSeg.GLOBAL)
  GLOBAL_LOAD_ADDTID_B32 = VMEM(opcode=40, seg=VMEMSeg.GLOBAL.value, vdst=xREG_opsel('v', 4, ArgInfo(False, True, None)),
                                saddr=xREG('s', 8, ArgInfo(True, False, None)))
  GLOBAL_STORE_ADDTID_B32 = VMEM(opcode=41, seg=VMEMSeg.GLOBAL.value, vdata=xREG_opsel('v', 4, ArgInfo(True, False, None)),
                                 saddr=xREG('s', 8, ArgInfo(True, False, None)))
  # --
  DS_STORE_B32 = LDS.store(13, 4)
  DS_STORE_B64 = LDS.store(77, 8)
  DS_STORE_B128 = LDS.store(223, 16)
  DS_LOAD_B32 = LDS.load(54, 4)
  DS_LOAD_B64 = LDS.load(118, 8)
  DS_LOAD_B128 = LDS.load(255, 16)
  # not real
  V_DUAL = DUAL()
  COMMENT = FakeOp()
  LABEL = FakeOp()

def is_valu(enc: OpEncoding):
  if isinstance(enc, VOP1): return True
  if isinstance(enc, VOP2): return True
  if isinstance(enc, VOP3): return True
  if isinstance(enc, VOP3SD): return True
  if isinstance(enc, MultiOpEncoding):
    return all([is_valu(x) for x in enc.encs])
  return False

DUAL_MAP = {RDNAOps.V_FMAC_F32: 0, RDNAOps.V_MUL_F32: 3, RDNAOps.V_ADD_F32: 4, RDNAOps.V_MOV_B32: 8, RDNAOps.V_CNDMASK_B32: 9, RDNAOps.V_MAX_F32: 10,
            RDNAOps.V_MIN_F32: 11, RDNAOps.V_DOT2ACC_F32_F16: 12, RDNAOps.V_ADD_NC_U32: 16, RDNAOps.V_LSHLREV_B32: 17, RDNAOps.V_AND_B32: 18}
RDNAOPS_VALU = set(filter(lambda x: is_valu(x.value), RDNAOps))
RDNAOPS_LOAD = set(filter(lambda x: 'LOAD' in x.name, RDNAOps))
RDNAOPS_STORE = set(filter(lambda x: 'STORE' in x.name, RDNAOps))
RDNAOPS_BRANCHES = {RDNAOps.S_CBRANCH_SCC1, RDNAOps.S_BRANCH}

class RDNAOp:
  def __init__(self, op: RDNAOps, *args: RDNAValue, note=''):
    self.op, self.args, self.note = op, list(args), note
  @staticmethod
  def from_text(text: str) -> RDNAOp:
    text = text.strip()
    if '::' in text:
      text_x, text_y = text.split('::')
      x, y = RDNAOp.from_text(text_x), RDNAOp.from_text(text_y)
      return RDNAOp(RDNAOps.V_DUAL, DUAL_OP(x.op), *x.args, DUAL_OP(y.op), *y.args)
    op_str_end: int|None = text.find(' ')
    if op_str_end == -1: op_str_end = None
    if text.startswith(';'): return RDNAOp(RDNAOps.COMMENT, note=text[op_str_end+1:] if op_str_end is not None else '')
    # print(text[op_str_end])
    if op_str_end is None and text[-1] == ':': return RDNAOp(RDNAOps.LABEL, note=text[:-1])
    op = getattr(RDNAOps, text[0:op_str_end].upper())
    args = [x.strip() for x in text[op_str_end+1:].split(',')] if op_str_end is not None else []
    # TODO: parse with ArgInfo
    if op in RDNAOPS_BRANCHES:
      assert len(args) == 1
      return RDNAOp(op, LBL(args[0]))
    return RDNAOp(op, *[RDNAImm.from_text(arg) if arg[0].isnumeric() or (arg[0] == '-' and arg[1].isnumeric()) or arg in {'True:bool', 'False:bool'} else RDNAReg.from_text(arg) for arg in args]) # noqa: E501
  def encode(self) -> bytes:
    encoded = self.op.value.encode(*self.args)
    if encoded is None: raise RuntimeError(f'failed to encode: {self.render_text()}')
    return encoded
  def render_text(self) -> str:
    if self.op is RDNAOps.COMMENT: return f'; {self.note}'
    if self.op is RDNAOps.LABEL: return f'{self.note}:'
    if self.op is RDNAOps.V_DUAL: return self.op.value.render_text(*self.args)
    return f"{self.op.name.lower()} {', '.join(map(str, self.args))}"
  def render_bytes(self) -> str:
    try:
      return f".byte {', '.join(map(hex, self.encode()))}"
    except Exception as e:
      print(f'Failed to encode: {self.render_text()}')
      raise e
  def argsinfo(self) -> list[ArgInfo]:
    return self.op.value.argsinfo(*self.args)

def serialize_tinyasm(rops: list[RDNAOp]):
  text = ''
  cpad = 0
  for rop in rops:
    text += ' '*cpad + rop.render_text() + '\n'
    if rop.op is RDNAOps.LABEL: cpad += 2
    if rop.op is RDNAOps.S_CBRANCH_SCC1: cpad -= 2
  return text[:-1]

# all llvm boilerplate is here to make debug output readable
def tinyasm_to_llvm(tasm: str) -> str:
  name = None
  arch = None
  # --
  kernarg_size = 0
  lds_size = 0
  # --
  rops: list[RDNAOp] = []
  last_reg = {'v': 0, 's': 0}
  # --
  for line in tasm.splitlines():
    components = line.split()
    if len(components) == 0: continue
    if components[0].startswith('.'):
      match components[0]:
        case '.kernel':
          assert name is None, 'redefining kernel'
          assert len(components) == 2, "kernel name can't contain whitespaces"
          name = components[1]
        case '.kernarg_size':
          assert kernarg_size == 0, 'redefining kernargs_size'
          assert len(components) == 2, "kernargs_size can't contain whitespaces"
          kernarg_size = int(components[1])
        case '.lds':
          assert lds_size == 0, 'redefining lds'
          assert len(components) == 2, "kernel lds can't contain whitespaces"
          lds_size = int(components[1])
        case '.arch':
          assert arch is None, 'redefining arch'
          assert len(components) == 2, "kernel arch can't contain whitespaces"
          arch = components[1]
        case _: raise RuntimeError(f'unknown tinyasm directive: {line}')
    else:
      rop = RDNAOp.from_text(line)
      for arg in rop.args:
        if not isinstance(arg, RDNAReg): continue
        assert arg.start is not None
        if not (arg.ns == 's' and arg.start > 105*4): # special registers like vcc, null, exec, ...
          last_reg[arg.ns] = max(arg.start+arg.size, last_reg[arg.ns])
      rops.append(rop)

  assert name is not None and arch is not None
  labels: dict[str, int] = {}
  relocs: dict[int, RDNAOp] = {}
  pos = 0
  for rop in rops:
    if rop.op is RDNAOps.LABEL: labels[rop.note] = pos
    if any(isinstance(arg, LBL) for arg in rop.args): relocs[pos] = rop
    pos += len(rop.encode())
  for pos,rop in relocs.items():
    for arg in rop.args:
      if not isinstance(arg, LBL): continue
      arg.val = labels[arg.dst] - pos
  print(last_reg)
  return f'''\
.text
.amdgcn_target "amdgcn-amd-amdhsa--{arch}"
.globl {name}
.p2align 8
.type {name},@function
{name}:
  {(chr(10) + '  ').join(flatten([['; ' + x.render_text(), x.render_bytes()] for x in rops]))}
  s_nop 0
  s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)
  s_endpgm
  .rept 64
  s_code_end
  .endr
.text
.Lfunc_end0:
  .size {name}, .Lfunc_end0-{name}

.rodata
.p2align 6
.amdhsa_kernel {name}
  .amdhsa_workgroup_processor_mode 0
  .amdhsa_wavefront_size32 1
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_kernarg_size {kernarg_size}
  .amdhsa_group_segment_fixed_size {lds_size}
  .amdhsa_system_sgpr_workgroup_id_x 1
  .amdhsa_system_sgpr_workgroup_id_y 1
  .amdhsa_system_sgpr_workgroup_id_z 1
  .amdhsa_system_vgpr_workitem_id 2
  .amdhsa_next_free_vgpr {(last_reg['v']+3)//4}
  .amdhsa_next_free_sgpr {(last_reg['s']+3)//4}
.end_amdhsa_kernel
'''

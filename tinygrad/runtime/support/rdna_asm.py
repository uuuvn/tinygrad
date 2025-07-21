from __future__ import annotations
import struct, functools
from tinygrad.device import Compiler, CompileError
from tinygrad.helpers import getbits, unwrap, flatten
from dataclasses import dataclass
from typing import Callable

class Register:
  def __init__(self, ns:str, size:int, start:int|None, base:Register|None=None, offset:int=0):
    self.ns, self.size, self._start, self.base, self.offset = ns, size, start, base, offset
  @staticmethod
  def parse(text:str) -> Register|None:
    text = text.strip()
    ns, text = text[:(ns_end:=next((i for i,c in enumerate(text) if not c.isalpha()), None))], text[ns_end:] if ns_end is not None else ''
    if ns not in {'s', 'v'}: return None
    if text.isnumeric():
      return Register(ns, 4, int(text) * 4)
    if text[-2:] in {'.l', '.h'}:
      return Register(ns, 2, int(text[:-2]) * 4 + (0 if text[-2:] == '.l' else 2))
    if (text[0], text[-1]) == ('[', ']') and text.count(':') == 1:
      start, end = text[1:-1].split(':')
      return Register(ns, (int(end) - int(start) + 1) * 4, int(start) * 4)
    return None
  @property
  def start(self) -> int|None:
    return self._start if self.base is None else None if self.base.start is None else self.base.start+self.offset
  def render(self):
    assert self.start is not None and self.start % self.size == 0, self.start
    return f"{self.ns}[{self.start//4}:{(self.start+self.size-4)//4}]" if self.size > 4 else \
           f"{self.ns}{self.start//4}" + ('' if self.size == 4 else '.l' if self.start%4 == 0 else '.h')
  def __repr__(self):
    return f"Register('{self.ns}', {self.size}, {self.start})"

Value = Register|float|int|bool|str

@dataclass(frozen=True)
class Encoding:
  name: str
  bitcount: int
  conditions: dict[str, Callable]
  bitfields: dict[str, tuple[int, int]]
  identifier_mask: int
  identifiers: set[int]
  def f2bf(self, fields:dict[str, int]) -> dict[str, int]|None:
    bitfields = fields.copy()
    if 'VOPD' in self.name:
      if bitfields['VDSTX'] & 1 == bitfields['VDSTY'] & 1: return None
      bitfields['VDSTY'] = bitfields['VDSTY'] >> 1
    return bitfields
  def check_condition(self, condition:str, bitfields:dict[str, int]) -> bool:
    return bool(self.conditions[condition]({'INST': {None: {**bitfields, None: 0}}, **{k:k for k in bitfields.keys()}}))
  def encode(self, bitfields:dict[str, int]) -> bytes:
    integer, matching_identifiers = 0, self.identifiers
    for k,v in bitfields.items():
      assert v & ~getbits(-1, 0, self.bitfields[k][1]-self.bitfields[k][0]) == 0, \
             f"Value {v} for bitfield {k} of {self.name} doesnt fit inside a mask: {getbits(-1, 0, self.bitfields[k][1]-self.bitfields[k][0])}"
      integer |= v << self.bitfields[k][0]
      if getbits(self.identifier_mask, *self.bitfields[k]):
        matching_identifiers = {ident for ident in matching_identifiers if getbits(ident, *self.bitfields[k]) == v}
    assert len(matching_identifiers) == 1, f"Ambiguous encoding, identifiers available: {[bin(x) for x in matching_identifiers]}"
    integer |= next(iter(matching_identifiers))
    assert self.bitcount % 32 == 0, f"Bitcount of {self.name} is {self.bitcount}, not a multiple of 32"
    return b''.join(struct.pack('<I', (integer >> i) & 0xFFFFFFFF) for i in range(0, self.bitcount, 32))

@dataclass(frozen=True)
class DataFormat:
  name: str
  datatype: str
  bitcount: int
  components: int

@dataclass(frozen=True)
class OperandType:
  name: str
  predef: dict[str, int]|None
  bitfields: dict[str, tuple[int, int]]|None

@dataclass(frozen=True)
class InstructionFlags:
  is_branch: bool
  is_cbranch: bool
  is_ibranch: bool

@dataclass(frozen=True)
class InstructionOperand:
  name: str|None
  dataformat: DataFormat
  operand_type: OperandType
  size: int
  input: bool
  output: bool
  implicit: bool
  def parse_value(self, text:str) -> Value|None:
    match self.operand_type.name:
      case 'OPR_VGPR': return reg if (reg:=Register.parse(text)) is not None and reg.ns == 'v' and reg.size*8 == self.size else None
      case _:
        print(f"WARNING: passthrough-parsing {repr(text)} ({self.name})")
        return text
  def render_value(self, val: Value) -> str:
    return unwrap(val.render()) if isinstance(val, Register) else repr(val)
  def arg2int(self, val: Value) -> int|None:
    print(self)
    return self.operand_type.predef.get(val, None) if self.operand_type.predef is not None else None

@dataclass(frozen=True)
class InstructionEncoding:
  encoding: Encoding
  condition: str
  opcode: int
  operands: list[InstructionOperand]
  @functools.cached_property
  def explicit_operands(self) -> list[InstructionOperand]: return [operand for operand in self.operands if not operand.implicit]
  def into_bitfields(self, *args) -> dict[str, int]|None:
    if len(args) != len(self.explicit_operands): return None
    bitfields = {}
    for operand, value in zip(self.explicit_operands, args):
      if (value_enc:=operand.arg2int(value)) is None: return None
      bitfields[unwrap(operand.name)] = value_enc
    return bitfields if self.encoding.check_condition(self.condition, bitfields) else None

@dataclass(frozen=True)
class InstructionOp:
  name: str
  flags: InstructionFlags
  encodings: list[InstructionEncoding]

class Instruction:
  def __init__(self, *args):
    self.opx, self.argx = args[0], args[1:(nxt:=next((i for i,x in enumerate(args[1:], start=1) if isinstance(x, InstructionOp)), None))]
    assert isinstance(self.opx, InstructionOp) and all(not isinstance(arg, InstructionOp) for arg in self.argx)
    self.opy, self.argy = args[nxt] if nxt is not None else None, args[nxt+1:] if nxt is not None else []
    assert self.opy is None or (isinstance(self.opy, InstructionOp) and all(not isinstance(arg, InstructionOp) for arg in self.argy))
  def encode(self) -> bytes:
    for encx in self.opx.encodings:
      if (fldx:=encx.into_bitfields(*self.argx)) is None: continue
      if self.opy is None:
        if (bf:=encx.encoding.f2bf({**fldx, 'OP': encx.opcode})) is None: continue
        return encx.encoding.encode(bf)
      for ency in self.opy.encodings:
        if encx.encoding != ency.encoding: continue
        if (fldy:=ency.into_bitfields(*self.argy)) is None or set.intersection(set(fldx.keys()), set(fldy.keys())): continue
        if (bf:=encx.encoding.f2bf({**fldx, **fldy, 'OPX': encx.opcode, 'OPY': ency.opcode})) is None: continue
        return encx.encoding.encode(bf)
    raise NotImplementedError(f"Couldn't encode: {self.pretty_print()}")
  def render_bytes(self) -> str: return f".byte {', '.join(map(hex, self.encode()))}"
  @staticmethod
  def from_text(mod, txt:str) -> Instruction:
    def _parse_part(part: str):
      op_str_end: int|None = pos if (pos:=part.find(' ')) != -1 else None
      iop = getattr(mod, part[:op_str_end].upper())
      argstrs = [x.strip() for x in part[op_str_end+1:].split(',')] if op_str_end is not None else []
      for enc in iop.encodings:
        if len(enc.explicit_operands) != len(argstrs): continue
        args = [opd.parse_value(astr) for opd,astr in zip(enc.explicit_operands, argstrs)]
        if any(arg is None for arg in args): continue
        return [iop, *args]
      raise NotImplementedError(part)
    return Instruction(*flatten([_parse_part(part.strip()) for part in txt.split('::')]))
  def pretty_print(self):
    return f"{self.opx.name.lower()} {', '.join(map(str, self.argx))}" + \
           (f" :: {self.opy.name.lower()} {', '.join(map(str, self.argy))}" if self.opy is not None else "")

from tinygrad.runtime.support.compiler_amd import compile_hip, amdgpu_disassemble

class RDNACompiler(Compiler):
  def __init__(self, arch:str):
    self.arch = arch
    super().__init__(f"compile_hip_{self.arch}")
  # all llvm boilerplate is here to make debug output readable
  def tinyasm_to_llvm(tasm: str) -> str:
    name = None
    arch = None
    # --
    kernarg_size = 0
    lds_size = 0
    # --
    rops: list[Instruction] = []
    last_reg = {'v': 128*4, 's': 100*4}
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
        rop = Instruction.from_text(line)
        # for arg in rop.args:
        #   if not isinstance(arg, RDNAReg): continue
        #   assert arg.start is not None
        #   if not (arg.ns == 's' and arg.start > 105*4): # special registers like vcc, null, exec, ...
        #     last_reg[arg.ns] = max(arg.start+arg.size, last_reg[arg.ns])
        rops.append(rop)

    assert name is not None and arch is not None
    return f'''\
  .text
  .amdgcn_target "amdgcn-amd-amdhsa--{arch}"
  .globl {name}
  .p2align 8
  .type {name},@function
  {name}:
    {(chr(10) + '  ').join(flatten([['; ' + x.pretty_print(), x.render_bytes()] for x in rops]))}
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
  def compile(self, src:str) -> bytes:
    try: return compile_hip(self.tinyasm_to_llvm(src), self.arch, True)
    except RuntimeError as e: raise CompileError(e) from e
  def disassemble(self, lib:bytes): amdgpu_disassemble(lib)

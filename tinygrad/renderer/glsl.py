from tinygrad.renderer.cstyle import CStyleLanguage
from tinygrad.dtype import DType, PtrDType, AddrSpace, dtypes
from tinygrad.uop.ops import UOp, UPat, Ops, GroupOp, PatternMatcher
from tinygrad.helpers import getbits, strip_parens

class GLSLRenderer(CStyleLanguage):
  device = "VK"
  suffix = "GLSL"

  global_max = (65535, 65535, 65535)
  local_max = (256, 256, 64)

  supports_float4 = False

  smem_prefix = "shared "
  barrier = "barrier();"

  infinity = "INF"
  nan = "NAN"

  string_rewrite = PatternMatcher([
    # bools are real
    (UPat(Ops.CONST, dtype=dtypes.bool, name="x"), lambda ctx,x: "true" if x.arg else "false"),
    (UPat(GroupOp.ALU, dtype=dtypes.bool, name="x"), lambda ctx,x: ctx.bool_code_for_op[x.op](
      *([strip_parens(ctx[v]) if v.op == x.op and x.op in {Ops.XOR, Ops.OR, Ops.AND} else ctx[v] for v in x.src]), x.dtype)
      if all([src.dtype == dtypes.bool for src in x.src]) else None),
    # no (u)ll suffix
    (UPat(Ops.CONST, dtype=dtypes.int64, name="x"),
      lambda ctx,x: f"int64_t(pack64(uvec2({getbits(x.arg, 0, 31)}u, {getbits(x.arg, 32, 63)}u)))"),
    (UPat(Ops.CONST, dtype=dtypes.uint64, name="x"),
      lambda ctx,x: f"pack64(uvec2({getbits(x.arg, 0, 31)}u, {getbits(x.arg, 32, 63)}u))"),
    # no pointers, bitcast is different
    (UPat(Ops.BITCAST, name="x"), lambda ctx,x: ctx.render_bitcast(x.dtype, x.src[0].dtype, ctx[x.src[0]])),
    # no pointers, buffers are arrays. also bool load/store is hacky
    (UPat(Ops.INDEX, src=(UPat.var("buf"), UPat.var('idx')), allow_any_len=True),
      lambda ctx,buf,idx: f"{ctx[buf]}{'.arr' if buf.dtype.addrspace is AddrSpace.GLOBAL else ''}[{strip_parens(ctx[idx])}]"),
    (UPat(Ops.LOAD, src=(UPat(Ops.INDEX, src=(UPat(), UPat(), UPat.var("gate"))).or_casted("bidx"), UPat.var("var")), dtype=dtypes.bool,
          allow_any_len=True),
      lambda ctx,bidx,var,gate: f"({ctx[gate]} ? {ctx.render_cast(dtypes.bool, ctx[bidx])} : {ctx[var]})"),
    (UPat(Ops.LOAD, src=(UPat(Ops.INDEX, src=(UPat(), UPat(), UPat.var("gate"))).or_casted("bidx"), UPat.var("var")), allow_any_len=True),
      lambda ctx,bidx,var,gate: f"({ctx[gate]} ? {ctx[bidx]} : {ctx[var]})"),
    (UPat(Ops.LOAD, src=(UPat.var('bidx'),), dtype=dtypes.bool, allow_any_len=True),
      lambda ctx,bidx: ctx.render_cast(dtypes.bool, ctx[bidx])),
    (UPat(Ops.LOAD, src=(UPat.var('bidx'),), allow_any_len=True),
      lambda ctx,bidx: ctx[bidx]),
    (UPat(Ops.STORE, src=(UPat.var('bidx'), UPat.var("var", dtypes.bool)), allow_any_len=True),
      lambda ctx,bidx,var: f"{ctx[bidx]} = {ctx.render_cast(dtypes.uint8, ctx[var])};"),
    (UPat(Ops.STORE, src=(UPat.var('bidx'), UPat.var("var")), allow_any_len=True),
      lambda ctx,bidx,var: f"{ctx[bidx]} = {strip_parens(ctx[var])};"),
    # more bool hacks
    (UPat(Ops.DEFINE_REG, name="x"),
      lambda ctx,x: f"{ctx.render_dtype(x.dtype.base, bool_hack=True)} {ctx[x]}[{x.dtype.size}];"),
    (UPat(Ops.DEFINE_LOCAL, name="x"),
      lambda ctx,x: f"{ctx.smem_align}{ctx.smem_prefix}{ctx.render_dtype(x.dtype.base, bool_hack=True)} {ctx[x]}[{x.dtype.size}];"),
  ]) + CStyleLanguage.string_rewrite

  code_for_workitem = {
    "g": lambda x: f"int32_t(gl_WorkGroupID.{'xyz'[int(x)]})",
    "l": lambda x: f"int32_t(gl_LocalInvocationID.{'xyz'[int(x)]})",
    "i": lambda x: f"int32_t(gl_GlobalInvocationID.{'xyz'[int(x)]})"
  }

  code_for_op: dict = {
    Ops.EXP2: lambda x,dtype: f"exp2({x})",
    Ops.LOG2: lambda x,dtype: f"log2({x})",
    Ops.SIN: lambda x,dtype: f"sin({x})",
    Ops.SQRT: lambda x,dtype: f"sqrt({x})",
    Ops.TRUNC: lambda x,dtype: f"trunc({x})",
    Ops.NEG: lambda x,dtype: f"-{x}",
    Ops.AND: lambda a,b,dtype: f"({a} & {b})",
    Ops.OR: lambda a,b,dtype: f"({a} | {b})",
    Ops.XOR: lambda a,b,dtype: f"({a} ^ {b})",
    Ops.ADD: lambda a,b,dtype: f"({a} + {b})",
    Ops.SUB: lambda a,b,dtype: f"({a} - {b})",
    Ops.MUL: lambda a,b,dtype: f"({a} * {b})",
    Ops.IDIV: lambda a,b,dtype: f"({a} / {b})",
    Ops.FDIV: lambda a,b,dtype: f"({a} / {b})",
    Ops.MOD: lambda a,b,dtype: f"({a} % {b})",
    Ops.SHL: lambda a,b,dtype: f"({a} << {b})",
    Ops.SHR: lambda a,b,dtype: f"({a} >> {b})",
    Ops.CMPLT: lambda a,b,dtype: f"({a} < {b})",
    Ops.CMPNE: lambda a,b,dtype: f"({a} != {b})",
    Ops.CMPEQ: lambda a,b,dtype: f"({a} == {b})",
    Ops.WHERE: lambda a,b,c,dtype: f"({a} ? {b} : {c})"
  }

  bool_code_for_op: dict = {
    **code_for_op,
    Ops.NEG: lambda x,dtype: f"!{x}",
    Ops.AND: lambda a,b,dtype: f"({a} && {b})",
    Ops.XOR: lambda a,b,dtype: f"({a} ^^ {b})",
    Ops.OR: lambda a,b,dtype: f"({a} || {b})",
    Ops.CMPLT: lambda a,b,dtype: f"(!{a} && {b})",
  }

  type_map = {
    dtypes.void: "void", dtypes.bool: "bool",
    dtypes.int8: "int8_t", dtypes.uint8: "uint8_t",
    dtypes.int16: "int16_t", dtypes.uint16: "uint16_t",
    dtypes.int32: "int32_t", dtypes.uint32: "uint32_t",
    dtypes.int64: "int64_t", dtypes.uint64: "uint64_t",
    dtypes.float16: "float16_t", dtypes.bfloat16: "bfloat16_t",
    dtypes.float32: "float32_t", dtypes.float64: "float64_t",
  }

  bitcast_map = {
    (dtypes.int16, dtypes.float16): "halfBitsToInt16", (dtypes.uint16, dtypes.float16): "halfBitsToUint16",
    (dtypes.float16, dtypes.int16): "int16BitsToHalf", (dtypes.float16, dtypes.uint16): "uint16BitsToHalf",
    (dtypes.int32, dtypes.float32): "floatBitsToInt", (dtypes.uint32, dtypes.float32): "floatBitsToUint",
    (dtypes.float32, dtypes.int32): "intBitsToFloat", (dtypes.float32, dtypes.uint32): "uintBitsToFloat",
    (dtypes.int64, dtypes.float64): "doubleBitsToInt64", (dtypes.uint64, dtypes.float64): "doubleBitsToUint64",
    (dtypes.float64, dtypes.int64): "int64BitsToDouble", (dtypes.float64, dtypes.uint64): "uint64BitsToDouble",
  }

  exts: list[str] = [
    "GL_EXT_buffer_reference",
    "GL_EXT_shader_explicit_arithmetic_types",
    *[f"GL_EXT_shader_explicit_arithmetic_types_{dt}" for dt in ["int8", "int16", "int32", "int64", "float16", "float32", "float64"]],
    "GL_EXT_bfloat16",
    "GL_EXT_shader_8bit_storage",
    "GL_EXT_shader_16bit_storage",
    "GL_EXT_scalar_block_layout",
  ]

  def render_kernel(self, function_name:str, kernel:list[str], args:list[tuple[str, tuple[DType, bool]]], uops:list[UOp], prefix=None) -> str:
    assert not prefix, f"empty prefix expected: {prefix}"

    local_usize = {int(u.arg[-1]): u.src[0] for u in uops if u.op is Ops.SPECIAL and u.arg[0] == "l"}
    assert all([u.op is Ops.CONST for u in local_usize.values()]), f"vulkan doesn't support symbolic local size: {local_usize=}"
    local_size = [local_usize[i].arg if i in local_usize else 1 for i in range(3)]

    buf_dts = {self.render_dtype(dt): self.render_dtype(dt.base, bool_hack=True) for _, (dt, _) in args if isinstance(dt, PtrDType)}

    k = (prefix or []).copy()
    k += ["#version 450", ""]
    k += [*[f"#extension {ext} : require" for ext in self.exts], ""]
    k += ["#define INF uintBitsToFloat(0x7f800000u)", "#define NAN uintBitsToFloat(0x7fc00000u)", ""]
    k += [f"layout(local_size_x = {local_size[0]}, local_size_y = {local_size[1]}, local_size_z = {local_size[2]}) in;", ""]
    k += [*(f"layout(buffer_reference, scalar) buffer {bt} {{ {it} arr[]; }};" for bt, it in buf_dts.items()), ""]
    k += [f"// SIZE: {sum([dt.itemsize if not isinstance(dt, PtrDType) else 8 for _, (dt, _) in args])}"]
    k += ["layout(set = 0, binding = 0, scalar) uniform args {"]
    k += [f"  {self.render_dtype(arg_dtype)} {arg_name};" for (arg_name, (arg_dtype, _)) in args]
    k += ["};", ""]
    if (shared:=[l[2:] for l in kernel if l.startswith("  shared ")]):
      k += [*shared, ""]
    k += ["void main() {", *[l for l in kernel if not l.startswith("  shared ")], "}"]
    return "\n".join(k)

  def render_dtype(self, dt:DType, mutable=True, bool_hack:bool=False) -> str:
    if isinstance(dt, PtrDType):
      return f"Buf{self.render_dtype(dt.base, mutable=mutable, bool_hack=bool_hack).capitalize()}"
    if dt.count > 1:
      raise NotImplementedError(f"vector {dt}")
    if dt == dtypes.bool and bool_hack:
      dt = dtypes.uint8
    return self.type_map[dt]

  def render_cast(self, dt:DType, val: str) -> str:
    return f"{self.render_dtype(dt)}({strip_parens(str(val))})"

  def render_bitcast(self, ddt:DType, sdt:DType, val: str) -> str:
    if dtypes.is_int(ddt) and dtypes.is_int(sdt):
      return self.render_cast(ddt, val)
    if (dtypes.is_int(ddt) and dtypes.is_float(sdt)) or (dtypes.is_float(ddt) and dtypes.is_int(sdt)):
      return f"{self.bitcast_map[(ddt, sdt)]}({strip_parens(val)})"
    raise NotImplementedError(f"bitcast {ddt} <= {sdt}")

import ctypes, ctypes.util

System = ctypes.CDLL(ctypes.util.find_library("System"))
NSConcreteGlobalBlock = ctypes.c_void_p.in_dll(System, "_NSConcreteGlobalBlock")

BLOCK_IS_GLOBAL = 1 << 28

# https://clang.llvm.org/docs/Block-ABI-Apple.html#high-level
def blockify(rtype, *argtypes):
  Invoke = ctypes.CFUNCTYPE(rtype, ctypes.c_void_p, *argtypes)

  class Descriptor(ctypes.Structure):
    _fields_ = [
      ('reserved', ctypes.c_ulong),
      ('size', ctypes.c_ulong),
    ]

  class Literal(ctypes.Structure):
    _fields_ = [
      ('isa', ctypes.c_void_p),
      ('flags', ctypes.c_int32),
      ('reserved', ctypes.c_int32),
      ('invoke', Invoke),
      ('descriptor', ctypes.POINTER(Descriptor)),
    ]

  def _blockify(fn):
    return ctypes.pointer(Literal(
      isa=NSConcreteGlobalBlock,
      flags=BLOCK_IS_GLOBAL,
      invoke=Invoke(lambda *args: fn(*args[1:])),
      descriptor=ctypes.pointer(Descriptor(
        size=ctypes.sizeof(Literal),
      )),
    ))

  return _blockify

import os, ctypes, ctypes.util, time
from tinygrad.helpers import Timing, to_mv, unwrap
from extra.macos.support.mtlcompiler import MTLRequestType, encode_request, decode_reply
from extra.macos.metal_compile.compile_common import test_compiler

with Timing("Loading MTLCompiler.framework"):
  compiler = ctypes.CDLL("/System/Library/PrivateFrameworks/MTLCompiler.framework/MTLCompiler")
  compiler.MTLCodeGenServiceCreate.restype = ctypes.c_void_p
  compiler.MTLCodeGenServiceBuildRequest.restype = ctypes.c_void_p

with Timing("Creating MTLCompilerObject"):
  cgs = compiler.MTLCodeGenServiceCreate(b"tinygrad")

def compile_direct(src:str, req_type:MTLRequestType=MTLRequestType.MTLBuildLibraryFromSourceToArchive) -> bytes:
  lib = None
  @ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_int32, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_char_p)
  def callback(blockptr, error, dataPtr, dataLen, errorMessage):
    nonlocal lib
    if error != 0: raise RuntimeError(errorMessage.decode())
    reply = bytes(to_mv(dataPtr, dataLen))
    lib = decode_reply(reply)
  data = encode_request(src)
  compiler.MTLCodeGenServiceBuildRequest(ctypes.c_void_p(cgs), None, req_type.value, data, len(data), ctypes.byref(callback, -0x10))
  return unwrap(lib)

if __name__ == '__main__': test_compiler(compile_direct)

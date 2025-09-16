import random, time, ctypes
from tinygrad.helpers import Timing
from tinygrad.runtime.ops_metal import MetalDevice, metal_src_to_library, msg, objc_instance
from extra.macos.metal_compile.compile_common import test_compiler
from typing import cast

dev = MetalDevice('METAL')

def compile_apple(src:str) -> bytes:
  library = metal_src_to_library(dev, src)
  library_contents = msg("libraryDataContents", objc_instance)(library)
  return ctypes.string_at(msg("bytes")(library_contents), cast(int, msg("length", ctypes.c_ulong)(library_contents)))

if __name__ == '__main__': test_compiler(compile_apple)

import random
from tinygrad.helpers import Timing
from typing import Callable

def get_test_kernel(n:float|None=None) -> str:
  if n is None: n = random.random()
  return f'''\
#include <metal_stdlib>
using namespace metal;
kernel void E_4_4(device float* data0, uint3 gid [[threadgroup_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]]) {{
  int lidx0 = lid.x; /* 4 */
  *((device float4*)((data0+(lidx0<<2)))) = float4({n}f,{n}f,{n}f,{n}f);
}}
'''
def test_compiler(compiler:Callable[[str], bytes]):
  src = get_test_kernel()
  with Timing(f"Compiling with {compiler.__name__}"):
    lib = compiler(src)
  # libraryDataContents returns something else instead of MTLB on conda and nix
  if compiler.__name__ != 'compile_apple':
    assert lib.startswith(b"MTLB") and lib.endswith(b"ENDT"), str(lib)
  return lib

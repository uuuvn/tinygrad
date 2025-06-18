from tinygrad import Tensor, TinyJit

@TinyJit
def x(x: Tensor) -> Tensor:
  return x.add(2.0).contiguous().realize().mul(2.0).contiguous().realize().sqrt().contiguous().realize().add(1.0).contiguous().realize()

for _ in range(10):
  x(Tensor.empty(512).contiguous().realize()).contiguous().realize()

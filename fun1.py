import random, numpy
from tinygrad import Tensor, dtypes

if __name__ == '__main__':
  dt = dtypes.float32
  x = Tensor.full((8192,), r:=random.random(), dtype=dt).contiguous().realize()
  print(r)
  print(x.numpy())
  numpy.testing.assert_allclose(x.numpy(), r, rtol=1e-3, atol=1e-3) # float16 tolerance
  y = (x + 1.0).contiguous().realize()
  print(r+1.0)
  print(y.numpy())
  numpy.testing.assert_allclose(y.numpy(), r+1.0, rtol=1e-3, atol=1e-3) # float16 tolerance

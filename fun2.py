import numpy as np
from tinygrad import Tensor, dtypes

if __name__ == '__main__':
  N = 4096
  dt, accdt = dtypes.float32, dtypes.float32
  an = (np.random.rand(N, N)-0.5)*10
  a = Tensor(an, dtype=dt).contiguous().realize()
  bn = (np.random.rand(N, N)-0.5)*10
  b = Tensor(bn, dtype=dt).contiguous().realize()
  cn = an @ bn
  c = a.matmul(b, acc_dtype=accdt).contiguous().realize()
  print(cn)
  print(c.numpy())


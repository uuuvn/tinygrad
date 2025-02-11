import numpy as np
from tinygrad import Tensor, dtypes

an = np.random.rand(1024, 1024)
a = Tensor(an, dtype=dtypes.float32).contiguous().realize()
bn = np.random.rand(1024, 1024)
b = Tensor(bn, dtype=dtypes.float32).contiguous().realize()
cn = np.random.rand(1024, 1024)
c = Tensor(bn, dtype=dtypes.float32).contiguous().realize()
xn = (an @ bn @ bn)
x = (a @ b @ c).contiguous().realize()
print(xn)
print(x.numpy())
np.testing.assert_allclose(xn, x.numpy(), atol=1e-5, rtol=1e-5)

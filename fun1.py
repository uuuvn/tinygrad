import numpy as np
from tinygrad import Tensor, dtypes

an = np.random.rand(1024, 1024)
a = Tensor(an, dtype=dtypes.float32).contiguous().realize()
bn = np.random.rand(1024, 1024)
b = Tensor(bn, dtype=dtypes.float32).contiguous().realize()
cn = an @ bn
c = (a @ b).contiguous().realize()
print(c.numpy())
print(cn)
np.testing.assert_allclose(cn, c.numpy(), atol=1e-5, rtol=1e-5)

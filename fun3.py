# import numpy as np
from tinygrad import Tensor, dtypes

Tensor.full((1*1024*1024*1024//4,), 1.0, dtype=dtypes.float32).contiguous().realize()

import random
from tinygrad import Tensor

a = Tensor([1.0]*4).contiguous().realize()
b = Tensor([1.0]*4).contiguous().realize()

n = random.random()
print(((a * b) + n).contiguous().realize())

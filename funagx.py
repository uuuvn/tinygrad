from tinygrad import Tensor

a = Tensor.full((64,), 1.0).contiguous().realize()
print(a.numpy())
b = Tensor.full((64,), 2.0).contiguous().realize()
print(b.numpy())
print(Tensor.arange(65536).contiguous().realize().numpy())
c = Tensor.cat(a.shrink(((0, 32),)), b.shrink(((0, 32),))).contiguous().realize()
print(c.numpy())

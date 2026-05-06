from tinygrad import Tensor, Variable

x = Variable("x", 0, 10).bind(1)
y = Variable("y", 0, 10).bind(3)
a = Tensor.full((4,), 5).contiguous().realize()
b = Tensor.full((4,), 21).contiguous().realize()
c = a + b + x + y

print(c.numpy())

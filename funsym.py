from tinygrad import Tensor, Variable

x = Variable("x", 0, 15).bind(2)
print(Tensor.arange(0, 16).contiguous().realize()[x].contiguous().realize().numpy())

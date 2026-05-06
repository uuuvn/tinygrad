from tinygrad import Tensor

def wtf(n):
  a = Tensor.full((n,), 1.0).contiguous().realize()
  print(a.numpy())
  b = Tensor.full((n,), 2.0).contiguous().realize()
  print(b.numpy())
  c = Tensor.cat(a, b).contiguous().realize()
  print(c.numpy())

if __name__ == "__main__":
  wtf(8)

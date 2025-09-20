import math, numpy as np
from tinygrad import Tensor, dtypes, TinyJit, Device, GlobalCounters, nn
from tinygrad.helpers import getenv, trange

class Layer:
  def __init__(self, d_model:int):
    bound = 1 / math.sqrt(d_model)
    self.weight = Tensor.uniform(d_model, d_model, low=-bound, high=bound)
    self.bias = Tensor.uniform(d_model, low=-bound, high=bound)
  def __call__(self, h:Tensor):
    return h + h.linear(self.weight.transpose(), self.bias)#.relu()

class Model:
  def __init__(self, d_model:int, n_layers:int):
    self.layers = [Layer(d_model=d_model) for _ in range(n_layers)]
  def __call__(self, h:Tensor):
    for layer in self.layers: h = layer(h)
    return h

if __name__ == "__main__":
  Tensor.manual_seed(31337)
  np.random.seed(31337)
  GPUS = tuple(f"{Device.DEFAULT}:{i}" for i in range(4))
  D_MODEL = getenv("D_MODEL", 512)
  N_LAYERS = getenv("N_LAYERS", 1)
  BS = getenv("BS", 64)

  model = Model(D_MODEL, N_LAYERS)
  opt = nn.optim.Adam(nn.state.get_parameters(model))
  nn.state.label_parameters({"model": model, "opt": opt})

  for x in nn.state.get_parameters(opt):
    x.shard_(GPUS, axis=0 if getenv("FSDP", 1) and x.numel() > 1 else None)

  # @TinyJit
  def train_step(batch) -> Tensor:
    with Tensor.train():
      opt.zero_grad()
      loss = (model(batch) - batch).pow(2).sum().backward()
      Tensor.realize(*opt.schedule_step(), loss)
      return loss

  for i in (t:=trange(getenv("STEPS", 2))):
    GlobalCounters.reset()
    loss = train_step(Tensor(np.random.uniform(size=(BS, D_MODEL)), dtype=dtypes.float32).shard(GPUS, axis=0).contiguous().realize())
    t.set_description(f"loss: {loss.item():6.2f} mem: {GlobalCounters.mem_used/1e9} gb")

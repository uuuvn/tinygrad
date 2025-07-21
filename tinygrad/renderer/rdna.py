from tinygrad.ops import UOp
from tinygrad.renderer import Renderer

class RDNARenderer(Renderer):
  device = "AMD"
  suffix= "RDMA"

  shared_max = 65536
  global_max = (2147483647, 65535, 65535)

  # pre_matcher: Optional[PatternMatcher] = None
  # extra_matcher: Optional[PatternMatcher] = None
  # code_for_op: dict[Ops, Callable] = {}

  def __init__(self, arch:str): self.arch = arch
  def __reduce__(self): return self.__class__, (self.arch,)

  def render(self, uops:list[UOp]) -> str:
    raise NotImplementedError("needs a renderer")

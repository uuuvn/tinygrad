import ctypes, struct, fcntl

LEAK = []

def msg(t: int, b: bytes):
  resp = fcntl.ioctl(0xdead, 0xc0de, struct.pack('<Q', t) + b)
  assert (resp_code:=struct.unpack('<Q', resp[:8])[0]) == 0, f'Non-zero response code {resp_code:#x}'
  return resp[8:]
def set_enabled(id, enabled):
  return bool(struct.unpack('<Q', msg(0x2, struct.pack('<2Q', id, int(enabled)))[:8])[0])
def backtrace():
  msg(0x3, b"")
def hook_address(s, d):
  LEAK.append(d)
  return struct.unpack('<Q', msg(0x10, struct.pack('<2Q', ctypes.cast(s, ctypes.c_void_p).value, ctypes.cast(d, ctypes.c_void_p).value))[:8])[0]
def hook_symbol(s, d):
  LEAK.append(d)
  return struct.unpack('<Q', msg(0x11, struct.pack('<Q', ctypes.cast(d, ctypes.c_void_p).value) + s.encode('ascii'))[:8])[0]

def override(*t_args, **t_kwargs):
  ftype = ctypes.CFUNCTYPE(*t_args, **t_kwargs)
  def _override(fn):
    bid = hook_symbol(fn.__name__, ftype(fn))
    def _original(*args, **kwargs):
      assert (old_enabled:=set_enabled(bid, False))
      r = ctypes.cast(ctypes.CDLL(None)[fn.__name__], ftype)(*args, **kwargs)
      assert not set_enabled(bid, old_enabled)
      return r
    return _original
  return _override

assert (r:=msg(1, b'ping')) == b'pong', f"Expected b'pong', got {r}"

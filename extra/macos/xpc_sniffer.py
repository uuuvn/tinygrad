import fnmatch, ctypes, extra.macos.support.xpc as pyxpc
from tinygrad.helpers import getenv
from extra.macos.tinyhook.lldb_client import override, backtrace

XPC_TRACE = getenv("XPC_TRACE", "com.apple.MTLCompilerService")
XPC_BT = getenv("XPC_BT", False)

INTERESTING_CONNECTIONS = set()

def is_interesting(name:str): return fnmatch.fnmatch(name, XPC_TRACE)

@override(ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p)
def xpc_connection_create(name: bytes, queue) -> ctypes.c_void_p:
  ptr = xpc_connection_create(name, queue)
  if is_interesting(name.decode()):
    if XPC_BT: backtrace()
    print(f'Opened connection to {name.decode()} ({ptr:#x})')
    INTERESTING_CONNECTIONS.add(ptr)
  return ptr

@override(ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p, ctypes.c_uint64)
def xpc_connection_create_mach_service(name: bytes, queue, flags) -> ctypes.c_void_p:
  ptr = xpc_connection_create_mach_service(name, queue, flags)
  if is_interesting(name.decode()):
    if XPC_BT: backtrace()
    print(f'Opened mach connection to {name.decode()} ({ptr:#x})')
    INTERESTING_CONNECTIONS.add(ptr)
  return ptr

@override(None, ctypes.c_void_p, ctypes.c_void_p)
def xpc_connection_send_message(conn, msg):
  if conn in INTERESTING_CONNECTIONS:
    if XPC_BT: backtrace()
    print(f'Sent message to ({conn:#x}):\n{pyxpc.xpc2py(msg)}')
  xpc_connection_send_message(conn, msg)

@override(None, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p)
def xpc_connection_send_message_with_reply(conn, msg, queue, handler):
  if conn in INTERESTING_CONNECTIONS:
    if XPC_BT: backtrace()
    print(f'Sent message (with reply, async) to ({conn:#x}):\n{pyxpc.xpc2py(msg)}')
  xpc_connection_send_message_with_reply(conn, msg, queue, handler)

@override(ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p)
def xpc_connection_send_message_with_reply_sync(conn, msg):
  if conn in INTERESTING_CONNECTIONS:
    if XPC_BT: backtrace()
    print(f'Sent message (with reply, sync) to ({conn:#x}):\n{pyxpc.xpc2py(msg)}')
  resp = xpc_connection_send_message_with_reply_sync(conn, msg)
  if conn in INTERESTING_CONNECTIONS:
    print(f'Got reply in ({conn:#x}):\n{pyxpc.xpc2py(resp)}')
  return resp

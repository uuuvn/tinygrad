import os, ctypes, ctypes.util, time
from tinygrad.helpers import Timing, unwrap
from extra.macos.support.block import blockify
from extra.macos.support.xpc import xpc_connection_create, xpc_connection_set_event_handler, xpc_connection_resume, \
                                    xpc_connection_send_message_with_reply_sync, py2xpc, xpc2py
from extra.macos.support.mtlcompiler import MTLRequestType, encode_request, decode_reply
from extra.macos.metal_compile.compile_common import test_compiler

def event_handler(object): raise NotImplementedError(f'Unexpected event {object}')

with Timing("Opening xpc connection"):
  connection = xpc_connection_create(b'com.apple.MTLCompilerService', None)
  xpc_connection_set_event_handler(connection, blockify(event_handler, None, ctypes.c_void_p))
  xpc_connection_resume(connection)

def compile_xpc(src:str, req_type:MTLRequestType=MTLRequestType.MTLBuildLibraryFromSourceToArchive) -> bytes:
  reply = xpc2py(xpc_connection_send_message_with_reply_sync(connection, py2xpc({
    'requestType': ('uint64', req_type.value),
    'llvmVersion': ('uint64', 32024),
    'data': encode_request(src)
  })))
  assert reply['error'] == ('uint64', 0), reply
  lib = decode_reply(reply['reply'])
  return unwrap(lib)

if __name__ == '__main__': test_compiler(compile_xpc)

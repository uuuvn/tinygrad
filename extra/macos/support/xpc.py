from tinygrad.helpers import to_mv
from extra.macos.support.block import blockify
from dataclasses import dataclass
import ctypes, ctypes.util

libSystem = ctypes.CDLL('/usr/lib/libSystem.dylib')

libSystem.xpc_connection_create.argtypes = [ctypes.c_char_p, ctypes.c_void_p]
libSystem.xpc_connection_create.restype = ctypes.c_void_p
xpc_connection_create = libSystem.xpc_connection_create

libSystem.xpc_connection_set_event_handler.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
libSystem.xpc_connection_set_event_handler.restype = None
xpc_connection_set_event_handler = libSystem.xpc_connection_set_event_handler

libSystem.xpc_connection_resume.argtypes = [ctypes.c_void_p]
libSystem.xpc_connection_resume.restype = None
xpc_connection_resume = libSystem.xpc_connection_resume

libSystem.xpc_connection_get_pid.argtypes = [ctypes.c_void_p]
libSystem.xpc_connection_get_pid.restype = ctypes.c_int32
xpc_connection_get_pid = libSystem.xpc_connection_get_pid

libSystem.xpc_connection_send_message_with_reply_sync.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
libSystem.xpc_connection_send_message_with_reply_sync.restype = ctypes.c_void_p
xpc_connection_send_message_with_reply_sync = libSystem.xpc_connection_send_message_with_reply_sync

libSystem.xpc_get_type.argtypes = [ctypes.c_void_p]
libSystem.xpc_get_type.restype = ctypes.c_void_p
xpc_get_type = libSystem.xpc_get_type

libSystem.xpc_type_get_name.argtypes = [ctypes.c_void_p]
libSystem.xpc_type_get_name.restype = ctypes.c_char_p
xpc_type_get_name = libSystem.xpc_type_get_name

libSystem.xpc_dictionary_create.argtypes = [ctypes.POINTER(ctypes.c_char_p), ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
libSystem.xpc_dictionary_create.restype = ctypes.c_void_p
xpc_dictionary_create = libSystem.xpc_dictionary_create

libSystem.xpc_dictionary_apply.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
libSystem.xpc_dictionary_apply.restype = ctypes.c_bool
xpc_dictionary_apply = libSystem.xpc_dictionary_apply

libSystem.xpc_bool_create.argtypes = [ctypes.c_bool]
libSystem.xpc_bool_create.restype = ctypes.c_void_p
xpc_bool_create = libSystem.xpc_bool_create

libSystem.xpc_bool_get_value.argtypes = [ctypes.c_void_p]
libSystem.xpc_bool_get_value.restype = ctypes.c_bool
xpc_bool_get_value = libSystem.xpc_bool_get_value

libSystem.xpc_int64_create.argtypes = [ctypes.c_int64]
libSystem.xpc_int64_create.restype = ctypes.c_void_p
xpc_int64_create = libSystem.xpc_int64_create

libSystem.xpc_int64_get_value.argtypes = [ctypes.c_void_p]
libSystem.xpc_int64_get_value.restype = ctypes.c_int64
xpc_int64_get_value = libSystem.xpc_int64_get_value

libSystem.xpc_uint64_create.argtypes = [ctypes.c_uint64]
libSystem.xpc_uint64_create.restype = ctypes.c_void_p
xpc_uint64_create = libSystem.xpc_uint64_create

libSystem.xpc_uint64_get_value.argtypes = [ctypes.c_void_p]
libSystem.xpc_uint64_get_value.restype = ctypes.c_uint64
xpc_uint64_get_value = libSystem.xpc_uint64_get_value

libSystem.xpc_array_create_empty.argtypes = []
libSystem.xpc_array_create_empty.restype = ctypes.c_void_p
xpc_array_create_empty = libSystem.xpc_array_create_empty

libSystem.xpc_array_create.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
libSystem.xpc_array_create.restype = ctypes.c_void_p
xpc_array_create = libSystem.xpc_array_create

libSystem.xpc_array_apply.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
libSystem.xpc_array_apply.restype = ctypes.c_bool
xpc_array_apply = libSystem.xpc_array_apply

libSystem.xpc_data_create.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
libSystem.xpc_data_create.restype = ctypes.c_void_p
xpc_data_create = libSystem.xpc_data_create

libSystem.xpc_data_get_bytes_ptr.argtypes = [ctypes.c_void_p]
libSystem.xpc_data_get_bytes_ptr.restype = ctypes.c_void_p
xpc_data_get_bytes_ptr = libSystem.xpc_data_get_bytes_ptr

libSystem.xpc_data_get_length.argtypes = [ctypes.c_void_p]
libSystem.xpc_data_get_length.restype = ctypes.c_size_t
xpc_data_get_length = libSystem.xpc_data_get_length

libSystem.xpc_string_create.argtypes = [ctypes.c_char_p]
libSystem.xpc_string_create.restype = ctypes.c_void_p
xpc_string_create = libSystem.xpc_string_create

libSystem.xpc_string_get_string_ptr.argtypes = [ctypes.c_void_p]
libSystem.xpc_string_get_string_ptr.restype = ctypes.c_char_p
xpc_string_get_string_ptr = libSystem.xpc_string_get_string_ptr

libSystem.xpc_copy_description.argtypes = [ctypes.c_void_p]
libSystem.xpc_copy_description.restype  = ctypes.c_char_p
xpc_copy_description = libSystem.xpc_copy_description

@dataclass(frozen=True)
class XPCObject:
  obj: ctypes.c_void_p
  def __repr__(self): return xpc_copy_description(self.obj).decode()

def xpc2py(obj: ctypes.c_void_p):
  obj_type = xpc_get_type(obj)
  obj_tname = xpc_type_get_name(obj_type).decode()
  if obj_tname == 'string': return xpc_string_get_string_ptr(obj).decode('ascii')
  elif obj_tname == 'data': return bytes(to_mv(xpc_data_get_bytes_ptr(obj), xpc_data_get_length(obj)))
  elif obj_tname == 'bool': return xpc_bool_get_value(obj)
  elif obj_tname == 'int64': return (obj_tname, int(xpc_int64_get_value(obj)))
  elif obj_tname == 'uint64': return (obj_tname, int(xpc_uint64_get_value(obj)))
  elif obj_tname == 'dictionary':
    d = {}
    def _dict_callback(key: bytes, obj: ctypes.c_void_p):
      d[key.decode()] = xpc2py(obj)
      return True
    assert xpc_dictionary_apply(obj, blockify(_dict_callback, ctypes.c_bool, ctypes.c_char_p, ctypes.c_void_p))
    return d
  elif obj_tname == 'array':
    a = []
    def _arr_callback(i, obj: ctypes.c_void_p):
      a.append(xpc2py(obj))
      return True
    assert xpc_array_apply(obj, blockify(_arr_callback, ctypes.c_bool, ctypes.c_size_t, ctypes.c_void_p))
    return a
  return XPCObject(obj)

def autotype(x):
  if isinstance(x, str): return 'string'
  elif isinstance(x, bool): return 'bool'
  elif isinstance(x, bytes): return 'data'
  elif isinstance(x, dict): return 'dictionary'
  elif isinstance(x, list): return 'array'
  raise NotImplementedError(f"couldn't autotype {x}")

def py2xpc(x) -> ctypes.c_void_p:
  tname, obj = x if isinstance(x, tuple) else (autotype(x), x)
  if tname == 'string': return xpc_string_create(obj.encode('ascii'))
  elif tname == 'data': return xpc_data_create(obj, len(obj))
  elif tname == 'bool': return xpc_bool_create(obj)
  elif tname == 'int64': return xpc_int64_create(obj)
  elif tname == 'uint64': return xpc_uint64_create(obj)
  elif tname == 'dictionary':
    keys, values = (ctypes.c_char_p * len(obj))(), (ctypes.c_void_p * len(obj))()
    for i,(k,v) in enumerate(obj.items()):
      keys[i], values[i] = k.encode('ascii'), py2xpc(v)
    return xpc_dictionary_create(keys, values, len(keys))
  elif tname == 'array':
    values = (ctypes.c_void_p * len(obj))()
    for i,v in enumerate(obj):
      values[i] = py2xpc(v)
    return xpc_array_create(values, len(values))
  raise NotImplementedError(f"Unknown type {tname} for {obj}")

import struct
from tinygrad.helpers import DEBUG
from lldb import SBDebugger, SBError, SBProcess, SBBreakpoint, SBBreakpointLocation, SBFrame, SBStructuredData

DISABLED = set()

def ioctl(frame: SBFrame, bp_loc: SBBreakpointLocation, dict):
  if int(frame.FindRegister('x0').GetValue(), 16) != 0xdead or \
     int(frame.FindRegister('x1').GetValue(), 16) != 0xc0de: return False
  proc: SBProcess = frame.GetThread().GetProcess()
  argp = proc.ReadPointerFromMemory(int(frame.FindRegister('sp').GetValue(), 16), (e:=SBError()))
  if e.Fail():
    print("LLDB_HELPER: failed to read argument pointer from stack")
    return False
  cmd = proc.ReadUnsignedFromMemory(argp, 8, (e:=SBError()))
  if e.Fail():
    print("LLDB_HELPER: failed to read command type from memory")
    return False
  if cmd == 0x1: ret = ioctl_alive(proc, frame, argp)
  elif cmd == 0x2: ret = ioctl_set_enabled(proc, frame, argp)
  elif cmd == 0x3: ret = ioctl_backtrace(proc, frame, argp)
  elif cmd == 0x10: ret = ioctl_hook_address(proc, frame, argp)
  elif cmd == 0x11: ret = ioctl_hook_symbol(proc, frame, argp)
  else:
    print(f"LLDB_HELPER: unknown command {cmd:#x}")
    ret = 0xbadc0de
    return False
  proc.WriteMemory(argp, struct.pack('<Q', ret), (e:=SBError()))
  if e.Fail(): print("LLDB_HELPER: failed to write return code to memory")
  frame.FindRegister('x0').SetValueFromCString('0')
  frame.FindRegister('pc').SetValueFromCString(frame.FindRegister('x30').GetValue())
  return False

def ioctl_alive(proc: SBProcess, frame: SBFrame, argp: int):
  arg = proc.ReadMemory(argp+8, 4, (e:=SBError()))
  if e.Fail():
    print("LLDB_HELPER (alive): failed to read ping from memory")
    return 0xbad0101
  if arg != b'ping':
    print(f"LLDB_HELPER (alive): got {arg} instead of b'ping'")
    return 0xbad0102
  proc.WriteMemory(argp+8, b"pong", (e:=SBError()))
  if e.Fail():
    print("LLDB_HELPER (alive): failed to write response to memory")
    return 0xbad0103
  return 0

def ioctl_set_enabled(proc: SBProcess, frame: SBFrame, argp: int):
  arg = proc.ReadMemory(argp+8, 16, (e:=SBError()))
  if e.Fail():
    print("LLDB_HELPER (set enabled): failed to read breakpoint id from memory")
    return 0xbad0201
  bid, enabled = struct.unpack('<2Q', arg)
  if enabled not in {0, 1}:
    print(f"LLDB_HELPER (set enabled): invalid value for eanbled {enabled}")
    return 0xbad0202
  tid = frame.GetThread().GetThreadID()
  old_enabled = (bid, tid) not in DISABLED
  if enabled: DISABLED.discard((bid, tid))
  else: DISABLED.add((bid, tid))
  proc.WriteMemory(argp+8, struct.pack('<Q', int(old_enabled)), (e:=SBError()))
  if e.Fail():
    print("LLDB_HELPER (set enabled): failed to write old enabled to memory")
    return 0xbad0203
  return 0

def ioctl_backtrace(proc: SBProcess, frame: SBFrame, argp: int):
  for f in reversed(list(iter(frame.GetThread()))): print(f)
  return 0

def ioctl_hook_address(proc: SBProcess, frame: SBFrame, argp: int):
  arg = proc.ReadMemory(argp+8, 16, (e:=SBError()))
  if e.Fail():
    print("LLDB_HELPER (hook address): failed to read argument struct from memory")
    return 0xbad1001
  p_loc, h_loc = struct.unpack('<2Q', arg)
  if DEBUG >= 2: print(f"LLDB_HELPER (hook address): setting up a hook from {p_loc:#x} to {h_loc:#x}")
  (data:=SBStructuredData()).SetFromJSON(str(h_loc))
  br: SBBreakpoint = proc.GetTarget().BreakpointCreateByAddress(p_loc)
  br.SetScriptCallbackFunction('lldb_helper.hook', data)
  proc.WriteMemory(argp+8, struct.pack('<Q', br.GetID()), (e:=SBError()))
  if e.Fail():
    print("LLDB_HELPER (hook address): failed to write breakpoint id to memory")
    return 0xbad1002
  return 0

def ioctl_hook_symbol(proc: SBProcess, frame: SBFrame, argp: int):
  h_loc = proc.ReadUnsignedFromMemory(argp+8, 8, (e:=SBError()))
  if e.Fail():
    print("LLDB_HELPER (hook symbol): failed to read hook address from memory")
    return 0xbad1001
  p_loc = proc.ReadCStringFromMemory(argp+16, 1024, (e:=SBError()))
  if e.Fail():
    print("LLDB_HELPER (hook symbol): failed to read target symbol from memory")
    return 0xbad1002
  if DEBUG >= 2: print(f"LLDB_HELPER (hook symbol): setting up a hook from {p_loc} to {h_loc:#x}")
  (data:=SBStructuredData()).SetFromJSON(str(h_loc))
  br: SBBreakpoint = proc.GetTarget().BreakpointCreateByName(p_loc)
  br.SetScriptCallbackFunction('lldb_helper.hook', data)
  proc.WriteMemory(argp+8, struct.pack('<Q', br.GetID()), (e:=SBError()))
  if e.Fail():
    print("LLDB_HELPER (hook symbol): failed to write breakpoint id to memory")
    return 0xbad1003
  return 0

def hook(frame: SBFrame, bp_loc: SBBreakpointLocation, data: SBStructuredData, dict):
  if (bp_loc.GetBreakpoint().GetID(), frame.GetThread().GetThreadID()) in DISABLED: return False
  frame.FindRegister('pc').SetValueFromCString(hex(data.GetIntegerValue()))
  return False

def __lldb_init_module(debugger: SBDebugger, dict):
  br: SBBreakpoint = debugger.GetSelectedTarget().BreakpointCreateByName('ioctl')
  br.SetScriptCallbackFunction('lldb_helper.ioctl')
  br.SetAutoContinue(True) # should be able to remove after llvm fixes bug with internal_dict

import os, struct
from enum import IntEnum
from tinygrad.helpers import cache_dir

# MTLCompiler wire protocol helpers

def lpad_atleastone(x, i, c): return x + c * (i-(len(x)%i))

class MTLRequestType(IntEnum):
    MTLBuildRequestTypeUnknown = -1
    MTLInvalidRequest = 0
    MTLBuildFunctions = 1
    MTLUnknown_2 = 2
    MTLBuildLibraryFromSource = 3
    MTLBuildOpaqueRequest = 4
    MTLBuildCISPIRequestToArchive = 5
    MTLSpecializeFunction = 6
    MTLDowngradeModule = 7
    MTLLogCompilerFailureRequest = 8
    MTLCompilerPingRequest = 9
    MTLStatelessBackendCompileRequest = 10
    MTLStitchFunctionDagRequest = 11
    MTLUnknown_12 = 12
    MTLBuildLibraryFromSourceToArchive = 13
    MTLStitchFunctionDagToArchive = 14
    MTLSpecializeFunctionToArchive = 15
    MTLGenerateMachO = 16
    MTLGenerateBinaryArchiveID = 17

DEFAULT_CMDLINE = f'-fno-fast-math -std=metal3.1 --driver-mode=metal -x metal -fmodules-cache-path="{os.path.join(cache_dir, "tinygrad")}"'

def encode_request(src: str, cmdline: str=DEFAULT_CMDLINE) -> bytes:
  e1, e2 = lpad_atleastone(src.encode(), 4, b'\x00'), cmdline.encode()+b'\x00'
  return struct.pack('<Q', len(e1)) + struct.pack('<Q', len(e2)) + e1 + e2

def decode_reply(reply:bytes) -> bytes:
  return reply[sum(struct.unpack('<LL', reply[8:16])):]

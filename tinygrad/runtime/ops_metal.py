import subprocess, pathlib, struct, ctypes, tempfile, functools, contextlib, decimal, platform
from typing import Any, cast
from tinygrad.helpers import prod, to_mv, getenv, round_up, cache_dir, T, init_c_struct_t, PROFILE, ProfileRangeEvent, cpu_profile, unwrap
from tinygrad.device import Compiled, Compiler, CompileError, LRUAllocator, ProfileDeviceEvent
from tinygrad.renderer.cstyle import MetalRenderer

class objc_id(ctypes.c_void_p): # This prevents ctypes from converting response to plain int, and dict.fromkeys() can use it to dedup
  def __hash__(self): return hash(self.value)
  def __eq__(self, other): return self.value == other.value

class objc_instance(objc_id): # method with name "new", "alloc" should be freed after use
  def __del__(self):
    # CPython doesn't make any guarantees about order in which globals (like `msg` or `libobjc`) are destroyed when the interpreter shuts down
    # https://github.com/tinygrad/tinygrad/pull/8949 triggered the unlucky ordering which lead to a bunch of errors at exit
    # TODO: Why isn't `sys.is_finalizing` working?
    if msg is not None and libobjc is not None: msg("release")(self)
  def __repr__(self): return from_ns_str(msg("description", objc_instance)(self))

class MTLFunctionOptions:
  MTLFunctionOptionCompileToBinary = 1 << 0
  MTLFunctionOptionPipelineIndependent = 1 << 3

class MTLResourceOptions:
  MTLResourceCPUCacheModeDefaultCache = 0
  MTLResourceStorageModeShared = 0 << 4

class MTLPipelineOption:
  MTLPipelineOptionNone = 0
  MTLPipelineOptionFailOnBinaryArchiveMiss = 1 << 2

# 13 is requestType that metal uses to compile source code into MTLB, there aren't any docs or symbols.
REQUEST_TYPE_COMPILE = 13

libobjc = ctypes.CDLL("/usr/lib/libobjc.dylib")
libmetal = ctypes.CDLL("/System/Library/Frameworks/Metal.framework/Metal")
# Must be loaded for default Metal Device: https://developer.apple.com/documentation/metal/1433401-mtlcreatesystemdefaultdevice?language=objc
ctypes.CDLL("/System/Library/Frameworks/CoreGraphics.framework/CoreGraphics")
libdispatch = ctypes.CDLL("/usr/lib/libSystem.dylib") # libdispatch is part of libSystem on mac
libobjc.objc_getClass.restype = objc_id
libobjc.sel_registerName.restype = objc_id
libmetal.MTLCreateSystemDefaultDevice.restype = objc_instance
libdispatch.dispatch_data_create.restype = objc_instance

@functools.cache
def msg(selector: str, restype: type[T] = objc_id):  # type: ignore [assignment]
  resname = libobjc.sel_registerName(selector.encode())
  sender = libobjc["objc_msgSend"] # Using attribute access returns a new reference so setting restype is safe
  sender.restype = restype
  def _msg(ptr: objc_id, *args: Any) -> T: return sender(ptr, resname, *args)
  return _msg

@functools.cache
def to_ns_str(s: str): return msg("stringWithUTF8String:", objc_instance)(libobjc.objc_getClass(b"NSString"), s.encode())
def from_ns_str(s): return bytes(msg("UTF8String", ctypes.c_char_p)(s)).decode()

def to_struct(*t: int, _type: type[ctypes._SimpleCData] = ctypes.c_ulong):
  return init_c_struct_t(tuple([(f"field{i}", _type) for i in range(len(t))]))(*t)

def wait_check(cbuf: Any):
  msg("waitUntilCompleted")(cbuf)
  error_check(msg("error", objc_instance)(cbuf))

def cmdbuf_label(cbuf: objc_id) -> str|None: return from_ns_str(label) if (label:=msg("label", objc_id)(cbuf)).value is not None else None
def cmdbuf_st_time(cbuf: objc_id) -> float: return cast(float, msg("GPUStartTime", ctypes.c_double)(cbuf))
def cmdbuf_en_time(cbuf: objc_id) -> float: return cast(float, msg("GPUEndTime", ctypes.c_double)(cbuf))

def error_check(error: objc_instance, error_constructor: type[Exception] = RuntimeError):
  if error.value is None: return None
  raise error_constructor(from_ns_str(msg("localizedDescription", objc_instance)(error)))

class MetalDevice(Compiled):
  def __init__(self, device:str):
    self.sysdevice = libmetal.MTLCreateSystemDefaultDevice()
    self.mtl_queue = msg("newCommandQueueWithMaxCommandBufferCount:", objc_instance)(self.sysdevice, 1024)
    if self.mtl_queue is None: raise RuntimeError("Cannot allocate a new command queue")
    self.mtl_buffers_in_flight: list[Any] = []
    self.timeline_signal = msg("newSharedEvent", objc_instance)(self.sysdevice)
    self.timeline_value = 0

    Compiled.profile_events += [ProfileDeviceEvent(device)]

    from tinygrad.runtime.graph.metal import MetalGraph
    # NOTE: GitHub CI macOS runners use paravirtualized metal which is broken with graph.
    # This can be reproduced locally with any virtualization software (like utm) that can create macOS VMs with apple's own virtualization framework.
    super().__init__(device, MetalAllocator(self), [(MetalRenderer, functools.partial(MetalCompiler, self)), (MetalRenderer, Compiler)],
                     functools.partial(MetalProgram, self), MetalGraph if 'virtual' not in from_ns_str(msg('name')(self.sysdevice)).lower() else None)

  def synchronize(self):
    for cbuf in self.mtl_buffers_in_flight:
      wait_check(cbuf)
      st, en = decimal.Decimal(cmdbuf_st_time(cbuf)) * 1000000, decimal.Decimal(cmdbuf_en_time(cbuf)) * 1000000
      # NOTE: command buffers from MetalGraph are not profiled here
      if PROFILE and (lb:=cmdbuf_label(cbuf)) is not None and not lb.startswith("batched"):
        Compiled.profile_events += [ProfileRangeEvent(self.device, lb, st, en, is_copy=lb.startswith("COPY"))]
    self.mtl_buffers_in_flight.clear()

def metal_src_to_library(device:MetalDevice, src:str) -> objc_instance:
  options = msg("new", objc_instance)(libobjc.objc_getClass(b"MTLCompileOptions"))
  msg("setFastMathEnabled:")(options, getenv("METAL_FAST_MATH"))
  library = msg("newLibraryWithSource:options:error:", objc_instance)(device.sysdevice, to_ns_str(src),
                                                                      options, ctypes.byref(compileError:=objc_instance()))
  error_check(compileError, CompileError)
  return library

class MetalCompiler(Compiler):
  # Opening METAL after LLVM doesn't fail because ctypes.CDLL opens with RTLD_LOCAL but MTLCompiler opens it's own llvm with RTLD_GLOBAL
  # This means that MTLCompiler's llvm will create it's own instances of global state because RTLD_LOCAL doesn't export symbols, but if RTLD_GLOBAL
  # library is loaded first then RTLD_LOCAL library will just use it's symbols. On linux there is RTLD_DEEPBIND to prevent that, but on macos there
  # doesn't seem to be anything we can do.
  with contextlib.suppress(FileNotFoundError, ModuleNotFoundError):
    import tinygrad.runtime.autogen.llvm # noqa: F401
  support = ctypes.CDLL("/System/Library/PrivateFrameworks/MTLCompiler.framework/MTLCompiler")
  support.MTLCodeGenServiceCreate.restype = ctypes.c_void_p

  def __init__(self, dev:MetalDevice):
    self.cgs = ctypes.c_void_p(MetalCompiler.support.MTLCodeGenServiceCreate(b"tinygrad"))
    self.sysdevice = dev.sysdevice
    super().__init__("compile_metal_direct_pipeline_state")
  def __reduce__(self): return (MetalCompiler,()) # force pickle to create new instance for each multiprocessing fork
  def compile(self, src:str) -> bytes:
    print("--- SOURCE:")
    print(src)
    mtlb: Exception|bytes = CompileError("MTLCodeGenServiceBuildRequest returned without calling the callback")
    @ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_int32, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_char_p)
    def callback(blockptr, error, dataPtr, dataLen, errorMessage):
      nonlocal mtlb
      if error == 0:
        reply = bytes(to_mv(dataPtr, dataLen))
        # offset from beginning to data = header size + warning size
        mtlb = reply[sum(struct.unpack('<LL', reply[8:16])):]
      else:
        mtlb = CompileError(errorMessage.decode())

    # no changes for compute in 2.0 - 2.4 specs, use 2.0 as default for old versions.
    macos_major = int(platform.mac_ver()[0].split('.')[0])
    metal_version = "metal3.1" if macos_major >= 14 else "metal3.0" if macos_major >= 13 else "macos-metal2.0"

    # llvm will create modules.timestamp in cache path and cache compilation of metal stdlib (250ms => 8ms compilation time)
    # note that llvm won't necessarily create anything else here as apple has prebuilt versions of many standard libraries
    params = f'-fno-fast-math -std={metal_version} --driver-mode=metal -x metal -fmodules-cache-path="{cache_dir}" -fno-caret-diagnostics'
    # source blob has to be padded to multiple of 4 but at least one 'b\x00' should be added, params blob just has to be null terminated
    src_padded, params_padded = src.encode() + b'\x00'*(round_up(len(src) + 1, 4) - len(src)), params.encode() + b'\x00'
    request = struct.pack('<QQ', len(src_padded), len(params_padded)) + src_padded + params_padded
    # The callback is actually not a callback but a block which is apple's non-standard extension to add closures to C.
    # See https://clang.llvm.org/docs/Block-ABI-Apple.html#high-level for struct layout.
    # Fields other than invoke are unused in this case so we can just use ctypes.byref with negative offset to invoke field, add blockptr as a first
    # argument and pretend it's a normal callback
    MetalCompiler.support.MTLCodeGenServiceBuildRequest(self.cgs, None, REQUEST_TYPE_COMPILE, request, len(request), ctypes.byref(callback, -0x10))
    if isinstance(mtlb, Exception): raise mtlb
    assert mtlb[:4] == b"MTLB" and mtlb[-4:] == b"ENDT", f"Invalid Metal library. {mtlb!r}"
    print("--- MTLB:")
    print(mtlb)
    print("--- CREATING BINARY ARCHIVE")
    binary_archive_descriptor = msg("new", objc_instance)(libobjc.objc_getClass(b"MTLBinaryArchiveDescriptor"))
    binary_archive  = msg("newBinaryArchiveWithDescriptor:error:", objc_instance)(self.sysdevice, binary_archive_descriptor,
                                                                                  ctypes.byref(error_archive:=objc_instance()))
    error_check(error_archive)
    print("--- COMPLING WITH APPLE APIS")
    mtlb_dispatch_data = libdispatch.dispatch_data_create(mtlb, len(mtlb), None, None)
    library = msg("newLibraryWithData:error:", objc_instance)(self.sysdevice, mtlb_dispatch_data, ctypes.byref(error_lib:=objc_instance()))
    error_check(error_lib)
    fxn = msg("newFunctionWithName:", objc_instance)(library, to_ns_str(_name:="E_4"))
    pipeline_descriptor = msg("new", objc_instance)(libobjc.objc_getClass(b"MTLComputePipelineDescriptor"))
    msg("setComputeFunction:")(pipeline_descriptor, fxn)
    msg("setSupportIndirectCommandBuffers:")(pipeline_descriptor, True)
    msg("addComputePipelineFunctionsWithDescriptor:error:")(binary_archive, pipeline_descriptor, ctypes.byref(error_add:=objc_instance()))
    error_check(error_add)
    print("--- SERIALIZING")
    with tempfile.NamedTemporaryFile(delete=True) as output_file:
      file_url = msg("fileURLWithPath:", objc_instance)(libobjc.objc_getClass(b"NSURL"), to_ns_str(f"{output_file.name}"))
      msg("serializeToURL:error:")(binary_archive, file_url, ctypes.byref(error_serialize:=objc_instance()))
      error_check(error_serialize)
      mtlpsbin = pathlib.Path(output_file.name).read_bytes()
      print("--- MTLPSBIN:")
      print(mtlpsbin)
      print("--- END COMPILE")
      return b"WRPR" + struct.pack("<II", len(mtlb), len(mtlpsbin)) + mtlb + mtlpsbin
  def disassemble(self, lib:bytes):
    with tempfile.NamedTemporaryFile(delete=True) as shader:
      shader.write(lib)
      shader.flush()
      proc = subprocess.Popen(f"cd {pathlib.Path(__file__).parents[2]}/extra/disassemblers/applegpu && python3 compiler_explorer.py {shader.name}",
                              stdout=subprocess.PIPE, shell=True, text=True, bufsize=1)
      for line in unwrap(proc.stdout): print(line, end="")
      ret = proc.wait()
      if ret: print("Disassembler Error: Make sure you have https://github.com/dougallj/applegpu cloned to tinygrad/extra/disassemblers/applegpu")

class MetalProgram:
  def __init__(self, dev:MetalDevice, name:str, lib:bytes):
    self.dev, self.name, self.lib = dev, name, lib
    if lib[:4] == b"WRPR":
      mtlb_len, mtlpsbin_len = struct.unpack("<II", lib[4:12])
      mtlb, mtlpsbin = lib[12:12+mtlb_len], lib[12+mtlb_len:12+mtlb_len+mtlpsbin_len]
      # binary metal library
      data = libdispatch.dispatch_data_create(mtlb, len(mtlb), None, None)
      self.library = msg("newLibraryWithData:error:", objc_instance)(self.dev.sysdevice, data, ctypes.byref(error_lib:=objc_instance()))
      error_check(error_lib)
      with tempfile.NamedTemporaryFile(delete=True) as input_file:
        pathlib.Path(input_file.name).write_bytes(mtlpsbin)
        mtlpsbin_url = msg("fileURLWithPath:", objc_instance)(libobjc.objc_getClass(b"NSURL"), to_ns_str(f"{input_file.name}"))
        binary_archive_descriptor = msg("new", objc_instance)(libobjc.objc_getClass(b"MTLBinaryArchiveDescriptor"))
        msg("setUrl:")(binary_archive_descriptor, mtlpsbin_url)
        self.binary_archive = msg("newBinaryArchiveWithDescriptor:error:", objc_instance)(self.dev.sysdevice, binary_archive_descriptor,
                                                                                     ctypes.byref(error_archive:=objc_instance()))
        error_check(error_archive)
    else:
      # metal source. rely on OS caching
      try: self.library = metal_src_to_library(self.dev, lib.decode())
      except CompileError as e: raise RuntimeError from e
      self.binary_archive = None
    self.fxn = msg("newFunctionWithName:", objc_instance)(self.library, to_ns_str(name))
    print('--- FUNCTION CREATED')
    descriptor = msg("new", objc_instance)(libobjc.objc_getClass(b"MTLComputePipelineDescriptor"))
    msg("setComputeFunction:")(descriptor, self.fxn)
    msg("setSupportIndirectCommandBuffers:")(descriptor, True)
    pipeline_state_option = MTLPipelineOption.MTLPipelineOptionNone
    if self.binary_archive is not None:
      msg("setBinaryArchives:")(descriptor, msg("arrayWithObject:", objc_instance)(libobjc.objc_getClass(b"NSArray"), self.binary_archive))
      pipeline_state_option = MTLPipelineOption.MTLPipelineOptionFailOnBinaryArchiveMiss
    print('--- PIPELINE DESCRIPTOR CREATED')
    self.pipeline_state = msg("newComputePipelineStateWithDescriptor:options:reflection:error:", objc_instance)(self.dev.sysdevice,
      descriptor, pipeline_state_option, None, ctypes.byref(error_pipeline_creation:=objc_instance()))
    error_check(error_pipeline_creation)
    print('--- PIPELINE STATE CREATED')
    # cache these msg calls
    self.max_total_threads: int = cast(int, msg("maxTotalThreadsPerThreadgroup", ctypes.c_ulong)(self.pipeline_state))

  def __call__(self, *bufs, global_size:tuple[int,int,int]=(1,1,1), local_size:tuple[int,int,int]=(1,1,1), vals:tuple[int, ...]=(), wait=False):
    if prod(local_size) > self.max_total_threads:
      exec_width = msg("threadExecutionWidth", ctypes.c_ulong)(self.pipeline_state)
      memory_length = msg("staticThreadgroupMemoryLength", ctypes.c_ulong)(self.pipeline_state)
      raise RuntimeError(f"local size {local_size} bigger than {self.max_total_threads} with exec width {exec_width} memory length {memory_length}")
    command_buffer = msg("commandBuffer", objc_instance)(self.dev.mtl_queue)
    encoder = msg("computeCommandEncoder", objc_instance)(command_buffer)
    msg("setComputePipelineState:")(encoder, self.pipeline_state)
    for i,a in enumerate(bufs): msg("setBuffer:offset:atIndex:")(encoder, a.buf, a.offset, i)
    for i,a in enumerate(vals, start=len(bufs)): msg("setBytes:length:atIndex:")(encoder, bytes(ctypes.c_int(a)), 4, i)
    msg("dispatchThreadgroups:threadsPerThreadgroup:")(encoder, to_struct(*global_size), to_struct(*local_size))
    msg("endEncoding")(encoder)
    msg("setLabel:")(command_buffer, to_ns_str(self.name)) # TODO: is this always needed?
    msg("commit")(command_buffer)
    self.dev.mtl_buffers_in_flight.append(command_buffer)
    if wait:
      wait_check(command_buffer)
      return cmdbuf_en_time(command_buffer) - cmdbuf_st_time(command_buffer)

class MetalBuffer:
  def __init__(self, buf:Any, size:int, offset=0): self.buf, self.size, self.offset = buf, size, offset

class MetalAllocator(LRUAllocator[MetalDevice]):
  def _alloc(self, size:int, options) -> MetalBuffer:
    if options.external_ptr: return MetalBuffer(objc_id(options.external_ptr), size)

    # Buffer is explicitly released in _free() rather than garbage collected via reference count
    ret = msg("newBufferWithLength:options:", objc_id)(self.dev.sysdevice, ctypes.c_ulong(size), MTLResourceOptions.MTLResourceStorageModeShared)
    if ret.value is None: raise MemoryError(f"Metal OOM while allocating {size=}")
    return MetalBuffer(ret, size)
  def _free(self, opaque:MetalBuffer, options):
    if msg is not None and libobjc is not None: msg("release")(opaque.buf)
  def _transfer(self, dest:MetalBuffer, src:MetalBuffer, sz:int, src_dev:MetalDevice, dest_dev:MetalDevice):
    dest_dev.synchronize()
    src_command_buffer = msg("commandBuffer", objc_instance)(src_dev.mtl_queue)
    encoder = msg("blitCommandEncoder", objc_instance)(src_command_buffer)
    msg("copyFromBuffer:sourceOffset:toBuffer:destinationOffset:size:")(encoder, src.buf, ctypes.c_ulong(src.offset),
        dest.buf, ctypes.c_ulong(dest.offset), ctypes.c_ulong(sz))
    msg("endEncoding")(encoder)
    if src_dev != dest_dev:
      msg("encodeSignalEvent:value:")(src_command_buffer, src_dev.timeline_signal, src_dev.timeline_value)
      dest_command_buffer = msg("commandBuffer", objc_instance)(dest_dev.mtl_queue)
      msg("encodeWaitForEvent:value:")(dest_command_buffer, src_dev.timeline_signal, src_dev.timeline_value)
      msg("commit")(dest_command_buffer)
      dest_dev.mtl_buffers_in_flight.append(dest_command_buffer)
      src_dev.timeline_value += 1
    msg("setLabel:")(src_command_buffer, to_ns_str(f"COPY {src_dev.device} -> {dest_dev.device}"))
    msg("commit")(src_command_buffer)
    src_dev.mtl_buffers_in_flight.append(src_command_buffer)
    # Transfers currently synchronize the completion. Otherwise, copies can sometimes lead to incorrect values.
    # There is no real metal multidevice support for now, so transfer is used only for tests.
    src_dev.synchronize()
  def _cp_mv(self, dst, src, prof_desc):
    with cpu_profile(prof_desc, self.dev.device, is_copy=True): dst[:] = src
  def _as_buffer(self, src:MetalBuffer) -> memoryview:
    self.dev.synchronize()
    return to_mv(cast(int, msg("contents", objc_id)(src.buf).value), src.size + src.offset)[src.offset:]
  def _copyin(self, dest:MetalBuffer, src:memoryview): self._cp_mv(self._as_buffer(dest), src, "TINY -> METAL")
  def _copyout(self, dest:memoryview, src:MetalBuffer): self._cp_mv(dest, self._as_buffer(src), "METAL -> TINY")
  def _offset(self, buf:MetalBuffer, size:int, offset:int): return MetalBuffer(buf.buf, size, offset)

from __future__ import annotations
import subprocess, time, functools, ctypes, struct, tinygrad.runtime.autogen.vulkan as vk
from typing import Any
from dataclasses import dataclass
from weakref import WeakSet
from tinygrad.device import LRUAllocator, BufferSpec, Compiler, Compiled
from tinygrad.renderer.glsl import GLSLRenderer
from tinygrad.helpers import fnone, to_mv, unwrap, getenv, DEBUG

def checkz(result:int):
  if result != vk.VK_SUCCESS:
    result_str = vk.VkResult__enumvalues.get(result, None)
    raise RuntimeError(f"{result}{f' ({result_str})' if result_str else ''}")

def pchain(structs: list[ctypes.Structure|None]) -> ctypes.c_void_p:
  ptr = ctypes.c_void_p(None)
  for s in reversed(fnone(structs)):
    assert hasattr(s, "pNext") and s.pNext is None
    s.pNext = ptr
    ptr = ctypes.cast(ctypes.pointer(s), ctypes.c_void_p)
  return ptr

class GLSLCompiler(Compiler):
  def __init__(self, debug: bool = False, cachekey:str|None=None):
    super().__init__(f"compile_glsl{'_debug' if debug else ''}" if cachekey is None else cachekey)
    self.debug = debug
  def compile(self, src:str) -> bytes:
    cmd = ["glslc", "--target-env=vulkan1.3", "-fshader-stage=compute", "-g" if self.debug else None, "-", "-o", "-"]
    lib = subprocess.check_output(fnone(cmd), input=src.encode("utf-8"))
    if self.debug:
      subprocess.run(["spirv-val", "--target-env", "vulkan1.3", "-"], input=lib, check=True)
    size_start = src.find("// SIZE: ") + 9
    size_end = src.find("\n", size_start)
    size = int(src[size_start:size_end])
    return struct.pack("<I", size) + lib
  def disassemble(self, lib:bytes):
    subprocess.run(["spirv-dis", "-"], input=lib[4:], check=True)

class VKProgram:
  def __init__(self, dev:VKDevice, name:str, lib:bytes):
    dev.live_programs.add(self)
    self.dev = dev

    self.kernarg_size = struct.unpack("<I", lib[:4])[0]
    self.spirv = lib[4:]

    shader_info = vk.VkShaderModuleCreateInfo(
      sType=vk.VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      codeSize=len(self.spirv),
      pCode=(ctypes.c_uint32 * (len(self.spirv) // 4)).from_buffer_copy(self.spirv),
    )
    self.shader = vk.VkShaderModule()
    checkz(vk.vkCreateShaderModule(self.dev.ldev, ctypes.byref(shader_info), None, ctypes.byref(self.shader)))

    descriptor_set_layout_create_info = vk.VkDescriptorSetLayoutCreateInfo(
      sType=vk.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      bindingCount=1,
      pBindings=ctypes.pointer(vk.VkDescriptorSetLayoutBinding(
        binding=0,
        descriptorType=vk.VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        descriptorCount=1,
        stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT,
      )),
    )
    self.descriptor_set_layout = vk.VkDescriptorSetLayout()
    checkz(vk.vkCreateDescriptorSetLayout(
      self.dev.ldev,
      ctypes.pointer(descriptor_set_layout_create_info),
      None,
      ctypes.byref(self.descriptor_set_layout)
    ))

    descriptor_pool_create_info = vk.VkDescriptorPoolCreateInfo(
      sType=vk.VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
      maxSets=1,
      poolSizeCount=1,
      pPoolSizes=ctypes.pointer(vk.VkDescriptorPoolSize(
        type=vk.VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        descriptorCount=1,
      )),
    )
    self.descriptor_pool = vk.VkDescriptorPool()
    checkz(vk.vkCreateDescriptorPool(self.dev.ldev, ctypes.pointer(descriptor_pool_create_info), None, ctypes.byref(self.descriptor_pool)))

    descriptor_set_allocate_info = vk.VkDescriptorSetAllocateInfo(
      sType=vk.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
      descriptorPool=self.descriptor_pool,
      descriptorSetCount=1,
      pSetLayouts=ctypes.pointer(self.descriptor_set_layout),
    )
    self.descriptor_set = vk.VkDescriptorSet()
    checkz(vk.vkAllocateDescriptorSets(self.dev.ldev, ctypes.pointer(descriptor_set_allocate_info), ctypes.byref(self.descriptor_set)))

    self.indirect_buffer = self.dev.allocator.alloc(self.kernarg_size + 12)

    dset_write = vk.VkWriteDescriptorSet(
      sType=vk.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
      dstSet=self.descriptor_set,
      dstBinding=0,
      descriptorCount=1,
      descriptorType=vk.VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
      pBufferInfo=ctypes.pointer(vk.VkDescriptorBufferInfo(
        buffer=self.indirect_buffer.buf,
        range=self.kernarg_size,
      )),
    )
    vk.vkUpdateDescriptorSets(self.dev.ldev, 1, ctypes.pointer(dset_write), 0, None)

    pipeline_layout_create_info = vk.VkPipelineLayoutCreateInfo(
      sType=vk.VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      setLayoutCount=1,
      pSetLayouts=ctypes.pointer(self.descriptor_set_layout),
    )
    self.pipeline_layout = vk.VkPipelineLayout()
    checkz(vk.vkCreatePipelineLayout(self.dev.ldev, ctypes.pointer(pipeline_layout_create_info), None, ctypes.byref(self.pipeline_layout)))

    pipeline_create_info = vk.VkComputePipelineCreateInfo(
      sType=vk.VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
      stage=vk.VkPipelineShaderStageCreateInfo(
        sType=vk.VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        stage=vk.VK_SHADER_STAGE_COMPUTE_BIT,
        module=self.shader,
        pName=vk.char_pointer_cast("main"),
      ),
      layout=self.pipeline_layout,
    )
    self.pipeline = vk.VkPipeline()
    checkz(vk.vkCreateComputePipelines(self.dev.ldev, None, 1, ctypes.pointer(pipeline_create_info), None, ctypes.byref(self.pipeline)))

    cbuf_allocate_info = vk.VkCommandBufferAllocateInfo(
      sType=vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      commandPool=self.dev.cq_pool,
      commandBufferCount=1,
    )
    self.cbuf = vk.VkCommandBuffer()
    checkz(vk.vkAllocateCommandBuffers(self.dev.ldev, ctypes.pointer(cbuf_allocate_info), ctypes.byref(self.cbuf)))

    cbuf_begin_info = vk.VkCommandBufferBeginInfo(
      sType=vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
    )
    checkz(vk.vkBeginCommandBuffer(self.cbuf, ctypes.pointer(cbuf_begin_info)))
    barrier = vk.VkMemoryBarrier(
      sType=vk.VK_STRUCTURE_TYPE_MEMORY_BARRIER,
      srcAccessMask=vk.VK_ACCESS_SHADER_READ_BIT | vk.VK_ACCESS_SHADER_WRITE_BIT,
      dstAccessMask=vk.VK_ACCESS_SHADER_READ_BIT | vk.VK_ACCESS_SHADER_WRITE_BIT,
    )
    vk.vkCmdPipelineBarrier(self.cbuf, vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, ctypes.pointer(barrier), 0, None, 0, None) # noqa: E501
    vk.vkCmdBindPipeline(self.cbuf, vk.VK_PIPELINE_BIND_POINT_COMPUTE, self.pipeline)
    vk.vkCmdBindDescriptorSets(self.cbuf, vk.VK_PIPELINE_BIND_POINT_COMPUTE, self.pipeline_layout, 0, 1, ctypes.pointer(self.descriptor_set), 0, None)
    vk.vkCmdDispatchIndirect(self.cbuf, self.indirect_buffer.buf, self.kernarg_size)
    vk.vkCmdPipelineBarrier(self.cbuf, vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, ctypes.pointer(barrier), 0, None, 0, None) # noqa: E501
    checkz(vk.vkEndCommandBuffer(self.cbuf))

    fence_create_info = vk.VkFenceCreateInfo(
      sType=vk.VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
      flags=vk.VK_FENCE_CREATE_SIGNALED_BIT,
    )
    self.fence = vk.VkFence()
    checkz(vk.vkCreateFence(self.dev.ldev, ctypes.pointer(fence_create_info), None, ctypes.byref(self.fence)))

  def __del__(self):
    if hasattr(self, "fence"):
      checkz(vk.vkWaitForFences(self.dev.ldev, 1, ctypes.pointer(self.fence), True, -1))
      checkz(vk.vkResetFences(self.dev.ldev, 1, ctypes.pointer(self.fence)))
      vk.vkDestroyFence(self.dev.ldev, self.fence, None)
      del self.fence
    if hasattr(self, "cbuf"):
      vk.vkFreeCommandBuffers(self.dev.ldev, self.dev.cq_pool, 1, ctypes.pointer(self.cbuf))
      del self.cbuf
    if hasattr(self, "indirect_buffer"):
      self.dev.allocator.free(self.indirect_buffer, self.indirect_buffer.sz)
      del self.indirect_buffer
    if hasattr(self, "pipeline"):
      vk.vkDestroyPipeline(self.dev.ldev, self.pipeline, None)
      del self.pipeline
    if hasattr(self, "pipeline_layout"):
      vk.vkDestroyPipelineLayout(self.dev.ldev, self.pipeline_layout, None)
      del self.pipeline_layout
    if hasattr(self, "descriptor_pool"):
      vk.vkDestroyDescriptorPool(self.dev.ldev, self.descriptor_pool, None)
      del self.descriptor_pool
      del self.descriptor_set
    if hasattr(self, "descriptor_set_layout"):
      vk.vkDestroyDescriptorSetLayout(self.dev.ldev, self.descriptor_set_layout, None)
      del self.descriptor_set_layout
    if hasattr(self, "shader"):
      vk.vkDestroyShaderModule(self.dev.ldev, self.shader, None)
      del self.shader
    if hasattr(self, "dev"):
      self.dev.live_programs.remove(self)
      del self.dev

  def __call__(self, *bufs:VKBuf, global_size:tuple[int,int,int]=(1,1,1), local_size:tuple[int,int,int]=(1,1,1), vals:tuple[int, ...]=(), wait=False):
    assert len(bufs) * 8 + len(vals) * 4 == self.kernarg_size

    checkz(vk.vkWaitForFences(self.dev.ldev, 1, ctypes.pointer(self.fence), True, -1))
    checkz(vk.vkResetFences(self.dev.ldev, 1, ctypes.pointer(self.fence)))

    st = time.perf_counter()

    self.indirect_buffer.as_mv()[:] = struct.pack(f"<{len(bufs)}Q{len(vals)}i3I", *(buf.gpu for buf in bufs), *vals, *global_size)

    submit_info = vk.VkSubmitInfo(
      sType=vk.VK_STRUCTURE_TYPE_SUBMIT_INFO,
      commandBufferCount=1,
      pCommandBuffers=ctypes.pointer(self.cbuf),
    )
    checkz(vk.vkQueueSubmit(self.dev.cq, 1, ctypes.pointer(submit_info), self.fence))

    if wait:
      checkz(vk.vkWaitForFences(self.dev.ldev, 1, ctypes.pointer(self.fence), True, -1))
      return time.perf_counter() - st

@dataclass(frozen=True)
class VKBuf:
  buf: Any
  mem: Any
  gpu: int
  cpu: int
  sz: int

  def __hash__(self): return id(self)

  def as_mv(self) -> memoryview:
    return to_mv(self.cpu, self.sz)

class VKAllocator(LRUAllocator['VKDevice']):
  def _find_memory_type(self, type_filter:int, properties:int) -> int:
    for i in range(self.dev.mem_props.memoryTypeCount):
      if (type_filter & (1 << i)) and (self.dev.mem_props.memoryTypes[i].propertyFlags & properties) == properties:
        return i
    raise RuntimeError("Failed to find suitable memory type")

  def _alloc(self, size:int, options:BufferSpec):
    buf_info = vk.VkBufferCreateInfo(
      sType=vk.VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      size=size,
      usage=vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | vk.VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | \
            vk.VK_BUFFER_USAGE_TRANSFER_SRC_BIT | vk.VK_BUFFER_USAGE_TRANSFER_DST_BIT | \
            vk.VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | vk.VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
    )

    checkz(vk.vkCreateBuffer(self.dev.ldev, ctypes.pointer(buf_info), None, ctypes.byref(buf:=vk.VkBuffer())))

    vk.vkGetBufferMemoryRequirements(self.dev.ldev, buf, ctypes.byref(mem_reqs:=vk.VkMemoryRequirements()))

    mem_info = vk.VkMemoryAllocateInfo(
      sType=vk.VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      allocationSize=mem_reqs.size,
      memoryTypeIndex=self._find_memory_type(
        mem_reqs.memoryTypeBits,
        vk.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | \
        vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | vk.VK_MEMORY_PROPERTY_HOST_CACHED_BIT,
      ),
      pNext=pchain([vk.VkMemoryAllocateFlagsInfo(
        sType=vk.VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO,
        flags=vk.VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT,
      )]),
    )

    checkz(vk.vkAllocateMemory(self.dev.ldev, ctypes.pointer(mem_info), None, ctypes.byref(mem:=vk.VkDeviceMemory())))

    checkz(vk.vkBindBufferMemory(self.dev.ldev, buf, mem, 0))

    bda_info = vk.VkBufferDeviceAddressInfo(
      sType=vk.VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
      buffer=buf,
    )
    gpu_ptr = vk.vkGetBufferDeviceAddress(self.dev.ldev, ctypes.pointer(bda_info))

    checkz(vk.vkMapMemory(self.dev.ldev, mem, 0, size, 0, ctypes.byref(cpu_ptr:=ctypes.c_void_p())))

    opaque = VKBuf(buf, mem, gpu_ptr, unwrap(cpu_ptr.value), size)
    self.dev.live_buffers.add(opaque)
    return opaque

  def _free(self, opaque:VKBuf, options:BufferSpec):
    self.dev.synchronize()
    vk.vkUnmapMemory(self.dev.ldev, opaque.mem)
    vk.vkDestroyBuffer(self.dev.ldev, opaque.buf, None)
    vk.vkFreeMemory(self.dev.ldev, opaque.mem, None)
    self.dev.live_buffers.remove(opaque)

  def _as_buffer(self, opaque:VKBuf) -> memoryview:
    self.dev.synchronize()
    return opaque.as_mv()

  def _copyin(self, dest:VKBuf, src:memoryview): self._as_buffer(dest)[:] = src
  def _copyout(self, dest:memoryview, src:VKBuf): dest[:] = self._as_buffer(src)

class VKDevice(Compiled):
  allocator: VKAllocator

  default_device = getenv("VKDEV", 0)
  graph_enabled = bool(getenv("VKGRAPH", 1))

  validation_level = getenv("VKVAL", 0)
  validation_enabled = validation_level >= 1
  validation_bestpractices = getenv("VKVAL_BESTPRACTICES", validation_level >= 2)
  validation_sync = getenv("VKVAL_SYNC", validation_level >= 3)
  validation_gpuassist = getenv("VKVAL_GPUASSIST", validation_level >= 4)
  validation_gpuassist_source_debug = getenv("VKVAL_GPUASSIST_SOURCE_DEBUG", validation_level >= 5)

  instance = None
  instance_refcount = 0
  devices = None

  def __init__(self, device:str):
    if VKDevice.instance is None:
      instance_layer_names: dict[str, None] = {}
      instance_create_info_next = None

      if VKDevice.validation_enabled:
        instance_layer_names["VK_LAYER_KHRONOS_validation"] = None
        feat_vals = fnone([
          vk.VK_VALIDATION_FEATURE_ENABLE_BEST_PRACTICES_EXT if VKDevice.validation_bestpractices else None,
          vk.VK_VALIDATION_FEATURE_ENABLE_SYNCHRONIZATION_VALIDATION_EXT if VKDevice.validation_sync else None,
          vk.VK_VALIDATION_FEATURE_ENABLE_GPU_ASSISTED_EXT if VKDevice.validation_gpuassist else None,
        ])
        instance_create_info_next = vk.VkValidationFeaturesEXT(
          sType = vk.VK_STRUCTURE_TYPE_VALIDATION_FEATURES_EXT,
          enabledValidationFeatureCount=len(feat_vals),
          pEnabledValidationFeatures=(vk.VkValidationFeatureEnableEXT * len(feat_vals))(*feat_vals),
        )

      instance_create_info = vk.VkInstanceCreateInfo(
        sType=vk.VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        pApplicationInfo=ctypes.pointer(vk.VkApplicationInfo(
          sType=vk.VK_STRUCTURE_TYPE_APPLICATION_INFO,
          apiVersion=vk.VK_API_VERSION_1_3,
        )),
        enabledLayerCount=(elcnt:=len(instance_layer_names)),
        ppEnabledLayerNames=(ctypes.POINTER(ctypes.c_char) * elcnt)(*(vk.char_pointer_cast(s) for s in instance_layer_names)),
        pNext=pchain([instance_create_info_next]),
      )
      VKDevice.instance = vk.VkInstance()
      checkz(vk.vkCreateInstance(ctypes.pointer(instance_create_info), None, ctypes.byref(VKDevice.instance)))

      checkz(vk.vkEnumeratePhysicalDevices(VKDevice.instance, ctypes.byref(n:=ctypes.c_uint32()), None))
      if n.value == 0: raise RuntimeError("No vulkan devices found!")

      VKDevice.devices = (vk.VkPhysicalDevice * n.value)()
      checkz(vk.vkEnumeratePhysicalDevices(VKDevice.instance, ctypes.byref(n), VKDevice.devices))

    self.device_id = VKDevice.default_device

    if self.device_id >= len(VKDevice.devices):
      raise RuntimeError(f"Tried opening {device}, but only {len(VKDevice.devices)} vulkan devices available!")

    self.pdev = VKDevice.devices[self.device_id]

    VKDevice.instance_refcount += 1

    self.dev_props = vk.VkPhysicalDeviceProperties()
    vk.vkGetPhysicalDeviceProperties(self.pdev, ctypes.byref(self.dev_props))
    self.vkdn = vk.string_cast(self.dev_props.deviceName)

    self.mem_props = vk.VkPhysicalDeviceMemoryProperties()
    vk.vkGetPhysicalDeviceMemoryProperties(self.pdev, ctypes.byref(self.mem_props))

    vk.vkGetPhysicalDeviceQueueFamilyProperties(self.pdev, ctypes.byref(n:=ctypes.c_uint32()), None)
    self.qf_properties = (vk.VkQueueFamilyProperties * n.value)()
    vk.vkGetPhysicalDeviceQueueFamilyProperties(self.pdev, ctypes.byref(n), self.qf_properties)

    checkz(vk.vkEnumerateDeviceExtensionProperties(self.pdev, None, ctypes.byref(n:=ctypes.c_uint32()), None))
    c_extension_properties = (vk.VkExtensionProperties * n.value)()
    checkz(vk.vkEnumerateDeviceExtensionProperties(self.pdev, None, ctypes.byref(n), c_extension_properties))
    self.extension_properties = {vk.string_cast(ext.extensionName): ext.specVersion for ext in c_extension_properties}

    self.enabled_extensions: dict[str, int] = {}

    def _ext(requested:dict[str, int], mandatory:bool=False):
      missing = {rn:(rv,dv) for rn,rv in requested.items() if not (dv:=self.extension_properties.get(rn, -1)) >= rv}
      if not missing: self.enabled_extensions.update({rn:self.extension_properties[rn] for rn in requested})
      if mandatory and missing:
        notsupp = [f"{rn} v{rv} ({'not found' if dv == -1 else f'device advertises v{dv}'})" for rn,(rv,dv) in missing.items()]
        raise RuntimeError(f"{device}: {self.vkdn} doesn't support {', '.join(notsupp)}")
      return not missing

    if DEBUG >= 1:
      str_exts = [f'{rn} v{rv}' for rn,rv in self.enabled_extensions.items()]
      print(f"{device}: opening {self.vkdn}{(' with ' + ', '.join(str_exts)) if str_exts else ''}")

    cq_fidx = next((i for i,q in enumerate(self.qf_properties) if q.queueFlags & vk.VK_QUEUE_COMPUTE_BIT and q.queueCount > 0), None)
    if cq_fidx is None: raise RuntimeError("No suitable compute queues found!")

    cq_info = vk.VkDeviceQueueCreateInfo(
      sType=vk.VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
      queueFamilyIndex=cq_fidx,
      queueCount=1,
      pQueuePriorities=ctypes.pointer(ctypes.c_float(1.0)),
    )
    base_feats = vk.VkPhysicalDeviceFeatures(
        shaderInt64=vk.VK_TRUE,
        shaderInt16=vk.VK_TRUE,
    )
    bdev_ref_feats = vk.VkPhysicalDeviceBufferDeviceAddressFeatures(
      sType=vk.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES,
      bufferDeviceAddress=vk.VK_TRUE,
    )
    f16i8_feats = vk.VkPhysicalDeviceShaderFloat16Int8Features(
        sType=vk.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES,
        shaderFloat16=vk.VK_TRUE,
        shaderInt8=vk.VK_TRUE,
    )
    storage8_feats = vk.VkPhysicalDevice8BitStorageFeatures(
        sType=vk.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_8BIT_STORAGE_FEATURES,
        storageBuffer8BitAccess=vk.VK_TRUE,
        uniformAndStorageBuffer8BitAccess=vk.VK_TRUE,
    )
    storage16_feats = vk.VkPhysicalDevice16BitStorageFeatures(
        sType=vk.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES,
        storageBuffer16BitAccess=vk.VK_TRUE,
        uniformAndStorageBuffer16BitAccess=vk.VK_TRUE,
    )
    scalar_feats = vk.VkPhysicalDeviceScalarBlockLayoutFeatures(
        sType=vk.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SCALAR_BLOCK_LAYOUT_FEATURES,
        scalarBlockLayout=vk.VK_TRUE,
    )
    ldev_info = vk.VkDeviceCreateInfo(
      sType=vk.VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
      enabledExtensionCount=(eecnt:=len(self.enabled_extensions)),
      ppEnabledExtensionNames=(ctypes.POINTER(ctypes.c_char) * eecnt)(*(vk.char_pointer_cast(s) for s in self.enabled_extensions)),
      queueCreateInfoCount=1,
      pQueueCreateInfos=ctypes.pointer(cq_info),
      pEnabledFeatures=ctypes.pointer(base_feats),
      pNext=pchain([bdev_ref_feats, f16i8_feats, storage8_feats, storage16_feats, scalar_feats]),
    )
    self.ldev = vk.VkDevice()
    checkz(vk.vkCreateDevice(self.pdev, ctypes.pointer(ldev_info), None, ctypes.byref(self.ldev)))

    self.cq = vk.VkQueue()
    vk.vkGetDeviceQueue(self.ldev, cq_fidx, 0, ctypes.byref(self.cq))

    cq_pool_info = vk.VkCommandPoolCreateInfo(
      sType=vk.VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      queueFamilyIndex=cq_fidx,
    )
    self.cq_pool = vk.VkCommandPool()
    checkz(vk.vkCreateCommandPool(self.ldev, ctypes.pointer(cq_pool_info), None, ctypes.byref(self.cq_pool)))

    self.live_buffers: WeakSet[VKBuf] = WeakSet()
    self.live_programs: WeakSet[Any] = WeakSet()

    VKGraph: type|None = None
    if VKDevice.graph_enabled:
      from tinygrad.runtime.graph.vk import VKGraph
    super().__init__(device, VKAllocator(self), [(GLSLRenderer, functools.partial(GLSLCompiler, VKDevice.validation_gpuassist_source_debug))],
                     functools.partial(VKProgram, self), VKGraph)

  @functools.cache
  def dynfn(self, name:str):
    return ctypes.cast(vk.vkGetDeviceProcAddr(self.ldev, vk.char_pointer_cast(name)), getattr(vk, f"PFN_{name}"))

  def synchronize(self):
    checkz(vk.vkQueueWaitIdle(self.cq))

  def finalize(self):
    self.synchronize()
    for prg in self.live_programs.copy(): prg.__del__()
    for buf in self.live_buffers.copy(): self.allocator._free(buf, BufferSpec())
    vk.vkDestroyCommandPool(self.ldev, self.cq_pool, None)
    vk.vkDestroyDevice(self.ldev, None)
    VKDevice.instance_refcount -= 1
    if VKDevice.instance_refcount == 0:
      vk.vkDestroyInstance(VKDevice.instance, None)
      VKDevice.instance = None
      VKDevice.devices = None

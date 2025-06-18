from __future__ import annotations
import subprocess, shutil, time, json, functools, ctypes, struct, tinygrad.runtime.autogen.vulkan as vk
from typing import Any
from dataclasses import dataclass
from weakref import WeakSet
from tinygrad.device import LRUAllocator, BufferSpec, Compiler, Compiled
from tinygrad.renderer.glsl import GLSLRenderer
from tinygrad.helpers import to_mv, unwrap, getenv, DEBUG

def checkz(result:int):
  if result != vk.VK_SUCCESS:
    result_str = vk.VkResult__enumvalues.get(result, None)
    raise RuntimeError(f"{result}{f' ({result_str})' if result_str else ''}")

class GLSLCompiler(Compiler):
  def __init__(self, cachekey:str|None=None):
    super().__init__("compile_glsl" if cachekey is None else cachekey)
  def compile(self, src:str) -> bytes:
    lib = subprocess.check_output(["glslc", "--target-env=vulkan1.3", "-fshader-stage=compute", "-", "-o", "-"], input=src.encode("utf-8"))
    if shutil.which("spirv-val"):
      subprocess.run(["spirv-val", "--target-env", "vulkan1.3", "-"], input=lib, check=True)
    meta_start = src.find("// META: ") + 9
    meta_end = src.find("\n", meta_start)
    meta = json.loads(src[meta_start:meta_end])
    return struct.pack("<II", meta["bufs"], meta["vars"]) + lib
  def disassemble(self, lib:bytes):
    subprocess.run(["spirv-dis", "-"], input=lib[8:], check=True)

class VKProgram:
  def __init__(self, dev:VKDevice, name:str, lib:bytes):
    self.dev = dev

    self.bufs, self.vars = struct.unpack("<II", lib[:8])
    self.spirv = lib[8:]

    shader_info = vk.VkShaderModuleCreateInfo(
      sType=vk.VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      codeSize=len(self.spirv),
      pCode=(ctypes.c_uint32 * (len(self.spirv) // 4)).from_buffer_copy(self.spirv),
    )
    self.shader = vk.VkShaderModule()
    checkz(vk.vkCreateShaderModule(self.dev.ldev, ctypes.byref(shader_info), None, ctypes.byref(self.shader)))

    bindings = [
      vk.VkDescriptorSetLayoutBinding(
        binding=i,
        descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        descriptorCount=1,
        stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT,
      ) for i in range(self.bufs)
    ]
    if self.vars:
      bindings += [
        vk.VkDescriptorSetLayoutBinding(
          binding=self.bufs,
          descriptorType=vk.VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
          descriptorCount=1,
          stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT,
        )
      ]
    descriptor_set_layout_create_info = vk.VkDescriptorSetLayoutCreateInfo(
      sType=vk.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      flags=vk.VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT,
      bindingCount=len(bindings),
      pBindings=(vk.VkDescriptorSetLayoutBinding * len(bindings))(*bindings),
      pNext=ctypes.cast(ctypes.pointer(vk.VkDescriptorSetLayoutBindingFlagsCreateInfo(
        sType=vk.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO,
        bindingCount=len(bindings),
        pBindingFlags=(vk.VkDescriptorBindingFlags * len(bindings))(*([vk.VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT] * len(bindings))),
      )), ctypes.c_void_p),
    )
    self.descriptor_set_layout = vk.VkDescriptorSetLayout()
    checkz(vk.vkCreateDescriptorSetLayout(
      self.dev.ldev,
      ctypes.pointer(descriptor_set_layout_create_info),
      None,
      ctypes.byref(self.descriptor_set_layout)
    ))

    descriptor_pool_create_info_pool_sizes = [
      vk.VkDescriptorPoolSize(
        type=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        descriptorCount=self.bufs,
      )
    ]
    if self.vars:
      descriptor_pool_create_info_pool_sizes += [
        vk.VkDescriptorPoolSize(
          type=vk.VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
          descriptorCount=1,
        )
      ]
    descriptor_pool_create_info = vk.VkDescriptorPoolCreateInfo(
      sType=vk.VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
      flags=vk.VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT,
      maxSets=1,
      poolSizeCount=(psc:=len(descriptor_pool_create_info_pool_sizes)),
      pPoolSizes=(vk.VkDescriptorPoolSize * psc)(*descriptor_pool_create_info_pool_sizes),
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

    self.indirect_buffer = self.dev.allocator.alloc(4 * (3 + self.vars))

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
    vk.vkCmdDispatchIndirect(self.cbuf, self.indirect_buffer.buf, 0)
    vk.vkCmdPipelineBarrier(self.cbuf, vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, ctypes.pointer(barrier), 0, None, 0, None) # noqa: E501
    checkz(vk.vkEndCommandBuffer(self.cbuf))

    fence_create_info = vk.VkFenceCreateInfo(
      sType=vk.VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
      flags=vk.VK_FENCE_CREATE_SIGNALED_BIT,
    )
    self.fence = vk.VkFence()
    checkz(vk.vkCreateFence(self.dev.ldev, ctypes.pointer(fence_create_info), None, ctypes.byref(self.fence)))

    self.dev.live_programs.add(self)

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
    assert len(bufs) == self.bufs
    assert len(vals) == self.vars

    checkz(vk.vkWaitForFences(self.dev.ldev, 1, ctypes.pointer(self.fence), True, -1))
    checkz(vk.vkResetFences(self.dev.ldev, 1, ctypes.pointer(self.fence)))

    st = time.perf_counter()

    arg_writes = [
      vk.VkWriteDescriptorSet(
        sType=vk.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        dstSet=self.descriptor_set,
        dstBinding=i,
        descriptorCount=1,
        descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        pBufferInfo=ctypes.pointer(vk.VkDescriptorBufferInfo(
          buffer=buf.buf,
          offset=0,
          range=vk.VK_WHOLE_SIZE,
        )),
      ) for i, buf in enumerate(bufs)
    ]
    if self.vars:
      arg_writes += [
        vk.VkWriteDescriptorSet(
          sType=vk.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
          dstSet=self.descriptor_set,
          dstBinding=self.bufs,
          descriptorCount=1,
          descriptorType=vk.VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
          pBufferInfo=ctypes.pointer(vk.VkDescriptorBufferInfo(
            buffer=self.indirect_buffer.buf,
            offset=12,
            range=vk.VK_WHOLE_SIZE,
          )),
        )
      ]
    vk.vkUpdateDescriptorSets(self.dev.ldev, len(arg_writes), (vk.VkWriteDescriptorSet * len(arg_writes))(*arg_writes), 0, None)
    self.indirect_buffer.as_mv()[:] = struct.pack(f"<{3+len(vals)}I", *global_size, *vals)

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
            vk.VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT,
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
    )

    checkz(vk.vkAllocateMemory(self.dev.ldev, ctypes.pointer(mem_info), None, ctypes.byref(mem:=vk.VkDeviceMemory())))

    checkz(vk.vkBindBufferMemory(self.dev.ldev, buf, mem, 0))

    checkz(vk.vkMapMemory(self.dev.ldev, mem, 0, size, 0, ctypes.byref(cpu_ptr:=ctypes.c_void_p())))

    opaque = VKBuf(buf, mem, unwrap(cpu_ptr.value), size)
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

  instance = None
  instance_refcount = 0
  devices = None

  def __init__(self, device:str):
    if VKDevice.instance is None:
      instance_layer_names: dict[str, None] = {}
      instance_create_info_next = None

      if getenv("VKVAL", 0):
        instance_layer_names["VK_LAYER_KHRONOS_validation"] = None
        feat_vals = (vk.VkValidationFeatureEnableEXT * 3)(
          vk.VK_VALIDATION_FEATURE_ENABLE_BEST_PRACTICES_EXT,
          vk.VK_VALIDATION_FEATURE_ENABLE_SYNCHRONIZATION_VALIDATION_EXT,
          vk.VK_VALIDATION_FEATURE_ENABLE_GPU_ASSISTED_EXT,
        )
        instance_create_info_next = vk.VkValidationFeaturesEXT(
          sType = vk.VK_STRUCTURE_TYPE_VALIDATION_FEATURES_EXT,
          enabledValidationFeatureCount = len(feat_vals),
          pEnabledValidationFeatures = feat_vals
        )

      instance_create_info = vk.VkInstanceCreateInfo(
        sType=vk.VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        pApplicationInfo=ctypes.pointer(vk.VkApplicationInfo(
          sType=vk.VK_STRUCTURE_TYPE_APPLICATION_INFO,
          apiVersion=vk.VK_API_VERSION_1_3,
        )),
        enabledLayerCount=(elcnt:=len(instance_layer_names)),
        ppEnabledLayerNames=(ctypes.POINTER(ctypes.c_char) * elcnt)(*(vk.char_pointer_cast(s) for s in instance_layer_names)),
        pNext=ctypes.cast(ctypes.pointer(instance_create_info_next), ctypes.c_void_p) if instance_create_info_next is not None else None,
      )
      VKDevice.instance = vk.VkInstance()
      checkz(vk.vkCreateInstance(ctypes.pointer(instance_create_info), None, ctypes.byref(VKDevice.instance)))

      checkz(vk.vkEnumeratePhysicalDevices(VKDevice.instance, ctypes.byref(n:=ctypes.c_uint32()), None))
      if n.value == 0: raise RuntimeError("No vulkan devices found!")

      VKDevice.devices = (vk.VkPhysicalDevice * n.value)()
      checkz(vk.vkEnumeratePhysicalDevices(VKDevice.instance, ctypes.byref(n), VKDevice.devices))

    self.device_id = getenv("VKDEV", 0)

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
    descr_idx_feats = vk.VkPhysicalDeviceDescriptorIndexingFeatures(
      sType=vk.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES,
      descriptorBindingStorageBufferUpdateAfterBind=vk.VK_TRUE,
    )
    ldev_info = vk.VkDeviceCreateInfo(
      sType=vk.VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
      enabledExtensionCount=(eecnt:=len(self.enabled_extensions)),
      ppEnabledExtensionNames=(ctypes.POINTER(ctypes.c_char) * eecnt)(*(vk.char_pointer_cast(s) for s in self.enabled_extensions)),
      queueCreateInfoCount=1,
      pQueueCreateInfos=ctypes.pointer(cq_info),
      pNext=ctypes.cast(ctypes.pointer(descr_idx_feats), ctypes.c_void_p),
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
    self.live_programs: WeakSet[VKProgram] = WeakSet()

    super().__init__(device, VKAllocator(self), [(GLSLRenderer, GLSLCompiler)], functools.partial(VKProgram, self))

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

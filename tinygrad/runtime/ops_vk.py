from __future__ import annotations
import os, subprocess, functools, weakref, ctypes, tinygrad.runtime.autogen.vulkan as vk
from typing import Any
from dataclasses import dataclass
from tinygrad.device import LRUAllocator, BufferSpec, DMAFdRef, Compiler, Compiled
from tinygrad.renderer.cstyle import OpenCLRenderer
from tinygrad.helpers import to_mv, unwrap, DEBUG

def checkz(result:int, ret=None):
  if result != 0: raise RuntimeError(f"{result}")
  return ret

class CLCompiler(Compiler):
  def compile(self, src:str) -> bytes:
    return subprocess.check_output("clang -x cl --target=spirv64-unknown-unknown -I/nix/store/939zjqv7wdnk5q1669vnay4n1v45786l-clang-20.1.5-lib/lib/clang/20/include - -o -".split(), input=src.encode("utf-8"))

class VKProgram:
  MAX_BUFFERS = 64

  def __init__(self, dev:VKDevice, name:str, lib:bytes):
    self.dev, self.name, self.lib = dev, name, lib

    shader_info = vk.VkShaderModuleCreateInfo(
      sType=vk.VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      pCode=ctypes.cast(lib, ctypes.POINTER(ctypes.c_uint32)),
      codeSize=len(lib),
    )
    self.shader = vk.VkShaderModule()
    checkz(vk.vkCreateShaderModule(dev.ldev, ctypes.byref(shader_info), None, ctypes.byref(self.shader)))

    bindings = [
      vk.VkDescriptorSetLayoutBinding(
        binding=i,
        descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        descriptorCount=1,
        stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT
      ) for i in range(VKProgram.MAX_BUFFERS)
    ]

    layout_info = vk.VkDescriptorSetLayoutCreateInfo(
      sType=vk.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      bindingCount=len(bindings),
      pBindings=(vk.VkDescriptorSetLayoutBinding * len(bindings))(*bindings)
    )

    self.desc_layout = vk.VkDescriptorSetLayout()
    checkz(vk.vkCreateDescriptorSetLayout(dev.ldev, ctypes.byref(layout_info), None, ctypes.byref(self.desc_layout)))

    pipeline_layout_info = vk.VkPipelineLayoutCreateInfo(
      sType=vk.VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      setLayoutCount=1,
      pSetLayouts=ctypes.pointer(self.desc_layout)
    )

    self.pipeline_layout = vk.VkPipelineLayout()
    checkz(vk.vkCreatePipelineLayout(dev.ldev, ctypes.byref(pipeline_layout_info), None, ctypes.byref(self.pipeline_layout)))

    stage_info = vk.VkPipelineShaderStageCreateInfo(
      sType=vk.VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
      stage=vk.VK_SHADER_STAGE_COMPUTE_BIT,
      module=self.shader,
      pName=vk.char_pointer_cast(name)
    )

    pipeline_info = vk.VkComputePipelineCreateInfo(
      sType=vk.VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
      stage=stage_info,
      layout=self.pipeline_layout
    )

    self.pipeline = vk.VkPipeline()
    checkz(vk.vkCreateComputePipelines(dev.ldev, None, 1, ctypes.byref(pipeline_info), None, ctypes.byref(self.pipeline)))

    pool_size = vk.VkDescriptorPoolSize(
      type=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      descriptorCount=VKProgram.MAX_BUFFERS
    )

    pool_info = vk.VkDescriptorPoolCreateInfo(
      sType=vk.VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
      maxSets=1,
      poolSizeCount=1,
      pPoolSizes=ctypes.pointer(pool_size)
    )

    self.desc_pool = vk.VkDescriptorPool()
    checkz(vk.vkCreateDescriptorPool(dev.ldev, ctypes.byref(pool_info), None, ctypes.byref(self.desc_pool)))

  def __del__(self):
    self.dev.synchronize()
    vk.vkDestroyDescriptorPool(self.dev.ldev, self.desc_pool, None)
    vk.vkDestroyPipeline(self.dev.ldev, self.pipeline, None)
    vk.vkDestroyPipelineLayout(self.dev.ldev, self.pipeline_layout, None)
    vk.vkDestroyDescriptorSetLayout(self.dev.ldev, self.desc_layout, None)
    vk.vkDestroyShaderModule(self.dev.ldev, self.shader, None)

  def __call__(self, *bufs:VKBuf, global_size:tuple[int,int,int]=(1,1,1), local_size:tuple[int,int,int]=(1,1,1), vals:tuple[int, ...]=(), wait=False):
    alloc_info = vk.VkDescriptorSetAllocateInfo(
      sType=vk.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
      descriptorPool=self.desc_pool,
      descriptorSetCount=1,
      pSetLayouts=ctypes.pointer(self.desc_layout)
    )

    checkz(vk.vkAllocateDescriptorSets(self.dev.ldev, ctypes.byref(alloc_info), ctypes.byref(desc_set:=vk.VkDescriptorSet())))

    writes = [
      vk.VkWriteDescriptorSet(
        sType=vk.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        dstSet=desc_set,
        dstBinding=i,
        dstArrayElement=0,
        descriptorCount=1,
        descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        pBufferInfo=ctypes.pointer(vk.VkDescriptorBufferInfo(buffer=buf.buf, offset=0, range=vk.VK_WHOLE_SIZE))
      ) for i, buf in enumerate(bufs)
    ]

    vk.vkUpdateDescriptorSets(self.dev.ldev, len(writes), (vk.VkWriteDescriptorSet * len(writes))(*writes), 0, None)

    alloc_info = vk.VkCommandBufferAllocateInfo(
      sType=vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      commandPool=self.dev.cq_pool,
      level=vk.VK_COMMAND_BUFFER_LEVEL_PRIMARY,
      commandBufferCount=1
    )

    checkz(vk.vkAllocateCommandBuffers(self.dev.ldev, ctypes.byref(alloc_info), ctypes.byref(cmd_buf:=vk.VkCommandBuffer())))

    begin_info = vk.VkCommandBufferBeginInfo(
      sType=vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
      flags=vk.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT
    )

    checkz(vk.vkBeginCommandBuffer(cmd_buf, ctypes.byref(begin_info)))

    vk.vkCmdBindPipeline(cmd_buf, vk.VK_PIPELINE_BIND_POINT_COMPUTE, self.pipeline)
    vk.vkCmdBindDescriptorSets(cmd_buf, vk.VK_PIPELINE_BIND_POINT_COMPUTE, self.pipeline_layout, 0, 1, ctypes.pointer(desc_set), 0, None)

    assert not vals, vals
    # TODO: vals
    # if vals:
    #   vals_data = (ctypes.c_uint32 * len(vals))(*vals)
    #   vk.vkCmdPushConstants(cmd_buf, self.pipeline_layout, vk.VK_SHADER_STAGE_COMPUTE_BIT, 0, len(vals) * 4, ctypes.cast(vals_data, ctypes.c_void_p))

    vk.vkCmdDispatch(cmd_buf, *global_size)

    checkz(vk.vkEndCommandBuffer(cmd_buf))

    submit_info = vk.VkSubmitInfo(
      sType=vk.VK_STRUCTURE_TYPE_SUBMIT_INFO,
      commandBufferCount=1,
      pCommandBuffers=ctypes.pointer(cmd_buf)
    )

    checkz(vk.vkQueueSubmit(self.dev.cq, 1, ctypes.byref(submit_info), None))

    if wait: self.dev.synchronize()

@dataclass(frozen=True)
class VKBuf:
  buf: Any
  mem: Any
  cpu: int
  sz: int
  def __hash__(self): return id(self)

class VKAllocator(LRUAllocator['VKDevice']):
  def __init__(self, dev:VKDevice):
    self.live_buffers: set[VKBuf] = set()
    if dev.extensions_dmaref: self._as_dmaref = self._as_dmaref_impl
    super().__init__(dev)

  def _find_memory_type(self, type_filter:int, properties:int) -> int:
    for i in range(self.dev.mem_props.memoryTypeCount):
      if (type_filter & (1 << i)) and (self.dev.mem_props.memoryTypes[i].propertyFlags & properties) == properties:
        return i
    raise RuntimeError("Failed to find suitable memory type")

  def _alloc(self, size:int, options:BufferSpec):
    buf_info = vk.VkBufferCreateInfo(
      sType=vk.VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      size=size,
      usage=vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | vk.VK_BUFFER_USAGE_TRANSFER_SRC_BIT | vk.VK_BUFFER_USAGE_TRANSFER_DST_BIT,
      pNext=ctypes.cast(ctypes.pointer(vk.VkExternalMemoryBufferCreateInfo(
        sType=vk.VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO,
        handleTypes=vk.VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT
      )), ctypes.c_void_p) if self.dev.extensions_dmaref else None,
    )

    checkz(vk.vkCreateBuffer(self.dev.ldev, buf_info, None, ctypes.byref(buf:=vk.VkBuffer())))

    vk.vkGetBufferMemoryRequirements(self.dev.ldev, buf, ctypes.byref(mem_reqs:=vk.VkMemoryRequirements()))

    mem_props = vk.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | vk.VK_MEMORY_PROPERTY_HOST_CACHED_BIT
    mem_info = vk.VkMemoryAllocateInfo(
      sType=vk.VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      allocationSize=mem_reqs.size,
      memoryTypeIndex=self._find_memory_type(mem_reqs.memoryTypeBits, mem_props),
      pNext=ctypes.cast(ctypes.pointer(vk.VkExportMemoryAllocateInfo(
        sType=vk.VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO,
        handleTypes=vk.VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT
      )), ctypes.c_void_p) if self.dev.extensions_dmaref else None,
    )

    checkz(vk.vkAllocateMemory(self.dev.ldev, ctypes.byref(mem_info), None, ctypes.byref(mem:=vk.VkDeviceMemory())))

    checkz(vk.vkBindBufferMemory(self.dev.ldev, buf, mem, 0))

    checkz(vk.vkMapMemory(self.dev.ldev, mem, 0, size, 0, ctypes.byref(cpu_ptr:=ctypes.c_void_p())))

    opaque = VKBuf(buf, mem, unwrap(cpu_ptr.value), size)
    self.live_buffers.add(opaque)
    return opaque

  def _free(self, opaque:VKBuf, options:BufferSpec):
    self.dev.synchronize()
    vk.vkUnmapMemory(self.dev.ldev, opaque.mem)
    vk.vkDestroyBuffer(self.dev.ldev, opaque.buf, None)
    vk.vkFreeMemory(self.dev.ldev, opaque.mem, None)
    self.live_buffers.remove(opaque)

  def _as_buffer(self, opaque:VKBuf) -> memoryview:
    self.dev.synchronize()
    return to_mv(opaque.cpu, opaque.sz)

  def _copyin(self, dest:VKBuf, src:memoryview): self._as_buffer(dest)[:] = src
  def _copyout(self, dest:memoryview, src:VKBuf): dest[:] = self._as_buffer(src)
  def _as_dmaref_impl(self, opaque:VKBuf):
    get_fd_info = vk.VkMemoryGetFdInfoKHR(
      sType=vk.VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR,
      memory=opaque.mem,
      handleType=vk.VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT
    )
    checkz(self.dev.dynfn('vkGetMemoryFdKHR')(self.dev.ldev, ctypes.byref(get_fd_info), ctypes.byref(dmabuf_fd:=ctypes.c_int32())))
    dmaref = DMAFdRef(dmabuf_fd.value, 0, opaque.sz)
    weakref.finalize(dmaref, os.close, dmabuf_fd.value)
    return dmaref

# VK_ADD_LAYER_PATH="$(nix eval --raw nixpkgs\#vulkan-validation-layers)/share/vulkan/explicit_layer.d"
# VK_LOADER_LAYERS_ENABLE=VK_LAYER_KHRONOS_validation
class VKDevice(Compiled):
  instance = None
  instance_refcount = 0
  devices = None

  def __init__(self, device:str):
    device_id = int(device.split(":")[1]) if ":" in device else 0

    if VKDevice.instance is None:
      app_info = vk.VkApplicationInfo(
        sType=vk.VK_STRUCTURE_TYPE_APPLICATION_INFO,
        pApplicationName=vk.char_pointer_cast("tinygrad"),
        pEngineName=vk.char_pointer_cast("tinygrad"),
        apiVersion=vk.VK_API_VERSION_1_3,
      )
      create_info = vk.VkInstanceCreateInfo(
        sType=vk.VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        pApplicationInfo=ctypes.pointer(app_info),
      )
      VKDevice.instance = vk.VkInstance()
      checkz(vk.vkCreateInstance(ctypes.pointer(create_info), None, ctypes.byref(VKDevice.instance)))

      checkz(vk.vkEnumeratePhysicalDevices(VKDevice.instance, ctypes.byref(n:=ctypes.c_uint32()), None))
      if n.value == 0: raise RuntimeError("No vulkan devices found!")
      VKDevice.devices = (vk.VkPhysicalDevice * n.value)()
      checkz(vk.vkEnumeratePhysicalDevices(VKDevice.instance, ctypes.byref(n), VKDevice.devices))

    if device_id >= len(VKDevice.devices): raise RuntimeError(f"Tried opening {device}, but only {len(VKDevice.devices)} vulkan devices available!")
    VKDevice.instance_refcount += 1
    self.pdev = VKDevice.devices[device_id]

    self.dev_props = vk.VkPhysicalDeviceProperties()
    vk.vkGetPhysicalDeviceProperties(self.pdev, ctypes.byref(self.dev_props))

    self.mem_props = vk.VkPhysicalDeviceMemoryProperties()
    vk.vkGetPhysicalDeviceMemoryProperties(self.pdev, ctypes.byref(self.mem_props))

    vk.vkGetPhysicalDeviceQueueFamilyProperties(self.pdev, ctypes.byref(n:=ctypes.c_uint32()), None)
    self.qf_properties = (vk.VkQueueFamilyProperties * n.value)()
    vk.vkGetPhysicalDeviceQueueFamilyProperties(self.pdev, ctypes.byref(n), self.qf_properties)

    checkz(vk.vkEnumerateDeviceExtensionProperties(self.pdev, None, ctypes.byref(n:=ctypes.c_uint32()), None))
    c_extension_properties = (vk.VkExtensionProperties * n.value)()
    checkz(vk.vkEnumerateDeviceExtensionProperties(self.pdev, None, ctypes.byref(n), c_extension_properties))
    extension_properties = {vk.string_cast(ext.extensionName): ext.specVersion for ext in c_extension_properties}

    self.enabled_extensions: dict[str, int] = {}

    def _ext(needed:dict[str, int]):
      supported = all(nn in extension_properties and extension_properties[nn] >= nv for nn,nv in needed.items())
      if supported: self.enabled_extensions.update(needed)
      return supported

    self.extensions_dmaref = _ext({"VK_EXT_external_memory_dma_buf": 1, "VK_KHR_external_memory_fd": 1})

    if DEBUG >= 1: print(f"{device} opening {vk.string_cast(self.dev_props.deviceName)} with {', '.join(self.enabled_extensions.keys())}")

    cq_fidx = next((i for i,q in enumerate(self.qf_properties) if q.queueFlags & vk.VK_QUEUE_COMPUTE_BIT and q.queueCount > 0), None)
    if cq_fidx is None: raise RuntimeError("No suitable compute queues found!")

    cq_info = vk.VkDeviceQueueCreateInfo(
      sType=vk.VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
      queueFamilyIndex=cq_fidx,
      queueCount=1,
      pQueuePriorities=ctypes.pointer(ctypes.c_float(0.0)),
    )

    ldev_info = vk.VkDeviceCreateInfo(
      sType=vk.VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
      enabledExtensionCount=(ecnt:=len(self.enabled_extensions)),
      ppEnabledExtensionNames=(ctypes.POINTER(ctypes.c_char) * ecnt)(*(vk.char_pointer_cast(s) for s in self.enabled_extensions)),
      queueCreateInfoCount=1,
      pQueueCreateInfos=ctypes.pointer(cq_info),
    )

    self.ldev = vk.VkDevice()
    checkz(vk.vkCreateDevice(self.pdev, ctypes.byref(ldev_info), None, ctypes.byref(self.ldev)))

    self.cq = vk.VkQueue()
    vk.vkGetDeviceQueue(self.ldev, cq_fidx, 0, ctypes.byref(self.cq))

    cq_pool_info = vk.VkCommandPoolCreateInfo(
      sType=vk.VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      queueFamilyIndex=cq_fidx,
    )

    self.cq_pool = vk.VkCommandPool()
    checkz(vk.vkCreateCommandPool(self.ldev, ctypes.byref(cq_pool_info), None, ctypes.byref(self.cq_pool)))

    super().__init__(device, VKAllocator(self), OpenCLRenderer(), CLCompiler(), functools.partial(VKProgram, self))

  @functools.cache
  def dynfn(self, name:str):
    return ctypes.cast(vk.vkGetDeviceProcAddr(self.ldev, vk.char_pointer_cast(name)), getattr(vk, f"PFN_{name}"))

  def synchronize(self):
    checkz(vk.vkQueueWaitIdle(self.cq))

  def finalize(self):
    self.synchronize()
    for opaque in self.allocator.live_buffers.copy(): self.allocator._free(opaque, BufferSpec())
    vk.vkDestroyCommandPool(self.ldev, self.cq_pool, None)
    vk.vkDestroyDevice(self.ldev, None)
    VKDevice.instance_refcount -= 1
    if VKDevice.instance_refcount == 0:
      vk.vkDestroyInstance(VKDevice.instance, None)
      VKDevice.instance = None
      VKDevice.devices = None

if __name__ == '__main__':
  from tinygrad import Tensor
  a = Tensor.full((4,), 1.0, device='VK').contiguous().realize()
  print(a.numpy())
  # from tinygrad.dtype import dtypes
  # from tinygrad.device import Buffer
  # from tinygrad.runtime.support.ib import IBCtx
  # ctx = IBCtx(0)
  # buf = Buffer("VK:0", 4096*4, dtypes.uint8, preallocate=True)
  # buf.copyin(memoryview(bytearray(b"asdf"*1024*4)))
  # ctx.reg(buf)
  # buf.copyout(mv:=memoryview(bytearray(4096*4)))
  # print(bytes(mv))
  # lib = dev.compiler.compile("""
  #   #version 450
  #   #extension GL_EXT_scalar_block_layout  : require
  #   #extension GL_EXT_shader_8bit_storage  : require
  #   #extension GL_EXT_shader_explicit_arithmetic_types_int8 : require
  #   layout(set=0, binding=0, scalar) buffer BufOut { uint8_t data[]; } buf0;
  #   layout(set=0, binding=1, scalar) readonly buffer BufIn { uint8_t data[]; } buf1;
  #   void main() {
  #     uint idx = gl_GlobalInvocationID.x;
  #     buf0.data[idx] = buf1.data[idx] + uint8_t(1);
  #   }
  # """)
  # dev.runtime('main', lib)(a, a, global_size=(4, 1, 1))

import functools, struct, ctypes, time, tinygrad.runtime.autogen.vulkan as vk
from typing import cast
from tinygrad.helpers import unwrap, round_up, all_same, merge_dicts
from tinygrad.device import Buffer
from tinygrad.engine.realize import ExecItem, CompiledRunner
from tinygrad.engine.jit import GraphRunner
from tinygrad.runtime.ops_vk import VKDevice, VKProgram, checkz

class VKGraph(GraphRunner):
  dev: VKDevice

  def __init__(self, jit_cache: list[ExecItem], input_rawbuffers: list[Buffer], var_vals: dict[str, int]):
    super().__init__(jit_cache, input_rawbuffers, var_vals)
    assert all_same([ji.prg.dev for ji in jit_cache])
    self.real_dev = jit_cache[0].prg.dev
    self.real_dev.live_programs.add(self)

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
      self.real_dev.ldev,
      ctypes.pointer(descriptor_set_layout_create_info),
      None,
      ctypes.byref(self.descriptor_set_layout)
    ))

    descriptor_pool_create_info = vk.VkDescriptorPoolCreateInfo(
      sType=vk.VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
      maxSets=len(jit_cache),
      poolSizeCount=1,
      pPoolSizes=ctypes.pointer(vk.VkDescriptorPoolSize(
        type=vk.VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        descriptorCount=len(jit_cache),
      )),
    )
    self.descriptor_pool = vk.VkDescriptorPool()
    checkz(vk.vkCreateDescriptorPool(self.real_dev.ldev, ctypes.pointer(descriptor_pool_create_info), None, ctypes.byref(self.descriptor_pool)))

    descriptor_set_allocate_info = vk.VkDescriptorSetAllocateInfo(
      sType=vk.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
      descriptorPool=self.descriptor_pool,
      descriptorSetCount=len(jit_cache),
      pSetLayouts=(vk.VkDescriptorSetLayout * len(jit_cache))(*([self.descriptor_set_layout] * len(jit_cache))),
    )
    self.descriptor_sets = (vk.VkDescriptorSet * len(jit_cache))()
    checkz(vk.vkAllocateDescriptorSets(self.real_dev.ldev, ctypes.pointer(descriptor_set_allocate_info), self.descriptor_sets))

    kernarg_sizes = [cast(CompiledRunner, ji.prg)._prg.kernarg_size + 12 for ji in jit_cache]
    indirect_buffer_size = functools.reduce(lambda acc, x: round_up(acc, 64) + x, kernarg_sizes)
    self.indirect_buffer = self.real_dev.allocator.alloc(indirect_buffer_size)

    cbuf_allocate_info = vk.VkCommandBufferAllocateInfo(
      sType=vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      commandPool=self.real_dev.cq_pool,
      commandBufferCount=1,
    )
    self.cbuf = vk.VkCommandBuffer()
    checkz(vk.vkAllocateCommandBuffers(self.real_dev.ldev, ctypes.pointer(cbuf_allocate_info), ctypes.byref(self.cbuf)))

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

    kernarg_offset = 0
    self.kernels_bufs = []
    self.kernels_vars = []
    self.kernels_globals = []
    for i, ji in enumerate(jit_cache):
      jprg: CompiledRunner = ji.prg # type: ignore
      prg: VKProgram = jprg._prg # type: ignore
      kernarg_offset = round_up(kernarg_offset, 64)

      kernel_bufs = [unwrap(buf)._buf.gpu for buf in ji.bufs]
      all_vals = merge_dicts([var_vals, ji.fixedvars])
      kernel_vals = [all_vals[k.expr] for k in jprg.p.vars]
      kernel_dims, _ = jprg.p.launch_dims(var_vals)
      kernel_blob = struct.pack(f"<{len(kernel_bufs)}Q{len(kernel_vals)}i3I", *kernel_bufs, *kernel_vals, *kernel_dims)
      self.indirect_buffer.as_mv()[kernarg_offset:kernarg_offset+prg.kernarg_size+12] = kernel_blob

      self.kernels_bufs.append(kernarg_offset)
      self.kernels_vars.append(kernarg_offset + len(ji.bufs) * 8)

      dset_write = vk.VkWriteDescriptorSet(
        sType=vk.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        dstSet=self.descriptor_sets[i],
        dstBinding=0,
        descriptorCount=1,
        descriptorType=vk.VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        pBufferInfo=ctypes.pointer(vk.VkDescriptorBufferInfo(
          buffer=self.indirect_buffer.buf,
          offset=kernarg_offset,
          range=prg.kernarg_size,
        )),
      )
      vk.vkUpdateDescriptorSets(self.real_dev.ldev, 1, ctypes.pointer(dset_write), 0, None)

      kernarg_offset += prg.kernarg_size
      self.kernels_globals.append(kernarg_offset)

      vk.vkCmdBindPipeline(self.cbuf, vk.VK_PIPELINE_BIND_POINT_COMPUTE, prg.pipeline)
      vk.vkCmdBindDescriptorSets(self.cbuf, vk.VK_PIPELINE_BIND_POINT_COMPUTE, prg.pipeline_layout, 0, 1, ctypes.pointer(self.descriptor_sets[i]), 0, None) # noqa: E501
      vk.vkCmdDispatchIndirect(self.cbuf, self.indirect_buffer.buf, kernarg_offset)
      vk.vkCmdPipelineBarrier(self.cbuf, vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, ctypes.pointer(barrier), 0, None, 0, None) # noqa: E501

      kernarg_offset += 12

    assert kernarg_offset == indirect_buffer_size

    checkz(vk.vkEndCommandBuffer(self.cbuf))

    fence_create_info = vk.VkFenceCreateInfo(
      sType=vk.VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
      flags=vk.VK_FENCE_CREATE_SIGNALED_BIT,
    )
    self.fence = vk.VkFence()
    checkz(vk.vkCreateFence(self.real_dev.ldev, ctypes.pointer(fence_create_info), None, ctypes.byref(self.fence)))

  def __del__(self):
    if hasattr(self, "fence"):
      checkz(vk.vkWaitForFences(self.real_dev.ldev, 1, ctypes.pointer(self.fence), True, -1))
      checkz(vk.vkResetFences(self.real_dev.ldev, 1, ctypes.pointer(self.fence)))
      vk.vkDestroyFence(self.real_dev.ldev, self.fence, None)
      del self.fence
    if hasattr(self, "cbuf"):
      vk.vkFreeCommandBuffers(self.real_dev.ldev, self.real_dev.cq_pool, 1, ctypes.pointer(self.cbuf))
      del self.cbuf
    if hasattr(self, "indirect_buffer"):
      self.real_dev.allocator.free(self.indirect_buffer, self.indirect_buffer.sz)
      del self.indirect_buffer
    if hasattr(self, "descriptor_pool"):
      vk.vkDestroyDescriptorPool(self.real_dev.ldev, self.descriptor_pool, None)
      del self.descriptor_pool
      del self.descriptor_sets
    if hasattr(self, "descriptor_set_layout"):
      vk.vkDestroyDescriptorSetLayout(self.real_dev.ldev, self.descriptor_set_layout, None)
      del self.descriptor_set_layout

  def __call__(self, rawbufs: list[Buffer], var_vals: dict[str, int], wait=False):
    checkz(vk.vkWaitForFences(self.real_dev.ldev, 1, ctypes.pointer(self.fence), True, -1))
    checkz(vk.vkResetFences(self.real_dev.ldev, 1, ctypes.pointer(self.fence)))

    st = time.perf_counter()

    imv = self.indirect_buffer.as_mv()

    for (j,i),input_idx in self.input_replace.items():
      idx = self.kernels_bufs[j] + i * 8
      imv[idx:idx + 8] = struct.pack("<Q", rawbufs[input_idx]._buf.gpu)

    for j,i,val in self.updated_vars(var_vals):
      idx = self.kernels_vars[j] + i * 4
      imv[idx:idx + 4] = struct.pack("<i", val)

    for j,global_dims,_ in self.updated_launch_dims(var_vals):
      idx = self.kernels_globals[j]
      imv[idx:idx + 12] = struct.pack("<3I", *global_dims)

    submit_info = vk.VkSubmitInfo(
      sType=vk.VK_STRUCTURE_TYPE_SUBMIT_INFO,
      commandBufferCount=1,
      pCommandBuffers=ctypes.pointer(self.cbuf),
    )
    checkz(vk.vkQueueSubmit(self.real_dev.cq, 1, ctypes.pointer(submit_info), self.fence))

    if wait:
      checkz(vk.vkWaitForFences(self.real_dev.ldev, 1, ctypes.pointer(self.fence), True, -1))
      return time.perf_counter() - st

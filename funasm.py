import time, struct, tinygrad.runtime.autogen.amd_gpu as amd_gpu
from tinygrad.renderer.support.rdna_asm import RDNACompiler
from tinygrad.runtime.ops_amd import AMDDevice, RDNAProgram

def trap(adev):
  adev.reg('regSPI_GDBG_TRAP_CONFIG').write(pipe0_en=1, pipe1_en=1, pipe2_en=1, pipe3_en=1)
  adev.reg('regSPI_GDBG_PER_VMID_CNTL').write(trap_en=1)
  adev.reg('regGRBM_GFX_INDEX').write(instance_broadcast_writes=1, se_broadcast_writes=1, sa_broadcast_writes=1)
  print(hex(adev.reg('regSQ_DEBUG_HOST_TRAP_STATUS').read()))
  adev.reg('regSQ_CMD').write(cmd=amd_gpu.SQ_IND_CMD_CMD_TRAP, mode=amd_gpu.SQ_IND_CMD_MODE_SINGLE, vm_id=0, wave_id=0, queue_id=0, data=0x4)
  print(hex(adev.reg('regSQ_DEBUG_HOST_TRAP_STATUS').read()))

dev = AMDDevice('AMD:0')
adev = dev.dev_iface.adev # type: ignore
compiler = RDNACompiler()
lib = compiler.compile(open('funasm.s').read())
compiler.disassemble(lib)
prg = RDNAProgram(dev, 'fun', lib)
# --
out = dev.allocator.alloc(4096)
dev.allocator._copyin(out, memoryview(bytearray(b'\x00'*4096)))
# --
prg(out, global_size=(1,1,1), local_size=(1,1,1), wait=False)
# --
time.sleep(0.1)
trap(adev)
time.sleep(0.1)
# --
dev.synchronize()
# --
out_cpu = memoryview(bytearray(b'\x00'*4096))
dev.allocator._copyout(out_cpu, out)
print(f"Magic read from shader: {struct.unpack('<Q', out_cpu[0:8])[0]:#x}")

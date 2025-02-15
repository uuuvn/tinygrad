.amdgcn_target "amdgcn-amd-amdhsa--gfx1100"

.text
.globl	E_4
.p2align	8
.type	E_4,@function
E_4:
  s_load_b64 s[0:1], s[0:1], 0x0
  v_mov_b32_e32 v0, 0x3effb4cb
  s_delay_alu instid0(VALU_DEP_1)
  v_dual_mov_b32 v4, 0 :: v_dual_mov_b32 v1, v0
  v_mov_b32_e32 v2, v0
  v_mov_b32_e32 v3, v0
  s_waitcnt lgkmcnt(0)
  global_store_b128 v4, v[0:3], s[0:1]
  s_nop 0
  s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)
  s_endpgm
.text
.Lfunc_end0:
  .size	E_4, .Lfunc_end0-E_4

.section	.rodata,#alloc
.p2align	6, 0x0
.amdhsa_kernel E_4
  .amdhsa_group_segment_fixed_size 0
  .amdhsa_private_segment_fixed_size 0
  .amdhsa_kernarg_size 8
  .amdhsa_user_sgpr_count 15
  .amdhsa_user_sgpr_dispatch_ptr 0
  .amdhsa_user_sgpr_queue_ptr 0
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_user_sgpr_dispatch_id 0
  .amdhsa_user_sgpr_private_segment_size 0
  .amdhsa_wavefront_size32 1
  .amdhsa_uses_dynamic_stack 0
  .amdhsa_enable_private_segment 0
  .amdhsa_system_sgpr_workgroup_id_x 1
  .amdhsa_system_sgpr_workgroup_id_y 0
  .amdhsa_system_sgpr_workgroup_id_z 0
  .amdhsa_system_sgpr_workgroup_info 0
  .amdhsa_system_vgpr_workitem_id 0
  .amdhsa_next_free_vgpr 5
  .amdhsa_next_free_sgpr 2
  .amdhsa_reserve_vcc 0
  .amdhsa_float_round_mode_32 0
  .amdhsa_float_round_mode_16_64 0
  .amdhsa_float_denorm_mode_32 3
  .amdhsa_float_denorm_mode_16_64 3
  .amdhsa_dx10_clamp 1
  .amdhsa_ieee_mode 1
  .amdhsa_fp16_overflow 0
  .amdhsa_workgroup_processor_mode 0
  .amdhsa_memory_ordered 1
  .amdhsa_forward_progress 0
  .amdhsa_shared_vgpr_count 0
  .amdhsa_exception_fp_ieee_invalid_op 0
  .amdhsa_exception_fp_denorm_src 0
  .amdhsa_exception_fp_ieee_div_zero 0
  .amdhsa_exception_fp_ieee_overflow 0
  .amdhsa_exception_fp_ieee_underflow 0
  .amdhsa_exception_fp_ieee_inexact 0
  .amdhsa_exception_int_div_zero 0
.end_amdhsa_kernel
.amdgpu_metadata
---
amdhsa.kernels:
  - .args:
      - .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 8
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 1
    .name:           E_4
    .private_segment_fixed_size: 0
    .sgpr_count:     2
    .sgpr_spill_count: 0
    .symbol:         E_4.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     5
    .vgpr_spill_count: 0
    .wavefront_size: 32
    .workgroup_processor_mode: 0
amdhsa.target:   amdgcn-amd-amdhsa--gfx1100
amdhsa.version:
  - 1
  - 2
...

.end_amdgpu_metadata

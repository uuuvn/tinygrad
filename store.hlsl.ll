; ModuleID = 'store.hlsl'
source_filename = "store.hlsl"
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64-G10"
target triple = "spirv1.6-unknown-vulkan1.3-compute"

; Function Attrs: alwaysinline mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: write, inaccessiblemem: none)
define hidden spir_func void @_Z11E_32_32_4_aDv3_jS_(<3 x i32> noundef %0, <3 x i32> noundef %1) local_unnamed_addr #0 {
  %3 = extractelement <3 x i32> %0, i64 0
  %4 = shl i32 %3, 5
  %5 = extractelement <3 x i32> %1, i64 0
  %6 = add i32 %4, %5
  %7 = tail call noundef align 16 dereferenceable(16) ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.VulkanBuffer_a0v4f32_12_1t(target("spirv.VulkanBuffer", [0 x <4 x float>], 12, 1) poison, i32 %6)
  store <4 x float> splat (float 0x3FC82CA4C0000000), ptr addrspace(11) %7, align 16, !tbaa !3
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(none)
declare ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.VulkanBuffer_a0v4f32_12_1t(target("spirv.VulkanBuffer", [0 x <4 x float>], 12, 1), i32) #1

attributes #0 = { alwaysinline mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: write, inaccessiblemem: none) "frame-pointer"="all" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind willreturn memory(none) }

!llvm.module.flags = !{!0, !1}
!llvm.ident = !{!2}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 7, !"frame-pointer", i32 2}
!2 = !{!"clang version 22.0.0git (https://github.com/llvm/llvm-project.git 8e383e22e2ba68556bf815d55312b190633dbed7)"}
!3 = !{!4, !4, i64 0}
!4 = !{!"omnipotent char", !5, i64 0}
!5 = !{!"Simple C++ TBAA"}

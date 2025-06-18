// HLSL (DXC -> SPIR-V) compute shader for Vulkan 1.3
// Each thread writes one float4 to an SSBO at index: gid.x * 32 + lid.x

[[vk::binding(0, 0)]]
RWStructuredBuffer<float4> gData0;

[numthreads(32, 1, 1)] // matches reqd_work_group_size(32,1,1)
void E_32_32_4_a(uint3 groupID       : SV_GroupID,        // workgroup id (x,y,z)
                 uint3 groupThreadID : SV_GroupThreadID)  // local id within the workgroup
{
    const float v = 0.1888624131679535f;

    // Manual 1D index from group and local ids (32 threads per group in X)
    const uint idx4 = groupID.x * 32u + groupThreadID.x;

    gData0[idx4] = float4(v, v, v, v);
}

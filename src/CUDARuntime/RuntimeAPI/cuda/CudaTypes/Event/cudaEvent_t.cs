using System;

namespace Hybridizer.Runtime.CUDAImports
{
    /// <summary>
    /// CUDA event types
    /// </summary>
    [IntrinsicType("cudaEvent_t")]
    public struct cudaEvent_t
    {
#pragma warning disable 0169
        IntPtr evt;
#pragma warning restore 0169
    }
}
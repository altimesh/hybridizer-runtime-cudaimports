using System;

namespace Hybridizer.Runtime.CUDAImports
{
    /// <summary>
    /// CUDA mipmapped array (as source argument)
    /// </summary>
    [IntrinsicType("cudaMipmappedArray_const_t")]
    public struct cudaMipmappedArray_const_t
    {
#pragma warning disable 0169
        IntPtr _inner;
#pragma warning restore 0169
    }
}
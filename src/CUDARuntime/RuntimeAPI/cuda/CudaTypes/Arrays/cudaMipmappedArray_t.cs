using System;

namespace Hybridizer.Runtime.CUDAImports
{
    /// <summary>
    ///  CUDA mipmapped array
    /// </summary>
    [IntrinsicType("cudaMipmappedArray_t")]
    public struct cudaMipmappedArray_t
    {
        IntPtr _inner;
    }
}
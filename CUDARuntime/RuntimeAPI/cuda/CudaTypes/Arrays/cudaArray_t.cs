using System;

namespace Hybridizer.Runtime.CUDAImports
{
    /// <summary>
    /// CUDA Array
    /// </summary>
    [IntrinsicType("cudaArray_t")]
    public struct cudaArray_t
    {
#pragma warning disable 0169
        IntPtr arr;
#pragma warning restore 0169
    }
}
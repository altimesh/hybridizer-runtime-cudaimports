using System;

namespace Hybridizer.Runtime.CUDAImports
{
    /// <summary>
    /// An opaque value that represents a CUDA texture object
    /// </summary>
    [IntrinsicType("cudaTextureObject_t")]
    public struct cudaTextureObject_t
    {
        ulong _inner;
    }
}
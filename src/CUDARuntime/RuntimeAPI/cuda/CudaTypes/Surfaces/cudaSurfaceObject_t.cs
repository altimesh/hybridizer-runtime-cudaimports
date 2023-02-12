using System;

namespace Hybridizer.Runtime.CUDAImports
{
    /// <summary>
    /// An opaque value that represents a CUDA Surface object
    /// </summary>
    [IntrinsicType("cudaSurfaceObject_t")]
    public struct cudaSurfaceObject_t
    {
        ulong _inner;
    }
}
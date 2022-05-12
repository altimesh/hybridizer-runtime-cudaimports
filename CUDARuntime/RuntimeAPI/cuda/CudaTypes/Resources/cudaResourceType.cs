using System.Runtime.InteropServices;
using System;

namespace Hybridizer.Runtime.CUDAImports
{
    /// <summary>
    /// CUDA resource types
    /// </summary>
    [IntrinsicType("")]
    public enum cudaResourceType : int
    {
        /// <summary>
        /// Array resource
        /// </summary>
        cudaResourceTypeArray = 0x00,
        /// <summary>
        /// Mipmapped array resource
        /// </summary>
        cudaResourceTypeMipmappedArray = 0x01,
        /// <summary>
        /// Linear resource
        /// </summary>
        cudaResourceTypeLinear = 0x02,
        /// <summary>
        /// Pitch 2D resource
        /// </summary>
        cudaResourceTypePitch2D = 0x03,
    }
}
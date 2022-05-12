using System;

namespace Hybridizer.Runtime.CUDAImports
{
    /// <summary>
    /// CUDA texture filter modes
    /// </summary>
    [IntrinsicType("cudaTextureFilterMode")]
    public enum cudaTextureFilterMode
    {
        /// <summary>
        /// Point filter mode
        /// </summary>
        cudaFilterModePoint = 0,
        /// <summary>
        /// Linear filter mode
        /// </summary>
        cudaFilterModeLinear = 1
    }
}
using System;

namespace Hybridizer.Runtime.CUDAImports
{
    /// <summary>
    /// CUDA texture address modes
    /// </summary>
    [IntrinsicType("cudaTextureAddressMode")]
    public enum cudaTextureAddressMode
    {
        /// <summary>
        /// Wrapping address mode
        /// </summary>
        cudaAddressModeWrap = 0,
        /// <summary>
        /// Clamp to edge address mode
        /// </summary>
        cudaAddressModeClamp = 1,
        /// <summary>
        /// Mirror address mode
        /// </summary>
        cudaAddressModeMirror = 2,
        /// <summary>
        /// Border address mode
        /// </summary>
        cudaAddressModeBorder = 3
    }
}
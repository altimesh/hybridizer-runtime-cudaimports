/* (c) ALTIMESH 2018 -- all rights reserved */
using System;
using System.Runtime.InteropServices;

namespace Hybridizer.Runtime.CUDAImports
{
    /// <summary>
    /// <see href="http://docs.nvidia.com/cuda/cufft/index.html#cufftcompatibility">online</see>
    /// </summary>
    [IntrinsicType("cufftCompatibility")]
    public enum cufftCompatibility : int
    {
        /// <summary>
        /// Compact data in native format (highest performance)
        /// </summary>
        CUFFT_COMPATIBILITY_NATIVE = 0,

        /// <summary>
        /// FFTW-compatible alignment (the default value)
        /// </summary>
        CUFFT_COMPATIBILITY_FFTW_PADDING = 1,

        /// <summary>
        /// Waives the C2R symmetry requirement input 
        /// </summary>
        CUFFT_COMPATIBILITY_FFTW_ASYMMETRIC = 2,

        /// <summary>
        /// ALL
        /// </summary>
        CUFFT_COMPATIBILITY_FFTW_ALL = CUFFT_COMPATIBILITY_FFTW_PADDING | CUFFT_COMPATIBILITY_FFTW_ASYMMETRIC
    }
}
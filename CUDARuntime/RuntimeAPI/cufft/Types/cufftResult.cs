/* (c) ALTIMESH 2018 -- all rights reserved */
using System;
using System.Runtime.InteropServices;

namespace Hybridizer.Runtime.CUDAImports
{
    /// <summary>
    /// 
    /// <see href="http://docs.nvidia.com/cuda/cufft/index.html#cufftresult">online</see>
    /// </summary>
    [IntrinsicType("cufftResult")]
    public enum cufftResult : int
    {
        /// <summary>
        /// The CUFFT operation was successful
        /// </summary>
        CUFFT_SUCCESS = 0, 
        /// <summary>
        /// CUFFT was passed an invalid plan handle
        /// </summary>
        CUFFT_INVALID_PLAN = 1,
        /// <summary>
        ///  CUFFT failed to allocate GPU or CPU memory
        /// </summary>
        CUFFT_ALLOC_FAILED = 2, 
        /// <summary>
        ///  No longer used 
        /// </summary>
        CUFFT_INVALID_TYPE = 3,
        /// <summary>
        ///  User specified an invalid pointer or parameter 
        /// </summary>
        CUFFT_INVALID_VALUE = 4, 
        /// <summary>
        ///  Driver or internal CUFFT library error
        /// </summary>
        CUFFT_INTERNAL_ERROR = 5, 
        /// <summary>
        ///  Failed to execute an FFT on the GPU 
        /// </summary>
        CUFFT_EXEC_FAILED = 6, 
        /// <summary>
        ///  The CUFFT library failed to initialize
        /// </summary>
        CUFFT_SETUP_FAILED = 7, 
        /// <summary>
        ///  User specified an invalid transform size
        /// </summary>
        CUFFT_INVALID_SIZE = 8, 
        /// <summary>
        ///  No longer used
        /// </summary>
        CUFFT_UNALIGNED_DATA = 9  
    }
}
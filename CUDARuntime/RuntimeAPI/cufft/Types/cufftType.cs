/* (c) ALTIMESH 2018 -- all rights reserved */
using System;
using System.Runtime.InteropServices;

namespace Hybridizer.Runtime.CUDAImports
{
    /// <summary>
    /// <see href="http://docs.nvidia.com/cuda/cufft/index.html#cufft-transform-types">online</see>
    /// </summary>
    [IntrinsicType("cufftType")]
    public enum cufftType : int
    {
        /// <summary>
        /// Real to complex (interleaved) 
        /// </summary>
        CUFFT_R2C = 0x2a,  
        /// <summary>
        /// Complex (interleaved) to real 
        /// </summary>
        CUFFT_C2R = 0x2c,  
        /// <summary>
        /// Complex to complex (interleaved) 
        /// </summary>
        CUFFT_C2C = 0x29,  
        /// <summary>
        /// Double to double-complex (interleaved) 
        /// </summary>
        CUFFT_D2Z = 0x6a,  
        /// <summary>
        /// Double-complex (interleaved) to double 
        /// </summary>
        CUFFT_Z2D = 0x6c,  
        /// <summary>
        /// Double-complex to double-complex (interleaved)
        /// </summary>
        CUFFT_Z2Z = 0x69   
    }
}
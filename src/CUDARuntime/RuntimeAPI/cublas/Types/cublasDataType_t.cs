/* (c) ALTIMESH 2018 -- all rights reserved */
using System;
using System.Runtime.InteropServices;
using System.Text;

namespace Hybridizer.Runtime.CUDAImports
{
    /// <summary>
    /// The cudaDataType_t type is an enumerant to specify the data precision. It is used when the data reference does not carry the type itself (e.g void *) 
    /// Complete documentation on <see href="https://docs.nvidia.com/cuda/cublas/index.html#cuda_datatype_t">Nvidia documentation</see>
    /// </summary>
    [IntrinsicType("cudaDataType_t")]
    public enum cudaDataType_t : int
    {
        /// <summary>
        /// read as a half
        /// </summary>
        CUDA_R_16F = 2,  
        /// <summary>
        ///  complex as a pair of half
        /// </summary>
        CUDA_C_16F = 6, 
        /// <summary>
        /// real as a float
        /// </summary>
        CUDA_R_32F = 0, 
        /// <summary>
        /// complex as a pair of float
        /// </summary>
        CUDA_C_32F = 4,  
        /// <summary>
        /// real as a double
        /// </summary>
        CUDA_R_64F = 1, 
        /// <summary>
        /// complex as a pair of double
        /// </summary>
        CUDA_C_64F = 5, 
        /// <summary>
        /// real as a signed char
        /// </summary>
        CUDA_R_8I = 3,  
        /// <summary>
        /// complex as a pair of signed char
        /// </summary>
        CUDA_C_8I = 7,  
        /// <summary>
        /// real as an unsigned char
        /// </summary>
        CUDA_R_8U = 8, 
        /// <summary>
        /// complex as a pair of unsigned char
        /// </summary>
        CUDA_C_8U = 9,  
        /// <summary>
        /// real as a signed int
        /// </summary>
        CUDA_R_32I = 10, 
        /// <summary>
        /// complex as a pair of signed int
        /// </summary>
        CUDA_C_32I = 11, 
        /// <summary>
        /// real as an unsigned int
        /// </summary>
        CUDA_R_32U = 12,
        /// <summary>
        /// complex as a pair of unsigned int numbers
        /// </summary>
        CUDA_C_32U = 13 
    }
}
/* (c) ALTIMESH 2018 -- all rights reserved */
using System;
using System.Runtime.InteropServices;
using System.Text;

namespace Hybridizer.Runtime.CUDAImports
{
    /// <summary>
    /// cublasMath_t enumerate type is used in cublasSetMathMode to choose whether or not to use Tensor Core operations in the library by setting the math mode to either CUBLAS_TENSOR_OP_MATH or CUBLAS_DEFAULT_MATH. 
    /// Complete documentation on <see href="https://docs.nvidia.com/cuda/cublas/index.html#cublasmath_t">Nvidia documentation</see>
    /// </summary>
    [IntrinsicType("cublasMath_t")]
    public enum cublasMath_t : int
    {
        /// <summary>
        ///  Prevent the library from using Tensor Core operations
        /// </summary>
        CUBLAS_DEFAULT_MATH,
        /// <summary>
        ///  Allows the library to use Tensor Core operations whenever possible
        /// </summary>
        CUBLAS_TENSOR_OP_MATH
    }
}
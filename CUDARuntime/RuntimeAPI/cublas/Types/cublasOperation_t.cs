/* (c) ALTIMESH 2018 -- all rights reserved */
using System;
using System.Runtime.InteropServices;
using System.Text;

namespace Hybridizer.Runtime.CUDAImports
{
	/// <summary>
    /// Indicates which operation needs to be performed with the dense matrix.
    /// Complete documentation on <see href="https://docs.nvidia.com/cuda/cublas/index.html#cublasoperation_t">Nvidia documentation</see>
    /// </summary>
    [IntrinsicType("cublasOperation_t")]
    public enum cublasOperation_t : int
    {
        /// <summary>
        /// the non-transpose operation is selected
        /// </summary>
        CUBLAS_OP_N,
        /// <summary>
        /// the transpose operation is selected
        /// </summary>
        CUBLAS_OP_T,
        /// <summary>
        /// the conjugate transpose operation is selected
        /// </summary>
        CUBLAS_OP_C,
    }
}
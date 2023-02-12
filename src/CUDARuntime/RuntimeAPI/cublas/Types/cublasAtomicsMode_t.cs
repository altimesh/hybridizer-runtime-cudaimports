/* (c) ALTIMESH 2018 -- all rights reserved */
using System;
using System.Runtime.InteropServices;
using System.Text;

namespace Hybridizer.Runtime.CUDAImports
{
    /// <summary>
    /// Indicates whether cuBLAS routines which has an alternate implementation using atomics can be used
    /// Complete documentation on <see href="https://docs.nvidia.com/cuda/cublas/index.html#cublasatomicsmode_t">Nvidia documentation</see>
    /// </summary>
    [IntrinsicType("cublasAtomicsMode_t")]
    public enum cublasAtomicsMode_t : int
    {
        /// <summary>
        /// the usage of atomics is not allowed
        /// </summary>
        CUBLAS_ATOMICS_NOT_ALLOWED,
        /// <summary>
        /// the usage of atomics is allowed
        /// </summary>
        CUBLAS_ATOMICS_ALLOWED,
    }
}
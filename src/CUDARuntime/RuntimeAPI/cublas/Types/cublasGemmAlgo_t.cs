/* (c) ALTIMESH 2018 -- all rights reserved */
using System;
using System.Runtime.InteropServices;
using System.Text;

namespace Hybridizer.Runtime.CUDAImports
{
    /// <summary>
    /// An enumerant to specify the algorithm for matrix-matrix multiplication
    /// Complete documentation on <see href="https://docs.nvidia.com/cuda/cublas/index.html#cublasgemmalgo_t">Nvidia documentation</see>
    /// </summary>
    [IntrinsicType("cublasGemmAlgo_t")]
    public enum cublasGemmAlgo_t : int
    {
        /// <summary>
        /// Apply Heuristics to select the GEMM algorithm
        /// </summary>
        CUBLAS_GEMM_DEFAULT,
        /// <summary>
        /// Explicitly choose Algorithm 0 
        /// </summary>
        CUBLAS_GEMM_ALGO0,
        /// <summary>
        /// Explicitly choose Algorithm 1 
        /// </summary>
        CUBLAS_GEMM_ALGO1,
        /// <summary>
        /// Explicitly choose Algorithm 2 
        /// </summary>
        CUBLAS_GEMM_ALGO2,
        /// <summary>
        /// Explicitly choose Algorithm 3 
        /// </summary>
        CUBLAS_GEMM_ALGO3,
        /// <summary>
        /// Explicitly choose Algorithm 4 
        /// </summary>
        CUBLAS_GEMM_ALGO4,
        /// <summary>
        /// Explicitly choose Algorithm 5 
        /// </summary>
        CUBLAS_GEMM_ALGO5,
        /// <summary>
        /// Explicitly choose Algorithm 6 
        /// </summary>
        CUBLAS_GEMM_ALGO6,
        /// <summary>
        /// Explicitly choose Algorithm 7 
        /// </summary>
        CUBLAS_GEMM_ALGO7,
        /// <summary>
        /// Explicitly choose Algorithm 8 
        /// </summary>
        CUBLAS_GEMM_ALGO8,
        /// <summary>
        /// Explicitly choose Algorithm 9 
        /// </summary>
        CUBLAS_GEMM_ALGO9,
        /// <summary>
        /// Explicitly choose Algorithm 10 
        /// </summary>
        CUBLAS_GEMM_ALGO10,
        /// <summary>
        /// Explicitly choose Algorithm 11 
        /// </summary>
        CUBLAS_GEMM_ALGO11,
        /// <summary>
        /// Explicitly choose Algorithm 12 
        /// </summary>
        CUBLAS_GEMM_ALGO12,
        /// <summary>
        /// Explicitly choose Algorithm 13 
        /// </summary>
        CUBLAS_GEMM_ALGO13,
        /// <summary>
        /// Explicitly choose Algorithm 14 
        /// </summary>
        CUBLAS_GEMM_ALGO14,
        /// <summary>
        /// Explicitly choose Algorithm 15 
        /// </summary>
        CUBLAS_GEMM_ALGO15,
        /// <summary>
        /// Explicitly choose Algorithm 16 
        /// </summary>
        CUBLAS_GEMM_ALGO16,
        /// <summary>
        /// Explicitly choose Algorithm 17 
        /// </summary>
        CUBLAS_GEMM_ALGO17,
        /// <summary>
        /// Apply Heuristics to select the GEMM algorithm, and allow the use of Tensor Core operations when possible
        /// </summary>
        CUBLAS_GEMM_DEFAULT_TENSOR_OP,
        /// <summary>
        /// Explicitly choose GEMM Algorithm 0 while allowing the use of Tensor Core operations when possible
        /// </summary>
        CUBLAS_GEMM_ALGO0_TENSOR_OP,
        /// <summary>
        /// Explicitly choose GEMM Algorithm 1 while allowing the use of Tensor Core operations when possible
        /// </summary>
        CUBLAS_GEMM_ALGO1_TENSOR_OP,
        /// <summary>
        /// Explicitly choose GEMM Algorithm 2 while allowing the use of Tensor Core operations when possible
        /// </summary>
        CUBLAS_GEMM_ALGO2_TENSOR_OP
    }
}
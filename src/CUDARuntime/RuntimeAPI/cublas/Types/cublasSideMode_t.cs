/* (c) ALTIMESH 2018 -- all rights reserved */
using System;
using System.Runtime.InteropServices;
using System.Text;

namespace Hybridizer.Runtime.CUDAImports
{
    /// <summary>
    /// Indicates whether the dense matrix is on the left or right side in the matrix equation solved by a particular function
    /// Complete documentation on <see href="https://docs.nvidia.com/cuda/cublas/index.html#cublassidemode_t">Nvidia documentation</see>
    /// </summary>
    [IntrinsicType("cublasSideMode_t")]
    public enum cublasSideMode_t : int
    {
        /// <summary>
        /// the matrix is on the left side in the equation
        /// </summary>
        CUBLAS_SIDE_LEFT,
        /// <summary>
        /// the matrix is on the right side in the equation
        /// </summary>
        CUBLAS_SIDE_RIGHT,
    }
}
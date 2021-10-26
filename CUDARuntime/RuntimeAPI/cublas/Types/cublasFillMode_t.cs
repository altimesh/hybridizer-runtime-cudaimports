/* (c) ALTIMESH 2018 -- all rights reserved */
using System;
using System.Runtime.InteropServices;
using System.Text;

namespace Hybridizer.Runtime.CUDAImports
{	
    /// <summary>
    /// The type indicates which part (lower or upper) of the dense matrix was filled and consequently should be used by the function.
    /// Complete documentation on <see href="https://docs.nvidia.com/cuda/cublas/index.html#cublasfillmode_t">Nvidia documentation</see>
    /// </summary>
    [IntrinsicType("cublasFillMode_t")]
    public enum cublasFillMode_t : int
    {
        /// <summary>
        /// the lower part of the matrix is filled
        /// </summary>
        CUBLAS_FILL_MODE_LOWER,
        /// <summary>
        /// the upper part of the matrix is filled
        /// </summary>
        CUBLAS_FILL_MODE_UPPER,
    }
}
/* (c) ALTIMESH 2018 -- all rights reserved */
using System;
using System.Runtime.InteropServices;
using System.Text;

namespace Hybridizer.Runtime.CUDAImports
{
    /// <summary>
    /// Indicates whether the main diagonal of the dense matrix is unity and consequently should not be touched or modified by the function
    /// Complete documentation on <see href="https://docs.nvidia.com/cuda/cublas/index.html#cublasdiagtype_t">Nvidia documentation</see>
    /// </summary>
    [IntrinsicType("cublasDiagType_t")]
    public enum cublasDiagType_t : int
    {
        /// <summary>
        /// the matrix diagonal has non-unit elements
        /// </summary>
        CUBLAS_DIAG_NON_UNIT,
        /// <summary>
        /// the matrix diagonal has unit elements
        /// </summary>
        CUBLAS_DIAG_UNIT,
    }
}
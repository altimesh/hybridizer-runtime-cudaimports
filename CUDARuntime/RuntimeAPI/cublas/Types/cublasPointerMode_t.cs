/* (c) ALTIMESH 2018 -- all rights reserved */
using System;
using System.Runtime.InteropServices;
using System.Text;

namespace Hybridizer.Runtime.CUDAImports
{
    /// <summary>
    /// Indicates whether the scalar values are passed by reference on the host or device
    /// Complete documentation on <see href="https://docs.nvidia.com/cuda/cublas/index.html#cublaspointermode_t">Nvidia documentation</see>
    /// </summary>
    [IntrinsicType("cublasPointerMode_t")]
    public enum cublasPointerMode_t : int
    {
        /// <summary>
        /// the scalars are passed by reference on the host
        /// </summary>
        CUBLAS_POINTER_MODE_HOST,
        /// <summary>
        /// the scalars are passed by reference on the device
        /// </summary>
        CUBLAS_POINTER_MODE_DEVICE
    }
}
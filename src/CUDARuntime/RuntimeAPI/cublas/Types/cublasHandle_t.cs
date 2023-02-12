/* (c) ALTIMESH 2018 -- all rights reserved */
using System;
using System.Runtime.InteropServices;
using System.Text;

namespace Hybridizer.Runtime.CUDAImports
{
	/// <summary>
    /// Opaque pointer holding the cuBLAS library context
    /// Complete documentation on <see href="https://docs.nvidia.com/cuda/cublas/index.html#cublashandle_t">Nvidia documentation</see>
    /// </summary>
    [IntrinsicType("cublasHandle_t")]
    public struct cublasHandle_t { IntPtr _inner; }
}
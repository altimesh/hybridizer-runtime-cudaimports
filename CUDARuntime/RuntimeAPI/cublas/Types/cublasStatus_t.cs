/* (c) ALTIMESH 2018 -- all rights reserved */
using System;
using System.Runtime.InteropServices;
using System.Text;

namespace Hybridizer.Runtime.CUDAImports
{
	/// <summary>
    /// The type is used for function status returns
    /// Complete documentation on <see href="https://docs.nvidia.com/cuda/cublas/index.html#cublasstatus_t">Nvidia documentation</see>
    /// </summary>
    [IntrinsicType("cublasStatus_t")]
    public enum cublasStatus_t : int
    {
        /// <summary>
        /// The operation completed successfully.
        /// </summary>
        CUBLAS_STATUS_SUCCESS,
        /// <summary>
        /// The cuBLAS library was not initialized.
        /// </summary>
        CUBLAS_STATUS_NOT_INITIALIZED,
        /// <summary>
        /// Resource allocation failed inside the cuBLAS library.
        /// </summary>
        CUBLAS_STATUS_ALLOC_FAILED,
        /// <summary>
        /// An unsupported value or parameter was passed to the function
        /// </summary>
        CUBLAS_STATUS_INVALID_VALUE,
        /// <summary>
        /// The function requires a feature absent from the device architecture
        /// </summary>
        CUBLAS_STATUS_ARCH_MISMATCH,
        /// <summary>
        /// An access to GPU memory space failed
        /// </summary>
        CUBLAS_STATUS_MAPPING_ERROR,
        /// <summary>
        /// The GPU program failed to execute. This is often caused by a launch failure of the kernel on the GPU, which can be caused by multiple reasons. 
        /// </summary>
        CUBLAS_STATUS_EXECUTION_FAILED,
        /// <summary>
        ///  An internal cuBLAS operation failed.
        /// </summary>
        CUBLAS_STATUS_INTERNAL_ERROR,
        /// <summary>
        /// The functionnality requested is not supported
        /// </summary>
        CUBLAS_STATUS_NOT_SUPPORTED,
        /// <summary>
        /// The functionnality requested requires some license and an error was detected when trying to check the current licensing.
        /// </summary>
        CUBLAS_STATUS_LICENSE_ERROR
    }
}
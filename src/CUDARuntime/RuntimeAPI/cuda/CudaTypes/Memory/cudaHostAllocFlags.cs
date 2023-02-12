using System;

namespace Hybridizer.Runtime.CUDAImports
{
    /// <summary>
    /// host allocation flags
    /// </summary>
    [IntrinsicType("cudaHostAllocFlags")]
    [Flags]
    public enum cudaHostAllocFlags : uint
    {
        /// <summary>
        /// Default page-locked allocation flag
        /// </summary>
        cudaHostAllocDefault = 0,
        /// <summary>
        /// Pinned memory accessible by all CUDA contexts
        /// </summary>
        cudaHostAllocPortable = 1,
        /// <summary>
        /// Map allocation into device space
        /// </summary>
        cudaHostAllocMapped = 2,
        /// <summary>
        /// Write-combined memory
        /// </summary>
        cudaHostAllocWriteCombined = 4
    }
}
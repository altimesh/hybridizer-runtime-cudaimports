using System;

namespace Hybridizer.Runtime.CUDAImports
{
    /// <summary>
    /// cuda event flags
    /// </summary>
    [IntrinsicType("cudaEventFlags")]
    [Flags]
    public enum cudaEventFlags : int
    {
        /// <summary>
        /// Default event flag
        /// </summary>
        cudaEventDefault = 0,
        /// <summary>
        /// Event uses blocking synchronization
        /// </summary>
        cudaEventBlockingSync = 1,
        /// <summary>
        /// Event will not record timing data
        /// </summary>
        cudaEventDisableTiming = 2,
        /// <summary>
        /// Event is suitable for interprocess use. cudaEventDisableTiming must be set
        /// </summary>
        cudaEventInterprocess = 4
    }
}
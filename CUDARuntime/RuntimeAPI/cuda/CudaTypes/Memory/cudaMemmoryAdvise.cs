using System;

namespace Hybridizer.Runtime.CUDAImports
{
    /// <summary>
    /// CUDA Memory Advise values 
    /// </summary>
    [IntrinsicType("")]
    public enum cudaMemmoryAdvise : int
    {
        /// <summary>
        /// Data will mostly be read and only occassionally be written to
        /// </summary>
        cudaMemAdviseSetReadMostly = 1,
        /// <summary>
        /// Undo the effect of ::cudaMemAdviseSetReadMostly
        /// </summary>
        cudaMemAdviseUnsetReadMostly = 2,
        /// <summary>
        /// Set the preferred location for the data as the specified device
        /// </summary>
        cudaMemAdviseSetPreferredLocation = 3,
        /// <summary>
        /// Clear the preferred location for the data
        /// </summary>
        cudaMemAdviseUnsetPreferredLocation = 4,
        /// <summary>
        /// Data will be accessed by the specified device, so prevent page faults as much as possible
        /// </summary>
        cudaMemAdviseSetAccessedBy = 5,
        /// <summary>
        /// Let the Unified Memory subsystem decide on the page faulting policy for the specified device
        /// </summary>
        cudaMemAdviseUnsetAccessedBy = 6,
    }
}
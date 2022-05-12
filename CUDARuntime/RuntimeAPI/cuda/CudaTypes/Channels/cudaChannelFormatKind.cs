using System;

namespace Hybridizer.Runtime.CUDAImports
{
    /// <summary>
    /// Channel format kind
    /// </summary>
    [IntrinsicType("cudaChannelFormatKind")]
    public enum cudaChannelFormatKind : int
    {
        /// <summary>
        /// Signed channel format
        /// </summary>
        cudaChannelFormatKindSigned = 0,
        /// <summary>
        /// Unsigned channel format
        /// </summary>
        cudaChannelFormatKindUnsigned = 1,
        /// <summary>
        /// Float channel format
        /// </summary>
        cudaChannelFormatKindFloat = 2,
        /// <summary>
        /// No channel format
        /// </summary>
        cudaChannelFormatKindNone = 3
    }
}
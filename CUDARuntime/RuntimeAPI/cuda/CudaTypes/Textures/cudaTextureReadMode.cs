namespace Hybridizer.Runtime.CUDAImports
{
    /// <summary>
    /// CUDA texture read modes
    /// </summary>
    [IntrinsicType("cudaTextureReadMode")]
    public enum cudaTextureReadMode
    {
        /// <summary>
        /// Read texture as specified element type
        /// </summary>
        cudaReadModeElementType = 0,
        /// <summary>
        /// Read texture as normalized float
        /// </summary>
        cudaReadModeNormalizedFloat = 1
    }
}
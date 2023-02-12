namespace Hybridizer.Runtime.CUDAImports
{
    /// <summary>
    /// array allocation flags
    /// </summary>
    [IntrinsicType("cudaMallocArrayFlags")]
    public enum cudaMallocArrayFlags : int
    {
        /// <summary>
        /// Default CUDA array allocation flag
        /// </summary>
        cudaArrayDefault = 0x00,
        /// <summary>
        /// Must be set in cudaMalloc3DArray to create a layered CUDA array
        /// </summary>
        cudaArrayLayered = 0x01,
        /// <summary>
        /// Must be set in cudaMallocArray or cudaMalloc3DArray in order to bind surfaces to the CUDA array
        /// </summary>
        cudaArraySurfaceLoadStore = 0x02,
        /// <summary>
        /// Must be set in cudaMalloc3DArray to create a cubemap CUDA array
        /// </summary>
        cudaArrayCubemap = 0x04,
        /// <summary>
        /// Must be set in cudaMallocArray or cudaMalloc3DArray in order to perform texture gather operations on the CUDA array
        /// </summary>
        cudaArrayTextureGather = 0x08,
    }
}
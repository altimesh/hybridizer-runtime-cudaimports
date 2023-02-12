namespace Hybridizer.Runtime.CUDAImports
{
    /// <summary>
    /// CUDA function cache configurations
    /// </summary>
    [IntrinsicType("cudaFuncCache")]
    public enum cudaFuncCache : int
    {
        /// <summary>
        /// Default function cache configuration, no preference
        /// </summary>
        cudaFuncCachePreferNone = 0,
        /// <summary>
        /// Prefer larger shared memory and smaller L1 cache
        /// </summary>
        cudaFuncCachePreferShared = 1,
        /// <summary>
        /// Prefer larger L1 cache and smaller shared memory
        /// </summary>
        cudaFuncCachePreferL1 = 2
    }
}
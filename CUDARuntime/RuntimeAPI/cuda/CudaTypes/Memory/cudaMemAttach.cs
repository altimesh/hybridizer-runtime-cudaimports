namespace Hybridizer.Runtime.CUDAImports
{
    /// <summary>
    /// cuda memory attach
    /// </summary>
    [IntrinsicType("")]
    public enum cudaMemAttach : int
    {
        /// <summary>
        /// Memory can be accessed by any stream on any device
        /// </summary>
        cudaMemAttachGlobal = 0x01,
        /// <summary>
        /// Memory cannot be accessed by any stream on any device 
        /// </summary>
        cudaMemAttachHost = 0x02,
        /// <summary>
        /// Memory can only be accessed by a single stream on the associated device 
        /// </summary>
        cudaMemAttachSingle = 0x04,
    }
}

namespace Hybridizer.Runtime.CUDAImports
{
    /// <summary>
    /// Defines the way in which copy is done
    /// </summary>
    [IntrinsicType("cudaMemcpyKind")]
    public enum cudaMemcpyKind : int
    {
        /// <summary>
        /// Host   -> Host
        /// </summary>
        cudaMemcpyHostToHost = 0,
        /// <summary>
        /// Host   -> Device
        /// </summary>
        cudaMemcpyHostToDevice = 1,
        /// <summary>
        /// Device -> Host
        /// </summary>
        cudaMemcpyDeviceToHost = 2,
        /// <summary>
        /// Device -> Device
        /// </summary>
        cudaMemcpyDeviceToDevice = 3
    }
}
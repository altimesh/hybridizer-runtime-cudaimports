using System.Runtime.InteropServices;

namespace Hybridizer.Runtime.CUDAImports
{
    /// <summary>
    /// CUDA 3D position
    /// </summary>
    [IntrinsicType("cudaPos")]
    [StructLayout(LayoutKind.Sequential)]
    public struct cudaPos
    {
        /// <summary>
        /// x
        /// </summary>
        size_t x;
        /// <summary>
        /// y
        /// </summary>
        size_t y;
        /// <summary>
        /// z
        /// </summary>
        size_t z;
    }
}
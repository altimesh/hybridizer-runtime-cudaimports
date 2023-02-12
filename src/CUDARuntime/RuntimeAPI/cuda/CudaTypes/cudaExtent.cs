using System.Runtime.InteropServices;

namespace Hybridizer.Runtime.CUDAImports
{
    /// <summary>
    /// CUDA extent
    /// </summary>
    [IntrinsicType("cudaExtent")]
    [StructLayout(LayoutKind.Sequential)]
    public struct cudaExtent
    {
        /// <summary>
        /// Width in elements when referring to array memory, in bytes when referring to linear memory 
        /// </summary>
        public size_t width;
        /// <summary>
        /// Height in elements
        /// </summary>
        public size_t height;
        /// <summary>
        /// Depth in elements 
        /// </summary>
        public size_t depth;
        /// <summary> </summary>
        public cudaExtent(size_t w, size_t h, size_t d)
        {
            width = w; height = h; depth = d;
        }
    }
}
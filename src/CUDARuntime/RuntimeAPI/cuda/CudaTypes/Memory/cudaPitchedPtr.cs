using System;
using System.Runtime.InteropServices;

namespace Hybridizer.Runtime.CUDAImports
{
    /// <summary>
    /// CUDA Pitched memory pointer
    /// </summary>
    [IntrinsicType("cudaPitchedPtr")]
    [StructLayout(LayoutKind.Sequential)]
    public struct cudaPitchedPtr
    {
        /// <summary>
        /// Pointer to allocated memory 
        /// </summary>
        public IntPtr ptr;
        /// <summary>
        /// Pitch of allocated memory in bytes
        /// </summary>
        public size_t pitch;
        /// <summary>
        /// Logical width of allocation in elements
        /// </summary>
        public size_t xsize;
        /// <summary>
        /// Logical height of allocation in elements 
        /// </summary>
        public size_t ysize;
        /// <summary>
        /// constructor
        /// </summary>
        public cudaPitchedPtr(IntPtr d, size_t p, size_t xsz, size_t ysz)
        {
            ptr = d;
            pitch = p;
            xsize = xsz;
            ysize = ysz;
        }
    }
}
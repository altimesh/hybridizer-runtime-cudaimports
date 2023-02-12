using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;

namespace Hybridizer.Runtime.CUDAImports
{
    /// <summary>
    /// 2D Size This struct typically represents the size of a a rectangular region in two space.
    /// </summary>
    [IntrinsicInclude("npp.h")]
    [IntrinsicType("NppiSize")]
    [StructLayout(LayoutKind.Explicit, Size = 8)]
    public struct NppiSize
    {
        /// <summary>
        /// Rectangle width.
        /// </summary>
        [FieldOffset(0)]
        public int width;
        /// <summary>
        /// Rectangle height.
        /// </summary>
        [FieldOffset(4)]
        public int height;
    }
}

using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;

namespace Hybridizer.Runtime.CUDAImports
{
    [IntrinsicInclude("npp.h")]
    [IntrinsicType("NppiSize")]
    [StructLayout(LayoutKind.Explicit, Size = 8)]
    public struct NppiSize
    {
        [FieldOffset(0)]
        public int width;
        [FieldOffset(4)]
        public int height;
    }
}

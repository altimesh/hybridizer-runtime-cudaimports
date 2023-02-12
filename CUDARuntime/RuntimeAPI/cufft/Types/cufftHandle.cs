/* (c) ALTIMESH 2018 -- all rights reserved */
using System;
using System.Runtime.InteropServices;

namespace Hybridizer.Runtime.CUDAImports
{
    /// <summary>
    /// Inner structure to carry handle.
    /// </summary>
    [IntrinsicType("cufftHandle"), StructLayout(LayoutKind.Explicit, Size=4, Pack=4)]
    public struct cufftHandle
    {
        [FieldOffset(0)]
        int _inner;
    }
}
/* (c) ALTIMESH 2018 -- all rights reserved */
//#define DEBUG
//#define DEBUG_ALLOC
//#define DEBUG_MEMCPY

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Diagnostics;
using System.Security;
using System.Text;
using System.Threading;
using NamingTools = Altimesh.Hybridizer.Runtime.NamingTools;

namespace Hybridizer.Runtime.CUDAImports
{
    /// <summary>
    /// Handle for delegate serialized as structs
    /// </summary>
    [StructLayout(LayoutKind.Explicit, Size = 24)]
    public struct HandleDelegate
    {
        /// <summary></summary>
        [FieldOffset(0)]
        public IntPtr Marshalled;
        /// <summary></summary>
        [FieldOffset(8)]
        public IntPtr FuncPtr;
        /// <summary></summary>
        [FieldOffset(16)]
        public bool _isStaticFunc;
    }
}
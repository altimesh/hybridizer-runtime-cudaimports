using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Hybridizer.Runtime.CUDAImports
{
    /// <summary>
    /// </summary>
    public struct CUlinkState
    {
        IntPtr _inner;
        /// <summary>
        /// </summary>
        public static implicit operator IntPtr(CUlinkState ctx) { return ctx._inner; }
        /// <summary>
        /// </summary>
        public bool Exists() { return _inner != IntPtr.Zero; }
        /// <summary>
        /// </summary>
        public CUlinkState(IntPtr inner) { _inner = inner; }
    }
}

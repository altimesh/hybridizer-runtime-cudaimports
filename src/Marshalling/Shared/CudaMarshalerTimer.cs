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
    /// internal
    /// </summary>
    public static class CudaMarshalerTimer
    {
#pragma warning disable 1591
        public static double ManagedToNative = 0.0;
        public static double NativeToManaged = 0.0;
        public static double Free = 0.0;
        public static void Init()
        {
            ManagedToNative = 0.0;
            NativeToManaged = 0.0;
            Free = 0.0;
        }
    }
#pragma warning restore 1591
}
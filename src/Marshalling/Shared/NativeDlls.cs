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
    /// wrapper class listing generated native dlls
    /// </summary>
    public static class NativeDlls
    {
        #region private fields
        private static string[] GetNativeDlls()
        {
            var result = new List<string>();
            return result.ToArray();
        }
        #endregion
        /// <summary>
        /// list of native dlls
        /// </summary>
        public static readonly string[] Dlls = GetNativeDlls();
    }
}
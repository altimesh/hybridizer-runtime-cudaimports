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
    /// Marshaler to main memory - to be used for OMP and AVX flavors
    /// 
    /// Usage example:
    /// 
    /// \begin{lstlisting}[style=customcs]
    /// 
    ///  [DllImport("{DLL name}.dll", 
    ///         EntryPoint = "{EntryPointName}", 
    ///         CallingConvention = CallingConvention.Cdecl)]
    ///  private static extern int methodName ( 
    ///      [MarshalAs(UnmanagedType.CustomMarshaler, 
    ///             MarshalTypeRef = typeof(MainMemoryMarshaler))] 
    ///             TypeToBemarshaled param);
    ///      
    /// \end{lstlisting}
    /// 
    /// </summary>
    public class MainMemoryMarshaler : AbstractNativeMarshaler
    {
        #region private fields
        private static MainMemoryMarshaler instance = Create(HybridizerFlavor.OMP);
        #endregion

        /// <summary>
        /// Create an instance of MainMemoryMarshaler for the given flavor
        /// </summary>
        /// <param name="flavor"></param>
        /// <returns></returns>
        public static MainMemoryMarshaler Create(HybridizerFlavor flavor)
        {
            NativePtrConverter ptrConverter = NativePtrConverter.Create(flavor, NativeDlls.Dlls);
            MainMemorySerializationState state = new MainMemorySerializationState(ptrConverter);
            return new MainMemoryMarshaler(state);
        }

        internal MainMemoryMarshaler(MainMemorySerializationState state)
            : base(state)
        {
        }

        /// <summary>
        /// Singleton implementation
        /// </summary>
        /// <param name="cookie"></param>
        /// <returns></returns>
        public static ICustomMarshaler GetInstance(string cookie)
        {
            return instance;
        }

        /// <summary>
        ///  current instance
        /// </summary>
        public static MainMemoryMarshaler Instance
        {
            get
            {
                return instance;
            }
            set
            {
                instance = value;
            }
        }

        /// <summary>
        ///  Marshals Managed to Native
        /// </summary>
        /// <param name="managedObj"></param>
        /// <returns></returns>
        public override IntPtr MarshalManagedToNative(object managedObj)
        {
            var intPtr = base.MarshalManagedToNative(managedObj);
            return intPtr;
        }
    }
}
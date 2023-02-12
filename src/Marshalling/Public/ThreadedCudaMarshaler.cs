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
    /// Cuda marshaler, threaded
    /// </summary>
    public class ThreadedCudaMarshaler : ICustomMarshaler
    {
        #region private fields
        static readonly ThreadedCudaMarshaler instance = new ThreadedCudaMarshaler();
        internal static readonly NativePtrConverter ptrConverter = NativePtrConverter.Create(HybridizerFlavor.CUDA, NativeDlls.Dlls);
        private static ThreadLocal<CudaMarshaler> tlInstance = new ThreadLocal<CudaMarshaler>(() => CudaMarshaler.Create(true));

        #endregion


        /// <summary>
        /// Mandatory when using CustomMarshaler
        /// </summary>
        public static ICustomMarshaler GetInstance(string cookie)
        {
            return instance;
        }

        /// <summary>
        /// current instance (thread)
        /// </summary>
        public static CudaMarshaler ThreadLocalInstance
        {
            get { return tlInstance.Value; }
            set
            {
                tlInstance.Value = value;
            }
        }

        /// <summary>
        /// Marshals Native to Managed
        /// </summary>
        public object MarshalNativeToManaged(IntPtr pNativeData)
        {
            return ThreadLocalInstance.MarshalNativeToManaged(pNativeData);
        }

        /// <summary>
        /// Marshals Managed to Native
        /// </summary>
        public IntPtr MarshalManagedToNative(object ManagedObj)
        {
            return ThreadLocalInstance.MarshalManagedToNative(ManagedObj);
        }

        /// <summary>
        /// cleanup native data
        /// </summary>
        /// <param name="pNativeData"></param>
        public void CleanUpNativeData(IntPtr pNativeData)
        {
            ThreadLocalInstance.CleanUpNativeData(pNativeData);
        }

        /// <summary>
        /// cleanup managed data
        /// </summary>
        public void CleanUpManagedData(object ManagedObj)
        {
            ThreadLocalInstance.CleanUpManagedData(ManagedObj);
        }

        /// <summary>
        /// Get Native Data Size
        /// </summary>
        /// <returns></returns>
        public int GetNativeDataSize()
        {
            return ThreadLocalInstance.GetNativeDataSize();
        }
    }
}
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
    /// Marshaler to CUDA device memory
    /// 
    /// Usage example:
    /// \begin{lstlisting}[style=customcs]
    /// 
    ///  [DllImport("{cuda DLL name}.dll", 
    ///             EntryPoint = "{EntryPointName}_ExternCWrapper_CUDA", 
    ///             CallingConvention = CallingConvention.Cdecl)]
    ///  private static extern int methodName(
    ///             int gridDimX, int gridDimY, 
    ///             int blockDimX, int blockDimY, int blockDimZ,
    ///             int shared,
    ///      [MarshalAs(UnmanagedType.CustomMarshaler, 
    ///             MarshalTypeRef = typeof(CudaMarshaler))] 
    ///             TypeToBemarshaled param);
    /// 
    /// \end{lstlisting}
    /// 
    /// </summary>
    public class CudaMarshaler : AbstractNativeMarshaler
    {
        /// <summary>
        /// Necessary because .Net runtime caches calls to GetInstance method and we need to change
        /// the implementation at runtime
        /// </summary>
        class CudaMarshalerFix : ICustomMarshaler
        {
            /// <summary>
            /// INTERNAL METHOD
            /// </summary>
            /// <param name="pNativeData"></param>
            /// <returns></returns>
            public object MarshalNativeToManaged(IntPtr pNativeData)
            {
                return globalInstance.MarshalNativeToManaged(pNativeData);
            }

            /// <summary>
            /// INTERNAL METHOD
            /// </summary>
            /// <param name="ManagedObj"></param>
            /// <returns></returns>
            public IntPtr MarshalManagedToNative(object ManagedObj)
            {
                return globalInstance.MarshalManagedToNative(ManagedObj);
            }

            /// <summary>
            /// INTERNAL METHOD
            /// </summary>
            /// <param name="pNativeData"></param>
            public void CleanUpNativeData(IntPtr pNativeData)
            {
                globalInstance.CleanUpNativeData(pNativeData);
            }

            /// <summary>
            /// INTERNAL METHOD
            /// </summary>
            /// <param name="ManagedObj"></param>
            public void CleanUpManagedData(object ManagedObj)
            {
                globalInstance.CleanUpManagedData(ManagedObj);
            }

            /// <summary>
            /// INTERNAL METHOD
            /// </summary>
            /// <returns></returns>
            public int GetNativeDataSize()
            {
                return globalInstance.GetNativeDataSize();
            }
        }

        private static CudaMarshalerFix fixInstance = new CudaMarshalerFix();


        #region private fields
        internal static readonly NativePtrConverter ptrConverter = NativePtrConverter.Create(HybridizerFlavor.CUDA, NativeDlls.Dlls);

        private static Boolean useAggregation;
        private static CudaMarshaler globalInstance = Create(false);

        #endregion

        /// <summary>
        /// Factory method
        /// </summary>
        /// <param name="useAggregation">True: malloc and memcopy will be aggregated in one single large block</param>
        /// <param name="stream">cuda stream for marshaller creation</param>
        /// <param name="cuda">Implementation of ICuda</param>
        /// <returns>The Instance of Marshaller (to be used with Instance property)</returns>
        public static CudaMarshaler Create(bool useAggregation, cudaStream_t stream, ICudaMarshalling cuda = null)
        {
            if (cuda == null)
            {
                cuda = global::Hybridizer.Runtime.CUDAImports.cuda.instance;
            }
            return Create(useAggregation, true, stream, cuda);
        }

        /// <summary>
        /// Factory method
        /// </summary>
        /// <param name="useAggregation">True: malloc and memcopy will be aggregated in one single large block</param>
        /// <param name="cuda">Implementation of ICuda</param>
        /// <returns>The Instance of Marshaller (to be used with Instance property)</returns>
        public static CudaMarshaler Create(bool useAggregation, ICudaMarshalling cuda = null)
        {
            if (cuda == null)
            {
                cuda = global::Hybridizer.Runtime.CUDAImports.cuda.instance;
            }
            return Create(useAggregation, false, new cudaStream_t(), cuda);
        }
        
        private static CudaMarshaler Create(bool useAggregation, bool useStream, cudaStream_t stream, ICudaMarshalling cuda)
        {
            NativeSerializerState state;
            if (useStream)
            {
                if (useAggregation)
                    state = new CudaAggregatedSerializationState(ptrConverter, cuda, stream);
                else
                    state = new CudaSerializationState(ptrConverter, cuda, stream);

            }
            else
            {
                if (useAggregation)
                    state = new CudaAggregatedSerializationState(ptrConverter, cuda);
                else
                    state = new CudaSerializationState(ptrConverter, cuda);
                
            }
            var cudaMarshaler = new CudaMarshaler(state);
            return cudaMarshaler;
        }


        private CudaMarshaler(NativeSerializerState state)
            : base(state)
        {
        }

        /// <summary>
        /// constructor
        /// </summary>
        protected CudaMarshaler() {}

        /// <summary>
        /// Necessary when using default DllImport-related marshaler.
        /// </summary>
        /// <param name="cookie"></param>
        /// <returns></returns>
        public static ICustomMarshaler GetInstance(string cookie)
        {
            return fixInstance;
        }

        /// <summary>
        /// Modify aggregation flag to enable marshaler aggregation
        /// </summary>
        /// <param name="useAggregation"></param>
        public static void changeAggregation(bool useAggregation)
        {
            if (useAggregation != CudaMarshaler.useAggregation)
            {
                globalInstance.Free();
                globalInstance = Create(useAggregation);
                CudaMarshaler.useAggregation = useAggregation;
            }
        }

        /// <summary>
        /// current instance
        /// </summary>
        public static CudaMarshaler Instance
        {
            get
            {
                return globalInstance;
            }
            set
            {
                globalInstance = value;
            }
        }
    }
}
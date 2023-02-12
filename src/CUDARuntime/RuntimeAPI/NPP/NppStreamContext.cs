using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;

namespace Hybridizer.Runtime.CUDAImports
{
    /// <summary>
    /// npp stream context
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct NppStreamContext
    {
#pragma warning disable 1591
        public cudaStream_t hStream;
        public int nCudaDeviceId;
        public int nMultiProcessorCount;
        public int nMaxThreadsPerMultiProcessor;
        public int nMaxThreadsPerBlock;
        public size_t nSharedMemPerBlock;
        public int nCudaDevAttrComputeCapabilityMajor;
        public int nCudaDevAttrComputeCapabilityMinor;
        int nReserved0;
        int nReserved1;
#pragma warning restore 1591
    }
}

/* (c) ALTIMESH 2018 -- all rights reserved */
using System;
using System.Collections.Generic;
using System.Configuration;
using System.Diagnostics;
using System.Text;
using System.Runtime.InteropServices;
using System.Reflection;

namespace Hybridizer.Runtime.CUDAImports
{
    /// <summary>
    /// CUDA Limits
    /// </summary>
    public enum cudaLimit
    {
        /// <summary>
        /// GPU thread stack size
        /// </summary>
        cudaLimitStackSize = 0x00, 
        /// <summary>
        /// GPU printf/fprintf FIFO size
        /// </summary>
        cudaLimitPrintfFifoSize = 0x01, 
        /// <summary>
        /// GPU malloc heap size
        /// </summary>
        cudaLimitMallocHeapSize = 0x02, 
        /// <summary>
        /// GPU device runtime synchronize depth
        /// </summary>
        cudaLimitDevRuntimeSyncDepth = 0x03, 
        /// <summary>
        /// GPU device runtime pending launch count
        /// </summary>
        cudaLimitDevRuntimePendingLaunchCount = 0x04,
        /// <summary>
        /// A value between 0 and 128 that indicates the maximum fetch granularity of L2 (in Bytes). This is a hint
        /// </summary>
        cudaLimitMaxL2FetchGranularity = 0x05
    }
}
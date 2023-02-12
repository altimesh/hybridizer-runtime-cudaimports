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
    /// CUDA device flags
    /// </summary>
    [Flags()]
    public enum deviceFlags : int
    {
        /// <summary>
        /// Automatic scheduling
        /// </summary>
        cudaDeviceScheduleAuto = 0,
        /// <summary>
        /// Spin default scheduling
        /// </summary>
        cudaDeviceScheduleSpin = 1,
        /// <summary>
        /// Yield default scheduling
        /// </summary>
        cudaDeviceScheduleYield = 2,
        /// <summary>
        /// Use blocking synchronization
        /// </summary>
        cudaDeviceScheduleBlockingSync = 4,
        /// <summary>
        /// Use blocking synchronization deprecated This flag was deprecated as of CUDA 4.0 and replaced with <see cref="cudaDeviceScheduleBlockingSync"></see>
        /// </summary>
        cudaDeviceBlockingSync = 4,
        /// <summary>
        /// Device schedule flags mask
        /// </summary>
        cudaDeviceScheduleMask = 7,
        /// <summary>
        /// Support mapped pinned allocations
        /// </summary>
        cudaDeviceMapHost = 8,
        /// <summary>
        /// Keep local memory allocation after launch
        /// </summary>
        cudaDeviceLmemResizeToMax = 0x10,
        /// <summary>
        /// Device flags mask
        /// </summary>
        cudaDeviceMask = 0x1F,
    }
}

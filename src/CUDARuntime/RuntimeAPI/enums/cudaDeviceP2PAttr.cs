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
    /// CUDA device P2P attributes 
    /// </summary>
    public enum cudaDeviceP2PAttr : int
    {
        /// <summary>
        /// A relative value indicating the performance of the link between two devices 
        /// </summary>
        cudaDevP2PAttrPerformanceRank = 1,
        /// <summary>
        /// Peer access is enabled 
        /// </summary>
        cudaDevP2PAttrAccessSupported = 2,
        /// <summary>
        /// Native atomic operation over the link supported 
        /// </summary>
        cudaDevP2PAttrNativeAtomicSupported = 3,
    }
}
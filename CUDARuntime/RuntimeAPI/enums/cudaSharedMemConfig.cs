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
    /// CUDA shared memory configuration 
    /// </summary>
    public enum cudaSharedMemConfig : int
    {
        /// <summary></summary>
        cudaSharedMemBankSizeDefault = 0,
        /// <summary></summary>
        cudaSharedMemBankSizeFourByte = 1,
        /// <summary></summary>
        cudaSharedMemBankSizeEightByte = 2
    }
}
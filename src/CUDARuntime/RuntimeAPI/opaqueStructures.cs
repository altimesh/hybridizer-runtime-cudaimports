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
    /// CUDA IPC event handle
    /// </summary>
    public struct cudaIpcEventHandle_t
    {
        IntPtr inner;
    }

    /// <summary>
    /// CUDA IPC memory handle
    /// </summary>
    public struct cudaIpcMemHandle_t
    {
        IntPtr inner;
    }

    /// <summary>
    /// Type of stream callback functions.
    /// </summary>
    public struct cudaStreamCallback_t
    {
#pragma warning disable 0169
        IntPtr inner;
#pragma warning restore 0169
    }
}
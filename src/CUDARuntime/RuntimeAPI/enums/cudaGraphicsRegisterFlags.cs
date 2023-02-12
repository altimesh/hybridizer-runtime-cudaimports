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
    /// CUDA Graphics register flags
    /// </summary>
    [Flags]
    public enum cudaGraphicsRegisterFlags : uint
    {
        /// <summary>
        /// Default
        /// </summary>
        None = 0,  
        /// <summary>
        /// CUDA will not write to this resource
        /// </summary>
        ReadOnly = 1,
        /// <summary>
        /// CUDA will only write to and will not read from this resource
        /// </summary>
        WriteDiscard = 2, 
        /// <summary>
        /// CUDA will bind this resource to a surface reference
        /// </summary>
        SurfaceLoadStore = 4,
        /// <summary>
        /// CUDA will perform texture gather operations on this resource
        /// </summary>
        TextureGather = 8
    }
}
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
    /// texture modes for opengl
    /// </summary>
    public enum GL_TEXTURE_MODE: uint
    {
        /// <summary>
        /// 2D texture
        /// </summary>
        GL_TEXTURE_2D = 0x0DE1
        // TODO: add others
    }
}
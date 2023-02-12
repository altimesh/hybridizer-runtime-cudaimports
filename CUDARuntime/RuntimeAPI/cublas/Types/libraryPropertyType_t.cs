/* (c) ALTIMESH 2018 -- all rights reserved */
using System;
using System.Runtime.InteropServices;
using System.Text;

namespace Hybridizer.Runtime.CUDAImports
{
    /// <summary>
    /// The libraryPropertyType_t is used as a parameter to specify which property is requested when using the routine cublasGetProperty
    /// Complete documentation on <see href="https://docs.nvidia.com/cuda/cublas/index.html#libraryPropertyType_t">Nvidia documentation</see>
    /// </summary>
    [IntrinsicType("libraryPropertyType_t")]
    public enum libraryPropertyType_t : int
    {
        /// <summary>
        /// enumerant to query the major version
        /// </summary>
        MAJOR_VERSION,
        /// <summary>
        /// enumerant to query the minor version
        /// </summary>
        MINOR_VERSION,
        /// <summary>
        /// number to identify the patch level
        /// </summary>
        PATCH_LEVEL
    }
}
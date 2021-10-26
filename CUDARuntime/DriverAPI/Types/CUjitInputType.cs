using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Hybridizer.Runtime.CUDAImports
{
    public enum CUjitInputType
    {
        /// <summary>
        /// Compiled device-class-specific device code\n
        /// Applicable options: none
        /// </summary>
        CU_JIT_INPUT_CUBIN = 0,

        /// <summary>
        /// PTX source code\n
        /// Applicable options: PTX compiler options
        /// </summary>
        CU_JIT_INPUT_PTX,

        /// <summary>
        /// Bundle of multiple cubins and/or PTX of some device code\n
        /// Applicable options: PTX compiler options, ::CU_JIT_FALLBACK_STRATEGY
        /// </summary>
        CU_JIT_INPUT_FATBINARY,

        /// <summary>
        /// Host object with embedded device code\n
        /// Applicable options: PTX compiler options, ::CU_JIT_FALLBACK_STRATEGY
        /// </summary>
        CU_JIT_INPUT_OBJECT,

        /// <summary>
        /// Archive of host objects with embedded device code\n
        /// Applicable options: PTX compiler options, ::CU_JIT_FALLBACK_STRATEGY
        /// </summary>
        CU_JIT_INPUT_LIBRARY,

        /// <summary>
        /// </summary>
        CU_JIT_NUM_INPUT_TYPES
    }
}

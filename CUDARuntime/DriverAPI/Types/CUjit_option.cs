using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Hybridizer.Runtime.CUDAImports
{
    /// <summary>
    /// cujit options
    /// </summary>
    public enum CUjit_option
    {
        /// <summary>
        /// Max number of registers that a thread may use.\n
        /// Option type: unsigned int\n
        /// Applies to: compiler only
        /// </summary>
        CU_JIT_MAX_REGISTERS = 0,
         
        /// <summary>
        /// IN: Specifies minimum number of threads per block to target compilation
        /// for\n
        /// OUT: Returns the number of threads the compiler actually targeted.
        /// This restricts the resource utilization fo the compiler (e.g. max
        /// registers) such that a block with the given number of threads should be
        /// able to launch based on register limitations. Note, this option does not
        /// currently take into account any other resource limitations, such as
        /// shared memory utilization.\n
        /// Cannot be combined with ::CU_JIT_TARGET.\n
        /// Option type: unsigned int\n
        /// Applies to: compiler only
        /// </summary>
        CU_JIT_THREADS_PER_BLOCK,

        /// <summary>
        /// Overwrites the option value with the total wall clock time, in
        /// milliseconds, spent in the compiler and linker\n
        /// Option type: float\n
        /// Applies to: compiler and linker
        /// </summary>
        CU_JIT_WALL_TIME,

        /// <summary>
        /// Pointer to a buffer in which to print any log messages
        /// that are informational in nature (the buffer size is specified via
        /// option ::CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES)\n
        /// Option type: char *\n
        /// Applies to: compiler and linker
        /// </summary>
        CU_JIT_INFO_LOG_BUFFER,

        /// <summary>
        /// IN: Log buffer size in bytes.  Log messages will be capped at this size
        /// (including null terminator)\n
        /// OUT: Amount of log buffer filled with messages\n
        /// Option type: unsigned int\n
        /// Applies to: compiler and linker
        /// </summary>
        CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES,

        /// <summary>
        /// Pointer to a buffer in which to print any log messages that
        /// reflect errors (the buffer size is specified via option
        /// ::CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES)\n
        /// Option type: char *\n
        /// Applies to: compiler and linker
        /// </summary>
        CU_JIT_ERROR_LOG_BUFFER,

        /// <summary>
        /// IN: Log buffer size in bytes.  Log messages will be capped at this size
        /// (including null terminator)\n
        /// OUT: Amount of log buffer filled with messages\n
        /// Option type: unsigned int\n
        /// Applies to: compiler and linker
        /// </summary>
        CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES,

        /// <summary>
        /// Level of optimizations to apply to generated code (0 - 4), with 4
        /// being the default and highest level of optimizations.\n
        /// Option type: unsigned int\n
        /// Applies to: compiler only
        /// </summary>
        CU_JIT_OPTIMIZATION_LEVEL,

        /// <summary>
        /// No option value required. Determines the target based on the current
        /// attached context (default)\n
        /// Option type: No option value needed\n
        /// Applies to: compiler and linker
        /// </summary>
        CU_JIT_TARGET_FROM_CUCONTEXT,

        /// <summary>
        /// Target is chosen based on supplied ::CUjit_target.  Cannot be
        /// combined with ::CU_JIT_THREADS_PER_BLOCK.\n
        /// Option type: unsigned int for enumerated type ::CUjit_target\n
        /// Applies to: compiler and linker
        /// </summary>
        CU_JIT_TARGET,

        /// <summary>
        /// Specifies choice of fallback strategy if matching cubin is not found.
        /// Choice is based on supplied ::CUjit_fallback.  This option cannot be
        /// used with cuLink* APIs as the linker requires exact matches.\n
        /// Option type: unsigned int for enumerated type ::CUjit_fallback\n
        /// Applies to: compiler only
        /// </summary>
        CU_JIT_FALLBACK_STRATEGY,

        /// <summary>
        /// Specifies whether to create debug information in output (-g)
        /// (0: false, default)\n
        /// Option type: int\n
        /// Applies to: compiler and linker
        /// </summary>
        CU_JIT_GENERATE_DEBUG_INFO,

        /// <summary>
        /// Generate verbose log messages (0: false, default)\n
        /// Option type: int\n
        /// Applies to: compiler and linker
        /// </summary>
        CU_JIT_LOG_VERBOSE,

        /// <summary>
        /// Generate line number information (-lineinfo) (0: false, default)\n
        /// Option type: int\n
        /// Applies to: compiler only
        /// </summary>
        CU_JIT_GENERATE_LINE_INFO,

        /// <summary>
        /// Specifies whether to enable caching explicitly (-dlcm) \n
        /// Choice is based on supplied ::CUjit_cacheMode_enum.\n
        /// Option type: unsigned int for enumerated type ::CUjit_cacheMode_enum\n
        /// Applies to: compiler only
        /// </summary>
        CU_JIT_CACHE_MODE,

        /// <summary>
        /// The below jit options are used for internal purposes only, in this version of CUDA
        /// </summary>
        CU_JIT_NEW_SM3X_OPT,
        /// <summary>
        /// fast compile
        /// </summary>
        CU_JIT_FAST_COMPILE,

        /// <summary>
        /// Array of device symbol names that will be relocated to the corresponing
        /// host addresses stored in ::CU_JIT_GLOBAL_SYMBOL_ADDRESSES.\n
        /// Must contain ::CU_JIT_GLOBAL_SYMBOL_COUNT entries.\n
        /// When loding a device module, driver will relocate all encountered
        /// unresolved symbols to the host addresses.\n
        /// It is only allowed to register symbols that correspond to unresolved
        /// global variables.\n
        /// It is illegal to register the same device symbol at multiple addresses.\n
        /// Option type: const char **\n
        /// Applies to: dynamic linker only
        /// </summary>
        CU_JIT_GLOBAL_SYMBOL_NAMES,

        /// <summary>
        /// Array of host addresses that will be used to relocate corresponding
        /// device symbols stored in ::CU_JIT_GLOBAL_SYMBOL_NAMES.\n
        /// Must contain ::CU_JIT_GLOBAL_SYMBOL_COUNT entries.\n
        /// Option type: void **\n
        /// Applies to: dynamic linker only
        /// </summary>
        CU_JIT_GLOBAL_SYMBOL_ADDRESSES,

        /// <summary>
        /// Number of entries in ::CU_JIT_GLOBAL_SYMBOL_NAMES and
        /// ::CU_JIT_GLOBAL_SYMBOL_ADDRESSES arrays.\n
        /// Option type: unsigned int\n
        /// Applies to: dynamic linker only
        /// </summary>
        CU_JIT_GLOBAL_SYMBOL_COUNT,

        /// <summary>
        /// </summary>
        CU_JIT_NUM_OPTIONS
    }
}

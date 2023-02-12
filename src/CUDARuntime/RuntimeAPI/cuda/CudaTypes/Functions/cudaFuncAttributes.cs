using System.Runtime.InteropServices;

namespace Hybridizer.Runtime.CUDAImports
{
    /// <summary>
    /// CUDA function attributes
    /// </summary>
    [IntrinsicType("cudaFuncAttributes")]
    [StructLayout(LayoutKind.Sequential)]
    public struct cudaFuncAttributes
    {
        /// <summary>
        ///  The size in bytes of statically-allocated shared memory per block
        ///  required by this function. This does not include dynamically-allocated
        ///  shared memory requested by the user at runtime.
        /// </summary>
        public size_t sharedSizeBytes;
        /// <summary>
        /// The size in bytes of user-allocated constant memory required by this function.
        /// </summary>
        public size_t constSizeBytes;
        /// <summary>
        /// The size in bytes of local memory used by each thread of this function.
        /// </summary>
        public size_t localSizeBytes;
        /// <summary>
        /// The maximum number of threads per block, beyond which a launch of the
        /// function would fail. This number depends on both the function and the
        /// device on which the function is currently loaded.
        /// </summary>
        public int maxThreadsPerBlock;
        /// <summary>
        /// The number of registers used by each thread of this function.
        /// </summary>
        public int numRegs;
        /// <summary>
        /// The PTX virtual architecture version for which the function was
        /// compiled. This value is the major PTX version * 10 + the minor PTX
        /// version, so a PTX version 1.3 function would return the value 13.
        /// </summary>
        public int ptxVersion;
        /// <summary>
        /// The binary architecture version for which the function was compiled.
        /// This value is the major binary version * 10 + the minor binary version,
        /// so a binary version 1.3 function would return the value 13.
        /// </summary>
        public int binaryVersion;
        /// <summary>
        /// The attribute to indicate whether the function has been compiled with
        /// user specified option "-Xptxas --dlcm=ca" set.
        /// </summary>
        public int cacheModeCA;
        /// <summary>
        /// The maximum size in bytes of dynamic shared memory per block for
        /// this function. Any launch must have a dynamic shared memory size
        /// smaller than this value.
        /// </summary>
        public int maxDynamicSharedSizeBytes;
        /// <summary>
        /// On devices where the L1 cache and shared memory use the same hardware resources, 
        /// this sets the shared memory carveout preference, in percent of the total resources. 
        /// This is only a hint, and the driver can choose a different ratio if required to execute the function.
        /// </summary>
        public int preferredShmemCarveout;
    }
}
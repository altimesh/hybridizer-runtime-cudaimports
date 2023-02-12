/* (c) ALTIMESH 2018 -- all rights reserved */
using System.Runtime.InteropServices;

namespace Hybridizer.Runtime.CUDAImports
{
    /// <summary>
    /// Supported flavors
    /// </summary>
    public enum HybridizerFlavor : int
    {
        /// <summary>
        /// Targets NVIDIA GPUS
        /// </summary>
        CUDA = 1,
        /// <summary>
        /// Simple C++ generation (no vectorization)
        /// </summary>
        OMP = 2,
        /// <summary>
        /// AVX and AVX2 C++ code generation
        /// </summary>
        AVX = 3,
        /// <summary>
        /// Vectorized CUDA leveraging ILP for kepler architectures (deprecated)
        /// </summary>
        KEPLER = 4,
        /// <summary>
        /// Vectorized C++ targeting MIC instruction set (KNC) (deprecated)
        /// </summary>
        PHI = 5,
        /// <summary>
        /// 
        /// </summary>
        JAVA = 7,
        /// <summary>
        /// Vectorized C++ targeting VSX instruction set (Power)
        /// </summary>
        VSX = 9,
        /// <summary>
        /// OpenCL target
        /// </summary>
        OPENCL = 8,
        /// <summary>
        /// 
        /// </summary>
        PTX = 6,
        /// <summary>
        /// 
        /// </summary>
        HYBOP = 10,
        /// <summary>
        /// Vectorized C++ targeting AVX512 instruction set (KNL, skylake avx512...)
        /// </summary>
        AVX512 = 11,
        /// <summary>
        /// 
        /// </summary>
        JAVA_BYTECODE = 12, // Allows to disable bytecode generation
        /// <summary>
        /// Targets AMD GPUS
        /// </summary>
        HIP = 13
    }

    /// <summary>
    /// Hybridizer properties at runtime
    /// </summary>
    [StructLayout(LayoutKind.Explicit,Size = 16)]
    public struct HybridizerProperties
    {
        [FieldOffset(0)]
        private int _useHybridArrays;
        [FieldOffset(4)]
        private int _flavor;
        [FieldOffset(8)]
        private int _delegateSupport;
        [FieldOffset(12)]
        private int _compatibilityMode;
        [FieldOffset(16)]
        private int _dummy;

        /// <summary>
        /// Use HybridArrays or C-style arrays (pointer to first element)
        /// </summary>
        public int UseHybridArrays
        {
            get { return _useHybridArrays; }
        }

        /// <summary>
        /// Target Flavor
        /// <see cref="HybridizerFlavor"/>
        /// </summary>
        public HybridizerFlavor Flavor
        {
            get { return (HybridizerFlavor) _flavor; }
        }

        /// <summary>
        /// internal
        /// </summary>
        public bool CompatibilityMode
        {
            get { return _compatibilityMode != 0; }
        }

        /// <summary>
        /// internal
        /// </summary>
        public bool DelegateSupport
        {
            get { return _delegateSupport != 0; }
        }
    }
}

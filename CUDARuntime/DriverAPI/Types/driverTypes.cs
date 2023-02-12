using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Hybridizer.Runtime.CUDAImports
{
    /// <summary>
    ///  CUDA device
    /// </summary>
    public struct CUdevice
    {
        IntPtr _inner;
        /// <summary>
        /// </summary>
        public static implicit operator IntPtr(CUdevice ctx) { return ctx._inner; }
        /// <summary>
        /// </summary>
        public bool Exists() { return _inner != IntPtr.Zero; }
        /// <summary>
        /// </summary>
        public CUdevice(IntPtr inner) { _inner = inner; }
    }
    /// <summary>
    /// CUDA context
    /// </summary>
    public struct CUcontext {
        IntPtr _inner;
        /// <summary>
        /// </summary>
        public static implicit operator IntPtr(CUcontext ctx) { return ctx._inner; }
        /// <summary>
        /// </summary>
        public bool Exists() { return _inner != IntPtr.Zero; }
        /// <summary>
        /// </summary>
        public CUcontext(IntPtr inner) { _inner = inner; }
    }
    /// <summary>
    /// CUDA module
    /// </summary>
    public struct CUmodule
    {
        IntPtr _inner;
        /// <summary>
        /// </summary>
        public static implicit operator IntPtr(CUmodule ctx) { return ctx._inner; }
        /// <summary>
        /// </summary>
        public bool Exists() { return _inner != IntPtr.Zero; }
        /// <summary>
        /// </summary>
        public CUmodule(IntPtr inner) { _inner = inner; }
    }
    /// <summary>
    /// CUDA function
    /// </summary>
    public struct CUfunction
    {
        IntPtr _inner;
        /// <summary>
        /// </summary>
        public static implicit operator IntPtr(CUfunction ctx) { return ctx._inner; }
        /// <summary>
        /// </summary>
        public bool Exists() { return _inner != IntPtr.Zero; }
        /// <summary>
        /// </summary>
        public CUfunction(IntPtr inner) { _inner = inner; }
    }
    /// <summary>
    /// CUDA array
    /// </summary>
    public struct CUarray
    {
        IntPtr _inner;
        /// <summary>
        /// </summary>
        public static implicit operator IntPtr(CUarray ctx) { return ctx._inner; }
        /// <summary>
        /// </summary>
        public bool Exists() { return _inner != IntPtr.Zero; }
        /// <summary>
        /// </summary>
        public CUarray(IntPtr inner) { _inner = inner; }
    }
    /// <summary>
    /// CUDA mipmapped array
    /// </summary>
    public struct CUmipmappedArray
    {
        IntPtr _inner;
        /// <summary>
        /// </summary>
        public static implicit operator IntPtr(CUmipmappedArray ctx) { return ctx._inner; }
        /// <summary>
        /// </summary>
        public bool Exists() { return _inner != IntPtr.Zero; }
        /// <summary>
        /// </summary>
        public CUmipmappedArray(IntPtr inner) { _inner = inner; }
    }
    /// <summary>
    /// CUDA texture reference
    /// </summary>
    public struct CUtexref
    {
        IntPtr _inner;
        /// <summary>
        /// </summary>
        public static implicit operator IntPtr(CUtexref ctx) { return ctx._inner; }
        /// <summary>
        /// </summary>
        public bool Exists() { return _inner != IntPtr.Zero; }
        /// <summary>
        /// </summary>
        public CUtexref(IntPtr inner) { _inner = inner; }
    }
    /// <summary>
    /// CUDA surface reference
    /// </summary>
    public struct CUsurfref
    {
        IntPtr _inner;
        /// <summary>
        /// </summary>
        public static implicit operator IntPtr(CUsurfref ctx) { return ctx._inner; }
        /// <summary>
        /// </summary>
        public bool Exists() { return _inner != IntPtr.Zero; }
        /// <summary>
        /// </summary>
        public CUsurfref(IntPtr inner) { _inner = inner; }
    }

    /// <summary>
    /// CUDA event
    /// </summary>
    public struct CUevent
    {
        IntPtr _inner;
        /// <summary>
        /// </summary>
        public static implicit operator IntPtr(CUevent ctx) { return ctx._inner; }
        /// <summary>
        /// </summary>
        public bool Exists() { return _inner != IntPtr.Zero; }
        /// <summary>
        /// </summary>
        public CUevent(IntPtr inner) { _inner = inner; }
    }

    /// <summary>
    /// CUDA stream
    /// </summary>
    public struct CUstream
    {
        IntPtr _inner;
        /// <summary>
        /// </summary>
        public static implicit operator IntPtr(CUstream ctx) { return ctx._inner; }

        /// <summary>
        /// converts a cuStream to a cudastream
        /// </summary>
        /// <param name="stream"></param>
        public static explicit operator cudaStream_t(CUstream stream)
        {
            return new cudaStream_t(stream._inner);
        }

        /// <summary>
        /// </summary>
        public bool Exists() { return _inner != IntPtr.Zero; }
        /// <summary>
        /// </summary>
        public CUstream(IntPtr inner) { _inner = inner; }

        /// <summary>
        /// The null stream (no stream used)
        /// </summary>
        public static CUstream NO_STREAM = new CUstream(IntPtr.Zero);
    }

    /// <summary>
    /// CUDA graphics interop resource
    /// </summary>
    public struct CUgraphicsResource
    {
        IntPtr _inner;
        /// <summary>
        /// </summary>
        public static implicit operator IntPtr(CUgraphicsResource ctx) { return ctx._inner; }
        /// <summary>
        /// </summary>
        public bool Exists() { return _inner != IntPtr.Zero; }
        /// <summary>
        /// </summary>
        public CUgraphicsResource(IntPtr inner) { _inner = inner; }
    }
    /// <summary>
    /// An opaque value that represents a CUDA texture object
    /// </summary>
    public struct CUtexObject
    {
        ulong _inner;
        /// <summary>
        /// </summary>
        public static implicit operator ulong(CUtexObject ctx) { return ctx._inner; }
        /// <summary>
        /// </summary>
        public bool Exists() { return _inner != 0UL; }
        /// <summary>
        /// </summary>
        public CUtexObject(ulong inner) { _inner = inner; }
    }
    /// <summary>
    /// An opaque value that represents a CUDA surface object
    /// </summary>
    public struct CUsurfObject
    {
        ulong _inner;
        /// <summary>
        /// </summary>
        public static implicit operator ulong(CUsurfObject ctx) { return ctx._inner; }
        /// <summary>
        /// </summary>
        public bool Exists() { return _inner != 0UL; }
        /// <summary>
        /// </summary>
        public CUsurfObject(ulong inner) { _inner = inner; }
    }
    /// <summary>
    /// CUDA external memory
    /// </summary>
    public struct CUexternalMemory
    {
        IntPtr _inner;
        /// <summary>
        /// </summary>
        public static implicit operator IntPtr(CUexternalMemory ctx) { return ctx._inner; }
        /// <summary>
        /// </summary>
        public bool Exists() { return _inner != IntPtr.Zero; }
        /// <summary>
        /// </summary>
        public CUexternalMemory(IntPtr inner) { _inner = inner; }
    }
    /// <summary>
    /// CUDA external semaphore
    /// </summary>
    public struct CUexternalSemaphore
    {
        IntPtr _inner;
        /// <summary>
        /// </summary>
        public static implicit operator IntPtr(CUexternalSemaphore ctx) { return ctx._inner; }
        /// <summary>
        /// </summary>
        public bool Exists() { return _inner != IntPtr.Zero; }
        /// <summary>
        /// </summary>
        public CUexternalSemaphore(IntPtr inner) { _inner = inner; }
    }
    /// <summary>
    /// CUDA  graph
    /// </summary>
    public struct CUgraph
    {
        IntPtr _inner;
        /// <summary>
        /// </summary>
        public static implicit operator IntPtr(CUgraph ctx) { return ctx._inner; }
        /// <summary>
        /// </summary>
        public bool Exists() { return _inner != IntPtr.Zero; }
        /// <summary>
        /// </summary>
        public CUgraph(IntPtr inner) { _inner = inner; }
    }
    /// <summary>
    ///  CUDA graph node
    /// </summary>
    public struct CUgraphNode
    {
        IntPtr _inner;
        /// <summary>
        /// </summary>
        public static implicit operator IntPtr(CUgraphNode ctx) { return ctx._inner; }
        /// <summary>
        /// </summary>
        public bool Exists() { return _inner != IntPtr.Zero; }
        /// <summary>
        /// </summary>
        public CUgraphNode(IntPtr inner) { _inner = inner; }
    }
    /// <summary>
    /// CUDA executable graph
    /// </summary>
    public struct CUgraphExec
    {
        IntPtr _inner;
        /// <summary>
        /// </summary>
        public static implicit operator IntPtr(CUgraphExec ctx) { return ctx._inner; }
        /// <summary>
        /// </summary>
        public bool Exists() { return _inner != IntPtr.Zero; }
        /// <summary>
        /// </summary>
        public CUgraphExec(IntPtr inner) { _inner = inner; }
    }

    /// <summary>
    /// Stream creation flags
    /// </summary>
    [Flags]
    public enum CUstream_flags: uint
    {
        /// <summary>
        /// Default stream flag 
        /// </summary>
        Default = 0,
        /// <summary>
        /// Stream does not synchronize with stream 0 (the NULL stream) 
        /// </summary>
        NonBlocking = 1
    }

    /// <summary>
    /// Event creation flags
    /// </summary>
    [Flags]
    public enum CUevent_flags : uint
    {
        /// <summary>
        /// Default event flag
        /// </summary>
        CU_EVENT_DEFAULT = 0x0,
        /// <summary>
        /// Event uses blocking synchronization
        /// </summary>
        CU_EVENT_BLOCKING_SYNC = 0x1,
        /// <summary>
        /// Event will not record timing data
        /// </summary>
        CU_EVENT_DISABLE_TIMING = 0x2,
        /// <summary>
        /// Event is suitable for interprocess use. CU_EVENT_DISABLE_TIMING must be set
        /// </summary>
        CU_EVENT_INTERPROCESS = 0x4  
    }

    /// <summary>
    /// Device attributes in driver API
    /// </summary>
    public enum CUdevice_attribute {
        /// <summary>
        ///  Maximum number of threads per block 
        /// </summary>
        CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 1,
        /// <summary>
        ///  Maximum block dimension X 
        /// </summary>
        CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X = 2,
        /// <summary>
        ///  Maximum block dimension Y 
        /// </summary>
        CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y = 3,
        /// <summary>
        ///  Maximum block dimension Z 
        /// </summary>
        CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z = 4,
        /// <summary>
        ///  Maximum grid dimension X 
        /// </summary>
        CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X = 5,
        /// <summary>
        ///  Maximum grid dimension Y 
        /// </summary>
        CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y = 6,
        /// <summary>
        ///  Maximum grid dimension Z 
        /// </summary>
        CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z = 7,
        /// <summary>
        ///  Maximum shared memory available per block in bytes 
        /// </summary>
        CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK = 8,
        /// <summary>
        ///  Deprecated, use CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK 
        /// </summary>
        CU_DEVICE_ATTRIBUTE_SHARED_MEMORY_PER_BLOCK = 8,
        /// <summary>
        ///  Memory available on device for __constant__ variables in a CUDA C kernel in bytes 
        /// </summary>
        CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY = 9,
        /// <summary>
        ///  Warp size in threads 
        /// </summary>
        CU_DEVICE_ATTRIBUTE_WARP_SIZE = 10,
        /// <summary>
        ///  Maximum pitch in bytes allowed by memory copies 
        /// </summary>
        CU_DEVICE_ATTRIBUTE_MAX_PITCH = 11,
        /// <summary>
        ///  Maximum number of 32-bit registers available per block 
        /// </summary>
        CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK = 12,
        /// <summary>
        ///  Deprecated, use CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK 
        /// </summary>
        CU_DEVICE_ATTRIBUTE_REGISTERS_PER_BLOCK = 12,
        /// <summary>
        ///  Typical clock frequency in kilohertz 
        /// </summary>
        CU_DEVICE_ATTRIBUTE_CLOCK_RATE = 13,
        /// <summary>
        ///  Alignment requirement for textures 
        /// </summary>
        CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT = 14,
        /// <summary>
        ///  Device can possibly copy memory and execute a kernel concurrently. Deprecated. Use instead CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT. 
        /// </summary>
        CU_DEVICE_ATTRIBUTE_GPU_OVERLAP = 15,
        /// <summary>
        ///  Number of multiprocessors on device 
        /// </summary>
        CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 16,
        /// <summary>
        ///  Specifies whether there is a run time limit on kernels 
        /// </summary>
        CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT = 17,
        /// <summary>
        ///  Device is integrated with host memory 
        /// </summary>
        CU_DEVICE_ATTRIBUTE_INTEGRATED = 18,
        /// <summary>
        ///  Device can map host memory into CUDA address space 
        /// </summary>
        CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY = 19,
        /// <summary>
        ///  Compute mode (See ::CUcomputemode for details) 
        /// </summary>
        CU_DEVICE_ATTRIBUTE_COMPUTE_MODE = 20,
        /// <summary>
        ///  Maximum 1D texture width 
        /// </summary>
        CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH = 21,
        /// <summary>
        ///  Maximum 2D texture width 
        /// </summary>
        CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH = 22,
        /// <summary>
        ///  Maximum 2D texture height 
        /// </summary>
        CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT = 23,
        /// <summary>
        ///  Maximum 3D texture width 
        /// </summary>
        CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH = 24,
        /// <summary>
        ///  Maximum 3D texture height 
        /// </summary>
        CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT = 25,
        /// <summary>
        ///  Maximum 3D texture depth 
        /// </summary>
        CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH = 26,
        /// <summary>
        ///  Maximum 2D layered texture width 
        /// </summary>
        CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_WIDTH = 27,
        /// <summary>
        ///  Maximum 2D layered texture height 
        /// </summary>
        CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_HEIGHT = 28,
        /// <summary>
        ///  Maximum layers in a 2D layered texture 
        /// </summary>
        CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_LAYERS = 29,
        /// <summary>
        ///  Deprecated, use CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_WIDTH 
        /// </summary>
        CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_WIDTH = 27,
        /// <summary>
        ///  Deprecated, use CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_HEIGHT 
        /// </summary>
        CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_HEIGHT = 28,
        /// <summary>
        ///  Deprecated, use CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_LAYERS 
        /// </summary>
        CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_NUMSLICES = 29,
        /// <summary>
        ///  Alignment requirement for surfaces 
        /// </summary>
        CU_DEVICE_ATTRIBUTE_SURFACE_ALIGNMENT = 30,
        /// <summary>
        ///  Device can possibly execute multiple kernels concurrently 
        /// </summary>
        CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS = 31,
        /// <summary>
        ///  Device has ECC support enabled 
        /// </summary>
        CU_DEVICE_ATTRIBUTE_ECC_ENABLED = 32,
        /// <summary>
        ///  PCI bus ID of the device 
        /// </summary>
        CU_DEVICE_ATTRIBUTE_PCI_BUS_ID = 33,
        /// <summary>
        ///  PCI device ID of the device 
        /// </summary>
        CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID = 34,
        /// <summary>
        ///  Device is using TCC driver model 
        /// </summary>
        CU_DEVICE_ATTRIBUTE_TCC_DRIVER = 35,
        /// <summary>
        ///  Peak memory clock frequency in kilohertz 
        /// </summary>
        CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE = 36,
        /// <summary>
        ///  Global memory bus width in bits 
        /// </summary>
        CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH = 37,
        /// <summary>
        ///  Size of L2 cache in bytes 
        /// </summary>
        CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE = 38,
        /// <summary>
        ///  Maximum resident threads per multiprocessor 
        /// </summary>
        CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR = 39,
        /// <summary>
        ///  Number of asynchronous engines 
        /// </summary>
        CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT = 40,
        /// <summary>
        ///  Device shares a unified address space with the host 
        /// </summary>
        CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING = 41,
        /// <summary>
        ///  Maximum 1D layered texture width 
        /// </summary>
        CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_WIDTH = 42,
        /// <summary>
        ///  Maximum layers in a 1D layered texture 
        /// </summary>
        CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_LAYERS = 43,
        /// <summary>
        ///  Deprecated, do not use. 
        /// </summary>
        CU_DEVICE_ATTRIBUTE_CAN_TEX2D_GATHER = 44,
        /// <summary>
        ///  Maximum 2D texture width if CUDA_ARRAY3D_TEXTURE_GATHER is set 
        /// </summary>
        CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_WIDTH = 45,
        /// <summary>
        ///  Maximum 2D texture height if CUDA_ARRAY3D_TEXTURE_GATHER is set 
        /// </summary>
        CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_HEIGHT = 46,
        /// <summary>
        ///  Alternate maximum 3D texture width 
        /// </summary>
        CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH_ALTERNATE = 47,
        /// <summary>
        ///  Alternate maximum 3D texture height 
        /// </summary>
        CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT_ALTERNATE = 48,
        /// <summary>
        ///  Alternate maximum 3D texture depth 
        /// </summary>
        CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH_ALTERNATE = 49,
        /// <summary>
        ///  PCI domain ID of the device 
        /// </summary>
        CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID = 50,
        /// <summary>
        ///  Pitch alignment requirement for textures 
        /// </summary>
        CU_DEVICE_ATTRIBUTE_TEXTURE_PITCH_ALIGNMENT = 51,
        /// <summary>
        ///  Maximum cubemap texture width/height 
        /// </summary>
        CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_WIDTH = 52,
        /// <summary>
        ///  Maximum cubemap layered texture width/height 
        /// </summary>
        CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_WIDTH = 53,
        /// <summary>
        ///  Maximum layers in a cubemap layered texture 
        /// </summary>
        CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_LAYERS = 54,
        /// <summary>
        ///  Maximum 1D surface width 
        /// </summary>
        CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_WIDTH = 55,
        /// <summary>
        ///  Maximum 2D surface width 
        /// </summary>
        CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_WIDTH = 56,
        /// <summary>
        ///  Maximum 2D surface height 
        /// </summary>
        CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_HEIGHT = 57,
        /// <summary>
        ///  Maximum 3D surface width 
        /// </summary>
        CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_WIDTH = 58,
        /// <summary>
        ///  Maximum 3D surface height 
        /// </summary>
        CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_HEIGHT = 59,
        /// <summary>
        ///  Maximum 3D surface depth 
        /// </summary>
        CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_DEPTH = 60,
        /// <summary>
        ///  Maximum 1D layered surface width 
        /// </summary>
        CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_WIDTH = 61,
        /// <summary>
        ///  Maximum layers in a 1D layered surface 
        /// </summary>
        CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_LAYERS = 62,
        /// <summary>
        ///  Maximum 2D layered surface width 
        /// </summary>
        CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_WIDTH = 63,
        /// <summary>
        ///  Maximum 2D layered surface height 
        /// </summary>
        CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_HEIGHT = 64,
        /// <summary>
        ///  Maximum layers in a 2D layered surface 
        /// </summary>
        CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_LAYERS = 65,
        /// <summary>
        ///  Maximum cubemap surface width 
        /// </summary>
        CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_WIDTH = 66,
        /// <summary>
        ///  Maximum cubemap layered surface width 
        /// </summary>
        CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_WIDTH = 67,
        /// <summary>
        ///  Maximum layers in a cubemap layered surface 
        /// </summary>
        CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_LAYERS = 68,
        /// <summary>
        ///  Maximum 1D linear texture width 
        /// </summary>
        CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LINEAR_WIDTH = 69,
        /// <summary>
        ///  Maximum 2D linear texture width 
        /// </summary>
        CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_WIDTH = 70,
        /// <summary>
        ///  Maximum 2D linear texture height 
        /// </summary>
        CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_HEIGHT = 71,
        /// <summary>
        ///  Maximum 2D linear texture pitch in bytes 
        /// </summary>
        CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_PITCH = 72,
        /// <summary>
        ///  Maximum mipmapped 2D texture width 
        /// </summary>
        CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_WIDTH = 73,
        /// <summary>
        ///  Maximum mipmapped 2D texture height 
        /// </summary>
        CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_HEIGHT = 74,
        /// <summary>
        ///  Major compute capability version number 
        /// </summary>
        CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75,
        /// <summary>
        ///  Minor compute capability version number 
        /// </summary>
        CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76,
        /// <summary>
        ///  Maximum mipmapped 1D texture width 
        /// </summary>
        CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_MIPMAPPED_WIDTH = 77,
        /// <summary>
        ///  Device supports stream priorities 
        /// </summary>
        CU_DEVICE_ATTRIBUTE_STREAM_PRIORITIES_SUPPORTED = 78,
        /// <summary>
        ///  Device supports caching globals in L1 
        /// </summary>
        CU_DEVICE_ATTRIBUTE_GLOBAL_L1_CACHE_SUPPORTED = 79,
        /// <summary>
        ///  Device supports caching locals in L1 
        /// </summary>
        CU_DEVICE_ATTRIBUTE_LOCAL_L1_CACHE_SUPPORTED = 80,
        /// <summary>
        ///  Maximum shared memory available per multiprocessor in bytes 
        /// </summary>
        CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR = 81,
        /// <summary>
        ///  Maximum number of 32-bit registers available per multiprocessor 
        /// </summary>
        CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR = 82,
        /// <summary>
        ///  Device can allocate managed memory on this system 
        /// </summary>
        CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY = 83,
        /// <summary>
        ///  Device is on a multi-GPU board 
        /// </summary>
        CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD = 84,
        /// <summary>
        ///  Unique id for a group of devices on the same multi-GPU board 
        /// </summary>
        CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD_GROUP_ID = 85,
        /// <summary>
        ///  Link between the device and the host supports native atomic operations (this is a placeholder attribute, and is not supported on any current hardware)
        /// </summary>
        CU_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED = 86,
        /// <summary>
        ///  Ratio of single precision performance (in floating-point operations per second) to double precision performance 
        /// </summary>
        CU_DEVICE_ATTRIBUTE_SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO = 87,
        /// <summary>
        ///  Device supports coherently accessing pageable memory without calling cudaHostRegister on it 
        /// </summary>
        CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS = 88,
        /// <summary>
        ///  Device can coherently access managed memory concurrently with the CPU 
        /// </summary>
        CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS = 89,
        /// <summary>
        ///  Device supports compute preemption. 
        /// </summary>
        CU_DEVICE_ATTRIBUTE_COMPUTE_PREEMPTION_SUPPORTED = 90,
        /// <summary>
        ///  Device can access host registered memory at the same virtual address as the CPU 
        /// </summary>
        CU_DEVICE_ATTRIBUTE_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM = 91,
        /// <summary>
        ///  ::cuStreamBatchMemOp and related APIs are supported. 
        /// </summary>
        CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_MEM_OPS = 92,
        /// <summary>
        ///  64-bit operations are supported in ::cuStreamBatchMemOp and related APIs. 
        /// </summary>
        CU_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS = 93,
        /// <summary>
        ///  ::CU_STREAM_WAIT_VALUE_NOR is supported. 
        /// </summary>
        CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_WAIT_VALUE_NOR = 94,
        /// <summary>
        ///  Device supports launching cooperative kernels via ::cuLaunchCooperativeKernel 
        /// </summary>
        CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH = 95,
        /// <summary>
        ///  Device can participate in cooperative kernels launched via ::cuLaunchCooperativeKernelMultiDevice 
        /// </summary>
        CU_DEVICE_ATTRIBUTE_COOPERATIVE_MULTI_DEVICE_LAUNCH = 96,
        /// <summary>
        ///  Maximum optin shared memory per block 
        /// </summary>
        CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN = 97,
        /// <summary>
        ///  Both the ::CU_STREAM_WAIT_VALUE_FLUSH flag and the ::CU_STREAM_MEM_OP_FLUSH_REMOTE_WRITES MemOp are supported on the device. See \ref CUDA_MEMOP for additional details. 
        /// </summary>
        CU_DEVICE_ATTRIBUTE_CAN_FLUSH_REMOTE_WRITES = 98,
        /// <summary>
        ///  Device supports host memory registration via ::cudaHostRegister. 
        /// </summary>
        CU_DEVICE_ATTRIBUTE_HOST_REGISTER_SUPPORTED = 99,
        /// <summary>
        ///  Device accesses pageable memory via the host's page tables. 
        /// </summary>
        CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES = 100,
        /// <summary>
        ///  The host can directly access managed memory on the device without migration. 
        /// </summary>
        CU_DEVICE_ATTRIBUTE_DIRECT_MANAGED_MEM_ACCESS_FROM_HOST = 101,
        /// <summary>
        ///  Upper boundary 
        /// </summary>
        CU_DEVICE_ATTRIBUTE_MAX,
    }
}

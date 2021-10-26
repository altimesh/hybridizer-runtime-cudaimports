﻿using System;
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

    [Flags]
    public enum CUstream_flags: uint
    {
        Default = 0,
        NonBlocking = 1
    }

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

    public enum CUdevice_attribute {
        CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 1,              /**< Maximum number of threads per block */
        CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X = 2,                    /**< Maximum block dimension X */
        CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y = 3,                    /**< Maximum block dimension Y */
        CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z = 4,                    /**< Maximum block dimension Z */
        CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X = 5,                     /**< Maximum grid dimension X */
        CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y = 6,                     /**< Maximum grid dimension Y */
        CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z = 7,                     /**< Maximum grid dimension Z */
        CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK = 8,        /**< Maximum shared memory available per block in bytes */
        CU_DEVICE_ATTRIBUTE_SHARED_MEMORY_PER_BLOCK = 8,            /**< Deprecated, use CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK */
        CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY = 9,              /**< Memory available on device for __constant__ variables in a CUDA C kernel in bytes */
        CU_DEVICE_ATTRIBUTE_WARP_SIZE = 10,                         /**< Warp size in threads */
        CU_DEVICE_ATTRIBUTE_MAX_PITCH = 11,                         /**< Maximum pitch in bytes allowed by memory copies */
        CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK = 12,           /**< Maximum number of 32-bit registers available per block */
        CU_DEVICE_ATTRIBUTE_REGISTERS_PER_BLOCK = 12,               /**< Deprecated, use CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK */
        CU_DEVICE_ATTRIBUTE_CLOCK_RATE = 13,                        /**< Typical clock frequency in kilohertz */
        CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT = 14,                 /**< Alignment requirement for textures */
        CU_DEVICE_ATTRIBUTE_GPU_OVERLAP = 15,                       /**< Device can possibly copy memory and execute a kernel concurrently. Deprecated. Use instead CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT. */
        CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 16,              /**< Number of multiprocessors on device */
        CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT = 17,               /**< Specifies whether there is a run time limit on kernels */
        CU_DEVICE_ATTRIBUTE_INTEGRATED = 18,                        /**< Device is integrated with host memory */
        CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY = 19,               /**< Device can map host memory into CUDA address space */
        CU_DEVICE_ATTRIBUTE_COMPUTE_MODE = 20,                      /**< Compute mode (See ::CUcomputemode for details) */
        CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH = 21,           /**< Maximum 1D texture width */
        CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH = 22,           /**< Maximum 2D texture width */
        CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT = 23,          /**< Maximum 2D texture height */
        CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH = 24,           /**< Maximum 3D texture width */
        CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT = 25,          /**< Maximum 3D texture height */
        CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH = 26,           /**< Maximum 3D texture depth */
        CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_WIDTH = 27,   /**< Maximum 2D layered texture width */
        CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_HEIGHT = 28,  /**< Maximum 2D layered texture height */
        CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_LAYERS = 29,  /**< Maximum layers in a 2D layered texture */
        CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_WIDTH = 27,     /**< Deprecated, use CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_WIDTH */
        CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_HEIGHT = 28,    /**< Deprecated, use CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_HEIGHT */
        CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_NUMSLICES = 29, /**< Deprecated, use CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_LAYERS */
        CU_DEVICE_ATTRIBUTE_SURFACE_ALIGNMENT = 30,                 /**< Alignment requirement for surfaces */
        CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS = 31,                /**< Device can possibly execute multiple kernels concurrently */
        CU_DEVICE_ATTRIBUTE_ECC_ENABLED = 32,                       /**< Device has ECC support enabled */
        CU_DEVICE_ATTRIBUTE_PCI_BUS_ID = 33,                        /**< PCI bus ID of the device */
        CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID = 34,                     /**< PCI device ID of the device */
        CU_DEVICE_ATTRIBUTE_TCC_DRIVER = 35,                        /**< Device is using TCC driver model */
        CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE = 36,                 /**< Peak memory clock frequency in kilohertz */
        CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH = 37,           /**< Global memory bus width in bits */
        CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE = 38,                     /**< Size of L2 cache in bytes */
        CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR = 39,    /**< Maximum resident threads per multiprocessor */
        CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT = 40,                /**< Number of asynchronous engines */
        CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING = 41,                /**< Device shares a unified address space with the host */
        CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_WIDTH = 42,   /**< Maximum 1D layered texture width */
        CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_LAYERS = 43,  /**< Maximum layers in a 1D layered texture */
        CU_DEVICE_ATTRIBUTE_CAN_TEX2D_GATHER = 44,                  /**< Deprecated, do not use. */
        CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_WIDTH = 45,    /**< Maximum 2D texture width if CUDA_ARRAY3D_TEXTURE_GATHER is set */
        CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_HEIGHT = 46,   /**< Maximum 2D texture height if CUDA_ARRAY3D_TEXTURE_GATHER is set */
        CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH_ALTERNATE = 47, /**< Alternate maximum 3D texture width */
        CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT_ALTERNATE = 48,/**< Alternate maximum 3D texture height */
        CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH_ALTERNATE = 49, /**< Alternate maximum 3D texture depth */
        CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID = 50,                     /**< PCI domain ID of the device */
        CU_DEVICE_ATTRIBUTE_TEXTURE_PITCH_ALIGNMENT = 51,           /**< Pitch alignment requirement for textures */
        CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_WIDTH = 52,      /**< Maximum cubemap texture width/height */
        CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_WIDTH = 53,  /**< Maximum cubemap layered texture width/height */
        CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_LAYERS = 54, /**< Maximum layers in a cubemap layered texture */
        CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_WIDTH = 55,           /**< Maximum 1D surface width */
        CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_WIDTH = 56,           /**< Maximum 2D surface width */
        CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_HEIGHT = 57,          /**< Maximum 2D surface height */
        CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_WIDTH = 58,           /**< Maximum 3D surface width */
        CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_HEIGHT = 59,          /**< Maximum 3D surface height */
        CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_DEPTH = 60,           /**< Maximum 3D surface depth */
        CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_WIDTH = 61,   /**< Maximum 1D layered surface width */
        CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_LAYERS = 62,  /**< Maximum layers in a 1D layered surface */
        CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_WIDTH = 63,   /**< Maximum 2D layered surface width */
        CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_HEIGHT = 64,  /**< Maximum 2D layered surface height */
        CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_LAYERS = 65,  /**< Maximum layers in a 2D layered surface */
        CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_WIDTH = 66,      /**< Maximum cubemap surface width */
        CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_WIDTH = 67,  /**< Maximum cubemap layered surface width */
        CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_LAYERS = 68, /**< Maximum layers in a cubemap layered surface */
        CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LINEAR_WIDTH = 69,    /**< Maximum 1D linear texture width */
        CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_WIDTH = 70,    /**< Maximum 2D linear texture width */
        CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_HEIGHT = 71,   /**< Maximum 2D linear texture height */
        CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_PITCH = 72,    /**< Maximum 2D linear texture pitch in bytes */
        CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_WIDTH = 73, /**< Maximum mipmapped 2D texture width */
        CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_HEIGHT = 74,/**< Maximum mipmapped 2D texture height */
        CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75,          /**< Major compute capability version number */
        CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76,          /**< Minor compute capability version number */
        CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_MIPMAPPED_WIDTH = 77, /**< Maximum mipmapped 1D texture width */
        CU_DEVICE_ATTRIBUTE_STREAM_PRIORITIES_SUPPORTED = 78,       /**< Device supports stream priorities */
        CU_DEVICE_ATTRIBUTE_GLOBAL_L1_CACHE_SUPPORTED = 79,         /**< Device supports caching globals in L1 */
        CU_DEVICE_ATTRIBUTE_LOCAL_L1_CACHE_SUPPORTED = 80,          /**< Device supports caching locals in L1 */
        CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR = 81,  /**< Maximum shared memory available per multiprocessor in bytes */
        CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR = 82,  /**< Maximum number of 32-bit registers available per multiprocessor */
        CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY = 83,                    /**< Device can allocate managed memory on this system */
        CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD = 84,                    /**< Device is on a multi-GPU board */
        CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD_GROUP_ID = 85,           /**< Unique id for a group of devices on the same multi-GPU board */
        CU_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED = 86,       /**< Link between the device and the host supports native atomic operations (this is a placeholder attribute, and is not supported on any current hardware)*/
        CU_DEVICE_ATTRIBUTE_SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO = 87,  /**< Ratio of single precision performance (in floating-point operations per second) to double precision performance */
        CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS = 88,            /**< Device supports coherently accessing pageable memory without calling cudaHostRegister on it */
        CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS = 89,         /**< Device can coherently access managed memory concurrently with the CPU */
        CU_DEVICE_ATTRIBUTE_COMPUTE_PREEMPTION_SUPPORTED = 90,      /**< Device supports compute preemption. */
        CU_DEVICE_ATTRIBUTE_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM = 91, /**< Device can access host registered memory at the same virtual address as the CPU */
        CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_MEM_OPS = 92,            /**< ::cuStreamBatchMemOp and related APIs are supported. */
        CU_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS = 93,     /**< 64-bit operations are supported in ::cuStreamBatchMemOp and related APIs. */
        CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_WAIT_VALUE_NOR = 94,     /**< ::CU_STREAM_WAIT_VALUE_NOR is supported. */
        CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH = 95,                /**< Device supports launching cooperative kernels via ::cuLaunchCooperativeKernel */
        CU_DEVICE_ATTRIBUTE_COOPERATIVE_MULTI_DEVICE_LAUNCH = 96,   /**< Device can participate in cooperative kernels launched via ::cuLaunchCooperativeKernelMultiDevice */
        CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN = 97, /**< Maximum optin shared memory per block */
        CU_DEVICE_ATTRIBUTE_CAN_FLUSH_REMOTE_WRITES = 98,           /**< Both the ::CU_STREAM_WAIT_VALUE_FLUSH flag and the ::CU_STREAM_MEM_OP_FLUSH_REMOTE_WRITES MemOp are supported on the device. See \ref CUDA_MEMOP for additional details. */
        CU_DEVICE_ATTRIBUTE_HOST_REGISTER_SUPPORTED = 99,           /**< Device supports host memory registration via ::cudaHostRegister. */
        CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES = 100, /**< Device accesses pageable memory via the host's page tables. */
        CU_DEVICE_ATTRIBUTE_DIRECT_MANAGED_MEM_ACCESS_FROM_HOST = 101, /**< The host can directly access managed memory on the device without migration. */
        CU_DEVICE_ATTRIBUTE_MAX
    }
}

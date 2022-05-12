using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Hybridizer.Runtime.CUDAImports
{
    /// <summary>
    /// result of driver api call
    /// </summary>
    public enum CUresult
    {
        /// <summary>
        /// The API call returned with no errors. In the case of query calls, this
        /// also means that the operation being queried is complete (see
        /// ::cuEventQuery() and ::cuStreamQuery()).
        /// </summary>
        CUDA_SUCCESS = 0,

        /// <summary>
        /// This indicates that one or more of the parameters passed to the API call
        /// is not within an acceptable range of values.
        /// </summary>
        CUDA_ERROR_INVALID_VALUE = 1,

        /// <summary>
        /// The API call failed because it was unable to allocate enough memory to
        /// perform the requested operation.
        /// </summary>
        CUDA_ERROR_OUT_OF_MEMORY = 2,

        /// <summary>
        /// This indicates that the CUDA driver has not been initialized with
        /// ::cuInit() or that initialization has failed.
        /// </summary>
        CUDA_ERROR_NOT_INITIALIZED = 3,

        /// <summary>
        /// This indicates that the CUDA driver is in the process of shutting down.
        /// </summary>
        CUDA_ERROR_DEINITIALIZED = 4,

        /// <summary>
        /// This indicates profiler is not initialized for this run. This can
        /// happen when the application is running with external profiling tools
        /// like visual profiler.
        /// </summary>
        CUDA_ERROR_PROFILER_DISABLED = 5,

        /// <summary>
        /// This error return is deprecated as of CUDA 5.0. It is no longer an error
        /// to attempt to enable/disable the profiling via ::cuProfilerStart or
        /// ::cuProfilerStop without initialization.
        /// </summary>
        [Obsolete]
        CUDA_ERROR_PROFILER_NOT_INITIALIZED = 6,

        /// <summary>
        /// This error return is deprecated as of CUDA 5.0. It is no longer an error
        /// to call cuProfilerStart() when profiling is already enabled.
        /// </summary>
        [Obsolete]
        CUDA_ERROR_PROFILER_ALREADY_STARTED = 7,

        /// <summary>
        /// This error return is deprecated as of CUDA 5.0. It is no longer an error
        /// to call cuProfilerStop() when profiling is already disabled.
        /// </summary>
        [Obsolete]
        CUDA_ERROR_PROFILER_ALREADY_STOPPED = 8,

        /// <summary>
        /// This indicates that no CUDA-capable devices were detected by the installed
        /// CUDA driver.
        /// </summary>
        CUDA_ERROR_NO_DEVICE = 100,

        /// <summary>
        /// This indicates that the device ordinal supplied by the user does not
        /// correspond to a valid CUDA device.
        /// </summary>
        CUDA_ERROR_INVALID_DEVICE = 101,


        /// <summary>
        /// This indicates that the device kernel image is invalid. This can also
        /// indicate an invalid CUDA module.
        /// </summary>
        CUDA_ERROR_INVALID_IMAGE = 200,

        /// <summary>
        /// This most frequently indicates that there is no context bound to the
        /// current thread. This can also be returned if the context passed to an
        /// API call is not a valid handle (such as a context that has had
        /// ::cuCtxDestroy() invoked on it). This can also be returned if a user
        /// mixes different API versions (i.e. 3010 context with 3020 API calls).
        /// See ::cuCtxGetApiVersion() for more details.
        /// </summary>
        CUDA_ERROR_INVALID_CONTEXT = 201,

        /// <summary>
        /// This indicated that the context being supplied as a parameter to the
        /// API call was already the active context.
        /// This error return is deprecated as of CUDA 3.2. It is no longer an
        /// error to attempt to push the active context via ::cuCtxPushCurrent().
        /// </summary>
        [Obsolete]
        CUDA_ERROR_CONTEXT_ALREADY_CURRENT = 202,

        /// <summary>
        /// This indicates that a map or register operation has failed.
        /// </summary>
        CUDA_ERROR_MAP_FAILED = 205,

        /// <summary>
        /// This indicates that an unmap or unregister operation has failed.
        /// </summary>
        CUDA_ERROR_UNMAP_FAILED = 206,

        /// <summary>
        /// This indicates that the specified array is currently mapped and thus
        /// cannot be destroyed.
        /// </summary>
        CUDA_ERROR_ARRAY_IS_MAPPED = 207,

        /// <summary>
        /// This indicates that the resource is already mapped.
        /// </summary>
        CUDA_ERROR_ALREADY_MAPPED = 208,

        /// <summary>
        /// This indicates that there is no kernel image available that is suitable
        /// for the device. This can occur when a user specifies code generation
        /// options for a particular CUDA source file that do not include the
        /// corresponding device configuration.
        /// </summary>
        CUDA_ERROR_NO_BINARY_FOR_GPU = 209,

        /// <summary>
        /// This indicates that a resource has already been acquired.
        /// </summary>
        CUDA_ERROR_ALREADY_ACQUIRED = 210,

        /// <summary>
        /// This indicates that a resource is not mapped.
        /// </summary>
        CUDA_ERROR_NOT_MAPPED = 211,

        /// <summary>
        /// This indicates that a mapped resource is not available for access as an
        /// array.
        /// </summary>
        CUDA_ERROR_NOT_MAPPED_AS_ARRAY = 212,

        /// <summary>
        /// This indicates that a mapped resource is not available for access as a
        /// pointer.
        /// </summary>
        CUDA_ERROR_NOT_MAPPED_AS_POINTER = 213,

        /// <summary>
        /// This indicates that an uncorrectable ECC error was detected during
        /// execution.
        /// </summary>
        CUDA_ERROR_ECC_UNCORRECTABLE = 214,

        /// <summary>
        /// This indicates that the ::CUlimit passed to the API call is not
        /// supported by the active device.
        /// </summary>
        CUDA_ERROR_UNSUPPORTED_LIMIT = 215,

        /// <summary>
        /// This indicates that the ::CUcontext passed to the API call can
        /// only be bound to a single CPU thread at a time but is already
        /// bound to a CPU thread.
        /// </summary>
        CUDA_ERROR_CONTEXT_ALREADY_IN_USE = 216,

        /// <summary>
        /// This indicates that peer access is not supported across the given
        /// devices.
        /// </summary>
        CUDA_ERROR_PEER_ACCESS_UNSUPPORTED = 217,

        /// <summary>
        /// This indicates that a PTX JIT compilation failed.
        /// </summary>
        CUDA_ERROR_INVALID_PTX = 218,

        /// <summary>
        /// This indicates an error with OpenGL or DirectX context.
        /// </summary>
        CUDA_ERROR_INVALID_GRAPHICS_CONTEXT = 219,

        /// <summary>
       /// This indicates that an uncorrectable NVLink error was detected during the
       /// execution.
        /// </summary>
        CUDA_ERROR_NVLINK_UNCORRECTABLE = 220,

        /// <summary>
       /// This indicates that the PTX JIT compiler library was not found.
        /// </summary>
        CUDA_ERROR_JIT_COMPILER_NOT_FOUND = 221,

        /// <summary>
        /// This indicates that the device kernel source is invalid.
        /// </summary>
        CUDA_ERROR_INVALID_SOURCE = 300,

        /// <summary>
        /// This indicates that the file specified was not found.
        /// </summary>
        CUDA_ERROR_FILE_NOT_FOUND = 301,

        /// <summary>
        /// This indicates that a link to a shared object failed to resolve.
        /// </summary>
        CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND = 302,

        /// <summary>
        /// This indicates that initialization of a shared object failed.
        /// </summary>
        CUDA_ERROR_SHARED_OBJECT_INIT_FAILED = 303,

        /// <summary>
        /// This indicates that an OS call failed.
        /// </summary>
        CUDA_ERROR_OPERATING_SYSTEM = 304,

        /// <summary>
        /// This indicates that a resource handle passed to the API call was not
        /// valid. Resource handles are opaque types like ::CUstream and ::CUevent.
        /// </summary>
        CUDA_ERROR_INVALID_HANDLE = 400,

        /// <summary>
        /// This indicates that a resource required by the API call is not in a
        /// valid state to perform the requested operation.
        /// </summary>
        CUDA_ERROR_ILLEGAL_STATE = 401,

        /// <summary>
        /// This indicates that a named symbol was not found. Examples of symbols
        /// are global/constant variable names, texture names, and surface names.
        /// </summary>
        CUDA_ERROR_NOT_FOUND = 500,

        /// <summary>
        /// This indicates that asynchronous operations issued previously have not
        /// completed yet. This result is not actually an error, but must be indicated
        /// differently than ::CUDA_SUCCESS (which indicates completion). Calls that
        /// may return this value include ::cuEventQuery() and ::cuStreamQuery().
        /// </summary>
        CUDA_ERROR_NOT_READY = 600,

        /// <summary>
        /// While executing a kernel, the device encountered a
        /// load or store instruction on an invalid memory address.
        /// This leaves the process in an inconsistent state and any further CUDA work
        /// will return the same error. To continue using CUDA, the process must be terminated
        /// and relaunched.
        /// </summary>
        CUDA_ERROR_ILLEGAL_ADDRESS = 700,

        /// <summary>
        /// This indicates that a launch did not occur because it did not have
        /// appropriate resources. This error usually indicates that the user has
        /// attempted to pass too many arguments to the device kernel, or the
        /// kernel launch specifies too many threads for the kernel's register
        /// count. Passing arguments of the wrong size (i.e. a 64-bit pointer
        /// when a 32-bit int is expected) is equivalent to passing too many
        /// arguments and can also result in this error.
        /// </summary>
        CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES = 701,

        /// <summary>
        /// This indicates that the device kernel took too long to execute. This can
        /// only occur if timeouts are enabled - see the device attribute
        /// ::CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT for more information.
        /// This leaves the process in an inconsistent state and any further CUDA work
        /// will return the same error. To continue using CUDA, the process must be terminated
        /// and relaunched.
        /// </summary>
        CUDA_ERROR_LAUNCH_TIMEOUT = 702,

        /// <summary>
        /// This error indicates a kernel launch that uses an incompatible texturing
        /// mode.
        /// </summary>
        CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING = 703,

        /// <summary>
        /// This error indicates that a call to ::cuCtxEnablePeerAccess() is
        /// trying to re-enable peer access to a context which has already
        /// had peer access to it enabled.
        /// </summary>
        CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED = 704,

        /// <summary>
        /// This error indicates that ::cuCtxDisablePeerAccess() is
        /// trying to disable peer access which has not been enabled yet
        /// via ::cuCtxEnablePeerAccess().
        /// </summary>
        CUDA_ERROR_PEER_ACCESS_NOT_ENABLED = 705,

        /// <summary>
        /// This error indicates that the primary context for the specified device
        /// has already been initialized.
        /// </summary>
        CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE = 708,

        /// <summary>
        /// This error indicates that the context current to the calling thread
        /// has been destroyed using ::cuCtxDestroy, or is a primary context which
        /// has not yet been initialized.
        /// </summary>
        CUDA_ERROR_CONTEXT_IS_DESTROYED = 709,

        /// <summary>
        /// A device-side assert triggered during kernel execution. The context
        /// cannot be used anymore, and must be destroyed. All existing device
        /// memory allocations from this context are invalid and must be
        /// reconstructed if the program is to continue using CUDA.
        /// </summary>
        CUDA_ERROR_ASSERT = 710,

        /// <summary>
        /// This error indicates that the hardware resources required to enable
        /// peer access have been exhausted for one or more of the devices
        /// passed to ::cuCtxEnablePeerAccess().
        /// </summary>
        CUDA_ERROR_TOO_MANY_PEERS = 711,

        /// <summary>
        /// This error indicates that the memory range passed to ::cuMemHostRegister()
        /// has already been registered.
        /// </summary>
        CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED = 712,

        /// <summary>
        /// This error indicates that the pointer passed to ::cuMemHostUnregister()
        /// does not correspond to any currently registered memory region.
        /// </summary>
        CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED = 713,

        /// <summary>
        /// While executing a kernel, the device encountered a stack error.
        /// This can be due to stack corruption or exceeding the stack size limit.
        /// This leaves the process in an inconsistent state and any further CUDA work
        /// will return the same error. To continue using CUDA, the process must be terminated
        /// and relaunched.
        /// </summary>
        CUDA_ERROR_HARDWARE_STACK_ERROR = 714,

        /// <summary>
        /// While executing a kernel, the device encountered an illegal instruction.
        /// This leaves the process in an inconsistent state and any further CUDA work
        /// will return the same error. To continue using CUDA, the process must be terminated
        /// and relaunched.
        /// </summary>
        CUDA_ERROR_ILLEGAL_INSTRUCTION = 715,

        /// <summary>
        /// While executing a kernel, the device encountered a load or store instruction
        /// on a memory address which is not aligned.
        /// This leaves the process in an inconsistent state and any further CUDA work
        /// will return the same error. To continue using CUDA, the process must be terminated
        /// and relaunched.
        /// </summary>
        CUDA_ERROR_MISALIGNED_ADDRESS = 716,

        /// <summary>
        /// While executing a kernel, the device encountered an instruction
        /// which can only operate on memory locations in certain address spaces
        /// (global, shared, or local), but was supplied a memory address not
        /// belonging to an allowed address space.
        /// This leaves the process in an inconsistent state and any further CUDA work
        /// will return the same error. To continue using CUDA, the process must be terminated
        /// and relaunched.
        /// </summary>
        CUDA_ERROR_INVALID_ADDRESS_SPACE = 717,

        /// <summary>
        /// While executing a kernel, the device program counter wrapped its address space.
        /// This leaves the process in an inconsistent state and any further CUDA work
        /// will return the same error. To continue using CUDA, the process must be terminated
        /// and relaunched.
        /// </summary>
        CUDA_ERROR_INVALID_PC = 718,

        /// <summary>
        /// An exception occurred on the device while executing a kernel. Common
        /// causes include dereferencing an invalid device pointer and accessing
        /// out of bounds shared memory. Less common cases can be system specific - more
        /// information about these cases can be found in the system specific user guide.
        /// This leaves the process in an inconsistent state and any further CUDA work
        /// will return the same error. To continue using CUDA, the process must be terminated
        /// and relaunched.
        /// </summary>
        CUDA_ERROR_LAUNCH_FAILED = 719,

        /// <summary>
        /// This error indicates that the number of blocks launched per grid for a kernel that was
        /// launched via either ::cuLaunchCooperativeKernel or ::cuLaunchCooperativeKernelMultiDevice
        /// exceeds the maximum number of blocks as allowed by ::cuOccupancyMaxActiveBlocksPerMultiprocessor
        /// or ::cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags times the number of multiprocessors
        /// as specified by the device attribute ::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT.
        /// </summary>
        CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE = 720,

        /// <summary>
        /// This error indicates that the attempted operation is not permitted.
        /// </summary>
        CUDA_ERROR_NOT_PERMITTED = 800,

        /// <summary>
        /// This error indicates that the attempted operation is not supported
        /// on the current system or device.
        /// </summary>
        CUDA_ERROR_NOT_SUPPORTED = 801,

        /// <summary>
        /// This error indicates that the system is not yet ready to start any CUDA
        /// work.  To continue using CUDA, verify the system configuration is in a
        /// valid state and all required driver daemons are actively running.
        /// More information about this error can be found in the system specific
        /// user guide.
        /// </summary>
        CUDA_ERROR_SYSTEM_NOT_READY = 802,

        /// <summary>
        /// This error indicates that there is a mismatch between the versions of
        /// the display driver and the CUDA driver. Refer to the compatibility documentation
        /// for supported versions.
        /// </summary>
        CUDA_ERROR_SYSTEM_DRIVER_MISMATCH = 803,

        /// <summary>
        /// This error indicates that the system was upgraded to run with forward compatibility
        /// but the visible hardware detected by CUDA does not support this configuration.
        /// Refer to the compatibility documentation for the supported hardware matrix or ensure
        /// that only supported hardware is visible during initialization via the CUDA_VISIBLE_DEVICES
        /// environment variable.
        /// </summary>
        CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE = 804,

        /// <summary>
        /// This error indicates that the operation is not permitted when
        /// the stream is capturing.
        /// </summary>
        CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED = 900,

        /// <summary>
        /// This error indicates that the current capture sequence on the stream
        /// has been invalidated due to a previous error.
        /// </summary>
        CUDA_ERROR_STREAM_CAPTURE_INVALIDATED = 901,

        /// <summary>
        /// This error indicates that the operation would have resulted in a merge
        /// of two independent capture sequences.
        /// </summary>
        CUDA_ERROR_STREAM_CAPTURE_MERGE = 902,

        /// <summary>
        /// This error indicates that the capture was not initiated in this stream.
        /// </summary>
        CUDA_ERROR_STREAM_CAPTURE_UNMATCHED = 903,

        /// <summary>
        /// This error indicates that the capture sequence contains a fork that was
        /// not joined to the primary stream.
        /// </summary>
        CUDA_ERROR_STREAM_CAPTURE_UNJOINED = 904,

        /// <summary>
        /// This error indicates that a dependency would have been created which
        /// crosses the capture sequence boundary. Only implicit in-stream ordering
        /// dependencies are allowed to cross the boundary.
        /// </summary>
        CUDA_ERROR_STREAM_CAPTURE_ISOLATION = 905,

        /// <summary>
        /// This error indicates a disallowed implicit dependency on a current capture
        /// sequence from cudaStreamLegacy.
        /// </summary>
        CUDA_ERROR_STREAM_CAPTURE_IMPLICIT = 906,

        /// <summary>
        /// This error indicates that the operation is not permitted on an event which
        /// was last recorded in a capturing stream.
        /// </summary>
        CUDA_ERROR_CAPTURED_EVENT = 907,

        /// <summary>
        /// A stream capture sequence not initiated with the ::CU_STREAM_CAPTURE_MODE_RELAXED
        /// argument to ::cuStreamBeginCapture was passed to ::cuStreamEndCapture in a
        /// different thread.
        /// </summary>
        CUDA_ERROR_STREAM_CAPTURE_WRONG_THREAD = 908,

        /// <summary>
        /// This indicates that an unknown internal error has occurred.
        /// </summary>
        CUDA_ERROR_UNKNOWN = 999
    }
}

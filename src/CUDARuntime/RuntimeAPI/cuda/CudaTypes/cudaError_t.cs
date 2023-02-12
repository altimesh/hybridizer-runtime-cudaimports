using System;

namespace Hybridizer.Runtime.CUDAImports
{
    /// <summary>
    /// cuda error codes
    /// <see href="https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gf599e5b8b829ce7db0f5216928f6ecb6"/>
    /// <see href="https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1g3f51e3575c2178246db0a94a430e0038"/>
    /// </summary>
    [IntrinsicType("cudaError_t")]
    public enum cudaError_t : int
    {
        /// <summary>
        /// The API call returned with no errors. In the case of query calls, this can also mean 
        /// that the operation being queried is complete (see cudaEventQuery() and 
        /// cudaStreamQuery()). 
        /// </summary>
        cudaSuccess = 0,

        /// <summary>
        /// The device function being invoked (usually via cudaLaunch()) was not previously configured 
        /// via the cudaConfigureCall() function. 
        /// </summary>
        cudaErrorMissingConfiguration = 1,

        /// <summary>
        /// The API call failed because it was unable to allocate enough memory to 
        /// perform the requested operation. 
        /// </summary>
        cudaErrorMemoryAllocation = 2,

        /// <summary>
        /// The API call failed because the CUDA driver and runtime could not be initialized. 
        /// </summary>
        cudaErrorInitializationError = 3,

        /// <summary>
        /// An exception occurred on the device while executing a kernel. Common causes 
        /// include dereferencing an invalid device pointer and accessing out of bounds 
        /// shared memory. The device cannot be used until cudaThreadExit() is called. 
        /// All existing device memory allocations are invalid and must be reconstructed if 
        /// the program is to continue using CUDA. 
        /// </summary>
        cudaErrorLaunchFailure = 4,

        /// <summary>
        /// Deprecated
        /// 
        /// This error return is deprecated as of CUDA 3.1. Device emulation mode was removed
        /// with the CUDA 3.1 release. 
        /// 
        /// This indicated that a previous kernel launch failed. This was previously used for 
        /// device emulation of kernel launches. 
        /// 
        /// </summary>
        cudaErrorPriorLaunchFailure = 5,

        /// <summary>
        /// This indicates that the device kernel took too long to execute. This can only occur
        /// if timeouts are enabled - see the device property kernelExecTimeoutEnabled for more
        /// information. The device cannot be used until cudaThreadExit() is called. All existing
        /// device memory allocations are invalid and must be reconstructed if the program is to
        /// continue using CUDA. 
        /// </summary>
        cudaErrorLaunchTimeout = 6,

        /// <summary>
        /// This indicates that a launch did not occur because it did not have appropriate 
        /// resources. Although this error is similar to cudaErrorInvalidConfiguration, this 
        /// error usually indicates that the user has attempted to pass too many arguments to
        /// the device kernel, or the kernel launch specifies too many threads for the kernel's
        /// register count. 
        /// </summary>
        cudaErrorLaunchOutOfResources = 7,

        /// <summary>
        /// The requested device function does not exist or is not compiled for the proper
        /// device architecture. 
        /// </summary>
        cudaErrorInvalidDeviceFunction = 8,

        /// <summary>
        /// This indicates that a kernel launch is requesting resources that can never 
        /// be satisfied by the current device. Requesting more shared memory per block
        /// than the device supports will trigger this error, as will requesting too many 
        /// threads or blocks. See cudaDeviceProp for more device limitations. 
        /// </summary>
        cudaErrorInvalidConfiguration = 9,

        /// <summary>
        /// This indicates that the device ordinal supplied by the user does not 
        /// correspond to a valid CUDA device. 
        /// </summary>
        cudaErrorInvalidDevice = 10,

        /// <summary>
        /// This indicates that one or more of the parameters passed to the API call
        /// is not within an acceptable range of values. 
        /// </summary>
        cudaErrorInvalidValue = 11,

        /// <summary>
        /// This indicates that one or more of the pitch-related parameters passed 
        /// to the API call is not within the acceptable range for pitch. 
        /// </summary>
        cudaErrorInvalidPitchValue = 12,


        /// <summary>
        /// This indicates that the symbol name/identifier passed to the API call
        /// is not a valid name or identifier.
        /// </summary>
        cudaErrorInvalidSymbol = 13,

        /// <summary>
        /// This indicates that the buffer object could not be mapped.
        /// </summary>
        cudaErrorMapBufferObjectFailed = 14,

        /// <summary>
        /// This indicates that the buffer object could not be unmapped.
        /// </summary>
        cudaErrorUnmapBufferObjectFailed = 15,

        /// <summary>
        /// This indicates that at least one host pointer passed to the API call is
        /// not a valid host pointer.
        /// </summary>
        cudaErrorInvalidHostPointer = 16,

        /// <summary>
        /// This indicates that at least one device pointer passed to the API call is
        /// not a valid device pointer.
        /// </summary>
        cudaErrorInvalidDevicePointer = 17,

        /// <summary>
        /// This indicates that the texture passed to the API call is not a valid
        /// texture.
        /// </summary>
        cudaErrorInvalidTexture = 18,

        /// <summary>
        /// This indicates that the texture binding is not valid. This occurs if you
        /// call ::cudaGetTextureAlignmentOffset() with an unbound texture.
        /// </summary>
        cudaErrorInvalidTextureBinding = 19,

        /// <summary>
        /// This indicates that the channel descriptor passed to the API call is not
        /// valid. This occurs if the format is not one of the formats specified by
        /// ::cudaChannelFormatKind, or if one of the dimensions is invalid.
        /// </summary>
        cudaErrorInvalidChannelDescriptor = 20,

        /// <summary>
        /// This indicates that the direction of the memcpy passed to the API call is
        /// not one of the types specified by ::cudaMemcpyKind.
        /// </summary>
        cudaErrorInvalidMemcpyDirection = 21,

        /// <summary>
        /// This indicated that the user has taken the address of a constant variable,
        /// which was forbidden up until the CUDA 3.1 release.
        /// This error return is deprecated as of CUDA 3.1. Variables in constant
        /// memory may now have their address taken by the runtime via
        /// ::cudaGetSymbolAddress().
        /// </summary>
        [Obsolete]
        cudaErrorAddressOfConstant = 22,

        /// <summary>
        /// This indicated that a texture fetch was not able to be performed.
        /// This was previously used for device emulation of texture operations.
        /// This error return is deprecated as of CUDA 3.1. Device emulation mode was
        /// removed with the CUDA 3.1 release.
        /// </summary>
        [Obsolete]
        cudaErrorTextureFetchFailed = 23,

        /// <summary>
        /// This indicated that a texture was not bound for access.
        /// This was previously used for device emulation of texture operations.
        /// This error return is deprecated as of CUDA 3.1. Device emulation mode was
        /// removed with the CUDA 3.1 release.
        /// </summary>
        [Obsolete]
        cudaErrorTextureNotBound = 24,

        /// <summary>
        /// This indicated that a synchronization operation had failed.
        /// This was previously used for some device emulation functions.
        /// This error return is deprecated as of CUDA 3.1. Device emulation mode was
        /// removed with the CUDA 3.1 release.
        /// </summary>
        [Obsolete]
        cudaErrorSynchronizationError = 25,

        /// <summary>
        /// This indicates that a non-float texture was being accessed with linear
        /// filtering. This is not supported by CUDA.
        /// </summary>
        cudaErrorInvalidFilterSetting = 26,

        /// <summary>
        /// This indicates that an attempt was made to read a non-float texture as a
        /// normalized float. This is not supported by CUDA.
        /// </summary>
        cudaErrorInvalidNormSetting = 27,

        /// <summary>
        /// Mixing of device and device emulation code was not allowed.
        /// This error return is deprecated as of CUDA 3.1. Device emulation mode was
        /// removed with the CUDA 3.1 release.
        /// </summary>
        [Obsolete]
        cudaErrorMixedDeviceExecution = 28,

        /// <summary>
        /// This indicates that a CUDA Runtime API call cannot be executed because
        /// it is being called during process shut down, at a point in time after
        /// CUDA driver has been unloaded.
        /// </summary>
        cudaErrorCudartUnloading = 29,

        /// <summary>
        /// This indicates that an unknown internal error has occurred.
        /// </summary>
        cudaErrorUnknown = 30,

        /// <summary>
        /// This indicates that the API call is not yet implemented. Production
        /// releases of CUDA will never return this error.
        /// This error return is deprecated as of CUDA 4.1.
        /// </summary>
        [Obsolete]
        cudaErrorNotYetImplemented = 31,

        /// <summary>
        /// This indicated that an emulated device pointer exceeded the 32-bit address
        /// range.
        /// This error return is deprecated as of CUDA 3.1. Device emulation mode was
        /// removed with the CUDA 3.1 release.
        /// </summary>
        [Obsolete]
        cudaErrorMemoryValueTooLarge = 32,

        /// <summary>
        /// This indicates that a resource handle passed to the API call was not
        /// valid. Resource handles are opaque types like ::cudaStream_t and
        /// ::cudaEvent_t.
        /// </summary>
        cudaErrorInvalidResourceHandle = 33,

        /// <summary>
        /// This indicates that asynchronous operations issued previously have not
        /// completed yet. This result is not actually an error, but must be indicated
        /// differently than ::cudaSuccess (which indicates completion). Calls that
        /// may return this value include ::cudaEventQuery() and ::cudaStreamQuery().
        /// </summary>
        cudaErrorNotReady = 34,

        /// <summary>
        /// This indicates that the installed NVIDIA CUDA driver is older than the
        /// CUDA runtime library. This is not a supported configuration. Users should
        /// install an updated NVIDIA display driver to allow the application to run.
        /// </summary>
        cudaErrorInsufficientDriver = 35,

        /// <summary>
        /// This indicates that the user has called ::cudaSetValidDevices(),
        /// ::cudaSetDeviceFlags(), ::cudaD3D9SetDirect3DDevice(),
        /// ::cudaD3D10SetDirect3DDevice, ::cudaD3D11SetDirect3DDevice(), or
        /// ::cudaVDPAUSetVDPAUDevice() after initializing the CUDA runtime by
        /// calling non-device management operations (allocating memory and
        /// launching kernels are examples of non-device management operations).
        /// This error can also be returned if using runtime/driver
        /// interoperability and there is an existing ::CUcontext active on the
        /// host thread.
        /// </summary>
        cudaErrorSetOnActiveProcess = 36,

        /// <summary>
        /// This indicates that the surface passed to the API call is not a valid
        /// surface.
        /// </summary>
        cudaErrorInvalidSurface = 37,

        /// <summary>
        /// This indicates that no CUDA-capable devices were detected by the installed
        /// CUDA driver.
        /// </summary>
        cudaErrorNoDevice = 38,

        /// <summary>
        /// This indicates that an uncorrectable ECC error was detected during
        /// execution.
        /// </summary>
        cudaErrorECCUncorrectable = 39,

        /// <summary>
        /// This indicates that a link to a shared object failed to resolve.
        /// </summary>
        cudaErrorSharedObjectSymbolNotFound = 40,

        /// <summary>
        /// This indicates that initialization of a shared object failed.
        /// </summary>
        cudaErrorSharedObjectInitFailed = 41,

        /// <summary>
        /// This indicates that the ::cudaLimit passed to the API call is not
        /// supported by the active device.
        /// </summary>
        cudaErrorUnsupportedLimit = 42,

        /// <summary>
        /// This indicates that multiple global or constant variables (across separate
        /// CUDA source files in the application) share the same string name.
        /// </summary>
        cudaErrorDuplicateVariableName = 43,

        /// <summary>
        /// This indicates that multiple textures (across separate CUDA source
        /// files in the application) share the same string name.
        /// </summary>
        cudaErrorDuplicateTextureName = 44,

        /// <summary>
        /// This indicates that multiple surfaces (across separate CUDA source
        /// files in the application) share the same string name.
        /// </summary>
        cudaErrorDuplicateSurfaceName = 45,

        /// <summary>
        /// This indicates that all CUDA devices are busy or unavailable at the current
        /// time. Devices are often busy/unavailable due to use of
        /// ::cudaComputeModeExclusive, ::cudaComputeModeProhibited or when long
        /// running CUDA kernels have filled up the GPU and are blocking new work
        /// from starting. They can also be unavailable due to memory constraints
        /// on a device that already has active CUDA work being performed.
        /// </summary>
        cudaErrorDevicesUnavailable = 46,

        /// <summary>
        /// This indicates that the device kernel image is invalid.
        /// </summary>
        cudaErrorInvalidKernelImage = 47,

        /// <summary>
        /// This indicates that there is no kernel image available that is suitable
        /// for the device. This can occur when a user specifies code generation
        /// options for a particular CUDA source file that do not include the
        /// corresponding device configuration.
        /// </summary>
        cudaErrorNoKernelImageForDevice = 48,

        /// <summary>
        /// This indicates that the current context is not compatible with this
        /// the CUDA Runtime. This can only occur if you are using CUDA
        /// Runtime/Driver interoperability and have created an existing Driver
        /// context using the driver API. The Driver context may be incompatible
        /// either because the Driver context was created using an older version 
        /// of the API, because the Runtime API call expects a primary driver 
        /// context and the Driver context is not primary, or because the Driver 
        /// context has been destroyed. Please see \ref CUDART_DRIVER "Interactions 
        /// with the CUDA Driver API" for more information.
        /// </summary>
        cudaErrorIncompatibleDriverContext = 49,

        /// <summary>
        /// This error indicates that a call to ::cudaDeviceEnablePeerAccess() is
        /// trying to re-enable peer addressing on from a context which has already
        /// had peer addressing enabled.
        /// </summary>
        cudaErrorPeerAccessAlreadyEnabled = 50,

        /// <summary>
        /// This error indicates that ::cudaDeviceDisablePeerAccess() is trying to 
        /// disable peer addressing which has not been enabled yet via 
        /// ::cudaDeviceEnablePeerAccess().
        /// </summary>
        cudaErrorPeerAccessNotEnabled = 51,

        /// <summary>
        /// This indicates that a call tried to access an exclusive-thread device that 
        /// is already in use by a different thread.
        /// </summary>
        cudaErrorDeviceAlreadyInUse = 54,

        /// <summary>
        /// This indicates profiler is not initialized for this run. This can
        /// happen when the application is running with external profiling tools
        /// like visual profiler.
        /// </summary>
        cudaErrorProfilerDisabled = 55,

        /// <summary>
        /// This error return is deprecated as of CUDA 5.0. It is no longer an error
        /// to attempt to enable/disable the profiling via ::cudaProfilerStart or
        /// ::cudaProfilerStop without initialization.
        /// </summary>
        [Obsolete]
        cudaErrorProfilerNotInitialized = 56,

        /// <summary>
        /// This error return is deprecated as of CUDA 5.0. It is no longer an error
        /// to call cudaProfilerStart() when profiling is already enabled.
        /// </summary>
        [Obsolete]
        cudaErrorProfilerAlreadyStarted = 57,

        /// <summary>
        /// This error return is deprecated as of CUDA 5.0. It is no longer an error
        /// to call cudaProfilerStop() when profiling is already disabled.
        /// </summary>
        [Obsolete]
        cudaErrorProfilerAlreadyStopped = 58,

        /// <summary>
        /// An assert triggered in device code during kernel execution. The device
        /// cannot be used again until ::cudaThreadExit() is called. All existing 
        /// allocations are invalid and must be reconstructed if the program is to
        /// continue using CUDA. 
        /// </summary>
        cudaErrorAssert = 59,

        /// <summary>
        /// This error indicates that the hardware resources required to enable
        /// peer access have been exhausted for one or more of the devices 
        /// passed to ::cudaEnablePeerAccess().
        /// </summary>
        cudaErrorTooManyPeers = 60,

        /// <summary>
        /// This error indicates that the memory range passed to ::cudaHostRegister()
        /// has already been registered.
        /// </summary>
        cudaErrorHostMemoryAlreadyRegistered = 61,

        /// <summary>
        /// This error indicates that the pointer passed to ::cudaHostUnregister()
        /// does not correspond to any currently registered memory region.
        /// </summary>
        cudaErrorHostMemoryNotRegistered = 62,

        /// <summary>
        /// This error indicates that an OS call failed.
        /// </summary>
        cudaErrorOperatingSystem = 63,

        /// <summary>
        /// This error indicates that P2P access is not supported across the given
        /// devices.
        /// </summary>
        cudaErrorPeerAccessUnsupported = 64,

        /// <summary>
        /// This error indicates that a device runtime grid launch did not occur 
        /// because the depth of the child grid would exceed the maximum supported
        /// number of nested grid launches. 
        /// </summary>
        cudaErrorLaunchMaxDepthExceeded = 65,

        /// <summary>
        /// This error indicates that a grid launch did not occur because the kernel 
        /// uses file-scoped textures which are unsupported by the device runtime. 
        /// Kernels launched via the device runtime only support textures created with 
        /// the Texture Object API's.
        /// </summary>
        cudaErrorLaunchFileScopedTex = 66,

        /// <summary>
        /// This error indicates that a grid launch did not occur because the kernel 
        /// uses file-scoped surfaces which are unsupported by the device runtime.
        /// Kernels launched via the device runtime only support surfaces created with
        /// the Surface Object API's.
        /// </summary>
        cudaErrorLaunchFileScopedSurf = 67,

        /// <summary>
        /// This error indicates that a call to ::cudaDeviceSynchronize made from
        /// the device runtime failed because the call was made at grid depth greater
        /// than than either the default (2 levels of grids) or user specified device 
        /// limit ::cudaLimitDevRuntimeSyncDepth. To be able to synchronize on 
        /// launched grids at a greater depth successfully, the maximum nested 
        /// depth at which ::cudaDeviceSynchronize will be called must be specified 
        /// with the ::cudaLimitDevRuntimeSyncDepth limit to the ::cudaDeviceSetLimit
        /// api before the host-side launch of a kernel using the device runtime. 
        /// Keep in mind that additional levels of sync depth require the runtime 
        /// to reserve large amounts of device memory that cannot be used for 
        /// user allocations.
        /// </summary>
        cudaErrorSyncDepthExceeded = 68,

        /// <summary>
        /// This error indicates that a device runtime grid launch failed because
        /// the launch would exceed the limit ::cudaLimitDevRuntimePendingLaunchCount.
        /// For this launch to proceed successfully, ::cudaDeviceSetLimit must be
        /// called to set the ::cudaLimitDevRuntimePendingLaunchCount to be higher 
        /// than the upper bound of outstanding launches that can be issued to the
        /// device runtime. Keep in mind that raising the limit of pending device
        /// runtime launches will require the runtime to reserve device memory that
        /// cannot be used for user allocations.
        /// </summary>
        cudaErrorLaunchPendingCountExceeded = 69,

        /// <summary>
        /// This error indicates the attempted operation is not permitted.
        /// </summary>
        cudaErrorNotPermitted = 70,

        /// <summary>
        /// This error indicates the attempted operation is not supported
        /// on the current system or device.
        /// </summary>
        cudaErrorNotSupported = 71,

        /// <summary>
        /// Device encountered an error in the call stack during kernel execution,
        /// possibly due to stack corruption or exceeding the stack size limit.
        /// This leaves the process in an inconsistent state and any further CUDA work
        /// will return the same error. To continue using CUDA, the process must be terminated
        /// and relaunched.
        /// </summary>
        cudaErrorHardwareStackError = 72,

        /// <summary>
        /// The device encountered an illegal instruction during kernel execution
        /// This leaves the process in an inconsistent state and any further CUDA work
        /// will return the same error. To continue using CUDA, the process must be terminated
        /// and relaunched.
        /// </summary>
        cudaErrorIllegalInstruction = 73,

        /// <summary>
        /// The device encountered a load or store instruction
        /// on a memory address which is not aligned.
        /// This leaves the process in an inconsistent state and any further CUDA work
        /// will return the same error. To continue using CUDA, the process must be terminated
        /// and relaunched.
        /// </summary>
        cudaErrorMisalignedAddress = 74,

        /// <summary>
        /// While executing a kernel, the device encountered an instruction
        /// which can only operate on memory locations in certain address spaces
        /// (global, shared, or local), but was supplied a memory address not
        /// belonging to an allowed address space.
        /// This leaves the process in an inconsistent state and any further CUDA work
        /// will return the same error. To continue using CUDA, the process must be terminated
        /// and relaunched.
        /// </summary>
        cudaErrorInvalidAddressSpace = 75,

        /// <summary>
        /// The device encountered an invalid program counter.
        /// This leaves the process in an inconsistent state and any further CUDA work
        /// will return the same error. To continue using CUDA, the process must be terminated
        /// and relaunched.
        /// </summary>
        cudaErrorInvalidPc = 76,

        /// <summary>
        /// The device encountered a load or store instruction on an invalid memory address.
        /// This leaves the process in an inconsistent state and any further CUDA work
        /// will return the same error. To continue using CUDA, the process must be terminated
        /// and relaunched.
        /// </summary>
        cudaErrorIllegalAddress = 77,

        /// <summary>
        /// A PTX compilation failed. The runtime may fall back to compiling PTX if
        /// an application does not contain a suitable binary for the current device.
        /// </summary>
        cudaErrorInvalidPtx = 78,

        /// <summary>
        /// This indicates an error with the OpenGL or DirectX context.
        /// </summary>
        cudaErrorInvalidGraphicsContext = 79,

        /// <summary>
        /// This indicates that an uncorrectable NVLink error was detected during the
        /// execution.
        /// </summary>
        cudaErrorNvlinkUncorrectable = 80,

        /// <summary>
        /// This indicates that the PTX JIT compiler library was not found. The JIT Compiler
        /// library is used for PTX compilation. The runtime may fall back to compiling PTX
        /// if an application does not contain a suitable binary for the current device.
        /// </summary>
        cudaErrorJitCompilerNotFound = 81,

        /// <summary>
        /// This error indicates that the number of blocks launched per grid for a kernel that was
        /// launched via either ::cudaLaunchCooperativeKernel or ::cudaLaunchCooperativeKernelMultiDevice
        /// exceeds the maximum number of blocks as allowed by ::cudaOccupancyMaxActiveBlocksPerMultiprocessor
        /// or ::cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags times the number of multiprocessors
        /// as specified by the device attribute ::cudaDevAttrMultiProcessorCount.
        /// </summary>
        cudaErrorCooperativeLaunchTooLarge = 82,

        /// <summary>
        /// This indicates an internal startup failure in the CUDA runtime.
        /// </summary>
        cudaErrorStartupFailure = 0x7f,

        /// <summary>
        /// The operation is not permitted when the stream is capturing.
        /// </summary>
        cudaErrorStreamCaptureUnsupported = 900,

        /// <summary>
        /// The current capture sequence on the stream has been invalidated due to
        /// a previous error.
        /// </summary>
        cudaErrorStreamCaptureInvalidated = 901,

        /// <summary>
        /// The operation would have resulted in a merge of two independent capture
        /// sequences.
        /// </summary>
        cudaErrorStreamCaptureMerge = 902,

        /// <summary>
        /// The capture was not initiated in this stream.
        /// </summary>
        cudaErrorStreamCaptureUnmatched = 903,

        /// <summary>
        /// The capture sequence contains a fork that was not joined to the primary
        /// stream.
        /// </summary>
        cudaErrorStreamCaptureUnjoined = 904,

        /// <summary>
        /// A dependency would have been created which crosses the capture sequence
        /// boundary. Only implicit in-stream ordering dependencies are allowed to
        /// cross the boundary.
        /// </summary>
        cudaErrorStreamCaptureIsolation = 905,

        /// <summary>
        /// The operation would have resulted in a disallowed implicit dependency on
        /// a current capture sequence from cudaStreamLegacy.
        /// </summary>
        cudaErrorStreamCaptureImplicit = 906,

        /// <summary>
        /// The operation is not permitted on an event which was last recorded in a
        /// capturing stream.
        /// </summary>
        cudaErrorCapturedEvent = 907,

        /// <summary>
        /// Any unhandled CUDA driver error is added to this value and returned via
        /// the runtime. Production releases of CUDA should not return such errors.
        /// This error return is deprecated as of CUDA 4.1.
        /// </summary>
        [Obsolete]
        cudaErrorApiFailureBase = 10000
    }
}
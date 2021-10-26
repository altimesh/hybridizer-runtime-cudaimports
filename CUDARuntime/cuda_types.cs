/* (c) ALTIMESH 2018 -- all rights reserved */
using System;
using System.Runtime.InteropServices;

namespace Hybridizer.Runtime.CUDAImports
{
    /// <summary>
    /// dimension structure.
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    [IntrinsicType("dim3")]
    public struct dim3
    {
        /// <summary>
        /// components
        /// </summary>
        public int x, y, z;

        /// <summary>
        /// Assignment constructor
        /// </summary>
        /// <param name="x"></param>
        /// <param name="y"></param>
        /// <param name="z"></param>
        public dim3(int x, int y, int z)
        {
            this.x = x;
            this.y = y;
            this.z = z;
        }
    }

    /// <summary>
    /// $size\_t$ type has different bit-size storage depending on architecture.
    /// </summary>
    [IntrinsicType("size_t")]
    [Guid("0F4E0F1A-A925-4A6B-9378-0F2AEBB3073B")]
    public struct size_t
    {
        IntPtr _inner;

        /// <summary>
        /// constructor from 32 bits signed integer
        /// </summary>
        public size_t(int val) { _inner = new IntPtr(val); }
        /// <summary>
        /// constructor from 32 bits sunigned integer
        /// </summary>
        public size_t(uint val) { _inner = new IntPtr((long)val); }
        /// <summary>
        /// constructor from 64 bits signed integer
        /// </summary>
        public size_t(long val) { _inner = new IntPtr(val); }

        /// <summary>
        /// implicit conversion operator
        /// </summary>
        public static implicit operator size_t(int val) { return new size_t(val); }
        /// <summary>
        /// implicit conversion operator
        /// </summary>
        public static implicit operator size_t(uint val) { return new size_t(val); }
        /// <summary>
        /// implicit conversion operator
        /// </summary>
        public static implicit operator size_t(long val) { return new size_t(val); }

        /// <summary>
        /// explicit conversion operator
        /// </summary>
        public static explicit operator int(size_t val) { return unchecked((int) val._inner.ToInt64()); }
        /// <summary>
        /// explicit conversion operator
        /// </summary>
        public static explicit operator uint(size_t val) { return unchecked((uint)val._inner.ToInt64()); }
        /// <summary>
        /// explicit conversion operator
        /// </summary>
        public static explicit operator long(size_t val) { return val._inner.ToInt64(); }
        /// <summary>
        /// explicit conversion operator
        /// </summary>
        public static explicit operator ulong(size_t val) { return (ulong) val._inner.ToInt64(); }
        /// <summary>
        /// Print contents of size\_t as a 64 bits integer
        /// </summary>
        /// <returns></returns>
        public override string ToString()
        {
            return _inner.ToInt64().ToString();
        }
    }

    /// <summary>
    /// CUDA stream
    /// </summary>
    [IntrinsicType("cudaStream_t")]
    public struct cudaStream_t
    {
        public IntPtr _inner;

        /// <summary>
        /// constructor from native pointer
        /// </summary>
        public cudaStream_t(IntPtr ptr)
        {
            this._inner = ptr;
        }

        /// <summary>
        /// void stream
        /// </summary>
        public static cudaStream_t NO_STREAM = new cudaStream_t(IntPtr.Zero);

        /// <summary>
        /// string representation
        /// </summary>
        public override string ToString()
        {
            return string.Format("Stream {0}", _inner.ToInt64());
        }

        /// <summary>
        /// convert a cudastream to a custream
        /// </summary>
        /// <param name="stream"></param>
        public static explicit operator CUstream(cudaStream_t stream)
        {
            return new CUstream(stream._inner);
        }

        public static implicit operator IntPtr(cudaStream_t stream)
        {
            return stream._inner;

        }

        /// <summary>
        /// equals operator (with other stream)
        /// </summary>
        public bool Equals(cudaStream_t other)
        {
            return _inner.Equals(other._inner);
        }

        /// <summary>
        /// equals operator (with object)
        /// </summary>
        public override bool Equals(object obj)
        {
            if (ReferenceEquals(null, obj)) return false;
            return obj is cudaStream_t && Equals((cudaStream_t) obj);
        }

        /// <summary>
        /// </summary>
        public override int GetHashCode()
        {
            return _inner.GetHashCode();
        }
    }

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

    /// <summary>
    /// Defines the way in which copy is done
    /// </summary>
    [IntrinsicType("cudaMemcpyKind")]
    public enum cudaMemcpyKind : int
    {
        /// <summary>
        /// Host   -> Host
        /// </summary>
        cudaMemcpyHostToHost = 0,      
        /// <summary>
        /// Host   -> Device
        /// </summary>
        cudaMemcpyHostToDevice = 1,      
        /// <summary>
        /// Device -> Host
        /// </summary>
        cudaMemcpyDeviceToHost = 2,      
        /// <summary>
        /// Device -> Device
        /// </summary>
        cudaMemcpyDeviceToDevice = 3       
    }

    /// <summary>
    /// CUDA Array
    /// </summary>
    [IntrinsicType("cudaArray_t")]
    public struct cudaArray_t
    {
        #pragma warning disable 0169
        IntPtr arr;
        #pragma warning restore 0169
    }

    /// <summary>
    /// CUDA 3D cross-device memory copying parameters
    /// </summary>
    [IntrinsicType("cudaMemcpy3DPeerParms")]
    public struct cudaMemcpy3DPeerParms
    {
        #pragma warning disable 0169
        /// <summary>
        /// Source memory address
        /// </summary>
        cudaArray_t            srcArray;  
        /// <summary>
        /// Source position offset
        /// </summary>
        cudaPos         srcPos;    
        /// <summary>
        /// Pitched source memory address
        /// </summary>
        cudaPitchedPtr  srcPtr;    
        /// <summary>
        /// Source device
        /// </summary>
        int                    srcDevice; 
  
        /// <summary>
        /// Destination memory address
        /// </summary>
        cudaArray_t            dstArray;  
        /// <summary>
        /// Destination position offset
        /// </summary>
        cudaPos         dstPos;    
        /// <summary>
        /// Pitched destination memory address
        /// </summary>
        cudaPitchedPtr  dstPtr;    
        /// <summary>
        /// Destination device
        /// </summary>
        int                    dstDevice; 
  
        /// <summary>
        /// Requested memory copy size
        /// </summary>
        cudaExtent      extent;    
        #pragma warning restore 0169
    }

    /// <summary>
    /// host allocation flags
    /// </summary>
    [IntrinsicType("cudaHostAllocFlags")]
    [Flags]
    public enum cudaHostAllocFlags : uint
    {
        /// <summary>
        /// Default page-locked allocation flag
        /// </summary>
        cudaHostAllocDefault = 0,   
        /// <summary>
        /// Pinned memory accessible by all CUDA contexts
        /// </summary>
        cudaHostAllocPortable = 1,   
        /// <summary>
        /// Map allocation into device space
        /// </summary>
        cudaHostAllocMapped = 2,   
        /// <summary>
        /// Write-combined memory
        /// </summary>
        cudaHostAllocWriteCombined = 4    
    }

    /// <summary>
    /// get device pointer flags
    /// </summary>
    [IntrinsicType("cudaGetDevicePointerFlags")]
    public enum cudaGetDevicePointerFlags : uint
    {
        /// <summary> </summary>
        cudaReserved = 0,
    }

    /// <summary>
    /// CUDA Pitched memory pointer
    /// </summary>
    [IntrinsicType("cudaPitchedPtr")]
    [StructLayout(LayoutKind.Sequential)]
    public struct cudaPitchedPtr
    {
        /// <summary>
        /// Pointer to allocated memory 
        /// </summary>
        public IntPtr ptr;
        /// <summary>
        /// Pitch of allocated memory in bytes
        /// </summary>
        public size_t pitch;
        /// <summary>
        /// Logical width of allocation in elements
        /// </summary>
        public size_t xsize;
        /// <summary>
        /// Logical height of allocation in elements 
        /// </summary>
        public size_t ysize;
        /// <summary>
        /// constructor
        /// </summary>
        public cudaPitchedPtr(IntPtr d, size_t p, size_t xsz, size_t ysz)
        {
            ptr = d;
            pitch = p;
            xsize = xsz;
            ysize = ysz;
        }
    }

    /// <summary>
    /// CUDA extent
    /// </summary>
    [IntrinsicType("cudaExtent")]
    [StructLayout(LayoutKind.Sequential)]
    public struct cudaExtent
    {
        /// <summary>
        /// Width in elements when referring to array memory, in bytes when referring to linear memory 
        /// </summary>
        public size_t width;
        /// <summary>
        /// Height in elements
        /// </summary>
        public size_t height;
        /// <summary>
        /// Depth in elements 
        /// </summary>
        public size_t depth;
        /// <summary> </summary>
        public cudaExtent(size_t w, size_t h, size_t d)
        {
            width = w; height = h; depth = d;
        }
    }

    /// <summary>
    /// An opaque value that represents a CUDA Surface object
    /// </summary>
    [IntrinsicType("cudaSurfaceObject_t")]
    public struct cudaSurfaceObject_t
    {
        ulong _inner;
    }

    /// <summary>
    /// An opaque value that represents a CUDA texture object
    /// </summary>
    [IntrinsicType("cudaTextureObject_t")]
    public struct cudaTextureObject_t
    {
        ulong _inner;
    }

    /// <summary>
    /// CUDA texture address modes
    /// </summary>
    [IntrinsicType("cudaTextureAddressMode")]
    public enum cudaTextureAddressMode
    {
        /// <summary>
        /// Wrapping address mode
        /// </summary>
        cudaAddressModeWrap   = 0,    
        /// <summary>
        /// Clamp to edge address mode
        /// </summary>
        cudaAddressModeClamp  = 1,    
        /// <summary>
        /// Mirror address mode
        /// </summary>
        cudaAddressModeMirror = 2,    
        /// <summary>
        /// Border address mode
        /// </summary>
        cudaAddressModeBorder = 3     
    }

    /// <summary>
    /// CUDA texture filter modes
    /// </summary>
    [IntrinsicType("cudaTextureFilterMode")]
    public enum cudaTextureFilterMode
    {
        /// <summary>
        /// Point filter mode
        /// </summary>
        cudaFilterModePoint  = 0,     
        /// <summary>
        /// Linear filter mode
        /// </summary>
        cudaFilterModeLinear = 1      
    }

    /// <summary>
    /// CUDA texture read modes
    /// </summary>
    [IntrinsicType("cudaTextureReadMode")]
    public enum cudaTextureReadMode
    {
        /// <summary>
        /// Read texture as specified element type
        /// </summary>
        cudaReadModeElementType     = 0,  
        /// <summary>
        /// Read texture as normalized float
        /// </summary>
        cudaReadModeNormalizedFloat = 1   
    }

    /// <summary>
    /// CUDA texture descriptor
    /// </summary>
    [IntrinsicType("cudaTextureDesc")]
    [StructLayout(LayoutKind.Explicit, Size = 64)]
    public unsafe struct cudaTextureDesc
    {
        /// <summary>
        /// Texture address mode for up to 3 dimensions
        /// </summary>
        [FieldOffset(0)]
        public fixed int addressMode[3];
        /// <summary>
        /// Texture filter mode
        /// </summary>
        [FieldOffset(12)]
        public cudaTextureFilterMode filterMode;
        /// <summary>
        /// Texture read mode
        /// </summary>
        [FieldOffset(16)]
        public cudaTextureReadMode readMode;
        /// <summary>
        /// Perform sRGB->linear conversion during texture read
        /// </summary>
        [FieldOffset(20)]
        public int sRGB;
        /// <summary>
        /// Texture Border Color
        /// </summary>
        [FieldOffset(24)]
        public fixed float borderColor[4];
        /// <summary>
        /// Indicates whether texture reads are normalized or not
        /// </summary>
        [FieldOffset(40)]
        public int normalizedCoords;
        /// <summary>
        /// Limit to the anisotropy ratio
        /// </summary>
        [FieldOffset(44)]
        public uint maxAnisotropy;
        /// <summary>
        /// Mipmap filter mode
        /// </summary>
        [FieldOffset(48)]
        public cudaTextureFilterMode mipmapFilterMode;
        /// <summary>
        /// Offset applied to the supplied mipmap level
        /// </summary>
        [FieldOffset(52)]
        public float mipmapLevelBias;
        /// <summary>
        /// Lower end of the mipmap level range to clamp access to
        /// </summary>
        [FieldOffset(56)]
        public float minMipmapLevelClamp;
        /// <summary>
        /// Upper end of the mipmap level range to clamp access to
        /// </summary>
        [FieldOffset(60)]
        public float maxMipmapLevelClamp;
    }

    /// <summary>
    ///  CUDA texture resource view formats
    /// </summary>
    [IntrinsicType("cudaResourceViewFormat")]
    public enum cudaResourceViewFormat
    {
        /// <summary>
        /// No resource view format (use underlying resource format)
        /// </summary>
        cudaResViewFormatNone                      = 0x00, 
        /// <summary>
        /// 1 channel unsigned 8-bit integers
        /// </summary>
        cudaResViewFormatUnsignedChar1             = 0x01, 
        /// <summary>
        /// 2 channel unsigned 8-bit integers
        /// </summary>
        cudaResViewFormatUnsignedChar2             = 0x02, 
        /// <summary>
        /// 4 channel unsigned 8-bit integers
        /// </summary>
        cudaResViewFormatUnsignedChar4             = 0x03, 
        /// <summary>
        /// 1 channel signed 8-bit integers
        /// </summary>
        cudaResViewFormatSignedChar1               = 0x04, 
        /// <summary>
        /// 2 channel signed 8-bit integers
        /// </summary>
        cudaResViewFormatSignedChar2               = 0x05, 
        /// <summary>
        /// 4 channel signed 8-bit integers
        /// </summary>
        cudaResViewFormatSignedChar4               = 0x06, 
        /// <summary>
        /// 1 channel unsigned 16-bit integers
        /// </summary>
        cudaResViewFormatUnsignedShort1            = 0x07, 
        /// <summary>
        /// 2 channel unsigned 16-bit integers
        /// </summary>
        cudaResViewFormatUnsignedShort2            = 0x08, 
        /// <summary>
        /// 4 channel unsigned 16-bit integers
        /// </summary>
        cudaResViewFormatUnsignedShort4            = 0x09, 
        /// <summary>
        /// 1 channel signed 16-bit integers
        /// </summary>
        cudaResViewFormatSignedShort1              = 0x0a, 
        /// <summary>
        /// 2 channel signed 16-bit integers
        /// </summary>
        cudaResViewFormatSignedShort2              = 0x0b, 
        /// <summary>
        /// 4 channel signed 16-bit integers
        /// </summary>
        cudaResViewFormatSignedShort4              = 0x0c, 
        /// <summary>
        /// 1 channel unsigned 32-bit integers
        /// </summary>
        cudaResViewFormatUnsignedInt1              = 0x0d, 
        /// <summary>
        /// 2 channel unsigned 32-bit integers
        /// </summary>
        cudaResViewFormatUnsignedInt2              = 0x0e, 
        /// <summary>
        /// 4 channel unsigned 32-bit integers
        /// </summary>
        cudaResViewFormatUnsignedInt4              = 0x0f, 
        /// <summary>
        /// 1 channel signed 32-bit integers
        /// </summary>
        cudaResViewFormatSignedInt1                = 0x10, 
        /// <summary>
        /// 2 channel signed 32-bit integers
        /// </summary>
        cudaResViewFormatSignedInt2                = 0x11, 
        /// <summary>
        /// 4 channel signed 32-bit integers
        /// </summary>
        cudaResViewFormatSignedInt4                = 0x12, 
        /// <summary>
        /// 1 channel 16-bit floating point
        /// </summary>
        cudaResViewFormatHalf1                     = 0x13, 
        /// <summary>
        /// 2 channel 16-bit floating point
        /// </summary>
        cudaResViewFormatHalf2                     = 0x14, 
        /// <summary>
        /// 4 channel 16-bit floating point
        /// </summary>
        cudaResViewFormatHalf4                     = 0x15, 
        /// <summary>
        /// 1 channel 32-bit floating point
        /// </summary>
        cudaResViewFormatFloat1                    = 0x16, 
        /// <summary>
        /// 2 channel 32-bit floating point
        /// </summary>
        cudaResViewFormatFloat2                    = 0x17, 
        /// <summary>
        /// 4 channel 32-bit floating point
        /// </summary>
        cudaResViewFormatFloat4                    = 0x18, 
        /// <summary>
        /// Block compressed 1
        /// </summary>
        cudaResViewFormatUnsignedBlockCompressed1  = 0x19, 
        /// <summary>
        /// Block compressed 2
        /// </summary>
        cudaResViewFormatUnsignedBlockCompressed2  = 0x1a, 
        /// <summary>
        /// Block compressed 3
        /// </summary>
        cudaResViewFormatUnsignedBlockCompressed3  = 0x1b, 
        /// <summary>
        /// Block compressed 4 unsigned
        /// </summary>
        cudaResViewFormatUnsignedBlockCompressed4  = 0x1c, 
        /// <summary>
        /// Block compressed 4 signed
        /// </summary>
        cudaResViewFormatSignedBlockCompressed4    = 0x1d, 
        /// <summary>
        /// Block compressed 5 unsigned
        /// </summary>
        cudaResViewFormatUnsignedBlockCompressed5  = 0x1e, 
        /// <summary>
        /// Block compressed 5 signed
        /// </summary>
        cudaResViewFormatSignedBlockCompressed5    = 0x1f, 
        /// <summary>
        /// Block compressed 6 unsigned half-float
        /// </summary>
        cudaResViewFormatUnsignedBlockCompressed6H = 0x20, 
        /// <summary>
        /// Block compressed 6 signed half-float
        /// </summary>
        cudaResViewFormatSignedBlockCompressed6H   = 0x21, 
        /// <summary>
        /// Block compressed 7
        /// </summary>
        cudaResViewFormatUnsignedBlockCompressed7  = 0x22  
    }

    /// <summary>
    /// CUDA resource view descriptor
    /// </summary>
    [IntrinsicType("cudaResourceViewDesc")]
    #if PLATFORM_X86
    [StructLayout(LayoutKind.Explicit, Size = 32)] 
    #elif PLATFORM_X64
    [StructLayout(LayoutKind.Explicit, Size = 48)] 
    #else
    #error Unsupported Platform
    #endif
    public struct cudaResourceViewDesc
    {
#if PLATFORM_X86
        [FieldOffset(0)]
        cudaResourceViewFormat format;
        [FieldOffset(4)]
        size_t width;
        [FieldOffset(8)]
        size_t height;
        [FieldOffset(12)]      
        size_t depth;
        [FieldOffset(16)]
        uint firstMipmapLevel;
        [FieldOffset(20)]
        uint lastMipmapLevel;
        [FieldOffset(24)]
        uint firstLayer;
        [FieldOffset(28)]
        uint lastLayer;
#elif PLATFORM_X64
        /// <summary>
        /// Resource view format
        /// </summary>
        [FieldOffset(0)]
        cudaResourceViewFormat format;
        /// <summary>
        /// Width of the resource view
        /// </summary>
        [FieldOffset(8)]
        size_t width;
        /// <summary>
        /// Height of the resource view
        /// </summary>
        [FieldOffset(16)]
        size_t height;
        /// <summary>
        /// Depth of the resource view
        /// </summary>
        [FieldOffset(24)]      
        size_t depth;
        /// <summary>
        /// First defined mipmap level
        /// </summary>
        [FieldOffset(32)]
        uint firstMipmapLevel;
        /// <summary>
        /// Last defined mipmap level
        /// </summary>
        [FieldOffset(36)]
        uint lastMipmapLevel;
        /// <summary>
        /// First layer index
        /// </summary>
        [FieldOffset(40)]
        uint firstLayer;
        /// <summary>
        /// Last layer index
        /// </summary>
        [FieldOffset(44)]
        uint lastlayer;
        #else
        #error Unsupported Platform
        #endif
    }

    /// <summary>
    /// Channel format kind
    /// </summary>
    [IntrinsicType("cudaChannelFormatKind")]
    public enum cudaChannelFormatKind : int
    {
        /// <summary>
        /// Signed channel format
        /// </summary>
        cudaChannelFormatKindSigned = 0,      
        /// <summary>
        /// Unsigned channel format
        /// </summary>
        cudaChannelFormatKindUnsigned = 1,      
        /// <summary>
        /// Float channel format
        /// </summary>
        cudaChannelFormatKindFloat = 2,      
        /// <summary>
        /// No channel format
        /// </summary>
        cudaChannelFormatKindNone = 3       
    }
    /// <summary>
    /// CUDA Channel format descriptor
    /// </summary>
    [IntrinsicType("cudaChannelFormatDesc")]
    #if PLATFORM_X86
    [StructLayout(LayoutKind.Explicit, Size = 20)] 
    #elif PLATFORM_X64
    [StructLayout(LayoutKind.Explicit, Size = 20)] 
    #else
    #error Unsupported Platform
    #endif
    public struct cudaChannelFormatDesc
    {
        /// <summary>
        /// x
        /// </summary>
        [FieldOffset(0)]
        public int x;
        /// <summary>
        /// y
        /// </summary>
        [FieldOffset(4)]
        public int y;
        /// <summary>
        /// z
        /// </summary>
        [FieldOffset(8)]
        public int z;
        /// <summary>
        /// w
        /// </summary>
        [FieldOffset(12)]
        public int w;
        /// <summary>
        /// Channel format kind
        /// </summary>
        [FieldOffset(16)]
        public cudaChannelFormatKind f; 
    }

    /// <summary>
    /// array allocation flags
    /// </summary>
    [IntrinsicType("cudaMallocArrayFlags")]
    public enum cudaMallocArrayFlags : int
    {
        /// <summary>
        /// Default CUDA array allocation flag
        /// </summary>
        cudaArrayDefault = 0x00,
        /// <summary>
        /// Must be set in cudaMalloc3DArray to create a layered CUDA array
        /// </summary>
        cudaArrayLayered = 0x01,
        /// <summary>
        /// Must be set in cudaMallocArray or cudaMalloc3DArray in order to bind surfaces to the CUDA array
        /// </summary>
        cudaArraySurfaceLoadStore = 0x02,
        /// <summary>
        /// Must be set in cudaMalloc3DArray to create a cubemap CUDA array
        /// </summary>
        cudaArrayCubemap = 0x04,
        /// <summary>
        /// Must be set in cudaMallocArray or cudaMalloc3DArray in order to perform texture gather operations on the CUDA array
        /// </summary>
        cudaArrayTextureGather = 0x08,
    }

    /// <summary>
    /// CUDA 3D position
    /// </summary>
    [IntrinsicType("cudaPos")]
    [StructLayout(LayoutKind.Sequential)]
    public struct cudaPos
    {
        /// <summary>
        /// x
        /// </summary>
        size_t x;     
        /// <summary>
        /// y
        /// </summary>
        size_t y;     
        /// <summary>
        /// z
        /// </summary>
        size_t z;     
    }

    /// <summary>
    /// CUDA 3D memory copying parameters
    /// </summary>
    [IntrinsicType("cudaMemcpy3DParms")]
    [StructLayout(LayoutKind.Sequential)]
    public struct cudaMemcpy3DParms
    {
        /// <summary>
        /// Source memory address
        /// </summary>
        cudaArray_t srcArray;  
        /// <summary>
        /// Source position offset
        /// </summary>
        cudaPos srcPos;    
        /// <summary>
        /// Pitched source memory address
        /// </summary>
        cudaPitchedPtr srcPtr;    

        /// <summary>
        /// Destination memory address
        /// </summary>
        cudaArray_t dstArray;  
        /// <summary>
        /// Destination position offset
        /// </summary>
        cudaPos dstPos;    
        /// <summary>
        /// Pitched destination memory address
        /// </summary>
        cudaPitchedPtr dstPtr;    

        /// <summary>
        /// Requested memory copy size
        /// </summary>
        cudaExtent extent;    
        /// <summary>
        /// Type of transfer
        /// </summary>
        cudaMemcpyKind kind;      
    }

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

    /// <summary>
    /// CUDA function cache configurations
    /// </summary>
    [IntrinsicType("cudaFuncCache")]
    public enum cudaFuncCache : int
    {
        /// <summary>
        /// Default function cache configuration, no preference
        /// </summary>
        cudaFuncCachePreferNone = 0,    
        /// <summary>
        /// Prefer larger shared memory and smaller L1 cache
        /// </summary>
        cudaFuncCachePreferShared = 1,    
        /// <summary>
        /// Prefer larger L1 cache and smaller shared memory
        /// </summary>
        cudaFuncCachePreferL1 = 2     
    }

    /// <summary>
    /// CUDA event types
    /// </summary>
    [IntrinsicType("cudaEvent_t")]
    public struct cudaEvent_t
    {
        #pragma warning disable 0169
        IntPtr evt;
        #pragma warning restore 0169
    }

    /// <summary>
    /// cuda event flags
    /// </summary>
    [IntrinsicType("cudaEventFlags")]
    [Flags]
    public enum cudaEventFlags : int
    {
        /// <summary>
        /// Default event flag
        /// </summary>
        cudaEventDefault = 0,
        /// <summary>
        /// Event uses blocking synchronization
        /// </summary>
        cudaEventBlockingSync = 1,
        /// <summary>
        /// Event will not record timing data
        /// </summary>
        cudaEventDisableTiming = 2,
        /// <summary>
        /// Event is suitable for interprocess use. cudaEventDisableTiming must be set
        /// </summary>
        cudaEventInterprocess = 4
    }

    /// <summary>
    ///  CUDA mipmapped array
    /// </summary>
    [IntrinsicType("cudaMipmappedArray_t")]
    public struct cudaMipmappedArray_t
    {
        IntPtr _inner;
    }

    /// <summary>
    /// CUDA mipmapped array (as source argument)
    /// </summary>
    [IntrinsicType("cudaMipmappedArray_const_t")]
    public struct cudaMipmappedArray_const_t
    {
#pragma warning disable 0169
        IntPtr _inner;
#pragma warning restore 0169
    }

    /// <summary>
    /// CUDA Memory Advise values 
    /// </summary>
    [IntrinsicType("")]
    public enum cudaMemmoryAdvise : int
    {
        /// <summary>
        /// Data will mostly be read and only occassionally be written to
        /// </summary>
        cudaMemAdviseSetReadMostly = 1,
        /// <summary>
        /// Undo the effect of ::cudaMemAdviseSetReadMostly
        /// </summary>
        cudaMemAdviseUnsetReadMostly = 2,
        /// <summary>
        /// Set the preferred location for the data as the specified device
        /// </summary>
        cudaMemAdviseSetPreferredLocation = 3,
        /// <summary>
        /// Clear the preferred location for the data
        /// </summary>
        cudaMemAdviseUnsetPreferredLocation = 4,
        /// <summary>
        /// Data will be accessed by the specified device, so prevent page faults as much as possible
        /// </summary>
        cudaMemAdviseSetAccessedBy = 5,
        /// <summary>
        /// Let the Unified Memory subsystem decide on the page faulting policy for the specified device
        /// </summary>
        cudaMemAdviseUnsetAccessedBy = 6,
    }

    /// <summary>
    /// cuda memory attach
    /// </summary>
    [IntrinsicType("")]
    public enum cudaMemAttach : int
    {
        /// <summary>
        /// Memory can be accessed by any stream on any device
        /// </summary>
        cudaMemAttachGlobal          =       0x01  , 
        /// <summary>
        /// Memory cannot be accessed by any stream on any device 
        /// </summary>
        cudaMemAttachHost            =       0x02  , 
        /// <summary>
        /// Memory can only be accessed by a single stream on the associated device 
        /// </summary>
        cudaMemAttachSingle          =       0x04  , 
    }

    /// <summary>
    /// CUDA resource types
    /// </summary>
    [IntrinsicType("")]
    public enum cudaResourceType : int
    {
        /// <summary>
        /// Array resource
        /// </summary>
        cudaResourceTypeArray = 0x00,
        /// <summary>
        /// Mipmapped array resource
        /// </summary>
        cudaResourceTypeMipmappedArray = 0x01,
        /// <summary>
        /// Linear resource
        /// </summary>
        cudaResourceTypeLinear = 0x02,
        /// <summary>
        /// Pitch 2D resource
        /// </summary>
        cudaResourceTypePitch2D = 0x03,
    }

    /// <summary>
    /// CUDA resource descriptor
    /// </summary>
    [IntrinsicType("cudaResourceDesc")]
    #if PLATFORM_X86
    [StructLayout(LayoutKind.Explicit, Size = 40)] 
    #elif PLATFORM_X64
    [StructLayout(LayoutKind.Explicit, Size = 64)]
    #else 
    #error Unsupported Platform
    #endif
    public struct cudaResourceDesc
    {
#if PLATFORM_X86
        [FieldOffset(0)]
        public cudaResourceType resType;
        [FieldOffset(4)]
        public cudaArray_t arrayStruct;
        [FieldOffset(4)]
        public cudaMipmappedArray_t mipmap;

        [StructLayout(LayoutKind.Explicit, Size = 28)]
        public struct cudaResourceDesc_linear
        {
            [FieldOffset(0)]
            public IntPtr devPtr;
            [FieldOffset(4)]
            public cudaChannelFormatDesc desc ;
            [FieldOffset(24)]
            public size_t sizeInBytes;
        }
        [FieldOffset(4)]
        public cudaResourceDesc_linear linear ;

        [StructLayout(LayoutKind.Explicit, Size = 36)]
        public struct cudaResourceDesc_pitch2D
        {
            [FieldOffset(0)]
            public IntPtr devPtr;
            [FieldOffset(4)]
            public cudaChannelFormatDesc desc ;
            [FieldOffset(24)]
            public size_t width;
            [FieldOffset(28)]
            public size_t height;
            [FieldOffset(32)]
            public size_t pitchInBytes;
        }
        [FieldOffset(4)]
        public cudaResourceDesc_pitch2D pitch2D;
#elif PLATFORM_X64
        /// <summary>
        /// Resource type
        /// </summary>
        [FieldOffset(0)]
        public cudaResourceType resType;
        /// <summary>
        /// CUDA array
        /// </summary>
        [FieldOffset(8)]
        public cudaArray_t arrayStruct;
        /// <summary>
        /// CUDA mipmapped array
        /// </summary>
        [FieldOffset(8)]
        public cudaMipmappedArray_t mipmap;

        /// <summary>
        /// linear
        /// </summary>
        [StructLayout(LayoutKind.Explicit, Size = 40)]
        public struct cudaResourceDesc_linear
        {
            /// <summary>
            /// device pointer
            /// </summary>
            [FieldOffset(0)]
            public IntPtr devPtr;
            /// <summary>
            /// Channel descriptor
            /// </summary>
            [FieldOffset(8)]
            public cudaChannelFormatDesc desc ;
            /// <summary>
            /// Size in bytes
            /// </summary>
            [FieldOffset(32)]
            public size_t sizeInBytes;
        }
        /// <summary>
        /// linear
        /// </summary>
        [FieldOffset(8)]
        public cudaResourceDesc_linear linear ;

        /// <summary>
        /// pitch2D
        /// </summary>
        [StructLayout(LayoutKind.Explicit, Size = 56)]
        public struct cudaResourceDesc_pitch2D
        {
            /// <summary>
            /// Device pointer
            /// </summary>
            [FieldOffset(0)]
            public IntPtr devPtr;
            /// <summary>
            /// Channel descriptor
            /// </summary>
            [FieldOffset(8)]
            public cudaChannelFormatDesc desc ;
            /// <summary>
            /// Width of the array in elements
            /// </summary>
            [FieldOffset(32)]
            public size_t width;
            /// <summary>
            /// Height of the array in elements
            /// </summary>
            [FieldOffset(40)]
            public size_t height;
            /// <summary>
            /// Pitch between two rows in bytes
            /// </summary>
            [FieldOffset(48)]
            public size_t pitchInBytes;
        }
        /// <summary>
        /// pitch2D
        /// </summary>
        [FieldOffset(8)]
        public cudaResourceDesc_pitch2D pitch2D;
    #else
    #error Unsupported Platform
    #endif
    }

    /// <summary>
    /// complex single-precision
    /// </summary>
    [IntrinsicInclude("<cublas.h>")]
    [IntrinsicType("cuComplex")]
    public struct cuComplex
    {
        /// <summary>
        /// real part
        /// </summary>
        /// 
        [IntrinsicRename("x")]
        public float re;
        /// <summary>
        /// imaginary part
        /// </summary>
        [IntrinsicRename("y")]
        public float im;
    }

    /// <summary>
    /// complex double-precision
    /// </summary>
    [IntrinsicInclude("cublas.h")]
    [IntrinsicType("cuDoubleComplex")]
    public struct cuDoubleComplex
    {
        /// <summary>
        /// real part
        /// </summary>
        public double re;
        /// <summary>
        /// imaginary part
        /// </summary>
        public double im;
    }

    /// <summary>
    /// 2 32 bits float, packed
    /// </summary>
    [IntrinsicType("float2")]
    [StructLayout(LayoutKind.Explicit)]
    public unsafe struct float2
    {
        /// <summary>
        /// x
        /// </summary>
        [FieldOffset(0)]
        public float x;
        /// <summary>
        /// y
        /// </summary>
        [FieldOffset(4)]
        public float y;

        /// <summary>
        /// conversion to signed 64 bits integer
        /// </summary>
        /// <param name="res"></param>
        public static explicit operator long(float2 res)
        {
            long* tmp = (long*)(&res);
            return *tmp;
        }

        /// <summary>
        /// conversion to int2
        /// </summary>
        public static explicit operator int2(float2 res)
        {
            int2* tmp = (int2*)(&res);
            return *tmp;
        }

        /// <summary>
        /// conversion to double
        /// </summary>
        public static explicit operator double(float2 res)
        {
            double* tmp = (double*)(&res);
            return *tmp;
        }

        /// <summary>
        /// conversion to short4
        /// </summary>
        public static explicit operator short4(float2 res)
        {
            short4* tmp = (short4*)(&res);
            return *tmp;
        }

        /// <summary>
        /// conversion to char8
        /// </summary>
        public static explicit operator char8(float2 res)
        {
            char8* tmp = (char8*)(&res);
            return *tmp;
        }

        /// <summary>
        /// constructor from 2 individual 32 bits floats
        /// </summary>
        private float2(float xx, float yy)
        {
            x = xx;
            y = yy;
        }

        [IntrinsicFunction("make_float2")]
        public static float2 make_float2(float xx, float yy)
        {
            return new float2(xx, yy);
        }

        /// <summary>
        /// constructor from int2
        /// </summary>
        public float2(int2 val)
        {
            x = (float)val.x;
            y = (float)val.y;
        }

        /// <summary>
        /// addition operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator+")]
        public static float2 operator +(float2 a, float2 b)
        {
            float2 res;
            res.x = a.x + b.x;
            res.y = a.y + b.y;
            return res;
        }

        /// <summary>
        /// addition operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator+")]
        public static float2 operator +(float a, float2 b)
        {
            float2 res;
            res.x = a + b.x;
            res.y = a + b.y;
            return res;
        }

        /// <summary>
        /// addition operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator+")]
        public static float2 operator +(float2 a, float b)
        {
            float2 res;
            res.x = a.x + b;
            res.y = a.y + b;
            return res;
        }

        /// <summary>
        /// substraction operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator-")]
        public static float2 operator -(float2 a, float2 b)
        {
            float2 res;
            res.x = a.x - b.x;
            res.y = a.y - b.y;
            return res;
        }

        /// <summary>
        /// substraction operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator-")]
        public static float2 operator -(float a, float2 b)
        {
            float2 res;
            res.x = a - b.x;
            res.y = a - b.y;
            return res;
        }

        /// <summary>
        /// substraction operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator-")]
        public static float2 operator -(float2 a, float b)
        {
            float2 res;
            res.x = a.x - b;
            res.y = a.y - b;
            return res;
        }

        /// <summary>
        /// multiplication operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator*")]
        public static float2 operator *(float2 a, float2 b)
        {
            float2 res;
            res.x = a.x * b.x;
            res.y = a.y * b.y;
            return res;
        }

        /// <summary>
        /// multiplication operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator*")]
        public static float2 operator *(float a, float2 b)
        {
            float2 res;
            res.x = a * b.x;
            res.y = a * b.y;
            return res;
        }

        /// <summary>
        /// multiplication operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator*")]
        public static float2 operator *(float2 a, float b)
        {
            float2 res;
            res.x = a.x * b;
            res.y = a.y * b;
            return res;
        }

        /// <summary>
        /// division operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator/")]
        public static float2 operator /(float2 a, float2 b)
        {
            float2 res;
            res.x = a.x / b.x;
            res.y = a.y / b.y;
            return res;
        }

        /// <summary>
        /// division operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator/")]
        public static float2 operator /(float a, float2 b)
        {
            float2 res;
            res.x = a / b.x;
            res.y = a / b.y;
            return res;
        }

        /// <summary>
        /// division operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator/")]
        public static float2 operator /(float2 a, float b)
        {
            float2 res;
            res.x = a.x / b;
            res.y = a.y / b;
            return res;
        }

        /// <summary>
        /// stores in memory
        /// </summary>
        /// <param name="ptr">destination pointer</param>
        /// <param name="val">value to store</param>
        /// <param name="alignment">has to be a compile time constant</param>
        [IntrinsicFunction(IsNaked=true, Name="__hybridizer_store_float2")]
        public unsafe static void Store(float2* ptr, float2 val, int alignment)
        {
            *ptr = val;
        }

        /// <summary>
        /// loads from memory
        /// </summary>
        /// <param name="ptr"></param>
        /// <param name="alignment">has to be a compile time constant</param>
        [IntrinsicFunction(IsNaked=true, Name="__hybridizer_load_float2")]
        public unsafe static float2 Load(float2* ptr, int alignment)
        {
            return *ptr;
        }
    }

    /// <summary>
    /// 3 32 bits floating points elements, packed
    /// </summary>
    [IntrinsicType("float3")]
    [StructLayout(LayoutKind.Explicit)]
    public unsafe struct float3
    {
        /// <summary>
        /// x
        /// </summary>
        [FieldOffset(0)]
        public float x;
        /// <summary>
        /// y
        /// </summary>
        [FieldOffset(4)]
        public float y;
        /// <summary>
        /// z
        /// </summary>
        [FieldOffset(8)]
        public float z;

        /// <summary>
        /// conversion to signed 64 bits integer
        /// </summary>
        public static explicit operator long(float3 res)
        {
            long* tmp = (long*)(&res);
            return *tmp;
        }

        /// <summary>
        /// constructor from 3 float
        /// </summary>
        /// <param name="xx"></param>
        /// <param name="yy"></param>
        /// <param name="zz"></param>
        private float3(float xx, float yy, float zz)
        {
            x = xx;
            y = yy;
            z = zz;
        }

        [IntrinsicFunction("make_float3")]
        public static float3 make_float3(float xx, float yy, float zz)
        {
            return new float3(xx, yy, zz);
        }

        /// <summary>
        /// addition operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator+")]
        public static float3 operator +(float3 a, float3 b)
        {
            float3 res;
            res.x = a.x + b.x;
            res.y = a.y + b.y;
            res.z = a.z + b.z;
            return res;
        }

        /// <summary>
        /// addition operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator+")]
        public static float3 operator +(float a, float3 b)
        {
            float3 res;
            res.x = a + b.x;
            res.y = a + b.y;
            res.z = a + b.z;
            return res;
        }

        /// <summary>
        /// addition operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator+")]
        public static float3 operator +(float3 a, float b)
        {
            float3 res;
            res.x = a.x + b;
            res.y = a.y + b;
            res.z = a.z + b;
            return res;
        }

        /// <summary>
        /// substraction operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator-")]
        public static float3 operator -(float3 a, float3 b)
        {
            float3 res;
            res.x = a.x - b.x;
            res.y = a.y - b.y;
            res.z = a.z - b.z;
            return res;
        }

        /// <summary>
        /// substraction operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator-")]
        public static float3 operator -(float a, float3 b)
        {
            float3 res;
            res.x = a - b.x;
            res.y = a - b.y;
            res.z = a - b.z;
            return res;
        }

        /// <summary>
        /// substraction operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator-")]
        public static float3 operator -(float3 a, float b)
        {
            float3 res;
            res.x = a.x - b;
            res.y = a.y - b;
            res.z = a.z - b;
            return res;
        }

        /// <summary>
        /// multiplication operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator*")]
        public static float3 operator *(float3 a, float3 b)
        {
            float3 res;
            res.x = a.x * b.x;
            res.y = a.y * b.y;
            res.z = a.z * b.y;
            return res;
        }

        /// <summary>
        /// multiplication operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator*")]
        public static float3 operator *(float a, float3 b)
        {
            float3 res;
            res.x = a * b.x;
            res.y = a * b.y;
            res.z = a * b.z;
            return res;
        }

        /// <summary>
        /// multiplication operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator*")]
        public static float3 operator *(float3 a, float b)
        {
            float3 res;
            res.x = a.x * b;
            res.y = a.y * b;
            res.z = a.z * b;
            return res;
        }

        /// <summary>
        /// division operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator/")]
        public static float3 operator /(float3 a, float3 b)
        {
            float3 res;
            res.x = a.x / b.x;
            res.y = a.y / b.y;
            res.z = a.z / b.y;
            return res;
        }

        /// <summary>
        /// division operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator/")]
        public static float3 operator /(float a, float3 b)
        {
            float3 res;
            res.x = a / b.x;
            res.y = a / b.y;
            res.z = a / b.z;
            return res;
        }

        /// <summary>
        /// division operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator/")]
        public static float3 operator /(float3 a, float b)
        {
            float3 res;
            res.x = a.x / b;
            res.y = a.y / b;
            res.z = a.z / b;
            return res;
        }

        /// <summary>
        /// stores in memory
        /// </summary>
        /// <param name="ptr">destination pointer</param>
        /// <param name="val">value to store</param>
        /// <param name="alignment">has to be a compile time constant</param>
        [IntrinsicFunction(IsNaked = true, Name = "__hybridizer_store_float3")]
        public unsafe static void Store(float3* ptr, float3 val, int alignment)
        {
            *ptr = val;
        }

        /// <summary>
        /// loads from memory
        /// </summary>
        /// <param name="ptr"></param>
        /// <param name="alignment">has to be a compile time constant</param>
        [IntrinsicFunction(IsNaked = true, Name = "__hybridizer_load_float3")]
        public unsafe static float3 Load(float3* ptr, int alignment)
        {
            return *ptr;
        }
    }

    /// <summary>
    /// 2 64 bits floating point elements, packed
    /// </summary>
    [IntrinsicType("double2")]
    [StructLayout(LayoutKind.Explicit)]
    public struct double2
    {
        /// <summary>
        /// x
        /// </summary>
        [FieldOffset(0)]
        public double x;
        /// <summary>
        /// y
        /// </summary>
        [FieldOffset(8)]
        public double y;

        /// <summary>
        /// constructor from 2 64 bits float
        /// </summary>
        /// <param name="a"></param>
        /// <param name="b"></param>
        private double2(double a, double b)
        {
            x = a;
            y = b;
        }

        [IntrinsicFunction("make_double2")]
        public static double2 make_double2(double xx, double yy)
        {
            return new double2(xx, yy);
        }

        /// <summary>
        /// copy constructor
        /// </summary>
        public double2(double2 a)
        {
            x = a.x;
            y = a.y;
        }

        /// <summary>
        /// addition operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator+")]
        public static double2 operator +(double2 a, double2 b)
        {
            double2 res;
            res.x = a.x + b.x;
            res.y = a.y + b.y;
            return res;
        }

        /// <summary>
        /// addition operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator+")]
        public static double2 operator +(double a, double2 b)
        {
            double2 res;
            res.x = a + b.x;
            res.y = a + b.y;
            return res;
        }

        /// <summary>
        /// addition operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator+")]
        public static double2 operator +(double2 a, double b)
        {
            double2 res;
            res.x = a.x + b;
            res.y = a.y + b;
            return res;
        }

        /// <summary>
        /// substraction operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator-")]
        public static double2 operator -(double2 a, double2 b)
        {
            double2 res;
            res.x = a.x - b.x;
            res.y = a.y - b.y;
            return res;
        }

        /// <summary>
        /// substraction operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator-")]
        public static double2 operator -(double a, double2 b)
        {
            double2 res;
            res.x = a - b.x;
            res.y = a - b.y;
            return res;
        }

        /// <summary>
        /// substraction operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator-")]
        public static double2 operator -(double2 a, double b)
        {
            double2 res;
            res.x = a.x - b;
            res.y = a.y - b;
            return res;
        }

        /// <summary>
        /// multiplication operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator*")]
        public static double2 operator *(double2 a, double2 b)
        {
            double2 res;
            res.x = a.x * b.x;
            res.y = a.y * b.y;
            return res;
        }

        /// <summary>
        /// multiplication operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator*")]
        public static double2 operator *(double a, double2 b)
        {
            double2 res;
            res.x = a * b.x;
            res.y = a * b.y;
            return res;
        }

        /// <summary>
        /// multiplication operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator*")]
        public static double2 operator *(double2 a, double b)
        {
            double2 res;
            res.x = a.x * b;
            res.y = a.y * b;
            return res;
        }

        /// <summary>
        /// division operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator/")]
        public static double2 operator /(double2 a, double2 b)
        {
            double2 res;
            res.x = a.x / b.x;
            res.y = a.y / b.y;
            return res;
        }

        /// <summary>
        /// division operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator/")]
        public static double2 operator /(double a, double2 b)
        {
            double2 res;
            res.x = a / b.x;
            res.y = a / b.y;
            return res;
        }

        /// <summary>
        /// division operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator/")]
        public static double2 operator /(double2 a, double b)
        {
            double2 res;
            res.x = a.x / b;
            res.y = a.y / b;
            return res;
        }

        /// <summary>
        /// greater or equal operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator>=")]
        public static bool2 operator >=(double2 l, double2 r)
        {
            bool2 res;
            res.x = l.x >= r.x;
            res.y = l.y >= r.y;
            return res;
        }

        /// <summary>
        /// strict greater operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator>")]
        public static bool2 operator >(double2 l, double2 r)
        {
            bool2 res;
            res.x = l.x > r.x;
            res.y = l.y > r.y;
            return res;
        }

        /// <summary>
        /// less or equal operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator<=")]
        public static bool2 operator <=(double2 l, double2 r)
        {
            bool2 res;
            res.x = l.x <= r.x;
            res.y = l.y <= r.y;
            return res;
        }

        /// <summary>
        /// strict less operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator>=")]
        public static bool2 operator <(double2 l, double2 r)
        {
            bool2 res;
            res.x = l.x < r.x;
            res.y = l.y < r.y;
            return res;
        }

        /// <summary>
        /// selects components from l or r, depending on mask value
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="hybridizer::select<double2>")]
        public static double2 Select(bool2 mask, double2 l, double2 r)
        {
            double2 res;
            res.x = mask.x ? l.x : r.x;
            res.y = mask.y ? l.y : r.y;
            return res;
        }

        /// <summary>
        /// stores in memory
        /// </summary>
        /// <param name="ptr">destination pointer</param>
        /// <param name="val">value to store</param>
        /// <param name="alignment">has to be a compile time constant</param>
        [IntrinsicFunction(IsNaked=true, Name="__hybridizer_store_double2")]
        public unsafe static void Store(double2* ptr, double2 val, int alignment)
        {
            *ptr = val;
        }

        /// <summary>
        /// loads from memory
        /// </summary>
        /// <param name="ptr"></param>
        /// <param name="alignment">has to be a compile time constant</param>
        [IntrinsicFunction(IsNaked=true, Name="__hybridizer_load_double2")]
        public unsafe static double2 Load(double2* ptr, int alignment)
        {
            return *ptr;
        }
    }

    /// <summary>
    /// 4 32 bits floats
    /// </summary>
    [IntrinsicType("float4")]
    [IntrinsicPrimitive("float4")]
    [StructLayout(LayoutKind.Explicit)]
    public struct float4
    {
        /// <summary>
        /// x
        /// </summary>
        [FieldOffset(0)]
        public float x;
        /// <summary>
        /// y
        /// </summary>
        [FieldOffset(4)]
        public float y;
        /// <summary>
        /// z
        /// </summary>
        [FieldOffset(8)]
        public float z;
        /// <summary>
        /// w
        /// </summary>
        [FieldOffset(12)]
        public float w;

        /// <summary>
        /// copy constructor
        /// </summary>
        public float4(float4 other)
        {
            x = other.x;
            y = other.y;
            z = other.z;
            w = other.w;
        }

        /// <summary>
        /// constructor from components
        /// </summary>
        private float4(float xx, float yy, float zz, float ww)
        {
            x = xx;
            y = yy;
            z = zz;
            w = ww;
        }

        [IntrinsicFunction("make_float4")]
        public static float4 make_float4(float xx, float yy, float zz, float ww)
        {
            return new float4(xx, yy, zz, ww);
        }

        /// <summary>
        /// addition operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator+")]
        public static float4 operator +(float4 a, float4 b)
        {
            float4 res;
            res.x = a.x + b.x;
            res.y = a.y + b.y;
            res.z = a.z + b.z;
            res.w = a.w + b.w;
            return res;
        }

        /// <summary>
        /// addition operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator+")]
        public static float4 operator +(float a, float4 b)
        {
            float4 res;
            res.x = a + b.x;
            res.y = a + b.y;
            res.z = a + b.z;
            res.w = a + b.w;
            return res;
        }

        /// <summary>
        /// addition operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator+")]
        public static float4 operator +(float4 a, float b)
        {
            float4 res;
            res.x = a.x + b;
            res.y = a.y + b;
            res.z = a.z + b;
            res.w = a.w + b;
            return res;
        }

        /// <summary>
        /// substraction operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator-")]
        public static float4 operator -(float4 a, float4 b)
        {
            float4 res;
            res.x = a.x - b.x;
            res.y = a.y - b.y;
            res.z = a.z - b.z;
            res.w = a.w - b.w;
            return res;
        }

        /// <summary>
        /// substraction operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator-")]
        public static float4 operator -(float a, float4 b)
        {
            float4 res;
            res.x = a - b.x;
            res.y = a - b.y;
            res.z = a - b.z;
            res.w = a - b.w;
            return res;
        }

        /// <summary>
        /// substraction operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator-")]
        public static float4 operator -(float4 a, float b)
        {
            float4 res;
            res.x = a.x - b;
            res.y = a.y - b;
            res.z = a.z - b;
            res.w = a.w - b;
            return res;
        }

        /// <summary>
        /// multiplication operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator*")]
        public static float4 operator *(float4 a, float4 b)
        {
            float4 res;
            res.x = a.x * b.x;
            res.y = a.y * b.y;
            res.z = a.z * b.z;
            res.w = a.w * b.w;
            return res;
        }

        /// <summary>
        /// multiplication operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator*")]
        public static float4 operator *(float a, float4 b)
        {
            float4 res;
            res.x = a * b.x;
            res.y = a * b.y;
            res.z = a * b.z;
            res.w = a * b.w;
            return res;
        }

        /// <summary>
        /// multiplication operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator*")]
        public static float4 operator *(float4 a, float b)
        {
            float4 res;
            res.x = a.x * b;
            res.y = a.y * b;
            res.z = a.z * b;
            res.w = a.w * b;
            return res;
        }

        /// <summary>
        /// division operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator/")]
        public static float4 operator /(float4 a, float4 b)
        {
            float4 res;
            res.x = a.x / b.x;
            res.y = a.y / b.y;
            res.z = a.z / b.z;
            res.w = a.w / b.w;
            return res;
        }

        /// <summary>
        /// division operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator/")]
        public static float4 operator /(float a, float4 b)
        {
            float4 res;
            res.x = a / b.x;
            res.y = a / b.y;
            res.z = a / b.z;
            res.w = a / b.w;
            return res;
        }

        /// <summary>
        /// division operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator/")]
        public static float4 operator /(float4 a, float b)
        {
            float4 res;
            res.x = a.x / b;
            res.y = a.y / b;
            res.z = a.z / b;
            res.w = a.w / b;
            return res;
        }

        /// <summary>
        /// Stores in memory
        /// </summary>
        /// <param name="ptr">destination pointer</param>
        /// <param name="val">value to store</param>
        /// <param name="alignment">has to be a compile time constant</param>
        [IntrinsicFunction(IsNaked=true, Name="__hybridizer_store_float4")]
        public unsafe static void Store(float4* ptr, float4 val, int alignment)
        {
            *ptr = val;
        }

        /// <summary>
        /// loads from memory
        /// </summary>
        /// <param name="ptr"></param>
        /// <param name="alignment">has to be a compile time constant</param>
        [IntrinsicFunction(IsNaked=true, Name="__hybridizer_load_float4")]
        public unsafe static float4 Load(float4* ptr, int alignment)
        {
            return *ptr;
        }
    }

    /// <summary>
    /// 8 32 bits floats
    /// </summary>
    [IntrinsicType("float8")]
    [StructLayout(LayoutKind.Explicit)]
    public struct float8
    {
        /// <summary>
        /// x
        /// </summary>
        [FieldOffset(0)]
        public float x;
        /// <summary>
        /// y
        /// </summary>
        [FieldOffset(4)]
        public float y;
        /// <summary>
        /// z
        /// </summary>
        [FieldOffset(8)]
        public float z;
        /// <summary>
        /// w
        /// </summary>
        [FieldOffset(12)]
        public float w;
        /// <summary>
        /// x2
        /// </summary>
        [FieldOffset(16)]
        public float x2;
        /// <summary>
        /// y2
        /// </summary>
        [FieldOffset(20)]
        public float y2;
        /// <summary>
        /// z2
        /// </summary>
        [FieldOffset(24)]
        public float z2;
        /// <summary>
        /// w2
        /// </summary>
        [FieldOffset(28)]
        public float w2;

        /// <summary>
        /// copy constructor
        /// </summary>
        public float8(float8 res)
        {
            x = res.x;
            y = res.y;
            z = res.z;
            w = res.w;
            x2 = res.x2;
            y2 = res.y2;
            z2 = res.z2;
            w2 = res.w2;
        }

        /// <summary>
        /// constructor from components
        /// </summary>
        public float8(float xx, float yy, float zz, float ww, float xx2, float yy2, float zz2, float ww2)
        {
            x = xx;
            y = yy;
            z = zz;
            w = ww;
            x2 = xx2;
            y2 = yy2;
            z2 = zz2;
            w2 = ww2;
        }

        /// <summary>
        /// selects components from l or r, depending on mask value
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="hybridizer::select<float8>")]
        public static float8 Select(bool8 mask, float8 l, float8 r)
        {
            float8 res;
            res.x = mask.x ? l.x : r.x;
            res.y = mask.y ? l.y : r.y;
            res.z = mask.z ? l.z : r.z;
            res.w = mask.w ? l.w : r.w;
            res.x2 = mask.x2 ? l.x2 : r.x2;
            res.y2 = mask.y2 ? l.y2 : r.y2;
            res.z2 = mask.z2 ? l.z2 : r.z2;
            res.w2 = mask.w2 ? l.w2 : r.w2;
            return res;
        }

        /// <summary>
        /// loads from memory
        /// </summary>
        /// <param name="ptr"></param>
        /// <param name="alignment">has to be a compile time constant</param>
        [IntrinsicFunction(IsNaked=true, Name="__hybridizer_load_float8")]
        public unsafe static float8 Load(float8* ptr, int alignment)
        {
            return *ptr;
        }

        /// <summary>
        /// stores in memory
        /// </summary>
        /// <param name="ptr">destination pointer</param>
        /// <param name="val">value to store</param>
        /// <param name="alignment">has to be a compile time constant</param>
        [IntrinsicFunction(IsNaked=true, Name="__hybridizer_store_float8")]
        public unsafe static void Store(float8* ptr, float8 val, int alignment)
        {
            *ptr = val;
        }

        /// <summary>
        /// addition operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator+")]
        public static float8 operator +(float8 a, float8 b)
        {
            float8 res;
            res.x = a.x + b.x;
            res.y = a.y + b.y;
            res.z = a.z + b.z;
            res.w = a.w + b.w;
            res.x2 = a.x2 + b.x2;
            res.y2 = a.y2 + b.y2;
            res.z2 = a.z2 + b.z2;
            res.w2 = a.w2 + b.w2;
            return res;
        }

        /// <summary>
        /// addition operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator+")]
        public static float8 operator +(float a, float8 b)
        {
            float8 res;
            res.x = a + b.x;
            res.y = a + b.y;
            res.z = a + b.z;
            res.w = a + b.w;
            res.x2 = a + b.x2;
            res.y2 = a + b.y2;
            res.z2 = a + b.z2;
            res.w2 = a + b.w2;
            return res;
        }

        /// <summary>
        /// strict less operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator<")]
        public static bool8 operator<(float8 l, float8 r)
        {
            bool8 res;
            res.x = l.x < r.x;
            res.y = l.y < r.y;
            res.z = l.z < r.z;
            res.w = l.w < r.w;
            res.x2 = l.x2 < r.x2;
            res.y2 = l.y2 < r.y2;
            res.z2 = l.z2 < r.z2;
            res.w2 = l.w2 < r.w2;
            return res;
        }

        /// <summary>
        /// less or equal operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator<=")]
        public static bool8 operator<=(float8 l, float8 r)
        {
            bool8 res;
            res.x = l.x <= r.x;
            res.y = l.y <= r.y;
            res.z = l.z <= r.z;
            res.w = l.w <= r.w;
            res.x2 = l.x2 <= r.x2;
            res.y2 = l.y2 <= r.y2;
            res.z2 = l.z2 <= r.z2;
            res.w2 = l.w2 <= r.w2;
            return res;
        }

        /// <summary>
        /// greater or equal operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator>=")]
        public static bool8 operator>=(float8 l, float8 r)
        {
            bool8 res;
            res.x = l.x >= r.x;
            res.y = l.y >= r.y;
            res.z = l.z >= r.z;
            res.w = l.w >= r.w;
            res.x2 = l.x2 >= r.x2;
            res.y2 = l.y2 >= r.y2;
            res.z2 = l.z2 >= r.z2;
            res.w2 = l.w2 >= r.w2;
            return res;
        }

        /// <summary>
        /// strict greater operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator>")]
        public static bool8 operator>(float8 l, float8 r)
        {
            bool8 res;
            res.x = l.x > r.x;
            res.y = l.y > r.y;
            res.z = l.z > r.z;
            res.w = l.w > r.w;
            res.x2 = l.x2 > r.x2;
            res.y2 = l.y2 > r.y2;
            res.z2 = l.z2 > r.z2;
            res.w2 = l.w2 > r.w2;
            return res;
        }

        /// <summary>
        /// strict greater operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator>")]
        public static bool8 operator >(float8 l, float r)
        {
            bool8 res;
            res.x = l.x > r;
            res.y = l.y > r;
            res.z = l.z > r;
            res.w = l.w > r;
            res.x2 = l.x2 > r;
            res.y2 = l.y2 > r;
            res.z2 = l.z2 > r;
            res.w2 = l.w2 > r;
            return res;
        }

        /// <summary>
        /// strict less operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator>")]
        public static bool8 operator<(float8 l, float r)
        {
            bool8 res;
            res.x = l.x < r;
            res.y = l.y < r;
            res.z = l.z < r;
            res.w = l.w < r;
            res.x2 = l.x2 < r;
            res.y2 = l.y2 < r;
            res.z2 = l.z2 < r;
            res.w2 = l.w2 < r;
            return res;
        }

        /// <summary>
        /// addition operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator+")]
        public static float8 operator +(float8 a, float b)
        {
            float8 res;
            res.x = a.x + b;
            res.y = a.y + b;
            res.z = a.z + b;
            res.w = a.w + b;
            res.x2 = a.x2 + b;
            res.y2 = a.y2 + b;
            res.z2 = a.z2 + b;
            res.w2 = a.w2 + b;
            return res;
        }

        /// <summary>
        /// substraction operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator-")]
        public static float8 operator -(float8 a, float8 b)
        {
            float8 res;
            res.x = a.x - b.x;
            res.y = a.y - b.y;
            res.z = a.z - b.z;
            res.w = a.w - b.w;
            res.x2 = a.x2 - b.x2;
            res.y2 = a.y2 - b.y2;
            res.z2 = a.z2 - b.z2;
            res.w2 = a.w2 - b.w2;
            return res;
        }

        /// <summary>
        /// substraction operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator-")]
        public static float8 operator -(float a, float8 b)
        {
            float8 res;
            res.x = a - b.x;
            res.y = a - b.y;
            res.z = a - b.z;
            res.w = a - b.w;
            res.x2 = a - b.x2;
            res.y2 = a - b.y2;
            res.z2 = a - b.z2;
            res.w2 = a - b.w2;
            return res;
        }

        /// <summary>
        /// substraction operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator-")]
        public static float8 operator -(float8 a, float b)
        {
            float8 res;
            res.x = a.x - b;
            res.y = a.y - b;
            res.z = a.z - b;
            res.w = a.w - b;
            res.x2 = a.x2 - b;
            res.y2 = a.y2 - b;
            res.z2 = a.z2 - b;
            res.w2 = a.w2 - b;
            return res;
        }

        /// <summary>
        /// multiplication operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator*")]
        public static float8 operator *(float8 a, float8 b)
        {
            float8 res;
            res.x = a.x * b.x;
            res.y = a.y * b.y;
            res.z = a.z * b.z;
            res.w = a.w * b.w;
            res.x2 = a.x2 * b.x2;
            res.y2 = a.y2 * b.y2;
            res.z2 = a.z2 * b.z2;
            res.w2 = a.w2 * b.w2;
            return res;
        }

        /// <summary>
        /// multiplication operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator*")]
        public static float8 operator *(float a, float8 b)
        {
            float8 res;
            res.x = a * b.x;
            res.y = a * b.y;
            res.z = a * b.z;
            res.w = a * b.w;
            res.x2 = a * b.x2;
            res.y2 = a * b.y2;
            res.z2 = a * b.z2;
            res.w2 = a * b.w2;
            return res;
        }

        /// <summary>
        /// multiplication operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator*")]
        public static float8 operator *(float8 a, float b)
        {
            float8 res;
            res.x = a.x * b;
            res.y = a.y * b;
            res.z = a.z * b;
            res.w = a.w * b;
            res.x2 = a.x2 * b;
            res.y2 = a.y2 * b;
            res.z2 = a.z2 * b;
            res.w2 = a.w2 * b;
            return res;
        }

        /// <summary>
        /// division operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator/")]
        public static float8 operator /(float8 a, float8 b)
        {
            float8 res;
            res.x = a.x / b.x;
            res.y = a.y / b.y;
            res.z = a.z / b.z;
            res.w = a.w / b.w;
            res.x2 = a.x2 / b.x2;
            res.y2 = a.y2 / b.y2;
            res.z2 = a.z2 / b.z2;
            res.w2 = a.w2 / b.w2;
            return res;
        }

        /// <summary>
        /// division operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator/")]
        public static float8 operator /(float a, float8 b)
        {
            float8 res;
            res.x = a / b.x;
            res.y = a / b.y;
            res.z = a / b.z;
            res.w = a / b.w;
            res.x2 = a / b.x2;
            res.y2 = a / b.y2;
            res.z2 = a / b.z2;
            res.w2 = a / b.w2;
            return res;
        }

        /// <summary>
        /// division operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator/")]
        public static float8 operator /(float8 a, float b)
        {
            float8 res;
            res.x = a.x / b;
            res.y = a.y / b;
            res.z = a.z / b;
            res.w = a.w / b;
            res.x2 = a.x2 / b;
            res.y2 = a.y2 / b;
            res.z2 = a.z2 / b;
            res.w2 = a.w2 / b;
            return res;
        }
    }

    /// <summary>
    /// 2 booleans
    /// </summary>
    [IntrinsicType("bool2")] // mask?
    public struct bool2
    {
        /// <summary>
        /// component
        /// </summary>
        public bool x, y;
    }

    /// <summary>
    /// 4 booleans
    /// </summary>
    [IntrinsicType("bool4")] // mask?
    public struct bool4
    {
        /// <summary>
        /// component
        /// </summary>
        public bool x, y, z, w;
    }

    /// <summary>
    /// 8 booleans
    /// </summary>
    [IntrinsicType("bool8")] // mask?
    public struct bool8
    {
        /// <summary>
        /// component
        /// </summary>
        public bool x, y, z, w, x2, y2, z2, w2; 
    }

    /// <summary>
    /// 4 64 bits floating points elements
    /// </summary>
    [IntrinsicType("double4")]
    [StructLayout(LayoutKind.Explicit)]
    public struct double4
    {
        /// <summary>
        /// x
        /// </summary>
        [FieldOffset(0)]
        public double x;
        /// <summary>
        /// y
        /// </summary>
        [FieldOffset(8)]
        public double y;
        /// <summary>
        /// z
        /// </summary>
        [FieldOffset(16)]
        public double z;
        /// <summary>
        /// w
        /// </summary>
        [FieldOffset(24)]
        public double w;

        private double4(double xx, double yy, double zz, double ww)
        {
            x = xx;
            y = yy;
            z = zz;
            w = ww;
        }

        [IntrinsicFunction("make_double4")]
        public static double4 make_double4(double xx, double yy, double zz, double ww)
        {
            return new double4(xx, yy, zz, ww);
        }

        /// <summary>
        /// addition operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator+")]
        public static double4 operator +(double4 a, double4 b)
        {
            double4 res;
            res.x = a.x + b.x;
            res.y = a.y + b.y;
            res.z = a.z + b.z;
            res.w = a.w + b.w;
            return res;
        }

        /// <summary>
        /// addition operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator+")]
        public static double4 operator +(double a, double4 b)
        {
            double4 res;
            res.x = a + b.x;
            res.y = a + b.y;
            res.z = a + b.z;
            res.w = a + b.w;
            return res;
        }

        /// <summary>
        /// addition operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator+")]
        public static double4 operator +(double4 a, double b)
        {
            double4 res;
            res.x = a.x + b;
            res.y = a.y + b;
            res.z = a.z + b;
            res.w = a.w + b;
            return res;
        }



        /// <summary>
        /// substraction operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator-")]
        public static double4 operator -(double4 a, double4 b)
        {
            double4 res;
            res.x = a.x - b.x;
            res.y = a.y - b.y;
            res.z = a.z - b.z;
            res.w = a.w - b.w;
            return res;
        }

        /// <summary>
        /// substraction operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator-")]
        public static double4 operator -(double a, double4 b)
        {
            double4 res;
            res.x = a - b.x;
            res.y = a - b.y;
            res.z = a - b.z;
            res.w = a - b.w;
            return res;
        }

        /// <summary>
        /// substraction operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator-")]
        public static double4 operator -(double4 a, double b)
        {
            double4 res;
            res.x = a.x - b;
            res.y = a.y - b;
            res.z = a.z - b;
            res.w = a.w - b;
            return res;
        }

        /// <summary>
        /// multiplication operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator*")]
        public static double4 operator *(double4 a, double4 b)
        {
            double4 res;
            res.x = a.x * b.x;
            res.y = a.y * b.y;
            res.z = a.z * b.z;
            res.w = a.w * b.w;
            return res;
        }

        /// <summary>
        /// multiplication operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator*")]
        public static double4 operator *(double a, double4 b)
        {
            double4 res;
            res.x = a * b.x;
            res.y = a * b.y;
            res.z = a * b.z;
            res.w = a * b.w;
            return res;
        }

        /// <summary>
        /// multiplication operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator*")]
        public static double4 operator *(double4 a, double b)
        {
            double4 res;
            res.x = a.x * b;
            res.y = a.y * b;
            res.z = a.z * b;
            res.w = a.w * b;
            return res;
        }

        /// <summary>
        /// division operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator/")]
        public static double4 operator /(double4 a, double4 b)
        {
            double4 res;
            res.x = a.x / b.x;
            res.y = a.y / b.y;
            res.z = a.z / b.z;
            res.w = a.w / b.w;
            return res;
        }

        /// <summary>
        /// division operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator/")]
        public static double4 operator /(double a, double4 b)
        {
            double4 res;
            res.x = a / b.x;
            res.y = a / b.y;
            res.z = a / b.z;
            res.w = a / b.w;
            return res;
        }

        /// <summary>
        /// division operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator/")]
        public static double4 operator /(double4 a, double b)
        {
            double4 res;
            res.x = a.x / b;
            res.y = a.y / b;
            res.z = a.z / b;
            res.w = a.w / b;
            return res;
        }
    }

    /// <summary>
    /// 2 32 bits integers
    /// </summary>
    [IntrinsicType("int2")]
    [StructLayout(LayoutKind.Explicit)]
    public unsafe struct int2
    {
        /// <summary>
        /// x
        /// </summary>
        [FieldOffset(0)]
        public int x;
        /// <summary>
        /// y
        /// </summary>
        [FieldOffset(4)]
        public int y;

        /// <summary>
        /// constructor from components
        /// </summary>
        private int2(int xx, int yy)
        {
            x = xx;
            y = yy;
        }

        [IntrinsicFunction("make_int2")]
        public static int2 make_int2(int xx, int yy)
        {
            return new int2(xx, yy);
        }
        
        /// <summary>
        /// constructor from float2
        /// </summary>
        public int2(float2 val)
        {
            x = (int)val.x;
            y = (int)val.y;
        }

        /// <summary>
        /// constructor from 64 bits integer
        /// </summary>
        /// <param name="val">lower part goes to x, high part to y</param>
        public int2(long val)
        {
            x = (int) (val & 0xFFFFFFFFL);
            y = (int) ((val >> 32) & 0xFFFFFFFFL);
        }

        /// <summary>
        /// conversion to 64 bits integer
        /// </summary>
        public static explicit operator long(int2 res) {
            long* ptr = (long*)&res;
            return *ptr;
        }

        /// <summary>
        /// conversion to float2
        /// </summary>
        public static explicit operator float2(int2 res)
        {
            float2* ptr = (float2*)&res;
            return *ptr;
        }

        /// <summary>
        /// conversion to 64 bits floating point
        /// </summary>
        public static explicit operator double(int2 res)
        {
            double* ptr = (double*)&res;
            return *ptr;
        }

        /// <summary>
        /// conversion to short4
        /// </summary>
        public static explicit operator short4(int2 res)
        {
            short4* ptr = (short4*)&res;
            return *ptr;
        }

        /// <summary>
        /// conversion to char8
        /// </summary>
        public static explicit operator char8(int2 res)
        {
            char8* ptr = (char8*)&res;
            return *ptr;
        }

        /// <summary>
        /// addition operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator+")]
        public static int2 operator +(int2 a, int2 b)
        {
            int2 res;
            res.x = a.x + b.x;
            res.y = a.y + b.y;
            return res;
        }

        /// <summary>
        /// addition operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator+")]
        public static int2 operator +(int a, int2 b)
        {
            int2 res;
            res.x = a + b.x;
            res.y = a + b.y;
            return res;
        }

        /// <summary>
        /// addition operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator+")]
        public static int2 operator +(int2 a, int b)
        {
            int2 res;
            res.x = a.x + b;
            res.y = a.y + b;
            return res;
        }

        /// <summary>
        /// substraction operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator-")]
        public static int2 operator -(int2 a, int2 b)
        {
            int2 res;
            res.x = a.x - b.x;
            res.y = a.y - b.y;
            return res;
        }

        /// <summary>
        /// substraction operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator-")]
        public static int2 operator -(int a, int2 b)
        {
            int2 res;
            res.x = a - b.x;
            res.y = a - b.y;
            return res;
        }

        /// <summary>
        /// substraction operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator-")]
        public static int2 operator -(int2 a, int b)
        {
            int2 res;
            res.x = a.x - b;
            res.y = a.y - b;
            return res;
        }

        /// <summary>
        /// multiplication operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator*")]
        public static int2 operator *(int2 a, int2 b)
        {
            int2 res;
            res.x = a.x * b.x;
            res.y = a.y * b.y;
            return res;
        }

        /// <summary>
        /// multiplication operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator*")]
        public static int2 operator *(int a, int2 b)
        {
            int2 res;
            res.x = a * b.x;
            res.y = a * b.y;
            return res;
        }

        /// <summary>
        /// multiplication operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator*")]
        public static int2 operator *(int2 a, int b)
        {
            int2 res;
            res.x = a.x * b;
            res.y = a.y * b;
            return res;
        }

        /// <summary>
        /// division operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator/")]
        public static int2 operator /(int2 a, int2 b)
        {
            int2 res;
            res.x = a.x / b.x;
            res.y = a.y / b.y;
            return res;
        }

        /// <summary>
        /// division operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator/")]
        public static int2 operator /(int a, int2 b)
        {
            int2 res;
            res.x = a / b.x;
            res.y = a / b.y;
            return res;
        }

        /// <summary>
        /// division operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator/")]
        public static int2 operator /(int2 a, int b)
        {
            int2 res;
            res.x = a.x / b;
            res.y = a.y / b;
            return res;
        }

        /// <summary>
        /// bitwise AND operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator&")]
        public static int2 operator &(int2 a, int2 b)
        {
            int2 res;
            res.x = a.x & b.x;
            res.y = a.y & b.y;
            return res;
        }

        /// <summary>
        /// bitwise AND operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator&")]
        public static int2 operator &(int a, int2 b)
        {
            int2 res;
            res.x = a & b.x;
            res.y = a & b.y;
            return res;
        }

        /// <summary>
        /// bitwise AND operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator&")]
        public static int2 operator &(int2 a, int b)
        {
            int2 res;
            res.x = a.x & b;
            res.y = a.y & b;
            return res;
        }

        /// <summary>
        /// bitwise OR operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator|")]
        public static int2 operator |(int2 a, int2 b)
        {
            int2 res;
            res.x = a.x | b.x;
            res.y = a.y | b.y;
            return res;
        }

        /// <summary>
        /// bitwise OR operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator|")]
        public static int2 operator |(int a, int2 b)
        {
            int2 res;
            res.x = a | b.x;
            res.y = a | b.y;
            return res;
        }

        /// <summary>
        /// bitwise OR operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator|")]
        public static int2 operator |(int2 a, int b)
        {
            int2 res;
            res.x = a.x | b;
            res.y = a.y | b;
            return res;
        }

        /// <summary>
        /// bitwise XOR operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator^")]
        public static int2 operator ^(int2 a, int2 b)
        {
            int2 res;
            res.x = a.x ^ b.x;
            res.y = a.y ^ b.y;
            return res;
        }

        /// <summary>
        /// bitwise XOR operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator^")]
        public static int2 operator ^(int a, int2 b)
        {
            int2 res;
            res.x = a ^ b.x;
            res.y = a ^ b.y;
            return res;
        }

        /// <summary>
        /// bitwise XOR operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator^")]
        public static int2 operator ^(int2 a, int b)
        {
            int2 res;
            res.x = a.x ^ b;
            res.y = a.y ^ b;
            return res;
        }
        
        /// <summary>
        /// Stores in memory
        /// </summary>
        /// <param name="ptr"></param>
        /// <param name="val"></param>
        /// <param name="alignment">has to be a compile time constant</param>
        [IntrinsicFunction(IsNaked=true, Name="__hybridizer_store_int2")]
        public unsafe static void Store(int2* ptr, int2 val, int alignment)
        {
            *ptr = val;
        }

        /// <summary>
        /// stores in memory
        /// </summary>
        /// <param name="ptr">destination pointer</param>
        /// <param name="val">value to store</param>
        /// <param name="alignment">has to be a compile time constant</param>
        [IntrinsicFunction(IsNaked=true, Name="__hybridizer_store_splat_int2")]
        public unsafe static void Store(int2* ptr, int val, int alignment)
        {
            *ptr = new int2(val);
        }

        /// <summary>
        /// loads from memory
        /// </summary>
        /// <param name="ptr"></param>
        /// <param name="alignment">has to be a compile time constant</param>
        [IntrinsicFunction(IsNaked=true, Name="__hybridizer_load_int2")]
        public unsafe static int2 Load(int2* ptr, int alignment)
        {
            return *ptr;
        }
    }

    /// <summary>
    /// 2 64 bits integers
    /// </summary>
    [IntrinsicType("long2")]
    [StructLayout(LayoutKind.Explicit)]
    public unsafe struct long2
    {
        /// <summary>
        /// x
        /// </summary>
        [FieldOffset(0)]
        public long x;
        /// <summary>
        /// y
        /// </summary>
        [FieldOffset(4)]
        public long y;

        /// <summary>
        /// constructor from components
        /// </summary>
        public long2(long xx, long yy)
        {
            x = xx;
            y = yy;
        }

        /// <summary>
        /// copy constructor
        /// </summary>
        public long2(long2 a)
        {
            x = a.x;
            y = a.y;
        }

        /// <summary>
        /// constructor from single component
        /// </summary>
        public long2(long val)
        {
            x = val;
            y = val;
        }

        /// <summary>
        /// addition operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator+")]
        public static long2 operator +(long2 a, long2 b)
        {
            long2 res;
            res.x = a.x + b.x;
            res.y = a.y + b.y;
            return res;
        }

        /// <summary>
        /// addition operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator+")]
        public static long2 operator +(int a, long2 b)
        {
            long2 res;
            res.x = a + b.x;
            res.y = a + b.y;
            return res;
        }

        /// <summary>
        /// addition operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator+")]
        public static long2 operator +(long2 a, int b)
        {
            long2 res;
            res.x = a.x + b;
            res.y = a.y + b;
            return res;
        }

        /// <summary>
        /// substraction operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator-")]
        public static long2 operator -(long2 a, long2 b)
        {
            long2 res;
            res.x = a.x - b.x;
            res.y = a.y - b.y;
            return res;
        }

        /// <summary>
        /// substraction operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator-")]
        public static long2 operator -(int a, long2 b)
        {
            long2 res;
            res.x = a - b.x;
            res.y = a - b.y;
            return res;
        }

        /// <summary>
        /// substraction operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator-")]
        public static long2 operator -(long2 a, int b)
        {
            long2 res;
            res.x = a.x - b;
            res.y = a.y - b;
            return res;
        }

        /// <summary>
        /// multiplication operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator*")]
        public static long2 operator *(long2 a, long2 b)
        {
            long2 res;
            res.x = a.x * b.x;
            res.y = a.y * b.y;
            return res;
        }

        /// <summary>
        /// multiplication operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator*")]
        public static long2 operator *(int a, long2 b)
        {
            long2 res;
            res.x = a * b.x;
            res.y = a * b.y;
            return res;
        }

        /// <summary>
        /// multiplication operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator*")]
        public static long2 operator *(long2 a, int b)
        {
            long2 res;
            res.x = a.x * b;
            res.y = a.y * b;
            return res;
        }

        /// <summary>
        /// division operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator/")]
        public static long2 operator /(long2 a, long2 b)
        {
            long2 res;
            res.x = a.x / b.x;
            res.y = a.y / b.y;
            return res;
        }

        /// <summary>
        /// division operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator/")]
        public static long2 operator /(int a, long2 b)
        {
            long2 res;
            res.x = a / b.x;
            res.y = a / b.y;
            return res;
        }

        /// <summary>
        /// division operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator/")]
        public static long2 operator /(long2 a, int b)
        {
            long2 res;
            res.x = a.x / b;
            res.y = a.y / b;
            return res;
        }

        /// <summary>
        /// bitwise AND operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator&")]
        public static long2 operator &(long2 a, long2 b)
        {
            long2 res;
            res.x = a.x & b.x;
            res.y = a.y & b.y;
            return res;
        }

        /// <summary>
        /// bitwise AND operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator&")]
        public static long2 operator &(int a, long2 b)
        {
            long2 res;
            res.x = a & b.x;
            res.y = a & b.y;
            return res;
        }

        /// <summary>
        /// bitwise AND operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator&")]
        public static long2 operator &(long2 a, int b)
        {
            long2 res;
            res.x = a.x & b;
            res.y = a.y & b;
            return res;
        }

        /// <summary>
        /// bitwise OR operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator|")]
        public static long2 operator |(long2 a, long2 b)
        {
            long2 res;
            res.x = a.x | b.x;
            res.y = a.y | b.y;
            return res;
        }

        /// <summary>
        /// bitwise OR operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator|")]
        public static long2 operator |(long a, long2 b)
        {
            long2 res;
            res.x = a | b.x;
            res.y = a | b.y;
            return res;
        }

        /// <summary>
        /// bitwise OR operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator|")]
        public static long2 operator |(long2 a, long b)
        {
            long2 res;
            res.x = a.x | b;
            res.y = a.y | b;
            return res;
        }

        /// <summary>
        /// bitwise XOR operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator^")]
        public static long2 operator ^(long2 a, long2 b)
        {
            long2 res;
            res.x = a.x ^ b.x;
            res.y = a.y ^ b.y;
            return res;
        }

        /// <summary>
        /// bitwise XOR operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator^")]
        public static long2 operator ^(int a, long2 b)
        {
            long2 res;
            res.x = a ^ b.x;
            res.y = a ^ b.y;
            return res;
        }

        /// <summary>
        /// bitwise XOR operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator^")]
        public static long2 operator ^(long2 a, int b)
        {
            long2 res;
            res.x = a.x ^ b;
            res.y = a.y ^ b;
            return res;
        }

        /// <summary>
        /// greater or equal operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator>=")]
        public static bool2 operator >=(long2 l, long2 r)
        {
            bool2 res;
            res.x = l.x >= r.x;
            res.y = l.y >= r.y;
            return res;
        }

        /// <summary>
        /// strict greater operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator>")]
        public static bool2 operator >(long2 l, long2 r)
        {
            bool2 res;
            res.x = l.x > r.x;
            res.y = l.y > r.y;
            return res;
        }

        /// <summary>
        /// less or equal operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator<=")]
        public static bool2 operator <=(long2 l, long2 r)
        {
            bool2 res;
            res.x = l.x <= r.x;
            res.y = l.y <= r.y;
            return res;
        }

        /// <summary>
        /// strict less operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator>=")]
        public static bool2 operator <(long2 l, long2 r)
        {
            bool2 res;
            res.x = l.x < r.x;
            res.y = l.y < r.y;
            return res;
        }

        /// <summary>
        /// left shift operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator<<")]
        public static long2 LeftShift(long2 a, long2 shift)
        {
            long2 res;
            res.x = a.x << (int)(shift.x);
            res.y = a.y << (int)(shift.y);
            return res;
        }

        /// <summary>
        /// right shift operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator>>")]
        public static long2 RightShift(long2 a, long2 shift)
        {
            long2 res;
            res.x = a.x >> (int)(shift.x);
            res.y = a.y >> (int)(shift.y);
            return res;
        }
        
        /// <summary>
        /// Stores in memory
        /// </summary>
        /// <param name="ptr"></param>
        /// <param name="val"></param>
        /// <param name="alignment">has to be a compile time constant</param>
        [IntrinsicFunction(IsNaked=true, Name="__hybridizer_store_int2")]
        public unsafe static void Store(long2* ptr, long2 val, int alignment)
        {
            *ptr = val;
        }

        /// <summary>
        /// stores in memory
        /// </summary>
        /// <param name="ptr">destination pointer</param>
        /// <param name="val">value to store</param>
        /// <param name="alignment">has to be a compile time constant</param>
        [IntrinsicFunction(IsNaked=true, Name="__hybridizer_store_splat_int2")]
        public unsafe static void Store(long2* ptr, int val, int alignment)
        {
            *ptr = new long2(val);
        }

        /// <summary>
        /// loads from memory
        /// </summary>
        /// <param name="ptr"></param>
        /// <param name="alignment">has to be a compile time constant</param>
        [IntrinsicFunction(IsNaked=true, Name="__hybridizer_load_int2")]
        public unsafe static long2 Load(long2* ptr, int alignment)
        {
            return *ptr;
        }

        /// <summary>
        /// selects components from l or r, depending on mask value
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="hybridizer::select<double2>")]
        public static long2 Select(bool2 mask, long2 l, long2 r)
        {
            long2 res;
            res.x = mask.x ? l.x : r.x;
            res.y = mask.y ? l.y : r.y;
            return res;
        }
    }

    /// <summary>
    /// 4 integers, packed
    /// </summary>
    [IntrinsicType("int4")]
    [StructLayout(LayoutKind.Explicit)]
    public struct int4
    {
        /// <summary>
        /// first component
        /// </summary>
        [FieldOffset(0)]
        public int x;
        /// <summary>
        /// second component
        /// </summary>
        [FieldOffset(4)]
        public int y;
        /// <summary>
        /// third component
        /// </summary>
        [FieldOffset(8)]
        public int z;
        /// <summary>
        /// fourth component
        /// </summary>
        [FieldOffset(12)]
        public int w;

        /// <summary>
        /// constructor from 4 distinc integers
        /// </summary>
        private int4(int xx, int yy, int zz, int ww)
        {
            x = xx;
            y = yy;
            z = zz;
            w = ww;
        }

        [IntrinsicFunction("make_int4")]
        public static int4 make_int4(int xx, int yy, int zz, int ww)
        {
            return new int4(xx, yy, zz, ww);
        }

        /// <summary>
        /// addition operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator+")]
        public static int4 operator +(int4 a, int4 b)
        {
            int4 res;
            res.x = a.x + b.x;
            res.y = a.y + b.y;
            res.z = a.z + b.z;
            res.w = a.w + b.w;
            return res;
        }

        /// <summary>
        /// addition operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator+")]
        public static int4 operator +(int a, int4 b)
        {
            int4 res;
            res.x = a + b.x;
            res.y = a + b.y;
            res.z = a + b.z;
            res.w = a + b.w;
            return res;
        }

        /// <summary>
        /// addition operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator+")]
        public static int4 operator +(int4 a, int b)
        {
            int4 res;
            res.x = a.x + b;
            res.y = a.y + b;
            res.z = a.z + b;
            res.w = a.w + b;
            return res;
        }

        /// <summary>
        /// substraction operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator-")]
        public static int4 operator -(int4 a, int4 b)
        {
            int4 res;
            res.x = a.x - b.x;
            res.y = a.y - b.y;
            res.z = a.z - b.z;
            res.w = a.w - b.w;
            return res;
        }

        /// <summary>
        /// substraction operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator-")]
        public static int4 operator -(int a, int4 b)
        {
            int4 res;
            res.x = a - b.x;
            res.y = a - b.y;
            res.z = a - b.z;
            res.w = a - b.w;
            return res;
        }

        /// <summary>
        /// substraction operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator-")]
        public static int4 operator -(int4 a, int b)
        {
            int4 res;
            res.x = a.x - b;
            res.y = a.y - b;
            res.z = a.z - b;
            res.w = a.w - b;
            return res;
        }

        /// <summary>
        /// multiplication operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator*")]
        public static int4 operator *(int4 a, int4 b)
        {
            int4 res;
            res.x = a.x * b.x;
            res.y = a.y * b.y;
            res.z = a.z * b.z;
            res.w = a.w * b.w;
            return res;
        }

        /// <summary>
        /// multiplication operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator*")]
        public static int4 operator *(int a, int4 b)
        {
            int4 res;
            res.x = a * b.x;
            res.y = a * b.y;
            res.z = a * b.z;
            res.w = a * b.w;
            return res;
        }

        /// <summary>
        /// multiplication operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator*")]
        public static int4 operator *(int4 a, int b)
        {
            int4 res;
            res.x = a.x * b;
            res.y = a.y * b;
            res.z = a.z * b;
            res.w = a.w * b;
            return res;
        }

        /// <summary>
        /// division operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator/")]
        public static int4 operator /(int4 a, int4 b)
        {
            int4 res;
            res.x = a.x / b.x;
            res.y = a.y / b.y;
            res.z = a.z / b.z;
            res.w = a.w / b.w;
            return res;
        }

        /// <summary>
        /// division operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator/")]
        public static int4 operator /(int a, int4 b)
        {
            int4 res;
            res.x = a / b.x;
            res.y = a / b.y;
            res.z = a / b.z;
            res.w = a / b.w;
            return res;
        }

        /// <summary>
        /// division operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator/")]
        public static int4 operator /(int4 a, int b)
        {
            int4 res;
            res.x = a.x / b;
            res.y = a.y / b;
            res.z = a.z / b;
            res.w = a.w / b;
            return res;
        }

        /// <summary>
        /// bitwise AND operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator&")]
        public static int4 operator &(int4 a, int4 b)
        {
            int4 res;
            res.x = a.x & b.x;
            res.y = a.y & b.y;
            res.z = a.z & b.z;
            res.w = a.w & b.w;
            return res;
        }

        /// <summary>
        /// bitwise AND operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator&")]
        public static int4 operator &(int a, int4 b)
        {
            int4 res;
            res.x = a & b.x;
            res.y = a & b.y;
            res.z = a & b.z;
            res.w = a & b.w;
            return res;
        }

        /// <summary>
        /// bitwise AND operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator&")]
        public static int4 operator &(int4 a, int b)
        {
            int4 res;
            res.x = a.x & b;
            res.y = a.y & b;
            res.z = a.z & b;
            res.w = a.w & b;
            return res;
        }



        /// <summary>
        /// bitwise OR operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator|")]
        public static int4 operator |(int4 a, int4 b)
        {
            int4 res;
            res.x = a.x | b.x;
            res.y = a.y | b.y;
            res.z = a.z | b.z;
            res.w = a.w | b.w;
            return res;
        }

        /// <summary>
        /// bitwise OR operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator|")]
        public static int4 operator |(int a, int4 b)
        {
            int4 res;
            res.x = a | b.x;
            res.y = a | b.y;
            res.z = a | b.z;
            res.w = a | b.w;
            return res;
        }

        /// <summary>
        /// bitwise OR operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator|")]
        public static int4 operator |(int4 a, int b)
        {
            int4 res;
            res.x = a.x | b;
            res.y = a.y | b;
            res.z = a.z | b;
            res.w = a.w | b;
            return res;
        }

        /// <summary>
        /// bitwise XOR operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator^")]
        public static int4 operator ^(int4 a, int4 b)
        {
            int4 res;
            res.x = a.x ^ b.x;
            res.y = a.y ^ b.y;
            res.z = a.z ^ b.z;
            res.w = a.w ^ b.w;
            return res;
        }

        /// <summary>
        /// bitwise XOR operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator^")]
        public static int4 operator ^(int a, int4 b)
        {
            int4 res;
            res.x = a ^ b.x;
            res.y = a ^ b.y;
            res.z = a ^ b.z;
            res.w = a ^ b.w;
            return res;
        }

        /// <summary>
        /// bitwise XOR operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator^")]
        public static int4 operator ^(int4 a, int b)
        {
            int4 res;
            res.x = a.x ^ b;
            res.y = a.y ^ b;
            res.z = a.z ^ b;
            res.w = a.w ^ b;
            return res;
        }
        /// <summary>
        /// stores in memory
        /// </summary>
        /// <param name="ptr">destination pointer</param>
        /// <param name="val">value to store</param>
        /// <param name="alignment">has to be a compile time constant</param>
        [IntrinsicFunction(IsNaked=true, Name="__hybridizer_store_int4")]
        public unsafe static void Store(int4* ptr, int4 val, int alignment)
        {
            *ptr = val;
        } 
    }

    /// <summary>
    /// 8 32 bits integers
    /// </summary>
    [IntrinsicType("int8")]
    [StructLayout(LayoutKind.Explicit)]
    public struct int8
    {
        /// <summary>
        /// x
        /// </summary>
        [FieldOffset(0)]
        public int x;
        /// <summary>
        /// y
        /// </summary>
        [FieldOffset(4)]
        public int y;
        /// <summary>
        /// z
        /// </summary>
        [FieldOffset(8)]
        public int z;
        /// <summary>
        /// w
        /// </summary>
        [FieldOffset(12)]
        public int w;
        /// <summary>
        /// x2
        /// </summary>
        [FieldOffset(16)]
        public int x2;
        /// <summary>
        /// y2
        /// </summary>
        [FieldOffset(20)]
        public int y2;
        /// <summary>
        /// z2
        /// </summary>
        [FieldOffset(24)]
        public int z2;
        /// <summary>
        /// w2
        /// </summary>
        [FieldOffset(28)]
        public int w2;

        /// <summary>
        /// constructor from single component
        /// </summary>
        public int8(int xx)
        {
            x =  xx;
            y =  xx;
            z =  xx;
            w =  xx;
            x2 = xx;
            y2 = xx;
            z2 = xx;
            w2 = xx;
        }
        
        /// <summary>
        /// constructor from components
        /// </summary>
        public int8(int xx, int yy, int zz, int ww, int xx2, int yy2, int zz2, int ww2)
        {
            x = xx;
            y = yy;
            z = zz;
            w = ww;
            x2 = xx2;
            y2 = yy2;
            z2 = zz2;
            w2 = ww2;
        }

        /// <summary>
        /// copy constructor
        /// </summary>
        public int8(int8 res)
        {
            x = res.x;
            y = res.y;
            z = res.z;
            w = res.w;
            x2 = res.x2;
            y2 = res.y2;
            z2 = res.z2;
            w2 = res.w2;
        }

        /// <summary>
        /// addition operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator+")]
        public static int8 operator +(int8 a, int8 b)
        {
            int8 res;
            res.x = a.x + b.x;
            res.y = a.y + b.y;
            res.z = a.z + b.z;
            res.w = a.w + b.w;
            res.x2 = a.x2 + b.x2;
            res.y2 = a.y2 + b.y2;
            res.z2 = a.z2 + b.z2;
            res.w2 = a.w2 + b.w2;
            return res;
        }

        /// <summary>
        /// bitwise XOR operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator^")]
        public static int8 operator ^(int8 a, int8 b)
        {
            int8 res;
            res.x = a.x ^ b.x;
            res.y = a.y ^ b.y;
            res.z = a.z ^ b.z;
            res.w = a.w ^ b.w;
            res.x2 = a.x2 ^ b.x2;
            res.y2 = a.y2 ^ b.y2;
            res.z2 = a.z2 ^ b.z2;
            res.w2 = a.w2 ^ b.w2;
            return res;
        }

        /// <summary>
        /// bitwise XOR operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator^")]
        public static int8 operator ^(int a, int8 b)
        {
            int8 res;
            res.x = a ^ b.x;
            res.y = a ^ b.y;
            res.z = a ^ b.z;
            res.w = a ^ b.w;
            res.x2 = a ^ b.x2;
            res.y2 = a ^ b.y2;
            res.z2 = a ^ b.z2;
            res.w2 = a ^ b.w2;
            return res;
        }

        /// <summary>
        /// bitwise XOR operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator^")]
        public static int8 operator ^(int8 a, int b)
        {
            int8 res;
            res.x = a.x ^ b;
            res.y = a.y ^ b;
            res.z = a.z ^ b;
            res.w = a.w ^ b;
            res.x2 = a.x2 ^ b;
            res.y2 = a.y2 ^ b;
            res.z2 = a.z2 ^ b;
            res.w2 = a.w2 ^ b;
            return res;
        }

        /// <summary>
        /// addition operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator+")]
        public static int8 operator +(int a, int8 b)
        {
            int8 res;
            res.x = a + b.x;
            res.y = a + b.y;
            res.z = a + b.z;
            res.w = a + b.w;
            res.x2 = a + b.x2;
            res.y2 = a + b.y2;
            res.z2 = a + b.z2;
            res.w2 = a + b.w2;
            return res;
        }

        /// <summary>
        /// addition operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator+")]
        public static int8 operator +(int8 a, int b)
        {
            int8 res;
            res.x = a.x + b;
            res.y = a.y + b;
            res.z = a.z + b;
            res.w = a.w + b;
            res.x2 = a.x2 + b;
            res.y2 = a.y2 + b;
            res.z2 = a.z2 + b;
            res.w2 = a.w2 + b;
            return res;
        }

        /// <summary>
        /// substraction operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator-")]
        public static int8 operator -(int8 a, int8 b)
        {
            int8 res;
            res.x = a.x - b.x;
            res.y = a.y - b.y;
            res.z = a.z - b.z;
            res.w = a.w - b.w;
            res.x2 = a.x2 - b.x2;
            res.y2 = a.y2 - b.y2;
            res.z2 = a.z2 - b.z2;
            res.w2 = a.w2 - b.w2;
            return res;
        }

        /// <summary>
        /// substraction operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator-")]
        public static int8 operator -(int a, int8 b)
        {
            int8 res;
            res.x = a - b.x;
            res.y = a - b.y;
            res.z = a - b.z;
            res.w = a - b.w;
            res.x2 = a - b.x2;
            res.y2 = a - b.y2;
            res.z2 = a - b.z2;
            res.w2 = a - b.w2;
            return res;
        }

        /// <summary>
        /// substraction operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator-")]
        public static int8 operator -(int8 a, int b)
        {
            int8 res;
            res.x = a.x - b;
            res.y = a.y - b;
            res.z = a.z - b;
            res.w = a.w - b;
            res.x2 = a.x2 - b;
            res.y2 = a.y2 - b;
            res.z2 = a.z2 - b;
            res.w2 = a.w2 - b;
            return res;
        }

        /// <summary>
        /// multiplication operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator*")]
        public static int8 operator *(int8 a, int8 b)
        {
            int8 res;
            res.x = a.x * b.x;
            res.y = a.y * b.y;
            res.z = a.z * b.z;
            res.w = a.w * b.w;
            res.x2 = a.x2 * b.x2;
            res.y2 = a.y2 * b.y2;
            res.z2 = a.z2 * b.z2;
            res.w2 = a.w2 * b.w2;
            return res;
        }

        /// <summary>
        /// multiplication operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator*")]
        public static int8 operator *(int a, int8 b)
        {
            int8 res;
            res.x = a * b.x;
            res.y = a * b.y;
            res.z = a * b.z;
            res.w = a * b.w;
            res.x2 = a * b.x2;
            res.y2 = a * b.y2;
            res.z2 = a * b.z2;
            res.w2 = a * b.w2;
            return res;
        }

        /// <summary>
        /// multiplication operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator*")]
        public static int8 operator *(int8 a, int b)
        {
            int8 res;
            res.x = a.x * b;
            res.y = a.y * b;
            res.z = a.z * b;
            res.w = a.w * b;
            res.x2 = a.x2 * b;
            res.y2 = a.y2 * b;
            res.z2 = a.z2 * b;
            res.w2 = a.w2 * b;
            return res;
        }

        /// <summary>
        /// division operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator/")]
        public static int8 operator /(int8 a, int8 b)
        {
            int8 res;
            res.x = a.x / b.x;
            res.y = a.y / b.y;
            res.z = a.z / b.z;
            res.w = a.w / b.w;
            res.x2 = a.x2 / b.x2;
            res.y2 = a.y2 / b.y2;
            res.z2 = a.z2 / b.z2;
            res.w2 = a.w2 / b.w2;
            return res;
        }

        /// <summary>
        /// division operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator/")]
        public static int8 operator /(int a, int8 b)
        {
            int8 res;
            res.x = a / b.x;
            res.y = a / b.y;
            res.z = a / b.z;
            res.w = a / b.w;
            res.x2 = a / b.x2;
            res.y2 = a / b.y2;
            res.z2 = a / b.z2;
            res.w2 = a / b.w2;
            return res;
        }

        /// <summary>
        /// division operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator/")]
        public static int8 operator /(int8 a, int b)
        {
            int8 res;
            res.x = a.x / b;
            res.y = a.y / b;
            res.z = a.z / b;
            res.w = a.w / b;
            res.x2 = a.x2 / b;
            res.y2 = a.y2 / b;
            res.z2 = a.z2 / b;
            res.w2 = a.w2 / b;
            return res;
        }

        /// <summary>
        /// stores in memory
        /// </summary>
        /// <param name="ptr">destination pointer</param>
        /// <param name="val">value to store</param>
        /// <param name="alignment">has to be a compile time constant</param>
        [IntrinsicFunction(IsNaked=true, Name="__hybridizer_store_int8")]
        public unsafe static void Store(int8* ptr, int8 val, int alignment)
        {
            *ptr = val;
        }

        /// <summary>
        /// stores in memory
        /// </summary>
        /// <param name="ptr">destination pointer</param>
        /// <param name="val">value to store</param>
        /// <param name="alignment">has to be a compile time constant</param>
        [IntrinsicFunction(IsNaked=true, Name="__hybridizer_store_splat_int8")]
        public unsafe static void Store(int8* ptr, int val, int alignment)
        {
            *ptr = new int8(val);
        }

        /// <summary>
        /// loads from memory
        /// </summary>
        /// <param name="ptr"></param>
        /// <param name="alignment">has to be a compile time constant</param>
        [IntrinsicFunction(IsNaked=true, Name="__hybridizer_load_int8")]
        public unsafe static int8 Load(int8* ptr, int alignment)
        {
            return *ptr;
        }
    }

    /// <summary>
    /// two signed bytes
    /// </summary>
    [IntrinsicType("char2")]
    [StructLayout(LayoutKind.Explicit)]
    public unsafe struct char2
    {
        /// <summary>
        /// x
        /// </summary>
        [FieldOffset(0)]
        public sbyte x;
        /// <summary>
        /// y
        /// </summary>
        [FieldOffset(1)]
        public sbyte y;

        /// <summary>
        /// constructor from 32 bits integer
        /// </summary>
        public char2(int val)
        {
            x = (sbyte)(val & 0xFF);
            y = (sbyte)((val >> 8) & 0xFF);
        }
        /// <summary>
        /// constructor from components
        /// </summary>
        public char2(sbyte xx, sbyte yy)
        {
            x = xx;
            y = yy;
        }

        /// <summary>
        /// constructor from single component
        /// </summary>
        /// <param name="val"></param>
        public char2(sbyte val)
        {
            x = val;
            y = val;
        }

        /// <summary>
        /// copy constructor
        /// </summary>
        public char2(char2 res)
        {
            x = res.x;
            y = res.y;
        }

        /// <summary>
        /// conversion to short
        /// </summary>
        public static explicit operator short(char2 res)
        {
            short* ptr = (short*)&res;
            return *ptr;
        }

        /// <summary>
        /// addition operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator+")]
        public static char2 operator +(char2 a, char2 b)
        {
            char2 res;
            res.x = (sbyte)(a.x + b.x);
            res.y = (sbyte)(a.y + b.y);
            return res;
        }

        /// <summary>
        /// addition operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator+")]
        public static char2 operator +(sbyte a, char2 b)
        {
            char2 res;
            res.x = (sbyte)(a + b.x);
            res.y = (sbyte)(a + b.y);
            return res;
        }

        /// <summary>
        /// addition operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator+")]
        public static char2 operator +(char2 a, sbyte b)
        {
            char2 res;
            res.x = (sbyte)(a.x + b);
            res.y = (sbyte)(a.y + b);
            return res;
        }

        /// <summary>
        /// substraction operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator-")]
        public static char2 operator -(char2 a, char2 b)
        {
            char2 res;
            res.x = (sbyte)(a.x - b.x);
            res.y = (sbyte)(a.y - b.y);
            return res;
        }

        /// <summary>
        /// substraction operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator-")]
        public static char2 operator -(sbyte a, char2 b)
        {
            char2 res;
            res.x = (sbyte)(a - b.x);
            res.y = (sbyte)(a - b.y);
            return res;
        }

        /// <summary>
        /// substraction operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator-")]
        public static char2 operator -(char2 a, sbyte b)
        {
            char2 res;
            res.x = (sbyte)(a.x - b);
            res.y = (sbyte)(a.y - b);
            return res;
        }

        /// <summary>
        /// multiplication operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator*")]
        public static char2 operator *(char2 a, char2 b)
        {
            char2 res;
            res.x = (sbyte)(a.x * b.x);
            res.y = (sbyte)(a.y * b.y);
            return res;
        }

        /// <summary>
        /// multiplication operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator*")]
        public static char2 operator *(sbyte a, char2 b)
        {
            char2 res;
            res.x = (sbyte)(a * b.x);
            res.y = (sbyte)(a * b.y);
            return res;
        }

        /// <summary>
        /// multiplication operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator*")]
        public static char2 operator *(char2 a, sbyte b)
        {
            char2 res;
            res.x = (sbyte)(a.x * b);
            res.y = (sbyte)(a.y * b);
            return res;
        }

        /// <summary>
        /// division operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator/")]
        public static char2 operator /(char2 a, char2 b)
        {
            char2 res;
            res.x = (sbyte)(a.x / b.x);
            res.y = (sbyte)(a.y / b.y);
            return res;
        }

        /// <summary>
        /// division operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator/")]
        public static char2 operator /(sbyte a, char2 b)
        {
            char2 res;
            res.x = (sbyte)(a / b.x);
            res.y = (sbyte)(a / b.y);
            return res;
        }

        /// <summary>
        /// division operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator/")]
        public static char2 operator /(char2 a, sbyte b)
        {
            char2 res;
            res.x = (sbyte)(a.x / b);
            res.y = (sbyte)(a.y / b);
            return res;
        }
    }

    /// <summary>
    /// four signed bytes
    /// </summary>
    [IntrinsicType("char4")]
    [StructLayout(LayoutKind.Explicit)]
    public struct char4
    {
        /// <summary>
        /// x
        /// </summary>
        [FieldOffset(0)]
        public sbyte x;
        /// <summary>
        /// y
        /// </summary>
        [FieldOffset(1)]
        public sbyte y;
        /// <summary>
        /// z
        /// </summary>
        [FieldOffset(2)]
        public sbyte z;
        /// <summary>
        /// w
        /// </summary>
        [FieldOffset(3)]
        public sbyte w;

        /// <summary>
        /// constructor from components
        /// </summary>
        /// <param name="xx"></param>
        /// <param name="yy"></param>
        /// <param name="zz"></param>
        /// <param name="ww"></param>
        public char4(sbyte xx, sbyte yy, sbyte zz, sbyte ww)
        {
            x = xx;
            y = yy;
            z = zz;
            w = ww;
        }

        /// <summary>
        ///  constructor from signed 32 bits integer
        /// </summary>
        /// <param name="val"></param>
        public char4(int val)
        { 
            // TODO: is that correct?? from pr60960 it looks like, but from logic it doesn't
            x = (sbyte)(val & 0xFF);
            y = (sbyte)((val >> 8) & 0xFF);
            z = (sbyte)((val >> 16) & 0xFF);
            w = (sbyte)((val >> 24) & 0xFF);
        }

        /// <summary>
        /// constructor from single component
        /// </summary>
        public char4(sbyte val)
        {
            x = val;
            y = val;
            z = val;
            w = val;
        }

        /// <summary>
        ///  copy constructor
        /// </summary>
        public char4(char4 res)
        {
            x = res.x;
            y = res.y;
            z = res.z;
            w = res.w;
        }

        /// <summary>
        /// addition operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator+")]
        public static char4 operator +(char4 a, char4 b)
        {
            char4 res;
            res.x = (sbyte)(a.x + b.x);
            res.y = (sbyte)(a.y + b.y);
            res.z = (sbyte)(a.z + b.z);
            res.w = (sbyte)(a.w + b.w);
            return res;
        }

        /// <summary>
        /// addition operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator+")]
        public static char4 operator +(sbyte a, char4 b)
        {
            char4 res;
            res.x = (sbyte)(a + b.x);
            res.y = (sbyte)(a + b.y);
            res.z = (sbyte)(a + b.z);
            res.w = (sbyte)(a + b.w);
            return res;
        }

        /// <summary>
        /// addition operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator+")]
        public static char4 operator +(char4 a, sbyte b)
        {
            char4 res;
            res.x = (sbyte)(a.x + b);
            res.y = (sbyte)(a.y + b);
            res.z = (sbyte)(a.z + b);
            res.w = (sbyte)(a.w + b);
            return res;
        }

        /// <summary>
        /// substraction operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator-")]
        public static char4 operator -(char4 a, char4 b)
        {
            char4 res;
            res.x = (sbyte)(a.x - b.x);
            res.y = (sbyte)(a.y - b.y);
            res.z = (sbyte)(a.z - b.z);
            res.w = (sbyte)(a.w - b.w);
            return res;
        }

        /// <summary>
        /// substraction operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator-")]
        public static char4 operator -(sbyte a, char4 b)
        {
            char4 res;
            res.x = (sbyte)(a - b.x);
            res.y = (sbyte)(a - b.y);
            res.z = (sbyte)(a - b.z);
            res.w = (sbyte)(a - b.w);
            return res;
        }

        /// <summary>
        /// substraction operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator-")]
        public static char4 operator -(char4 a, sbyte b)
        {
            char4 res;
            res.x = (sbyte)(a.x - b);
            res.y = (sbyte)(a.y - b);
            res.z = (sbyte)(a.z - b);
            res.w = (sbyte)(a.w - b);
            return res;
        }

        /// <summary>
        /// multiplication operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator*")]
        public static char4 operator *(char4 a, char4 b)
        {
            char4 res;
            res.x = (sbyte)(a.x * b.x);
            res.y = (sbyte)(a.y * b.y);
            res.z = (sbyte)(a.z * b.z);
            res.w = (sbyte)(a.w * b.w);
            return res;
        }

        /// <summary>
        /// multiplication operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator*")]
        public static char4 operator *(sbyte a, char4 b)
        {
            char4 res;
            res.x = (sbyte)(a * b.x);
            res.y = (sbyte)(a * b.y);
            res.z = (sbyte)(a * b.z);
            res.w = (sbyte)(a * b.w);
            return res;
        }

        /// <summary>
        /// multiplication operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator*")]
        public static char4 operator *(char4 a, sbyte b)
        {
            char4 res;
            res.x = (sbyte)(a.x * b);
            res.y = (sbyte)(a.y * b);
            res.z = (sbyte)(a.z * b);
            res.w = (sbyte)(a.w * b);
            return res;
        }

        /// <summary>
        /// division operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator/")]
        public static char4 operator /(char4 a, char4 b)
        {
            char4 res;
            res.x = (sbyte)(a.x / b.x);
            res.y = (sbyte)(a.y / b.y);
            res.z = (sbyte)(a.z / b.z);
            res.w = (sbyte)(a.w / b.w);
            return res;
        }

        /// <summary>
        /// division operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator/")]
        public static char4 operator /(sbyte a, char4 b)
        {
            char4 res;
            res.x = (sbyte)(a / b.x);
            res.y = (sbyte)(a / b.y);
            res.z = (sbyte)(a / b.z);
            res.w = (sbyte)(a / b.w);
            return res;
        }

        /// <summary>
        /// division operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator/")]
        public static char4 operator /(char4 a, sbyte b)
        {
            char4 res;
            res.x = (sbyte)(a.x / b);
            res.y = (sbyte)(a.y / b);
            res.z = (sbyte)(a.z / b);
            res.w = (sbyte)(a.w / b);
            return res;
        }

        /// <summary>
        /// stores in memory
        /// </summary>
        /// <param name="ptr">destination pointer</param>
        /// <param name="val">value to store</param>
        /// <param name="alignment">has to be a compile time constant</param>
        [IntrinsicFunction(IsNaked=true, Name="__hybridizer_store_char4")]
        public unsafe static void Store(char4* ptr, char4 val, int alignment)
        {
            *ptr = val;
        }

        /// <summary>
        /// stores in memory
        /// </summary>
        /// <param name="ptr">destination pointer</param>
        /// <param name="val">value to store</param>
        /// <param name="alignment">has to be a compile time constant</param>
        [IntrinsicFunction(IsNaked=true, Name="__hybridizer_store_splat_char4")]
        public unsafe static void Store(char4* ptr, sbyte val, int alignment)
        {
            *ptr = new char4(val);
        }

        /// <summary>
        /// loads from memory
        /// </summary>
        /// <param name="ptr"></param>
        /// <param name="alignment">has to be a compile time constant</param>
        [IntrinsicFunction(IsNaked=true, Name="__hybridizer_load_char4")]
        public unsafe static char4 Load(char4* ptr, int alignment)
        {
            return *ptr;
        }
    }

    /// <summary>
    /// four unsigned signed bytes
    /// </summary>
    [IntrinsicType("uchar4")]
    [IntrinsicPrimitive("uchar4")]
    [StructLayout(LayoutKind.Explicit)]
    public struct uchar4
    {
        /// <summary>
        /// x
        /// </summary>
        [FieldOffset(0)]
        public byte x;
        /// <summary>
        /// y
        /// </summary>
        [FieldOffset(1)]
        public byte y;
        /// <summary>
        /// z
        /// </summary>
        [FieldOffset(2)]
        public byte z;
        /// <summary>
        /// w
        /// </summary>
        [FieldOffset(3)]
        public byte w;

        /// <summary>
        /// constructor from components
        /// </summary>
        /// <param name="xx"></param>
        /// <param name="yy"></param>
        /// <param name="zz"></param>
        /// <param name="ww"></param>
        public uchar4(byte xx, byte yy, byte zz, byte ww)
        {
            x = xx;
            y = yy;
            z = zz;
            w = ww;
        }

        /// <summary>
        ///  constructor from signed 32 bits integer
        /// </summary>
        /// <param name="val"></param>
        public uchar4(int val)
        {
            // TODO: is that correct?? from pr60960 it looks like, but from logic it doesn't
            x = (byte)(val & 0xFF);
            y = (byte)((val >> 8) & 0xFF);
            z = (byte)((val >> 16) & 0xFF);
            w = (byte)((val >> 24) & 0xFF);
        }

        /// <summary>
        /// constructor from single component
        /// </summary>
        public uchar4(byte val)
        {
            x = val;
            y = val;
            z = val;
            w = val;
        }

        /// <summary>
        ///  copy constructor
        /// </summary>
        public uchar4(uchar4 res)
        {
            x = res.x;
            y = res.y;
            z = res.z;
            w = res.w;
        }

        /// <summary>
        /// addition operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator+")]
        public static uchar4 operator +(uchar4 a, uchar4 b)
        {
            uchar4 res;
            res.x = (byte)(a.x + b.x);
            res.y = (byte)(a.y + b.y);
            res.z = (byte)(a.z + b.z);
            res.w = (byte)(a.w + b.w);
            return res;
        }

        /// <summary>
        /// addition operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator+")]
        public static uchar4 operator +(byte a, uchar4 b)
        {
            uchar4 res;
            res.x = (byte)(a + b.x);
            res.y = (byte)(a + b.y);
            res.z = (byte)(a + b.z);
            res.w = (byte)(a + b.w);
            return res;
        }

        /// <summary>
        /// addition operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator+")]
        public static uchar4 operator +(uchar4 a, sbyte b)
        {
            uchar4 res;
            res.x = (byte)(a.x + b);
            res.y = (byte)(a.y + b);
            res.z = (byte)(a.z + b);
            res.w = (byte)(a.w + b);
            return res;
        }

        /// <summary>
        /// substraction operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator-")]
        public static uchar4 operator -(uchar4 a, uchar4 b)
        {
            uchar4 res;
            res.x = (byte)(a.x - b.x);
            res.y = (byte)(a.y - b.y);
            res.z = (byte)(a.z - b.z);
            res.w = (byte)(a.w - b.w);
            return res;
        }

        /// <summary>
        /// substraction operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator-")]
        public static uchar4 operator -(byte a, uchar4 b)
        {
            uchar4 res;
            res.x = (byte)(a - b.x);
            res.y = (byte)(a - b.y);
            res.z = (byte)(a - b.z);
            res.w = (byte)(a - b.w);
            return res;
        }

        /// <summary>
        /// substraction operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator-")]
        public static uchar4 operator -(uchar4 a, byte b)
        {
            uchar4 res;
            res.x = (byte)(a.x - b);
            res.y = (byte)(a.y - b);
            res.z = (byte)(a.z - b);
            res.w = (byte)(a.w - b);
            return res;
        }

        /// <summary>
        /// multiplication operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator*")]
        public static uchar4 operator *(uchar4 a, uchar4 b)
        {
            uchar4 res;
            res.x = (byte)(a.x * b.x);
            res.y = (byte)(a.y * b.y);
            res.z = (byte)(a.z * b.z);
            res.w = (byte)(a.w * b.w);
            return res;
        }

        /// <summary>
        /// multiplication operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator*")]
        public static uchar4 operator *(byte a, uchar4 b)
        {
            uchar4 res;
            res.x = (byte)(a * b.x);
            res.y = (byte)(a * b.y);
            res.z = (byte)(a * b.z);
            res.w = (byte)(a * b.w);
            return res;
        }

        /// <summary>
        /// multiplication operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator*")]
        public static uchar4 operator *(uchar4 a, byte b)
        {
            uchar4 res;
            res.x = (byte)(a.x * b);
            res.y = (byte)(a.y * b);
            res.z = (byte)(a.z * b);
            res.w = (byte)(a.w * b);
            return res;
        }

        /// <summary>
        /// division operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator/")]
        public static uchar4 operator /(uchar4 a, uchar4 b)
        {
            uchar4 res;
            res.x = (byte)(a.x / b.x);
            res.y = (byte)(a.y / b.y);
            res.z = (byte)(a.z / b.z);
            res.w = (byte)(a.w / b.w);
            return res;
        }

        /// <summary>
        /// division operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator/")]
        public static uchar4 operator /(byte a, uchar4 b)
        {
            uchar4 res;
            res.x = (byte)(a / b.x);
            res.y = (byte)(a / b.y);
            res.z = (byte)(a / b.z);
            res.w = (byte)(a / b.w);
            return res;
        }

        /// <summary>
        /// division operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator/")]
        public static uchar4 operator /(uchar4 a, byte b)
        {
            uchar4 res;
            res.x = (byte)(a.x / b);
            res.y = (byte)(a.y / b);
            res.z = (byte)(a.z / b);
            res.w = (byte)(a.w / b);
            return res;
        }

        /// <summary>
        /// stores in memory
        /// </summary>
        /// <param name="ptr">destination pointer</param>
        /// <param name="val">value to store</param>
        /// <param name="alignment">has to be a compile time constant</param>
        [IntrinsicFunction(IsNaked = true, Name = "__hybridizer_store_uchar4")]
        public unsafe static void Store(uchar4* ptr, uchar4 val, int alignment)
        {
            *ptr = val;
        }

        /// <summary>
        /// stores in memory
        /// </summary>
        /// <param name="ptr">destination pointer</param>
        /// <param name="val">value to store</param>
        /// <param name="alignment">has to be a compile time constant</param>
        [IntrinsicFunction(IsNaked = true, Name = "__hybridizer_store_splat_uchar4")]
        public unsafe static void Store(uchar4* ptr, byte val, int alignment)
        {
            *ptr = new uchar4(val);
        }

        /// <summary>
        /// loads from memory
        /// </summary>
        /// <param name="ptr"></param>
        /// <param name="alignment">has to be a compile time constant</param>
        [IntrinsicFunction(IsNaked = true, Name = "__hybridizer_load_uchar4")]
        public unsafe static uchar4 Load(uchar4* ptr, int alignment)
        {
            return *ptr;
        }
    }

    /// <summary>
    /// 8 signed bytes
    /// </summary>
    [IntrinsicType("char8")]
    [StructLayout(LayoutKind.Explicit)]
    public struct char8
    {
        /// <summary>
        /// x
        /// </summary>
        [FieldOffset(0)]
        public sbyte x;
        /// <summary>
        /// y
        /// </summary>
        [FieldOffset(1)]
        public sbyte y;
        /// <summary>
        /// z
        /// </summary>
        [FieldOffset(2)]
        public sbyte z;
        /// <summary>
        /// w
        /// </summary>
        [FieldOffset(3)]
        public sbyte w;
        /// <summary>
        /// x2
        /// </summary>
        [FieldOffset(4)]
        public sbyte x2;
        /// <summary>
        /// y2
        /// </summary>
        [FieldOffset(5)]
        public sbyte y2;
        /// <summary>
        /// z2
        /// </summary>
        [FieldOffset(6)]
        public sbyte z2;
        /// <summary>
        /// w2
        /// </summary>
        [FieldOffset(7)]
        public sbyte w2;

        /// <summary>
        /// constructor from components
        /// </summary>
        public char8(sbyte xx, sbyte yy, sbyte zz, sbyte ww, sbyte xx2, sbyte yy2, sbyte zz2, sbyte ww2)
        {
            x = xx;
            y = yy;
            z = zz;
            w = ww;
            x2 = xx2;
            y2 = yy2;
            z2 = zz2;
            w2 = ww2;
        }

        /// <summary>
        /// constructor from single component
        /// </summary>
        public char8(sbyte val)
        {
            x = val;
            y = val;
            z = val;
            w = val;
            x2 = val;
            y2 = val;
            z2 = val;
            w2 = val;
        }

        /// <summary>
        /// copy constructor
        /// </summary>
        public char8(char8 res)
        {
            x = res.x;
            y = res.y;
            z = res.z;
            w = res.w;
            x2 = res.x2;
            y2 = res.y2;
            z2 = res.z2;
            w2 = res.w2;
        }

        /// <summary>
        /// addition operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator+")]
        public static char8 operator +(char8 a, char8 b)
        {
            char8 res;
            res.x = (sbyte)(a.x + b.x);
            res.y = (sbyte)(a.y + b.y);
            res.z = (sbyte)(a.z + b.z);
            res.w = (sbyte)(a.w + b.w);
            res.x2 = (sbyte)(a.x2 + b.x2);
            res.y2 = (sbyte)(a.y2 + b.y2);
            res.z2 = (sbyte)(a.z2 + b.z2);
            res.w2 = (sbyte)(a.w2 + b.w2);
            return res;
        }

        /// <summary>
        /// addition operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator+")]
        public static char8 operator +(sbyte a, char8 b)
        {
            char8 res;
            res.x = (sbyte)(a + b.x);
            res.y = (sbyte)(a + b.y);
            res.z = (sbyte)(a + b.z);
            res.w = (sbyte)(a + b.w);
            res.x2 = (sbyte)(a + b.x2);
            res.y2 = (sbyte)(a + b.y2);
            res.z2 = (sbyte)(a + b.z2);
            res.w2 = (sbyte)(a + b.w2);
            return res;
        }

        /// <summary>
        /// addition operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator+")]
        public static char8 operator +(char8 a, sbyte b)
        {
            char8 res;
            res.x = (sbyte)(a.x + b);
            res.y = (sbyte)(a.y + b);
            res.z = (sbyte)(a.z + b);
            res.w = (sbyte)(a.w + b);
            res.x2 = (sbyte)(a.x2 + b);
            res.y2 = (sbyte)(a.y2 + b);
            res.z2 = (sbyte)(a.z2 + b);
            res.w2 = (sbyte)(a.w2 + b);
            return res;
        }

        /// <summary>
        /// substraction operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator-")]
        public static char8 operator -(char8 a, char8 b)
        {
            char8 res;
            res.x = (sbyte)(a.x - b.x);
            res.y = (sbyte)(a.y - b.y);
            res.z = (sbyte)(a.z - b.z);
            res.w = (sbyte)(a.w - b.w);
            res.x2 = (sbyte)(a.x2 - b.x2);
            res.y2 = (sbyte)(a.y2 - b.y2);
            res.z2 = (sbyte)(a.z2 - b.z2);
            res.w2 = (sbyte)(a.w2 - b.w2);
            return res;
        }

        /// <summary>
        /// substraction operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator-")]
        public static char8 operator -(sbyte a, char8 b)
        {
            char8 res;
            res.x = (sbyte)(a - b.x);
            res.y = (sbyte)(a - b.y);
            res.z = (sbyte)(a - b.z);
            res.w = (sbyte)(a - b.w);
            res.x2 = (sbyte)(a - b.x2);
            res.y2 = (sbyte)(a - b.y2);
            res.z2 = (sbyte)(a - b.z2);
            res.w2 = (sbyte)(a - b.w2);
            return res;
        }

        /// <summary>
        /// substraction operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator-")]
        public static char8 operator -(char8 a, sbyte b)
        {
            char8 res;
            res.x = (sbyte)(a.x - b);
            res.y = (sbyte)(a.y - b);
            res.z = (sbyte)(a.z - b);
            res.w = (sbyte)(a.w - b);
            res.x2 = (sbyte)(a.x2 - b);
            res.y2 = (sbyte)(a.y2 - b);
            res.z2 = (sbyte)(a.z2 - b);
            res.w2 = (sbyte)(a.w2 - b);
            return res;
        }

        /// <summary>
        /// multiplication operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator*")]
        public static char8 operator *(char8 a, char8 b)
        {
            char8 res;
            res.x = (sbyte)(a.x * b.x);
            res.y = (sbyte)(a.y * b.y);
            res.z = (sbyte)(a.z * b.z);
            res.w = (sbyte)(a.w * b.w);
            res.x2 = (sbyte)(a.x2 * b.x2);
            res.y2 = (sbyte)(a.y2 * b.y2);
            res.z2 = (sbyte)(a.z2 * b.z2);
            res.w2 = (sbyte)(a.w2 * b.w2);
            return res;
        }

        /// <summary>
        /// multiplication operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator*")]
        public static char8 operator *(sbyte a, char8 b)
        {
            char8 res;
            res.x = (sbyte)(a * b.x);
            res.y = (sbyte)(a * b.y);
            res.z = (sbyte)(a * b.z);
            res.w = (sbyte)(a * b.w);
            res.x2 = (sbyte)(a * b.x2);
            res.y2 = (sbyte)(a * b.y2);
            res.z2 = (sbyte)(a * b.z2);
            res.w2 = (sbyte)(a * b.w2);
            return res;
        }

        /// <summary>
        /// multiplication operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator*")]
        public static char8 operator *(char8 a, sbyte b)
        {
            char8 res;
            res.x = (sbyte)(a.x * b);
            res.y = (sbyte)(a.y * b);
            res.z = (sbyte)(a.z * b);
            res.w = (sbyte)(a.w * b);
            res.x2 = (sbyte)(a.x2 * b);
            res.y2 = (sbyte)(a.y2 * b);
            res.z2 = (sbyte)(a.z2 * b);
            res.w2 = (sbyte)(a.w2 * b);
            return res;
        }

        /// <summary>
        /// division operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator/")]
        public static char8 operator /(char8 a, char8 b)
        {
            char8 res;
            res.x = (sbyte)(a.x / b.x);
            res.y = (sbyte)(a.y / b.y);
            res.z = (sbyte)(a.z / b.z);
            res.w = (sbyte)(a.w / b.w);
            res.x2 = (sbyte)(a.x2 / b.x2);
            res.y2 = (sbyte)(a.y2 / b.y2);
            res.z2 = (sbyte)(a.z2 / b.z2);
            res.w2 = (sbyte)(a.w2 / b.w2);
            return res;
        }

        /// <summary>
        /// division operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator/")]
        public static char8 operator /(sbyte a, char8 b)
        {
            char8 res;
            res.x = (sbyte)(a / b.x);
            res.y = (sbyte)(a / b.y);
            res.z = (sbyte)(a / b.z);
            res.w = (sbyte)(a / b.w);
            res.x2 = (sbyte)(a / b.x2);
            res.y2 = (sbyte)(a / b.y2);
            res.z2 = (sbyte)(a / b.z2);
            res.w2 = (sbyte)(a / b.w2);
            return res;
        }

        /// <summary>
        /// division operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator/")]
        public static char8 operator /(char8 a, sbyte b)
        {
            char8 res;
            res.x = (sbyte)(a.x / b);
            res.y = (sbyte)(a.y / b);
            res.z = (sbyte)(a.z / b);
            res.w = (sbyte)(a.w / b);
            res.x2 = (sbyte)(a.x2 / b);
            res.y2 = (sbyte)(a.y2 / b);
            res.z2 = (sbyte)(a.z2 / b);
            res.w2 = (sbyte)(a.w2 / b);
            return res;
        }

        /// <summary>
        /// stores in memory
        /// </summary>
        /// <param name="ptr">destination pointer</param>
        /// <param name="val">value to store</param>
        /// <param name="alignment">has to be a compile time constant</param>
        [IntrinsicFunction(IsNaked=true, Name="__hybridizer_store_char8")]
        public unsafe static void Store(char8* ptr, char8 val, int alignment)
        {
            *ptr = val;
        }

        /// <summary>
        /// stores in memory
        /// </summary>
        /// <param name="ptr">destination pointer</param>
        /// <param name="val">value to store</param>
        /// <param name="alignment">has to be a compile time constant</param>
        [IntrinsicFunction(IsNaked=true, Name="__hybridizer_store_splat_char8")]
        public unsafe static void Store(char8* ptr, sbyte val, int alignment)
        {
            *ptr = new char8(val);
        }

        /// <summary>
        /// loads from memory
        /// </summary>
        /// <param name="ptr"></param>
        /// <param name="alignment">has to be a compile time constant</param>
        [IntrinsicFunction(IsNaked=true, Name="__hybridizer_load_char8")]
        public unsafe static char8 Load(char8* ptr, int alignment)
        {
            return *ptr;
        }
    }

    /// <summary>
    /// 2 26 bits integers
    /// </summary>
    [IntrinsicType("short2")]
    [StructLayout(LayoutKind.Explicit)]
    public unsafe struct short2
    {
        /// <summary>
        /// x
        /// </summary>
        [FieldOffset(0)]
        public short x;
        /// <summary>
        /// y
        /// </summary>
        [FieldOffset(2)]
        public short y;

        /// <summary>
        /// constructor from 32 bits integer
        /// </summary>
        public short2(int val)
        {
            x = (short) (val & 0xFFFF);
            y = (short) ((val >> 16) & 0xFFFF);
        }

        /// <summary>
        /// constructor from components
        /// </summary>
        public short2(short xx, short yy)
        {
            x = xx;
            y = yy;
        }

        /// <summary>
        /// constructor from single component
        /// </summary>
        public short2(short val)
        {
            x = val;
            y = val;
        }

        /// <summary>
        /// copy constructor
        /// </summary>
        public short2(short2 res)
        {
            x = res.x;
            y = res.y;
        }

        /// <summary>
        /// conversion to 32 bits integer
        /// </summary>
        /// <param name="res"></param>
        public static explicit operator int(short2 res)
        {
            int* ptr = (int*)&res;
            return *ptr;
        }

        /// <summary>
        /// addition operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator+")]
        public static short2 operator +(short2 a, short2 b)
        {
            short2 res;
            res.x = (short)(a.x + b.x);
            res.y = (short)(a.y + b.y);
            return res;
        }

        /// <summary>
        /// addition operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator+")]
        public static short2 operator +(short a, short2 b)
        {
            short2 res;
            res.x = (short)(a + b.x);
            res.y = (short)(a + b.y);
            return res;
        }

        /// <summary>
        /// addition operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator+")]
        public static short2 operator +(short2 a, short b)
        {
            short2 res;
            res.x = (short)(a.x + b);
            res.y = (short)(a.y + b);
            return res;
        }

        /// <summary>
        /// substraction operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator-")]
        public static short2 operator -(short2 a, short2 b)
        {
            short2 res;
            res.x = (short)(a.x - b.x);
            res.y = (short)(a.y - b.y);
            return res;
        }

        /// <summary>
        /// substraction operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator-")]
        public static short2 operator -(short a, short2 b)
        {
            short2 res;
            res.x = (short)(a - b.x);
            res.y = (short)(a - b.y);
            return res;
        }

        /// <summary>
        /// substraction operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator-")]
        public static short2 operator -(short2 a, short b)
        {
            short2 res;
            res.x = (short)(a.x - b);
            res.y = (short)(a.y - b);
            return res;
        }

        /// <summary>
        /// multiplication operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator*")]
        public static short2 operator *(short2 a, short2 b)
        {
            short2 res;
            res.x = (short)(a.x * b.x);
            res.y = (short)(a.y * b.y);
            return res;
        }

        /// <summary>
        /// multiplication operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator*")]
        public static short2 operator *(short a, short2 b)
        {
            short2 res;
            res.x = (short)(a * b.x);
            res.y = (short)(a * b.y);
            return res;
        }

        /// <summary>
        /// multiplication operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator*")]
        public static short2 operator *(short2 a, short b)
        {
            short2 res;
            res.x = (short)(a.x * b);
            res.y = (short)(a.y * b);
            return res;
        }

        /// <summary>
        /// division operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator/")]
        public static short2 operator /(short2 a, short2 b)
        {
            short2 res;
            res.x = (short)(a.x / b.x);
            res.y = (short)(a.y / b.y);
            return res;
        }

        /// <summary>
        /// division operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator/")]
        public static short2 operator /(short a, short2 b)
        {
            short2 res;
            res.x = (short)(a / b.x);
            res.y = (short)(a / b.y);
            return res;
        }

        /// <summary>
        /// division operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator/")]
        public static short2 operator /(short2 a, short b)
        {
            short2 res;
            res.x = (short)(a.x / b);
            res.y = (short)(a.y / b);
            return res;
        }
        
        /// <summary>
        /// stores in memory
        /// </summary>
        /// <param name="ptr">destination pointer</param>
        /// <param name="val">value to store</param>
        /// <param name="alignment">has to be a compile time constant</param>
        [IntrinsicFunction(IsNaked=true, Name="__hybridizer_store_short2")]
        public unsafe static void Store(short2* ptr, short2 val, int alignment)
        {
            *ptr = val;
        }

        /// <summary>
        /// stores in memory
        /// </summary>
        /// <param name="ptr">destination pointer</param>
        /// <param name="val">value to store</param>
        /// <param name="alignment">has to be a compile time constant</param>
        [IntrinsicFunction(IsNaked=true, Name="__hybridizer_store_splat_short2")]
        public unsafe static void Store(short2* ptr, sbyte val, int alignment)
        {
            *ptr = new short2(val);
        }

        /// <summary>
        /// loads from memory
        /// </summary>
        /// <param name="ptr"></param>
        /// <param name="alignment">has to be a compile time constant</param>
        [IntrinsicFunction(IsNaked=true, Name="__hybridizer_load_short2")]
        public unsafe static short2 Load(short2* ptr, int alignment)
        {
            return *ptr;
        }
    }

    /// <summary>
    /// 4 16 bits integers
    /// </summary>
    [IntrinsicType("short4")]
    [StructLayout(LayoutKind.Explicit)]
    public struct short4
    {
        /// <summary>
        /// x
        /// </summary>
        [FieldOffset(0)]
        public short x;
        /// <summary>
        /// y
        /// </summary>
        [FieldOffset(2)]
        public short y;
        /// <summary>
        /// z
        /// </summary>
        [FieldOffset(4)]
        public short z;
        /// <summary>
        /// w
        /// </summary>
        [FieldOffset(6)]
        public short w;

        /// <summary>
        /// constructor from components
        /// </summary>
        public short4(short xx, short yy, short zz, short ww)
        {
            x = xx;
            y = yy;
            z = zz;
            w = ww;
        }

        /// <summary>
        /// constructor from single component
        /// </summary>
        public short4(short val)
        {
            x = val;
            y = val;
            z = val;
            w = val;
        }

        /// <summary>
        /// copy constructor
        /// </summary>
        /// <param name="res"></param>
        public short4(short4 res)
        {
            x = res.x;
            y = res.y;
            z = res.z;
            w = res.w;
        }

        /// <summary>
        /// addition operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator+")]
        public static short4 operator +(short4 a, short4 b)
        {
            short4 res;
            res.x = (short)(a.x + b.x);
            res.y = (short)(a.y + b.y);
            res.z = (short)(a.z + b.z);
            res.w = (short)(a.w + b.w);
            return res;
        }

        /// <summary>
        /// addition operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator+")]
        public static short4 operator +(short a, short4 b)
        {
            short4 res;
            res.x = (short)(a + b.x);
            res.y = (short)(a + b.y);
            res.z = (short)(a + b.z);
            res.w = (short)(a + b.w);
            return res;
        }

        /// <summary>
        /// addition operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator+")]
        public static short4 operator +(short4 a, short b)
        {
            short4 res;
            res.x = (short)(a.x + b);
            res.y = (short)(a.y + b);
            res.z = (short)(a.z + b);
            res.w = (short)(a.w + b);
            return res;
        }

        /// <summary>
        /// substraction operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator-")]
        public static short4 operator -(short4 a, short4 b)
        {
            short4 res;
            res.x = (short)(a.x - b.x);
            res.y = (short)(a.y - b.y);
            res.z = (short)(a.z - b.z);
            res.w = (short)(a.w - b.w);
            return res;
        }

        /// <summary>
        /// substraction operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator-")]
        public static short4 operator -(short a, short4 b)
        {
            short4 res;
            res.x = (short)(a - b.x);
            res.y = (short)(a - b.y);
            res.z = (short)(a - b.z);
            res.w = (short)(a - b.w);
            return res;
        }

        /// <summary>
        /// substraction operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator-")]
        public static short4 operator -(short4 a, short b)
        {
            short4 res;
            res.x = (short)(a.x - b);
            res.y = (short)(a.y - b);
            res.z = (short)(a.z - b);
            res.w = (short)(a.w - b);
            return res;
        }

        /// <summary>
        /// multiplication operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator*")]
        public static short4 operator *(short4 a, short4 b)
        {
            short4 res;
            res.x = (short)(a.x * b.x);
            res.y = (short)(a.y * b.y);
            res.z = (short)(a.z * b.z);
            res.w = (short)(a.w * b.w);
            return res;
        }

        /// <summary>
        /// multiplication operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator*")]
        public static short4 operator *(short a, short4 b)
        {
            short4 res;
            res.x = (short)(a * b.x);
            res.y = (short)(a * b.y);
            res.z = (short)(a * b.z);
            res.w = (short)(a * b.w);
            return res;
        }

        /// <summary>
        /// multiplication operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator*")]
        public static short4 operator *(short4 a, short b)
        {
            short4 res;
            res.x = (short)(a.x * b);
            res.y = (short)(a.y * b);
            res.z = (short)(a.z * b);
            res.w = (short)(a.w * b);
            return res;
        }

        /// <summary>
        /// division operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator/")]
        public static short4 operator /(short4 a, short4 b)
        {
            short4 res;
            res.x = (short)(a.x / b.x);
            res.y = (short)(a.y / b.y);
            res.z = (short)(a.z / b.z);
            res.w = (short)(a.w / b.w);
            return res;
        }

        /// <summary>
        /// division operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator/")]
        public static short4 operator /(short a, short4 b)
        {
            short4 res;
            res.x = (short)(a / b.x);
            res.y = (short)(a / b.y);
            res.z = (short)(a / b.z);
            res.w = (short)(a / b.w);
            return res;
        }

        /// <summary>
        /// division operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator/")]
        public static short4 operator /(short4 a, short b)
        {
            short4 res;
            res.x = (short)(a.x / b);
            res.y = (short)(a.y / b);
            res.z = (short)(a.z / b);
            res.w = (short)(a.w / b);
            return res;
        }

        /// <summary>
        /// loads from memory
        /// </summary>
        /// <param name="ptr"></param>
        /// <param name="alignment">has to be a compile time constant</param>
        [IntrinsicFunction(IsNaked=true, Name="__hybridizer_load_short4")]
        public unsafe static short4 Load(short4* ptr, int alignment)
        {
            return *ptr;
        }

        /// <summary>
        /// stores in memory
        /// </summary>
        /// <param name="ptr">destination pointer</param>
        /// <param name="val">value to store</param>
        /// <param name="alignment">has to be a compile time constant</param>
        [IntrinsicFunction(IsNaked=true, Name="__hybridizer_store_short4")]
        public unsafe static void Store(short4* ptr, short4 val, int alignment)
        {
            *ptr = val;
        }
    }

    /// <summary>
    /// 8 16 bits integers
    /// </summary>
    [IntrinsicType("short8")]
    [StructLayout(LayoutKind.Explicit)]
    public struct short8
    {
        /// <summary>
        /// x
        /// </summary>
        [FieldOffset(0)]
        public short x;
        /// <summary>
        /// y
        /// </summary>
        [FieldOffset(2)]
        public short y;
        /// <summary>
        /// z
        /// </summary>
        [FieldOffset(4)]
        public short z;
        /// <summary>
        /// w
        /// </summary>
        [FieldOffset(6)]
        public short w;
        /// <summary>
        /// x2
        /// </summary>
        [FieldOffset(8)]
        public short x2;
        /// <summary>
        /// y2
        /// </summary>
        [FieldOffset(10)]
        public short y2;
        /// <summary>
        /// z2
        /// </summary>
        [FieldOffset(12)]
        public short z2;
        /// <summary>
        /// w2
        /// </summary>
        [FieldOffset(14)]
        public short w2;

        /// <summary>
        /// constructor from components
        /// </summary>
        public short8(short xx, short yy, short zz, short ww, short xx2, short yy2, short zz2, short ww2)
        {
            x = xx;
            y = yy;
            z = zz;
            w = ww;
            x2 = xx2;
            y2 = yy2;
            z2 = zz2;
            w2 = ww2;
        }

        /// <summary>
        /// constructor from single component
        /// </summary>
        public short8(short val)
        {
            x = val;
            y = val;
            z = val;
            w = val;
            x2 = val;
            y2 = val;
            z2 = val;
            w2 = val;
        }

        /// <summary>
        /// copy constructor
        /// </summary>
        public short8(short8 res)
        {
            x = res.x;
            y = res.y;
            z = res.z;
            w = res.w;
            x2 = res.x2;
            y2 = res.y2;
            z2 = res.z2;
            w2 = res.w2;
        }

        /// <summary>
        /// addition operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator+")]
        public static short8 operator +(short8 a, short8 b)
        {
            short8 res;
            res.x = (short)(a.x + b.x);
            res.y = (short)(a.y + b.y);
            res.z = (short)(a.z + b.z);
            res.w = (short)(a.w + b.w);
            res.x2 = (short)(a.x2 + b.x2);
            res.y2 = (short)(a.y2 + b.y2);
            res.z2 = (short)(a.z2 + b.z2);
            res.w2 = (short)(a.w2 + b.w2);
            return res;
        }

        /// <summary>
        /// addition operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator+")]
        public static short8 operator +(short a, short8 b)
        {
            short8 res;
            res.x = (short)(a + b.x  );
            res.y = (short)(a + b.y  );
            res.z = (short)(a + b.z  );
            res.w = (short)(a + b.w  );
            res.x2 =(short)( a + b.x2);
            res.y2 =(short)( a + b.y2);
            res.z2 =(short)( a + b.z2);
            res.w2 =(short)( a + b.w2);
            return res;
        }

        /// <summary>
        /// addition operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator+")]
        public static short8 operator +(short8 a, short b)
        {
            short8 res;
            res.x = (short)(a.x + b);
            res.y = (short)(a.y + b);
            res.z = (short)(a.z + b);
            res.w = (short)(a.w + b);
            res.x2 =(short)( a.x2 + b);
            res.y2 =(short)( a.y2 + b);
            res.z2 =(short)( a.z2 + b);
            res.w2 =(short)( a.w2 + b);
            return res;
        }

        /// <summary>
        /// substraction operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator-")]
        public static short8 operator -(short8 a, short8 b)
        {
            short8 res;
            res.x = (short)(a.x - b.x);
            res.y = (short)(a.y - b.y);
            res.z = (short)(a.z - b.z);
            res.w = (short)(a.w - b.w);
            res.x2 =(short)( a.x2 - b.x2);
            res.y2 =(short)( a.y2 - b.y2);
            res.z2 =(short)( a.z2 - b.z2);
            res.w2 =(short)( a.w2 - b.w2);
            return res;
        }

        /// <summary>
        /// substraction operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator-")]
        public static short8 operator -(short a, short8 b)
        {
            short8 res;
            res.x = (short)(a - b.x);
            res.y = (short)(a - b.y);
            res.z = (short)(a - b.z);
            res.w = (short)(a - b.w);
            res.x2 =(short)( a - b.x2);
            res.y2 =(short)( a - b.y2);
            res.z2 =(short)( a - b.z2);
            res.w2 =(short)( a - b.w2);
            return res;
        }

        /// <summary>
        /// substraction operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator-")]
        public static short8 operator -(short8 a, short b)
        {
            short8 res;
            res.x = (short)(a.x - b);
            res.y = (short)(a.y - b);
            res.z = (short)(a.z - b);
            res.w = (short)(a.w - b);
            res.x2 =(short)( a.x2 - b);
            res.y2 =(short)( a.y2 - b);
            res.z2 =(short)( a.z2 - b);
            res.w2 =(short)( a.w2 - b);
            return res;
        }

        /// <summary>
        /// multiplication operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator*")]
        public static short8 operator *(short8 a, short8 b)
        {
            short8 res;
            res.x = (short)(a.x * b.x);
            res.y = (short)(a.y * b.y);
            res.z = (short)(a.z * b.z);
            res.w = (short)(a.w * b.w);
            res.x2 =(short)( a.x2 * b.x2);
            res.y2 =(short)( a.y2 * b.y2);
            res.z2 =(short)( a.z2 * b.z2);
            res.w2 =(short)( a.w2 * b.w2);
            return res;
        }

        /// <summary>
        /// multiplication operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator*")]
        public static short8 operator *(short a, short8 b)
        {
            short8 res;
            res.x = (short)(a * b.x);
            res.y = (short)(a * b.y);
            res.z = (short)(a * b.z);
            res.w = (short)(a * b.w);
            res.x2 =(short)( a * b.x2);
            res.y2 =(short)( a * b.y2);
            res.z2 =(short)( a * b.z2);
            res.w2 =(short)( a * b.w2);
            return res;
        }

        /// <summary>
        /// multiplication operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator*")]
        public static short8 operator *(short8 a, short b)
        {
            short8 res;
            res.x = (short)(a.x * b);
            res.y = (short)(a.y * b);
            res.z = (short)(a.z * b);
            res.w = (short)(a.w * b);
            res.x2 =(short)( a.x2 * b);
            res.y2 =(short)( a.y2 * b);
            res.z2 =(short)( a.z2 * b);
            res.w2 =(short)( a.w2 * b);
            return res;
        }

        /// <summary>
        /// division operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator/")]
        public static short8 operator /(short8 a, short8 b)
        {
            short8 res;
            res.x = (short)(a.x / b.x);
            res.y = (short)(a.y / b.y);
            res.z = (short)(a.z / b.z);
            res.w = (short)(a.w / b.w);
            res.x2 =(short)( a.x2 / b.x2);
            res.y2 =(short)( a.y2 / b.y2);
            res.z2 =(short)( a.z2 / b.z2);
            res.w2 =(short)( a.w2 / b.w2);
            return res;
        }

        /// <summary>
        /// division operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator/")]
        public static short8 operator /(short a, short8 b)
        {
            short8 res;
            res.x = (short)(a / b.x);
            res.y = (short)(a / b.y);
            res.z = (short)(a / b.z);
            res.w = (short)(a / b.w);
            res.x2 =(short)( a / b.x2);
            res.y2 =(short)( a / b.y2);
            res.z2 =(short)( a / b.z2);
            res.w2 =(short)( a / b.w2);
            return res;
        }

        /// <summary>
        /// division operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator/")]
        public static short8 operator /(short8 a, short b)
        {
            short8 res;
            res.x = (short)(a.x / b);
            res.y = (short)(a.y / b);
            res.z = (short)(a.z / b);
            res.w = (short)(a.w / b);
            res.x2 =(short)( a.x2 / b);
            res.y2 =(short)( a.y2 / b);
            res.z2 =(short)( a.z2 / b);
            res.w2 =(short)( a.w2 / b);
            return res;
        }

        /// <summary>
        /// bitwise AND operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator&")]
        public static short8 operator &(short8 a, short8 b)
        {
            short8 res;
            res.x = (short)(a.x & b.x);
            res.y = (short)(a.y & b.y);
            res.z = (short)(a.z & b.z);
            res.w = (short)(a.w & b.w);
            res.x2 = (short)(a.x2 & b.x2);
            res.y2 = (short)(a.y2 & b.y2);
            res.z2 = (short)(a.z2 & b.z2);
            res.w2 = (short)(a.w2 & b.w2);
            return res;
        }

        /// <summary>
        /// bitwise AND operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator&")]
        public static short8 operator &(short a, short8 b)
        {
            short8 res;
            res.x = (short)(a & b.x);
            res.y = (short)(a & b.y);
            res.z = (short)(a & b.z);
            res.w = (short)(a & b.w);
            res.x2 = (short)(a & b.x2);
            res.y2 = (short)(a & b.y2);
            res.z2 = (short)(a & b.z2);
            res.w2 = (short)(a & b.w2);
            return res;
        }

        /// <summary>
        /// bitwise AND operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator&")]
        public static short8 operator &(short8 a, short b)
        {
            short8 res;
            res.x = (short)(a.x & b);
            res.y = (short)(a.y & b);
            res.z = (short)(a.z & b);
            res.w = (short)(a.w & b);
            res.x2 = (short)(a.x2 & b);
            res.y2 = (short)(a.y2 & b);
            res.z2 = (short)(a.z2 & b);
            res.w2 = (short)(a.w2 & b);
            return res;
        }

        /// <summary>
        /// bitwise OR operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator|")]
        public static short8 operator |(short8 a, short8 b)
        {
            short8 res;
            res.x = (short)(a.x | b.x);
            res.y = (short)(a.y | b.y);
            res.z = (short)(a.z | b.z);
            res.w = (short)(a.w | b.w);
            res.x2 = (short)(a.x2 | b.x2);
            res.y2 = (short)(a.y2 | b.y2);
            res.z2 = (short)(a.z2 | b.z2);
            res.w2 = (short)(a.w2 | b.w2);
            return res;
        }

        /// <summary>
        /// bitwise OR operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator|")]
        public static short8 operator |(short a, short8 b)
        {
            short8 res;
            res.x = (short)(a | b.x);
            res.y = (short)(a | b.y);
            res.z = (short)(a | b.z);
            res.w = (short)(a | b.w);
            res.x2 = (short)(a | b.x2);
            res.y2 = (short)(a | b.y2);
            res.z2 = (short)(a | b.z2);
            res.w2 = (short)(a | b.w2);
            return res;
        }

        /// <summary>
        /// bitwise OR operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator|")]
        public static short8 operator |(short8 a, short b)
        {
            short8 res;
            res.x = (short)(a.x | b);
            res.y = (short)(a.y | b);
            res.z = (short)(a.z | b);
            res.w = (short)(a.w | b);
            res.x2 = (short)(a.x2 | b);
            res.y2 = (short)(a.y2 | b);
            res.z2 = (short)(a.z2 | b);
            res.w2 = (short)(a.w2 | b);
            return res;
        }

        /// <summary>
        /// bitwise XOR operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator^")]
        public static short8 operator ^(short8 a, short8 b)
        {
            short8 res;
            res.x = (short)(a.x ^ b.x);
            res.y = (short)(a.y ^ b.y);
            res.z = (short)(a.z ^ b.z);
            res.w = (short)(a.w ^ b.w);
            res.x2 = (short)(a.x2 ^ b.x2);
            res.y2 = (short)(a.y2 ^ b.y2);
            res.z2 = (short)(a.z2 ^ b.z2);
            res.w2 = (short)(a.w2 ^ b.w2);
            return res;
        }

        /// <summary>
        /// bitwise XOR operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator^")]
        public static short8 operator ^(short a, short8 b)
        {
            short8 res;
            res.x = (short)(a ^ b.x);
            res.y = (short)(a ^ b.y);
            res.z = (short)(a ^ b.z);
            res.w = (short)(a ^ b.w);
            res.x2 = (short)(a ^ b.x2);
            res.y2 = (short)(a ^ b.y2);
            res.z2 = (short)(a ^ b.z2);
            res.w2 = (short)(a ^ b.w2);
            return res;
        }
        
        /// <summary>
        /// bitwise XOR operator
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="operator^")]
        public static short8 operator ^(short8 a, short b)
        {
            short8 res;
            res.x = (short)(a.x ^ b);
            res.y = (short)(a.y ^ b);
            res.z = (short)(a.z ^ b);
            res.w = (short)(a.w ^ b);
            res.x2 = (short)(a.x2 ^ b);
            res.y2 = (short)(a.y2 ^ b);
            res.z2 = (short)(a.z2 ^ b);
            res.w2 = (short)(a.w2 ^ b);
            return res;
        }

        /// <summary>
        /// loads from memory
        /// </summary>
        /// <param name="ptr"></param>
        /// <param name="alignment">has to be a compile time constant</param>
        [IntrinsicFunction(IsNaked=true, Name="__hybridizer_load_short8")]
        public unsafe static short8 Load(short8* ptr, int alignment)
        {
            return *ptr;
        }

        /// <summary>
        /// stores in memory
        /// </summary>
        /// <param name="ptr">destination pointer</param>
        /// <param name="val">value to store</param>
        /// <param name="alignment">has to be a compile time constant</param>
        [IntrinsicFunction(IsNaked=true, Name="__hybridizer_store_short8")]
        public unsafe static void Store(short8* ptr, short8 val, int alignment)
        {
            *ptr = val;
        }
    }

    /// <summary>
    /// 
    /// </summary>
    public class LLVMVectorIntrinsics {
        /// <summary>
        /// The ‘insertelement‘ instruction inserts a scalar element into a vector at a specified index.
        /// Documentation <see href="https://llvm.org/docs/LangRef.html#insertelement-instruction">here</see>
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="hybridizer::insertElement<int8>")]
        public unsafe static int8 InsertElement(int8 vector, int valueToInsert, int index)
        {
            int8 result = new int8(vector);
            int* ptr = (int*)&result;
            ptr[index] = valueToInsert;
            return result;
        }

        /// <summary>
        /// The ‘insertelement‘ instruction inserts a scalar element into a vector at a specified index.
        /// Documentation <see href="https://llvm.org/docs/LangRef.html#insertelement-instruction">here</see>
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="hybridizer::insertElement<float8>")]
        public unsafe static float8 InsertElement(float8 vector, float valueToInsert, int index)
        {
            float8 result = new float8(vector);
            float* ptr = (float*)&result;
            ptr[index] = valueToInsert;
            return result;
        }

        /// <summary>
        /// The ‘insertelement‘ instruction inserts a scalar element into a vector at a specified index.
        /// Documentation <see href="https://llvm.org/docs/LangRef.html#insertelement-instruction">here</see>
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="hybridizer::insertElement<float4>")]
        public unsafe static float4 InsertElement(float4 vector, float valueToInsert, int index)
        {
            float4 result = new float4(vector);
            float* ptr = (float*)&result;
            ptr[index] = valueToInsert;
            return result;
        }

        /// <summary>
        /// The ‘insertelement‘ instruction inserts a scalar element into a vector at a specified index.
        /// Documentation <see href="https://llvm.org/docs/LangRef.html#insertelement-instruction">here</see>
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="hybridizer::insertElement<double2>")]
        public unsafe static double2 InsertElement(double2 vector, double valueToInsert, int index)
        {
            double2 result = new double2(vector);
            double* ptr = (double*)&result;
            ptr[index] = valueToInsert;
            return result;
        }

        /// <summary>
        /// The ‘insertelement‘ instruction inserts a scalar element into a vector at a specified index.
        /// Documentation <see href="https://llvm.org/docs/LangRef.html#insertelement-instruction">here</see>
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="hybridizer::insertElement<long2>")]
        public unsafe static long2 InsertElement(long2 vector, long valueToInsert, int index)
        {
            long2 result = new long2(vector);
            long* ptr = (long*)&result;
            ptr[index] = valueToInsert;
            return result;
        }

        /// <summary>
        /// The ‘extractelement‘ instruction extracts a single scalar element from a vector at a specified index.
        /// Documentation <see href="https://llvm.org/docs/LangRef.html#extractelement-instruction">here</see>
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="hybridizer::extractElement<int8, int>")]
        public unsafe static int ExtractElement(int8 vector, int index)
        {
            int* ptr = (int*)&vector;
            return ptr[index];
        }

        /// <summary>
        /// The ‘extractelement‘ instruction extracts a single scalar element from a vector at a specified index.
        /// Documentation <see href="https://llvm.org/docs/LangRef.html#extractelement-instruction">here</see>
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="hybridizer::extractElement<float8, float>")]
        public unsafe static float ExtractElement(float8 vector, int index)
        {
            float* ptr = (float*)&vector;
            return ptr[index];
        }

        /// <summary>
        /// The ‘extractelement‘ instruction extracts a single scalar element from a vector at a specified index.
        /// Documentation <see href="https://llvm.org/docs/LangRef.html#extractelement-instruction">here</see>
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="hybridizer::extractElement<float4, float>")]
        public unsafe static float ExtractElement(float4 vector, int index)
        {
            float* ptr = (float*)&vector;
            return ptr[index];
        }

        /// <summary>
        /// The ‘extractelement‘ instruction extracts a single scalar element from a vector at a specified index.
        /// Documentation <see href="https://llvm.org/docs/LangRef.html#extractelement-instruction">here</see>
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="hybridizer::extractElement<double2, double>")]
        public unsafe static double ExtractElement(double2 vector, int index)
        {
            double* ptr = (double*)&vector;
            return ptr[index];
        }

        /// <summary>
        /// The ‘extractelement‘ instruction extracts a single scalar element from a vector at a specified index.
        /// Documentation <see href="https://llvm.org/docs/LangRef.html#extractelement-instruction">here</see>
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="hybridizer::extractElement<long2, float>")]
        public unsafe static long ExtractElement(long2 vector, int index)
        {
            long* ptr = (long*)&vector;
            return ptr[index];
        }

        /// <summary>
        /// The ‘shufflevector‘ instruction constructs a permutation of elements from two input vectors, returning a vector with the same element type as the input and length that is the same as the shuffle mask.
        /// Documentation <see href="https://llvm.org/docs/LangRef.html#shufflevector-instruction">here</see>
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="hybridizer::shuffleVector<int8, int8, int8>")]
        public unsafe static int8 ShuffleVector(int8 left, int8 right, int8 mask)
        {
            int8 res;
            int* resptr = (int*)&res;
            int* leftptr = (int*)&left;
            int* rightptr = (int*)&right;
            int* maskptr = (int*)&mask;
            for (int i = 0; i < 8; ++i)
            {
                int index = maskptr[i];
                if(index >= 0 && index < 8) {
                    resptr[i] = leftptr[index];
                }
                else if (index >= 0) 
                {
                    resptr[i] = leftptr[index - 8];
                }
            }

            return res;
        }

        /// <summary>
        /// The ‘shufflevector‘ instruction constructs a permutation of elements from two input vectors, returning a vector with the same element type as the input and length that is the same as the shuffle mask.
        /// Documentation <see href="https://llvm.org/docs/LangRef.html#shufflevector-instruction">here</see>
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="hybridizer::shuffleVector<float8, float8, int8>")]
        public unsafe static float8 ShuffleVector(float8 left, float8 right, int8 mask)
        {
            float8 res;
            float* resptr = (float*)&res;
            float* leftptr = (float*)&left;
            float* rightptr = (float*)&right;
            int* maskptr = (int*)&mask;
            for (int i = 0; i < 8; ++i)
            {
                int index = maskptr[i];
                if (index >= 0 && index < 8)
                {
                    resptr[i] = leftptr[index];
                }
                else if (index >= 0)
                {
                    resptr[i] = leftptr[index - 8];
                }
            }

            return res;
        }

        /// <summary>
        /// The ‘shufflevector‘ instruction constructs a permutation of elements from two input vectors, returning a vector with the same element type as the input and length that is the same as the shuffle mask.
        /// Documentation <see href="https://llvm.org/docs/LangRef.html#shufflevector-instruction">here</see>
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="hybridizer::shuffleVector<long2, long2, int2>")]
        public unsafe static long2 ShuffleVector(long2 left, long2 right, int2 mask)
        {
            long2 res;
            long* resptr = (long*)&res;
            long* leftptr = (long*)&left;
            long* rightptr = (long*)&right;
            int* maskptr = (int*)&mask;
            for (int i = 0; i < 2; ++i)
            {
                int index = maskptr[i];
                if (index >= 0 && index < 2)
                {
                    resptr[i] = leftptr[index];
                }
                else if (index >= 0)
                {
                    resptr[i] = leftptr[index - 2];
                }
            }

            return res;
        }

        /// <summary>
        /// The ‘shufflevector‘ instruction constructs a permutation of elements from two input vectors, returning a vector with the same element type as the input and length that is the same as the shuffle mask.
        /// Documentation <see href="https://llvm.org/docs/LangRef.html#shufflevector-instruction">here</see>
        /// </summary>
        [IntrinsicFunction(IsNaked=true, Name="hybridizer::shuffleVector<double2, double2, int2>")]
        public unsafe static double2 ShuffleVector(double2 left, double2 right, int2 mask)
        {
            double2 res;
            double* resptr = (double*)&res;
            double* leftptr = (double*)&left;
            double* rightptr = (double*)&right;
            int* maskptr = (int*)&mask;
            for (int i = 0; i < 2; ++i)
            {
                int index = maskptr[i];
                if (index >= 0 && index < 2)
                {
                    resptr[i] = leftptr[index];
                }
                else if (index >= 0)
                {
                    resptr[i] = leftptr[index - 2];
                }
            }

            return res;
        }
    }
}

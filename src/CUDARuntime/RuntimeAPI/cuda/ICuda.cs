/* (c) ALTIMESH 2018 -- all rights reserved */
using System;
using System.Collections.Generic;
using System.Configuration;
using System.Diagnostics;
using System.Text;
using System.Runtime.InteropServices;
using System.Reflection;

namespace Hybridizer.Runtime.CUDAImports
{   
	/// <summary>
	/// interface wrapping all cuda versions
	/// </summary>
	public interface ICuda : ICudaMarshalling
	{
		#region Device Management

		/// <summary>
		///  Set a list of devices that can be used for CUDA
		/// </summary>
		cudaError_t SetValidDevices(int[] devs);
		/// <summary>
		///  Select compute-device which best matches criteria. 
		/// </summary>
		cudaError_t ChooseDevice(out int device, ref cudaDeviceProp prop);
		/// <summary>
		///  Returns information about the device. 
		/// </summary>
		cudaError_t DeviceGetAttribute(out int value, cudaDeviceAttr attr, int device);
		/// <summary>
		///  Returns a handle to a compute device. 
		/// </summary>
		cudaError_t DeviceGetByPCIBusId(out int device, string pciBusId);
		/// <summary>
		///  Returns the preferred cache configuration for the current device. 
		/// </summary>
		cudaError_t DeviceGetCacheConfig(IntPtr /* cudaFuncCache ** */ pCacheConfig);
		/// <summary>
		///  Returns resource limits. 
		/// </summary>
		cudaError_t DeviceGetLimit(out size_t pValue, cudaLimit limit);
		/// <summary>
		///  Returns a PCI Bus Id string for the device. 
		/// </summary>
		cudaError_t DeviceGetPCIBusId(StringBuilder pciBusId, int len, int device);
		/// <summary>
		///  Returns the shared memory configuration for the current device. 
		/// </summary>
		cudaError_t DeviceGetSharedMemConfig(IntPtr /* cudaSharedMemConfig ** */ pConfig);
		/// <summary>
		///  Destroy all allocations and reset all state on the current device in the current process. 
		/// </summary>
		cudaError_t DeviceReset();
		/// <summary>
		///  Sets the preferred cache configuration for the current device. 
		/// </summary>
		cudaError_t DeviceSetCacheConfig(cudaFuncCache cacheConfig);
		/// <summary>
		///  Set resource limits. 
		/// </summary>
		cudaError_t DeviceSetLimit(cudaLimit limit, size_t value);
		/// <summary>
		///  Sets the shared memory configuration for the current device. 
		/// </summary>
		cudaError_t DeviceSetSharedMemConfig(cudaSharedMemConfig config);
		/// <summary>
		///  Wait for compute device to finish. 
		/// </summary>
		cudaError_t DeviceSynchronize();
		/// <summary>
		///  Returns which device is currently being used. 
		/// </summary>
		cudaError_t GetDevice(out int device);
		/// <summary>
		///  Returns the number of compute-capable devices. 
		/// </summary>
		cudaError_t GetDeviceCount(out int count);
		/// <summary>
		///  Returns information about the compute-device. 
		/// </summary>
		cudaError_t GetDeviceProperties(out cudaDeviceProp prop, int device);
		/// <summary>
		///  Close memory mapped with cudaIpcOpenMemHandle. 
		/// </summary>
		cudaError_t IpcCloseMemHandle(IntPtr devPtr);
		/// <summary>
		///  Gets an interprocess handle for a previously allocated event. 
		/// </summary>
		cudaError_t IpcGetEventHandle(out cudaIpcEventHandle_t handle, cudaEvent_t evt);
		/// <summary>
		///   Gets an interprocess memory handle for an existing device memory allocation
		/// </summary>
		cudaError_t IpcGetMemHandle(out cudaIpcMemHandle_t handle, IntPtr devPtr);
		/// <summary>
		///  Opens an interprocess event handle for use in the current process. 
		/// </summary>
		cudaError_t IpcOpenEventHandle(out cudaEvent_t evt, cudaIpcEventHandle_t handle);
		/// <summary>
		///  Opens an interprocess memory handle exported from another process and returns a device pointer usable in the local process.
		/// </summary>
		cudaError_t IpcOpenMemHandle(out IntPtr devPtr, cudaIpcMemHandle_t handle, uint flags);
		/// <summary>
		///  Set device to be used for GPU executions. 
		/// </summary>
		cudaError_t SetDevice(int device);
		/// <summary>
		///  Sets flags to be used for device executions. 
		/// </summary>
		cudaError_t SetDeviceFlags(deviceFlags flags);
		/// <summary>
		///  Gets the flags for the current device
		/// </summary>
		cudaError_t GetDeviceFlags(out uint flags);
		/// <summary>
		///  Returns numerical values that correspond to the least and greatest stream priorities.
		/// </summary>
		cudaError_t DeviceGetStreamPriorityRange(out int leastPriority, out int greatestPriority);
		/// <summary>
		///  Queries attributes of the link between two devices.
		/// </summary>
		cudaError_t DeviceGetP2PAttribute(out int value, cudaDeviceP2PAttr attr, int srcDevice, int dstDevice);

		#endregion

		#region Thread Management

		/// <summary>
		/// Exit and clean up from CUDA launches
		/// </summary>
		[Obsolete]
		cudaError_t ThreadExit() ;
		/// <summary>
		/// Returns the preferred cache configuration for the current device.
		/// </summary>
		[Obsolete]
		cudaError_t ThreadGetLimit(out size_t value, cudaLimit limit) ;
		/// <summary>
		/// Set resource limits
		/// </summary>
		[Obsolete]
		cudaError_t ThreadSetLimit(cudaLimit limit, size_t value) ;
		/// <summary>
		///  Wait for compute device to finish
		/// </summary>
		/// <returns></returns>
		[Obsolete]
		cudaError_t ThreadSynchronize() ;

		#endregion

		#region Stream Management

		/// <summary>
		/// Queries an asynchronous stream for completion status
		/// </summary>
		cudaError_t StreamQuery(cudaStream_t stream) ;
		/// <summary>
		/// Query the flags of a stream
		/// </summary>
		cudaError_t StreamGetFlags(cudaStream_t hStream, out uint flags);
		/// <summary>
		/// Create an asynchronous stream
		/// </summary>
		cudaError_t StreamCreateWithFlags(out cudaStream_t pStream, uint flags);
		/// <summary>
		/// Attach memory to a stream asynchronously
		/// </summary>
		cudaError_t StreamAttachMemAsync(cudaStream_t stream, IntPtr devPtr, size_t length, uint flags);
		/// <summary>
		/// Make a compute stream wait on an event
		/// </summary>
		cudaError_t StreamWaitEvent(cudaStream_t stream, cudaEvent_t evt, uint flags);
		/// <summary>
		/// Create an asynchronous stream with the specified priority
		/// </summary>
		cudaError_t StreamCreateWithPriority(out cudaStream_t pStream, uint flags, int priority);
		/// <summary>
		/// Query the priority of a stream
		/// </summary>
		cudaError_t StreamGetPriority(cudaStream_t hStream, out int priority);

		#endregion

		#region Event Management

		/// <summary>
		/// Creates an event object
		/// </summary>
		cudaError_t EventCreate(out cudaEvent_t evt);
		/// <summary>
		/// Creates an event object with the specified flags
		/// </summary>
		cudaError_t EventCreateWithFlags(out cudaEvent_t evt, cudaEventFlags flags);
		/// <summary>
		/// Destroys an event object
		/// </summary>
		cudaError_t EventDestroy(cudaEvent_t evt);
		/// <summary>
		/// Computes the elapsed time between events
		/// </summary>
		cudaError_t EventElapsedTime(out float ms, cudaEvent_t start, cudaEvent_t stop);
		/// <summary>
		/// Queries an event's status
		/// </summary>
		cudaError_t EventQuery(cudaEvent_t evt);
		/// <summary>
		/// Records an event
		/// </summary>
		cudaError_t EventRecord(cudaEvent_t evt, cudaStream_t stream);
		/// <summary>
		/// Waits for an event to complete
		/// </summary>
		cudaError_t EventSynchronize(cudaEvent_t evt);

		#endregion

		#region Execution Control

		/// <summary>
		/// Configure a device-launch
		/// </summary>
		cudaError_t ConfigureCall(dim3 gridDim, dim3 blockDim, size_t sharedMemory, cudaStream_t stream) ;
		/// <summary>
		/// Find out attributes for a given function. 
		/// </summary>
		cudaError_t FuncGetAttributes(out cudaFuncAttributes attr, string func) ;
		/// <summary>
		/// Sets the preferred cache configuration for a device function. 
		/// </summary>
		cudaError_t FuncSetCacheConfig(IntPtr func, cudaFuncCache cacheConfig) ;
		/// <summary>
		/// Launches a device function. 
		/// </summary>
		cudaError_t Launch(string func) ;
		/// <summary>
		/// Converts a double argument to be executed on a device. 
		/// </summary>
		[Obsolete]
		cudaError_t SetDoubleForDevice(ref double d);
		/// <summary>
		/// Converts a double argument after execution on a device. 
		/// </summary>
		[Obsolete]
		cudaError_t SetDoubleForHost(ref double d) ;
		/// <summary>
		/// Configure a device launch. 
		/// </summary>
		[Obsolete]
		cudaError_t SetupArgument(IntPtr arg, size_t size, size_t offset) ;
		/// <summary>
		/// Launches a device function. 
		/// </summary>
		cudaError_t LaunchKernel(IntPtr func, dim3 gridDim, dim3 blockDim, IntPtr args, size_t sharedMem, cudaStream_t stream);
		/// <summary>
		/// Sets the shared memory configuration for a device function. 
		/// </summary>
		cudaError_t FuncSetSharedMemConfig(IntPtr func, cudaSharedMemConfig config);

		#endregion

		#region Memory Management

		/// <summary>
		/// Frees an array on the device
		/// </summary>
		cudaError_t FreeArray (cudaArray_t arr) ;
		/// <summary>
		/// Finds the address associated with a CUDA symbol
		/// </summary>
		cudaError_t GetSymbolAddress (out IntPtr devPtr, string symbol) ;
		/// <summary>
		/// Finds the size of the object associated with a CUDA symbol
		/// </summary>
		cudaError_t GetSymbolSize (out size_t size, string symbol);
		/// <summary>
		/// Passes back device pointer of mapped host memory allocated by cudaHostAlloc or registered by cudaHostRegister
		/// </summary>
		cudaError_t HostGetDevicePointer (out IntPtr pdev, IntPtr phost, cudaGetDevicePointerFlags flags);
		/// <summary>
		/// Passes back flags used to allocate pinned host memory allocated by cudaHostAlloc
		/// </summary>
		cudaError_t HostGetFlags (out cudaHostAllocFlags flags, IntPtr phost) ;
		/// <summary>
		/// Allocates logical 1D, 2D, or 3D memory objects on the device
		/// </summary>
		cudaError_t Malloc3D (ref cudaPitchedPtr ptr, cudaFuncAttributes extent) ;
		/// <summary>
		/// Allocate an array on the device
		/// </summary>
		cudaError_t Malloc3DArray (out cudaArray_t arr, ref cudaChannelFormatDesc chan, cudaFuncAttributes extent, cudaMallocArrayFlags flags) ;
		/// <summary>
		/// Allocate an array on the device
		/// </summary>
		cudaError_t MallocArray (out cudaArray_t arr, ref cudaChannelFormatDesc chan, size_t width, size_t height, cudaMallocArrayFlags flags) ;
		/// <summary>
		/// Allocates page-locked memory on the host
		/// </summary>
		cudaError_t MallocHost (out IntPtr ptr, size_t size) ;
		/// <summary>
		/// Allocates pitched memory on the device
		/// </summary>
		cudaError_t MallocPitch (out IntPtr dptr, out size_t pitch, size_t width, size_t height) ;
		/// <summary>
		/// Copies data between host and device
		/// </summary>
		cudaError_t Memcpy2D (IntPtr dest, size_t dpitch, IntPtr src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind) ;
		/// <summary>
		/// Copies data between host and device
		/// </summary>
		cudaError_t Memcpy2DArrayToArray (cudaArray_t dest, size_t wOffsetDest, size_t hOffsetDest, cudaArray_t src, size_t wOffsetSrc, size_t hOffsetSrc, size_t width, size_t height, cudaMemcpyKind kind) ;
		/// <summary>
		/// Copies data between host and device
		/// </summary>
		cudaError_t Memcpy2DAsync (IntPtr dest, size_t dpitch, IntPtr src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind, cudaStream_t stream);
		/// <summary>
		/// Copies data between host and device
		/// </summary>
		cudaError_t Memcpy2DFromArray (IntPtr dest, size_t dpitch, cudaArray_t src, size_t wOffset, size_t hOffset, size_t width, size_t height, cudaMemcpyKind kind);
		/// <summary>
		/// Copies data between host and device
		/// </summary>
		cudaError_t Memcpy2DFromArrayAsync (IntPtr dest, size_t dpitch, cudaArray_t src, size_t wOffset, size_t hOffset, size_t width, size_t height, cudaMemcpyKind kind, cudaStream_t stream);
		/// <summary>
		/// Copies data between host and device
		/// </summary>
		cudaError_t Memcpy2DToArray (cudaArray_t dest, size_t wOffsetDest, size_t hOffsetDest, IntPtr src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind);
		/// <summary>
		/// Copies data between host and device
		/// </summary>
		cudaError_t Memcpy2DToArrayAsync (cudaArray_t dest, size_t wOffsetDest, size_t hOffsetDest, IntPtr src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind, cudaStream_t stream);
		/// <summary>
		/// Copies data between host and device
		/// </summary>
		cudaError_t Memcpy3D (ref cudaMemcpy3DParms par);
		/// <summary>
		/// Copies data between host and device
		/// </summary>
		cudaError_t Memcpy3DAsync (ref cudaMemcpy3DParms par, cudaStream_t stream);
		/// <summary>
		/// Copies data between host and device
		/// </summary>
		cudaError_t MemcpyArrayToArray (cudaArray_t dest, size_t wOffsetDst, size_t hOffsetDst, cudaArray_t src, size_t wOffsetSrc, size_t hOffsetSrc, size_t count, cudaMemcpyKind kind);
		/// <summary>
		/// Copies data between host and device
		/// </summary>
		cudaError_t MemcpyFromArray (IntPtr dest, cudaArray_t src, size_t wOffset, size_t hOffset, size_t count, cudaMemcpyKind kind);
		/// <summary>
		/// Copies data between host and device
		/// </summary>
		cudaError_t MemcpyFromArrayAsync (IntPtr dest, cudaArray_t src, size_t wOffset, size_t hOffset, size_t count, cudaMemcpyKind kind, cudaStream_t stream);
		/// <summary>
		/// Copies data between host and device
		/// </summary>
		cudaError_t MemcpyFromSymbol (IntPtr dest, string symbol, size_t count, size_t offset, cudaMemcpyKind kind);
		/// <summary>
		/// Copies data between host and device
		/// </summary>
		cudaError_t MemcpyFromSymbolAsync (IntPtr dest, string symbol, size_t count, size_t offset, cudaMemcpyKind kind, cudaStream_t stream);
		/// <summary>
		/// Copies data between host and device
		/// </summary>
		cudaError_t MemcpyToArray (cudaArray_t dest, size_t wOffset, size_t hOffset, IntPtr src, size_t count, cudaMemcpyKind kind);
		/// <summary>
		/// Copies data between host and device
		/// </summary>
		cudaError_t MemcpyToArrayAsync (cudaArray_t dest, size_t wOffset, size_t hOffset, IntPtr src, size_t count, cudaMemcpyKind kind, cudaStream_t stream);
		/// <summary>
		/// Copies data between host and device
		/// </summary>
		cudaError_t MemcpyToSymbol (string symbol, IntPtr src, size_t count, size_t offset, cudaMemcpyKind kind);
		/// <summary>
		/// Copies data between host and device
		/// </summary>
		cudaError_t MemcpyToSymbolAsync (string symbol, IntPtr src, size_t count, size_t offset, cudaMemcpyKind kind, cudaStream_t stream);
		/// <summary>
		/// Gets free and total device memory
		/// </summary>
		cudaError_t MemGetInfo (out size_t free, out size_t total) ;
		/// <summary>
		/// Initializes or sets device memory to a value
		/// </summary>
		cudaError_t Memset (IntPtr devPtr, int value, size_t count);
		/// <summary>
		/// Initializes or sets device memory to a value
		/// </summary>
		cudaError_t Memset2D (IntPtr devPtr, size_t pitch, int value, size_t width, size_t height);
		/// <summary>
		/// Initializes or sets device memory to a value
		/// </summary>
		cudaError_t Memset3D (cudaPitchedPtr devPtr, int value, cudaFuncAttributes extent) ;
		/// <summary>
		/// Gets info about the specified cudaArray
		/// </summary>
		cudaError_t ArrayGetInfo(out cudaChannelFormatDesc desc, out cudaFuncAttributes extent, out uint flags, cudaArray_t array);
		/// <summary>
		/// Frees a mipmapped array on the device
		/// </summary>
		cudaError_t FreeMipmappedArray(cudaMipmappedArray_t mipmappedArray);
		/// <summary>
		/// Gets a mipmap level of a CUDA mipmapped array
		/// </summary>
		cudaError_t GetMipmappedArrayLevel(out cudaArray_t levelArray, cudaMipmappedArray_const_t mipmappedArray, uint level);
		/// <summary>
		///  Allocates memory that will be automatically managed by the Unified Memory system
		/// </summary>
		cudaError_t MallocManaged(out IntPtr devPtr, size_t size, uint flags);
		/// <summary>
		/// Allocate a mipmapped array on the device
		/// </summary>
		cudaError_t MallocMipmappedArray(out cudaMipmappedArray_t mipmappedArray, ref cudaChannelFormatDesc desc, cudaFuncAttributes extent, uint numLevels, uint flags);
		/// <summary>
		/// Advise about the usage of a given memory range
		/// </summary>
		cudaError_t MemAdvise(IntPtr devptr, size_t count, cudaMemmoryAdvise advice, int device);
		/// <summary>
		/// Prefetches memory to the specified destination device
		/// </summary>
		cudaError_t MemPrefetchAsync(IntPtr devptr, size_t count, int dstDevice, cudaStream_t stream);
		/// <summary>
		/// Prefetches memory to the specified destination device
		/// </summary>
		cudaError_t MemPrefetchAsync(IntPtr devptr, size_t count, int dstDevice);
		/// <summary>
		/// Copies memory between devices
		/// </summary>
		cudaError_t Memcpy3DPeer(ref cudaMemcpy3DPeerParms par);
		/// <summary>
		/// Copies memory between devices
		/// </summary>
		cudaError_t Memcpy3DPeerAsync(ref cudaMemcpy3DPeerParms par, cudaStream_t stream);
		/// <summary>
		/// Copies memory between devices
		/// </summary>
		cudaError_t MemcpyPeer(IntPtr dest, int dstDevice, IntPtr src, int srcDevice, size_t count);
		/// <summary>
		/// Copies memory between devices
		/// </summary>
		cudaError_t MemcpyPeerAsync(IntPtr dest, int dstDevice, IntPtr src, int srcDevice, size_t count, cudaStream_t stream);
		/// <summary>
		/// Copies memory between devices
		/// </summary>
		cudaError_t Memset2DAsync(IntPtr devPtr, size_t pitch, int value, size_t width, size_t height, cudaStream_t stream);
		/// <summary>
		/// Copies memory between devices
		/// </summary>
		cudaError_t Memset3DAsync(cudaPitchedPtr devPtr, int value, cudaFuncAttributes extent, cudaStream_t stream);
		/// <summary>
		/// Copies memory between devices
		/// </summary>
		cudaError_t MemsetAsync(IntPtr devPtr, int value, size_t count, cudaStream_t stream);

		#endregion

		#region Surface Management

		/// <summary>
		/// Creates a surface object
		/// </summary>
		cudaError_t CreateSurfaceObject(out cudaSurfaceObject_t surface, ref cudaResourceDesc resDesc) ;
		/// <summary>
		/// Destroys a surface object
		/// </summary>
		cudaError_t DestroySurfaceObject(cudaSurfaceObject_t surface) ;
		/// <summary>
		///  Returns a surface object's resource descriptor
		/// </summary>
		cudaError_t GetSurfaceObjectResourceDesc(out cudaResourceDesc resDesc, cudaSurfaceObject_t surface) ;

		#endregion        

		#region Texture Management

		/// <summary>
		/// Creates a texture object
		/// </summary>
		cudaError_t CreateTextureObject(out cudaTextureObject_t texture, ref cudaResourceDesc resDesc, ref cudaTextureDesc texDesc, ref cudaResourceViewDesc ResViewDesc);
		/// <summary>
		/// Creates a texture object
		/// </summary>
		cudaError_t CreateTextureObject(out cudaTextureObject_t texture, ref cudaResourceDesc resDesc, ref cudaTextureDesc texDesc);
		/// <summary>
		/// Destroys a texture object
		/// </summary>
		cudaError_t DestroyTextureObject(cudaTextureObject_t texture);
		/// <summary>
		/// 
		/// </summary>
		cudaError_t GetTextureObjectResourceDesc(out cudaResourceDesc resDesc, cudaTextureObject_t texture);

		#endregion

		#region OPENGL interop
		/// <summary>
		/// Registers a buffer object for access by CUDA. 
		/// </summary>
		/// <param name="buffer">Buffer object ID to register</param>
		/// <see href="https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__OPENGL__DEPRECATED.html#group__CUDART__OPENGL__DEPRECATED_1gb835a92a340e999f4eaa55a8d57e122c">nvidia documentation</see>
		cudaError_t GLRegisterBufferObject(uint buffer);
		/// <summary>
		/// Registers an OpenGL buffer object. 
		/// </summary>
		/// <param name="pCudaResource"> Pointer to the returned object handle </param>
		/// <param name="buffer">name of buffer object to be registered</param>
		/// <param name="Flags">Register flags</param>
		/// <see href="https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__OPENGL.html#group__CUDART__OPENGL_1g0fd33bea77ca7b1e69d1619caf44214b">nvidia documentation</see>
		cudaError_t GraphicsGLRegisterBuffer(out IntPtr pCudaResource, uint buffer, uint Flags);
		/// <summary>
		/// Unregisters a graphics resource for access by CUDA. 
		/// </summary>
		/// <param name="resource">Resource to unregister</param>
		/// <see href="https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__INTEROP.html#group__CUDART__INTEROP_1gc65d1f2900086747de1e57301d709940">nvidia documentation</see>
		cudaError_t GraphicsUnregisterResource(IntPtr resource);

		/// <summary>
		/// Unmaps a buffer object for access by CUDA. 
		/// </summary>
		/// <param name="buffer">Buffer object to unmap </param>
		/// <see href="https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__OPENGL__DEPRECATED.html#group__CUDART__OPENGL__DEPRECATED_1g5ce0566e8543a8c7677b619acfefd5b5">nvidia documentation</see>
		cudaError_t GLUnregisterBufferObject(uint buffer);
		/// <summary>
		/// Get an device pointer through which to access a mapped graphics resource. 
		/// </summary>
		/// <param name="devPtr"> Returned pointer through which resource may be accessed </param>
		/// <param name="size"> Returned size of the buffer accessible starting at *devPtr</param>
		/// <param name="resource">Mapped resource to access</param>
		/// <see href="https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__INTEROP.html#group__CUDART__INTEROP_1ga36881081c8deb4df25c256158e1ac99">nvidia documentation</see>
		cudaError_t GraphicsResourceGetMappedPointer(out IntPtr devPtr, out size_t size, IntPtr resource);
		/// <summary>
		/// Set usage flags for mapping a graphics resource. 
		/// </summary>
		/// <param name="resource">Registered resource to set flags for</param>
		/// <param name="flags"> Parameters for resource mapping</param>
		/// <see href="https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__INTEROP.html#group__CUDART__INTEROP_1g5f94a0043909fddc100ab5f0c2476b9f">nvidia documentation</see>
		cudaError_t GraphicsResourceSetMapFlags(IntPtr resource, uint flags);
		/// <summary>
		/// Map graphics resources for access by CUDA. 
		/// </summary>
		/// <param name="count"> Number of resources to map </param>
		/// <param name="resources">Resources to map for CUDA </param>
		/// <param name="stream">Stream for synchronization</param>
		/// <see href="https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__INTEROP.html#group__CUDART__INTEROP_1gad8fbe74d02adefb8e7efb4971ee6322">nvidia documentation</see>
		cudaError_t GraphicsMapResources(int count, IntPtr[] resources, cudaStream_t stream);
		/// <summary>
		/// Unmap graphics resources. 
		/// </summary>
		/// <param name="count"> Number of resources to map </param>
		/// <param name="resources">Resources to map for CUDA </param>
		/// <param name="stream">Stream for synchronization</param>
		/// <see href="https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__INTEROP.html#group__CUDART__INTEROP_1g11988ab4431b11ddb7cbde7aedb60491">nvidia documentation</see>
		cudaError_t GraphicsUnmapResources(int count, IntPtr[] resources, cudaStream_t stream);

		/// <summary>
		/// Register an OpenGL texture or renderbuffer object.
		/// </summary>
		/// <param name="cudaGraphicsResource">Pointer to the returned object handle </param>
		/// <param name="image">name of texture or renderbuffer object to be registered</param>
		/// <param name="target">Identifies the type of object specified by image</param>
		/// <param name="flags">Register flags</param>
		/// <returns>cudaErrorInvalidDevice, cudaErrorInvalidValue, cudaErrorInvalidResourceHandle, cudaErrorUnknown</returns>
		/// <see href="https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__OPENGL.html#group__CUDART__OPENGL_1g80d12187ae7590807c7676697d9fe03d">nvidia documentation</see>
		cudaError_t GraphicsGLRegisterImage(out IntPtr cudaGraphicsResource, uint image, uint target, uint flags);

		/// <summary>
		/// Get an array through which to access a subresource of a mapped graphics resource. 
		/// </summary>
		/// <param name="array">Returned array through which a subresource of resource may be accessed</param>
		/// <param name="resource"> Mapped resource to access </param>
		/// <param name="arrayIndex">Array index for array textures or cubemap face index as defined by cudaGraphicsCubeFace for cubemap textures for the subresource to access </param>
		/// <param name="mipLevel">Mipmap level for the subresource to access</param>
		/// <returns>cudaSuccess, cudaErrorInvalidValue, cudaErrorInvalidResourceHandle, cudaErrorUnknown</returns>
		/// <see href="https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__INTEROP.html#group__CUDART__INTEROP_1g0dd6b5f024dfdcff5c28a08ef9958031">nvidia documentation</see>
		cudaError_t GraphicsSubResourceGetMappedArray(out cudaArray_t array, IntPtr resource, uint arrayIndex, uint mipLevel);

		#endregion

	}
}

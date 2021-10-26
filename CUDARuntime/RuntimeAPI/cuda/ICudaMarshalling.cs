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
	/// ICuda simplified for marshalling only
	/// </summary>
	public interface ICudaMarshalling
	{
		/// <summary>
		/// Free memory allocated on device
		/// </summary>
		/// <remarks> <see href="http://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY"/></remarks>
		/// <param name="dev">Device pointer to free</param>
		/// <returns>cudaSuccess, cudaErrorInvalidDevicePointer, cudaErrorInitializationError</returns>
		cudaError_t Free(IntPtr dev);
		/// <summary>
		/// Register host memory onto the device
		/// </summary>
		/// <remarks> <see href="http://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY"/></remarks>
		/// <param name="ptr">Host pointer to memory to page-lock </param>
		/// <param name="size">Size in bytes of the address range to page-lock in bytes </param>
		/// <param name="flags">Flags for allocation request</param>
		/// <returns>cudaSuccess, cudaErrorInvalidValue, cudaErrorMemoryAllocation</returns>
		cudaError_t HostRegister(IntPtr ptr, size_t size, uint flags);
		/// <summary>
		/// Unregister host memory
		/// </summary>
		/// <remarks> <see href="http://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY"/></remarks>
		/// <param name="ptr">Host pointer to memory to unregister</param>
		/// <returns>cudaSuccess, cudaErrorInvalidValue</returns>
		cudaError_t HostUnregister(IntPtr ptr);
		/// <summary>
		/// Allocate memory on the device
		/// </summary>
		/// <remarks> <see href="http://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY"/></remarks>
		/// <param name="dev">Pointer to allocated device memory</param>
		/// <param name="size">Requested allocation size in bytes</param>
		/// <returns>cudaSuccess, cudaErrorMemoryAllocation</returns>
		cudaError_t Malloc(out IntPtr dev, size_t size);
		/// <summary>
		/// Copies data between host and device
		/// </summary>
		/// <remarks> <see href="http://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY"/></remarks>
		/// <param name="dest">Destination memory address </param>
		/// <param name="src">Source memory address </param>
		/// <param name="size">Size in bytes to copy </param>
		/// <param name="kind">Type of transfer</param>
		/// <returns>cudaSuccess, cudaErrorInvalidValue, cudaErrorInvalidDevicePointer, cudaErrorInvalidMemcpyDirection</returns>
		cudaError_t Memcpy(IntPtr dest, IntPtr src, size_t size, cudaMemcpyKind kind);
		/// <summary>
		/// Copies data between host and device
		/// </summary>
		/// <remarks> <see href="http://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY"/></remarks>
		/// <param name="dest">Destination memory address </param>
		/// <param name="src">Source memory address </param>
		/// <param name="size">Size in bytes to copy </param>
		/// <param name="kind">Type of transfer</param>
		/// <param name="stream">Stream identifier</param>
		/// <returns>cudaSuccess, cudaErrorInvalidValue, cudaErrorInvalidDevicePointer, cudaErrorInvalidMemcpyDirection</returns>
		cudaError_t MemcpyAsync(IntPtr dest, IntPtr src, size_t size, cudaMemcpyKind kind, cudaStream_t stream);
		/// <summary>
		/// Frees page-locked memory.
		/// </summary>
		/// <remarks> <see href="http://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY"/></remarks>
		/// <param name="ptr">Pointer to memory to free</param>
		/// <returns>cudaSuccess, cudaErrorInitializationError</returns>
		cudaError_t FreeHost(IntPtr ptr);
		/// <summary>
		/// Allocates page-locked memory on the host
		/// </summary>
		/// <remarks> <see href="http://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY"/></remarks>
		/// <param name="pHost">Device pointer to allocated memory </param>
		/// <param name="size">Requested allocation size in bytes </param>
		/// <param name="flags">Requested properties of allocated memory</param>
		/// <returns>cudaSuccess, cudaErrorMemoryAllocation</returns>
		cudaError_t HostAlloc(out IntPtr pHost, size_t size, cudaHostAllocFlags flags);
		/// <summary>
		/// Get last cuda error
		/// </summary>
		/// <remarks> <see href="http://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__ERROR.html#group__CUDART__ERROR"/></remarks>
		/// <returns>cudaSuccess, cudaErrorMissingConfiguration, cudaErrorMemoryAllocation, 
		/// cudaErrorInitializationError, cudaErrorLaunchFailure, cudaErrorLaunchTimeout, 
		/// cudaErrorLaunchOutOfResources, cudaErrorInvalidDeviceFunction, 
		/// cudaErrorInvalidConfiguration, cudaErrorInvalidDevice, cudaErrorInvalidValue, 
		/// cudaErrorInvalidPitchValue, cudaErrorInvalidSymbol, cudaErrorUnmapBufferObjectFailed, 
		/// cudaErrorInvalidHostPointer, cudaErrorInvalidDevicePointer, cudaErrorInvalidTexture, 
		/// cudaErrorInvalidTextureBinding, cudaErrorInvalidChannelDescriptor, 
		/// cudaErrorInvalidMemcpyDirection, cudaErrorInvalidFilterSetting, 
		/// cudaErrorInvalidNormSetting, cudaErrorUnknown, cudaErrorInvalidResourceHandle, 
		/// cudaErrorInsufficientDriver, cudaErrorSetOnActiveProcess, 
		/// cudaErrorStartupFailure, </returns>
		cudaError_t GetLastError();
		/// <summary>
		/// Peek last cuda error
		/// </summary>
		/// <remarks> <see href="http://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__ERROR.html#group__CUDART__ERROR"/></remarks>
		/// <returns>cudaSuccess, cudaErrorMissingConfiguration, cudaErrorMemoryAllocation, 
		/// cudaErrorInitializationError, cudaErrorLaunchFailure, cudaErrorLaunchTimeout,
		/// cudaErrorLaunchOutOfResources, cudaErrorInvalidDeviceFunction,
		/// cudaErrorInvalidConfiguration, cudaErrorInvalidDevice, cudaErrorInvalidValue, 
		/// cudaErrorInvalidPitchValue, cudaErrorInvalidSymbol, cudaErrorUnmapBufferObjectFailed, 
		/// cudaErrorInvalidHostPointer, cudaErrorInvalidDevicePointer, cudaErrorInvalidTexture, 
		/// cudaErrorInvalidTextureBinding, cudaErrorInvalidChannelDescriptor, 
		/// cudaErrorInvalidMemcpyDirection, cudaErrorInvalidFilterSetting, 
		/// cudaErrorInvalidNormSetting, cudaErrorUnknown, cudaErrorInvalidResourceHandle, 
		/// cudaErrorInsufficientDriver, cudaErrorSetOnActiveProcess, 
		/// cudaErrorStartupFailure, </returns>
		cudaError_t GetPeekAtLastError();
		/// <summary>
		/// Returns the description string for an error code
		/// </summary>
		/// <remarks> <see href="http://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__ERROR.html#group__CUDART__ERROR"/></remarks>
		/// <param name="err">Error code to convert to string</param>
		/// <returns>char* pointer to a NULL-terminated string, or NULL if the error code is not valid. </returns>
		string GetErrorString(cudaError_t err);

		/// <summary>
		/// Returns the string representation of an error code enum name
		/// </summary>
		string GetErrorName(cudaError_t err);

		/// <summary>
		/// Create an asynchronous stream
		/// </summary>
		/// <remarks> <see href="http://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM"/></remarks>
		/// <param name="stream">Pointer to new stream identifier</param>
		/// <returns>cudaSuccess, cudaErrorInvalidValue</returns>
		cudaError_t StreamCreate(out cudaStream_t stream);
		/// <summary>
		/// Destroy cuda steam
		/// </summary>
		/// <remarks> <see href="http://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM"/></remarks>
		/// <param name="stream">Stream identifier</param>
		/// <returns>cudaSuccess, cudaErrorInvalidResourceHandle</returns>
		cudaError_t StreamDestroy(cudaStream_t stream);
		/// <summary>
		/// Synchronize cuda steam
		/// </summary>
		/// <remarks> <see href="http://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM"/></remarks>
		/// <param name="stream">Stream identifier</param>
		/// <returns>cudaSuccess, cudaErrorInvalidResourceHandle</returns>
		cudaError_t StreamSynchronize(cudaStream_t stream);
	}
}

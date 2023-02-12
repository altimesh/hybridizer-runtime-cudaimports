/* (c) ALTIMESH 2018 -- all rights reserved */
using System;
using System.Runtime.InteropServices;

using size_t = System.UIntPtr ;

namespace Hybridizer.Runtime.CUDAImports
{
	/// <summary>
	/// curand status
	/// </summary>
	[IntrinsicType("curandStatus_t")]
	public enum curandStatus_t : int
	{
		/// <summary>
		/// No errors. 
		/// </summary>
		CURAND_STATUS_SUCCESS = 0, 
		/// <summary>
		/// Header file and linked library version do not match. 
		/// </summary>
		CURAND_STATUS_VERSION_MISMATCH = 100,
		/// <summary>
		/// Generator not initialized. 
		/// </summary>
		CURAND_STATUS_NOT_INITIALIZED = 101, 
		/// <summary>
		/// Memory allocation failed. 
		/// </summary>
		CURAND_STATUS_ALLOCATION_FAILED = 102, 
		/// <summary>
		/// Generator is wrong type. 
		/// </summary>
		CURAND_STATUS_TYPE_ERROR = 103,
		/// <summary>
		/// Argument out of range. 
		/// </summary>
		CURAND_STATUS_OUT_OF_RANGE = 104,
		/// <summary>
		/// Length requested is not a multple of dimension. 
		/// </summary>
		CURAND_STATUS_LENGTH_NOT_MULTIPLE = 105, 
		/// <summary>
		/// GPU does not have double precision required by MRG32k3a. 
		/// </summary>
		CURAND_STATUS_DOUBLE_PRECISION_REQUIRED = 106,
		/// <summary>
		/// Kernel launch failure. 
		/// </summary>
		CURAND_STATUS_LAUNCH_FAILURE = 201,
		/// <summary>
		/// Preexisting failure on library entry. 
		/// </summary>
		CURAND_STATUS_PREEXISTING_FAILURE = 202, 
		/// <summary>
		/// Initialization of CUDA failed. 
		/// </summary>
		CURAND_STATUS_INITIALIZATION_FAILED = 203,
		/// <summary>
		/// Architecture mismatch, GPU does not support requested feature. 
		/// </summary>
		CURAND_STATUS_ARCH_MISMATCH = 204, 
		/// <summary>
		/// Internal library error. 
		/// </summary>
		CURAND_STATUS_INTERNAL_ERROR = 999, 
	}
}

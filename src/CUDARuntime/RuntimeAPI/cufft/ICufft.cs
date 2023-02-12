/* (c) ALTIMESH 2018 -- all rights reserved */
using System;
using System.Runtime.InteropServices;

namespace Hybridizer.Runtime.CUDAImports
{
#pragma warning disable CS1591 // Missing XML comment for publicly visible type or member

	public interface ICUFFT
	{
		cufftResult Create(out cufftHandle plan);
		cufftResult Destroy(cufftHandle plan);
		cufftResult Estimate1d(int nx, cufftType type, int batch, out size_t workSize);
		cufftResult Estimate2d(int nx, int ny, cufftType type, out size_t workSize);
		cufftResult Estimate3d(int nx, int ny, int nz, cufftType type, out size_t workSize);
		cufftResult EstimateMany(int rank, IntPtr n, IntPtr inembed, int istride, int idist, IntPtr onembed, int ostride, int odist, cufftType type, int batch, out size_t workSize);
		cufftResult ExecC2C(cufftHandle plan, IntPtr idata, IntPtr odata, int direction);
		cufftResult ExecC2R(cufftHandle plan, IntPtr idata, IntPtr odata);
		cufftResult ExecD2Z(cufftHandle plan, IntPtr idata, IntPtr odata);
		cufftResult ExecR2C(cufftHandle plan, IntPtr idata, IntPtr odata);
		cufftResult ExecZ2D(cufftHandle plan, IntPtr idata, IntPtr odata);
		cufftResult ExecZ2Z(cufftHandle plan, IntPtr idata, IntPtr odata, int direction);
		cufftResult GetProperty(libraryPropertyType_t type, out int val);
		cufftResult GetSize(cufftHandle handle, out size_t workSize);
		cufftResult GetSize1d(cufftHandle handle, int nx, cufftType type, int batch, out size_t workSize);
		cufftResult GetSize2d(cufftHandle handle, int nx, int ny, cufftType type, out size_t workSize);
		cufftResult GetSize3d(cufftHandle handle, int nx, int ny, int nz, cufftType type, out size_t workSize);
		cufftResult GetSizeMany(cufftHandle plan, int rank, IntPtr n, IntPtr inembed, int istride, int idist, IntPtr onembed, int ostride, int odist, cufftType type, int batch, out size_t workSize);
		cufftResult GetSizeMany64(cufftHandle plan, int rank, IntPtr n, IntPtr inembed, long istride, long idist, IntPtr onembed, long ostride, long odist, cufftType type, long batch, out size_t workSize);
		cufftResult GetVersion(out int version);
		cufftResult MakePlan1d(cufftHandle plan, int nx, cufftType type, int batch, out size_t workSize);
		cufftResult MakePlan2d(cufftHandle plan, int nx, int ny, cufftType type, out size_t workSize);
		cufftResult MakePlan3d(cufftHandle plan, int nx, int ny, int nz, cufftType type, out size_t workSize);
		cufftResult MakePlanMany(cufftHandle plan, int rank, IntPtr n, IntPtr inembed, int istride, int idist, IntPtr onembed, int ostride, int odist, cufftType type, int batch, out size_t workSize);
		cufftResult MakePlanMany64(cufftHandle plan, int rank, IntPtr n, IntPtr inembed, long istride, long idist, IntPtr onembed, long ostride, long odist, cufftType type, long batch, out size_t workSize);
		cufftResult Plan1d(out cufftHandle plan, int nx, cufftType type, int batch);
		cufftResult Plan2d(out cufftHandle plan, int nx, int ny, cufftType type);
		cufftResult Plan3d(out cufftHandle plan, int nx, int ny, int nz, cufftType type);
		cufftResult PlanMany(out cufftHandle plan, int rank, IntPtr n, IntPtr inembed, int istride, int idist, IntPtr onembed, int ostride, int odist, cufftType type, int batch);
		cufftResult SetAutoAllocate(cufftHandle plan, int autoAllocate);
		cufftResult SetCompatibilityMode(cufftHandle plan, cufftCompatibility mode);
		cufftResult SetStream(cufftHandle plan, cudaStream_t stream);
		cufftResult SetWorkArea(cufftHandle plan, IntPtr workArea);
		cufftResult XtMakePlanMany(cufftHandle plan, int rank, IntPtr n, IntPtr inembed, long istride, long idist, cudaDataType_t inputtype, IntPtr onembed, long ostride, long odist, cudaDataType_t outputtype, long batch, out size_t workSize, cudaDataType_t executiontype);
		// TODO: cufftXt...
	}
#pragma warning restore CS1591 // Missing XML comment for publicly visible type or member

}
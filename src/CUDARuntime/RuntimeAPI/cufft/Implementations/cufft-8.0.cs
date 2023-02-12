/* (c) ALTIMESH 2018 -- all rights reserved */
using System;
using System.Runtime.InteropServices;

namespace Hybridizer.Runtime.CUDAImports
{
    /// <summary>
    /// cufft wrapper
    /// Complete documentation <see href="https://docs.nvidia.com/cuda/cufft/index.html">here</see>
    /// </summary>
    internal partial class cufftImplem
    {
        internal class CUFFT_64_80 : ICUFFT
        {
            public const string CUFFT_DLL = "cufft_64_80.dll";

            [DllImport(CUFFT_DLL, EntryPoint = "cufftPlanMany")]
            public static extern cufftResult cufftPlanMany(out cufftHandle plan, int rank, IntPtr n, IntPtr inembed, int istride, int idist, IntPtr onembed, int ostride, int odist, cufftType type, int batch);
            public cufftResult PlanMany(out cufftHandle plan, int rank, IntPtr n, IntPtr inembed, int istride, int idist, IntPtr onembed, int ostride, int odist, cufftType type, int batch) { return cufftPlanMany(out plan, rank, n, inembed, istride, idist, onembed, ostride, odist, type, batch); }

            [DllImport(CUFFT_DLL, EntryPoint = "cufftPlan1d")]
            public static extern cufftResult cufftPlan1d(out cufftHandle plan, int nx, cufftType type, int batch);
            public cufftResult Plan1d(out cufftHandle plan, int nx, cufftType type, int batch) { return cufftPlan1d(out plan, nx, type, batch); }

            [DllImport(CUFFT_DLL, EntryPoint = "cufftPlan2d")]
            public static extern cufftResult cufftPlan2d(out cufftHandle plan, int nx, int ny, cufftType type);
            public cufftResult Plan2d(out cufftHandle plan, int nx, int ny, cufftType type) { return cufftPlan2d(out plan, nx, ny, type); }

            [DllImport(CUFFT_DLL, EntryPoint = "cufftPlan3d")]
            public static extern cufftResult cufftPlan3d(out cufftHandle plan, int nx, int ny, int nz, cufftType type);
            public cufftResult Plan3d(out cufftHandle plan, int nx, int ny, int nz, cufftType type) { return cufftPlan3d(out plan, nx, ny, nz, type); }

            [DllImport(CUFFT_DLL, EntryPoint = "cufftDestroy")]
            public static extern cufftResult cufftDestroy(cufftHandle plan);
            public cufftResult Destroy(cufftHandle plan) { return cufftDestroy(plan); }

            [DllImport(CUFFT_DLL, EntryPoint = "cufftExecC2C")]
            public static extern cufftResult cufftExecC2C(cufftHandle plan, IntPtr idata, IntPtr odata, int direction);
            public cufftResult ExecC2C(cufftHandle plan, IntPtr idata, IntPtr odata, int direction) { return cufftExecC2C(plan, idata, odata, direction); }

            [DllImport(CUFFT_DLL, EntryPoint = "cufftExecZ2Z")]
            public static extern cufftResult cufftExecZ2Z(cufftHandle plan, IntPtr idata, IntPtr odata, int direction);
            public cufftResult ExecZ2Z(cufftHandle plan, IntPtr idata, IntPtr odata, int direction) { return cufftExecZ2Z(plan, idata, odata, direction); }

            [DllImport(CUFFT_DLL, EntryPoint = "cufftExecR2C")]
            public static extern cufftResult cufftExecR2C(cufftHandle plan, IntPtr idata, IntPtr odata);
            public cufftResult ExecR2C(cufftHandle plan, IntPtr idata, IntPtr odata) { return cufftExecR2C(plan, idata, odata); }

            [DllImport(CUFFT_DLL, EntryPoint = "cufftExecD2Z")]
            public static extern cufftResult cufftExecD2Z(cufftHandle plan, IntPtr idata, IntPtr odata);
            public cufftResult ExecD2Z(cufftHandle plan, IntPtr idata, IntPtr odata) { return cufftExecD2Z(plan, idata, odata); }

            [DllImport(CUFFT_DLL, EntryPoint = "cufftExecC2R")]
            public static extern cufftResult cufftExecC2R(cufftHandle plan, IntPtr idata, IntPtr odata);
            public cufftResult ExecC2R(cufftHandle plan, IntPtr idata, IntPtr odata) { return cufftExecC2R(plan, idata, odata); }

            [DllImport(CUFFT_DLL, EntryPoint = "cufftExecZ2D")]
            public static extern cufftResult cufftExecZ2D(cufftHandle plan, IntPtr idata, IntPtr odata);
            public cufftResult ExecZ2D(cufftHandle plan, IntPtr idata, IntPtr odata) { return cufftExecZ2D(plan, idata, odata); }


            [DllImport(CUFFT_DLL, EntryPoint = "cufftSetStream")]
            public static extern cufftResult cufftSetStream(cufftHandle plan, cudaStream_t stream);
            public cufftResult SetStream(cufftHandle plan, cudaStream_t stream) { return cufftSetStream(plan, stream); }

            [DllImport(CUFFT_DLL, EntryPoint = "cufftSetCompatibilityMode")]
            public static extern cufftResult cufftSetCompatibilityMode(cufftHandle plan, cufftCompatibility mode);
            public cufftResult SetCompatibilityMode(cufftHandle plan, cufftCompatibility mode) { return cufftSetCompatibilityMode(plan, mode); }

            public cufftResult Create(out cufftHandle plan)
            {
                throw new NotImplementedException();
            }

            public cufftResult Estimate1d(int nx, cufftType type, int batch, out size_t workSize)
            {
                throw new NotImplementedException();
            }

            public cufftResult Estimate2d(int nx, int ny, cufftType type, out size_t workSize)
            {
                throw new NotImplementedException();
            }

            public cufftResult Estimate3d(int nx, int ny, int nz, cufftType type, out size_t workSize)
            {
                throw new NotImplementedException();
            }

            public cufftResult EstimateMany(int rank, IntPtr n, IntPtr inembed, int istride, int idist, IntPtr onembed, int ostride, int odist, cufftType type, int batch, out size_t workSize)
            {
                throw new NotImplementedException();
            }

            public cufftResult GetProperty(libraryPropertyType_t type, out int val)
            {
                throw new NotImplementedException();
            }

            public cufftResult GetSize(cufftHandle handle, out size_t workSize)
            {
                throw new NotImplementedException();
            }

            public cufftResult GetSize1d(cufftHandle handle, int nx, cufftType type, int batch, out size_t workSize)
            {
                throw new NotImplementedException();
            }

            public cufftResult GetSize2d(cufftHandle handle, int nx, int ny, cufftType type, out size_t workSize)
            {
                throw new NotImplementedException();
            }

            public cufftResult GetSize3d(cufftHandle handle, int nx, int ny, int nz, cufftType type, out size_t workSize)
            {
                throw new NotImplementedException();
            }

            public cufftResult GetSizeMany(cufftHandle plan, int rank, IntPtr n, IntPtr inembed, int istride, int idist, IntPtr onembed, int ostride, int odist, cufftType type, int batch, out size_t workSize)
            {
                throw new NotImplementedException();
            }

            public cufftResult GetSizeMany64(cufftHandle plan, int rank, IntPtr n, IntPtr inembed, long istride, long idist, IntPtr onembed, long ostride, long odist, cufftType type, long batch, out size_t workSize)
            {
                throw new NotImplementedException();
            }

            public cufftResult GetVersion(out int version)
            {
                throw new NotImplementedException();
            }

            public cufftResult MakePlan1d(cufftHandle plan, int nx, cufftType type, int batch, out size_t workSize)
            {
                throw new NotImplementedException();
            }

            public cufftResult MakePlan2d(cufftHandle plan, int nx, int ny, cufftType type, out size_t workSize)
            {
                throw new NotImplementedException();
            }

            public cufftResult MakePlan3d(cufftHandle plan, int nx, int ny, int nz, cufftType type, out size_t workSize)
            {
                throw new NotImplementedException();
            }

            public cufftResult MakePlanMany(cufftHandle plan, int rank, IntPtr n, IntPtr inembed, int istride, int idist, IntPtr onembed, int ostride, int odist, cufftType type, int batch, out size_t workSize)
            {
                throw new NotImplementedException();
            }

            public cufftResult MakePlanMany64(cufftHandle plan, int rank, IntPtr n, IntPtr inembed, long istride, long idist, IntPtr onembed, long ostride, long odist, cufftType type, long batch, out size_t workSize)
            {
                throw new NotImplementedException();
            }

            public cufftResult SetAutoAllocate(cufftHandle plan, int autoAllocate)
            {
                throw new NotImplementedException();
            }

            public cufftResult SetWorkArea(cufftHandle plan, IntPtr workArea)
            {
                throw new NotImplementedException();
            }

            public cufftResult XtMakePlanMany(cufftHandle plan, int rank, IntPtr n, IntPtr inembed, long istride, long idist, cudaDataType_t inputtype, IntPtr onembed, long ostride, long odist, cudaDataType_t outputtype, long batch, out size_t workSize, cudaDataType_t executiontype)
            {
                throw new NotImplementedException();
            }
        }

    }
}
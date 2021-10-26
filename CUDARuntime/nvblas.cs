/* (c) ALTIMESH 2018 -- all rights reserved */
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;

namespace Hybridizer.Runtime.CUDAImports
{
    /// <summary>
    /// Complete documentation on <see href="https://docs.nvidia.com/cuda/nvblas/index.html">NVidia documentation</see>
    /// </summary>
    public interface INVBLAS
    {
#pragma warning disable 1591
        void cgemm(IntPtr transa, IntPtr transb, IntPtr m, IntPtr n, IntPtr k, IntPtr alpha, IntPtr a, IntPtr lda, IntPtr b, IntPtr ldb, IntPtr beta, IntPtr c, IntPtr ldc);
        void cgemm_(IntPtr transa, IntPtr transb, IntPtr m, IntPtr n, IntPtr k, IntPtr alpha, IntPtr a, IntPtr lda, IntPtr b, IntPtr ldb, IntPtr beta, IntPtr c, IntPtr ldc);
        void chemm(IntPtr side, IntPtr uplo, IntPtr m, IntPtr n, IntPtr alpha, IntPtr a, IntPtr lda, IntPtr b, IntPtr ldb, IntPtr beta, IntPtr c, IntPtr ldc);
        void chemm_(IntPtr side, IntPtr uplo, IntPtr m, IntPtr n, IntPtr alpha, IntPtr a, IntPtr lda, IntPtr b, IntPtr ldb, IntPtr beta, IntPtr c, IntPtr ldc);
        void cher2k(IntPtr uplo, IntPtr trans, IntPtr n, IntPtr k, IntPtr alpha, IntPtr a, IntPtr lda, IntPtr b, IntPtr ldb, IntPtr beta, IntPtr c, IntPtr ldc);
        void cher2k_(IntPtr uplo, IntPtr trans, IntPtr n, IntPtr k, IntPtr alpha, IntPtr a, IntPtr lda, IntPtr b, IntPtr ldb, IntPtr beta, IntPtr c, IntPtr ldc);
        void cherk(IntPtr uplo, IntPtr trans, IntPtr n, IntPtr k, IntPtr alpha, IntPtr a, IntPtr lda, IntPtr beta, IntPtr c, IntPtr ldc);
        void cherk_(IntPtr uplo, IntPtr trans, IntPtr n, IntPtr k, IntPtr alpha, IntPtr a, IntPtr lda, IntPtr beta, IntPtr c, IntPtr ldc);
        void csymm(IntPtr side, IntPtr uplo, IntPtr m, IntPtr n, IntPtr alpha, IntPtr a, IntPtr lda, IntPtr b, IntPtr ldb, IntPtr beta, IntPtr c, IntPtr ldc);
        void csymm_(IntPtr side, IntPtr uplo, IntPtr m, IntPtr n, IntPtr alpha, IntPtr a, IntPtr lda, IntPtr b, IntPtr ldb, IntPtr beta, IntPtr c, IntPtr ldc);
        void csyr2k(IntPtr uplo, IntPtr trans, IntPtr n, IntPtr k, IntPtr alpha, IntPtr a, IntPtr lda, IntPtr b, IntPtr ldb, IntPtr beta, IntPtr c, IntPtr ldc);
        void csyr2k_(IntPtr uplo, IntPtr trans, IntPtr n, IntPtr k, IntPtr alpha, IntPtr a, IntPtr lda, IntPtr b, IntPtr ldb, IntPtr beta, IntPtr c, IntPtr ldc);
        void csyrk(IntPtr uplo, IntPtr trans, IntPtr n, IntPtr k, IntPtr alpha, IntPtr a, IntPtr lda, IntPtr beta, IntPtr c, IntPtr ldc);
        void csyrk_(IntPtr uplo, IntPtr trans, IntPtr n, IntPtr k, IntPtr alpha, IntPtr a, IntPtr lda, IntPtr beta, IntPtr c, IntPtr ldc);
        void ctrmm(IntPtr side, IntPtr uplo, IntPtr transa, IntPtr diag, IntPtr m, IntPtr n, IntPtr alpha, IntPtr a, IntPtr lda, IntPtr b, IntPtr ldb);
        void ctrmm_(IntPtr side, IntPtr uplo, IntPtr transa, IntPtr diag, IntPtr m, IntPtr n, IntPtr alpha, IntPtr a, IntPtr lda, IntPtr b, IntPtr ldb);
        void ctrsm(IntPtr side, IntPtr uplo, IntPtr transa, IntPtr diag, IntPtr m, IntPtr n, IntPtr alpha, IntPtr a, IntPtr lda, IntPtr b, IntPtr ldb);
        void ctrsm_(IntPtr side, IntPtr uplo, IntPtr transa, IntPtr diag, IntPtr m, IntPtr n, IntPtr alpha, IntPtr a, IntPtr lda, IntPtr b, IntPtr ldb);
        void dgemm(IntPtr transa, IntPtr transb, IntPtr m, IntPtr n, IntPtr k, IntPtr alpha, IntPtr a, IntPtr lda, IntPtr b, IntPtr ldb, IntPtr beta, IntPtr c, IntPtr ldc);
        void dgemm_(IntPtr transa, IntPtr transb, IntPtr m, IntPtr n, IntPtr k, IntPtr alpha, IntPtr a, IntPtr lda, IntPtr b, IntPtr ldb, IntPtr beta, IntPtr c, IntPtr ldc);
        void dsymm(IntPtr side, IntPtr uplo, IntPtr m, IntPtr n, IntPtr alpha, IntPtr a, IntPtr lda, IntPtr b, IntPtr ldb, IntPtr beta, IntPtr c, IntPtr ldc);
        void dsymm_(IntPtr side, IntPtr uplo, IntPtr m, IntPtr n, IntPtr alpha, IntPtr a, IntPtr lda, IntPtr b, IntPtr ldb, IntPtr beta, IntPtr c, IntPtr ldc);
        void dsyr2k(IntPtr uplo, IntPtr trans, IntPtr n, IntPtr k, IntPtr alpha, IntPtr a, IntPtr lda, IntPtr b, IntPtr ldb, IntPtr beta, IntPtr c, IntPtr ldc);
        void dsyr2k_(IntPtr uplo, IntPtr trans, IntPtr n, IntPtr k, IntPtr alpha, IntPtr a, IntPtr lda, IntPtr b, IntPtr ldb, IntPtr beta, IntPtr c, IntPtr ldc);
        void dsyrk(IntPtr uplo, IntPtr trans, IntPtr n, IntPtr k, IntPtr alpha, IntPtr a, IntPtr lda, IntPtr beta, IntPtr c, IntPtr ldc);
        void dsyrk_(IntPtr uplo, IntPtr trans, IntPtr n, IntPtr k, IntPtr alpha, IntPtr a, IntPtr lda, IntPtr beta, IntPtr c, IntPtr ldc);
        void dtrmm(IntPtr side, IntPtr uplo, IntPtr transa, IntPtr diag, IntPtr m, IntPtr n, IntPtr alpha, IntPtr a, IntPtr lda, IntPtr b, IntPtr ldb);
        void dtrmm_(IntPtr side, IntPtr uplo, IntPtr transa, IntPtr diag, IntPtr m, IntPtr n, IntPtr alpha, IntPtr a, IntPtr lda, IntPtr b, IntPtr ldb);
        void dtrsm(IntPtr side, IntPtr uplo, IntPtr transa, IntPtr diag, IntPtr m, IntPtr n, IntPtr alpha, IntPtr a, IntPtr lda, IntPtr b, IntPtr ldb);
        void dtrsm_(IntPtr side, IntPtr uplo, IntPtr transa, IntPtr diag, IntPtr m, IntPtr n, IntPtr alpha, IntPtr a, IntPtr lda, IntPtr b, IntPtr ldb);
        void sgemm(IntPtr transa, IntPtr transb, IntPtr m, IntPtr n, IntPtr k, IntPtr alpha, IntPtr a, IntPtr lda, IntPtr b, IntPtr ldb, IntPtr beta, IntPtr c, IntPtr ldc);
        void sgemm_(IntPtr transa, IntPtr transb, IntPtr m, IntPtr n, IntPtr k, IntPtr alpha, IntPtr a, IntPtr lda, IntPtr b, IntPtr ldb, IntPtr beta, IntPtr c, IntPtr ldc);
        void ssymm(IntPtr side, IntPtr uplo, IntPtr m, IntPtr n, IntPtr alpha, IntPtr a, IntPtr lda, IntPtr b, IntPtr ldb, IntPtr beta, IntPtr c, IntPtr ldc);
        void ssymm_(IntPtr side, IntPtr uplo, IntPtr m, IntPtr n, IntPtr alpha, IntPtr a, IntPtr lda, IntPtr b, IntPtr ldb, IntPtr beta, IntPtr c, IntPtr ldc);
        void ssyr2k(IntPtr uplo, IntPtr trans, IntPtr n, IntPtr k, IntPtr alpha, IntPtr a, IntPtr lda, IntPtr b, IntPtr ldb, IntPtr beta, IntPtr c, IntPtr ldc);
        void ssyr2k_(IntPtr uplo, IntPtr trans, IntPtr n, IntPtr k, IntPtr alpha, IntPtr a, IntPtr lda, IntPtr b, IntPtr ldb, IntPtr beta, IntPtr c, IntPtr ldc);
        void ssyrk(IntPtr uplo, IntPtr trans, IntPtr n, IntPtr k, IntPtr alpha, IntPtr a, IntPtr lda, IntPtr beta, IntPtr c, IntPtr ldc);
        void ssyrk_(IntPtr uplo, IntPtr trans, IntPtr n, IntPtr k, IntPtr alpha, IntPtr a, IntPtr lda, IntPtr beta, IntPtr c, IntPtr ldc);
        void strmm(IntPtr side, IntPtr uplo, IntPtr transa, IntPtr diag, IntPtr m, IntPtr n, IntPtr alpha, IntPtr a, IntPtr lda, IntPtr b, IntPtr ldb);
        void strmm_(IntPtr side, IntPtr uplo, IntPtr transa, IntPtr diag, IntPtr m, IntPtr n, IntPtr alpha, IntPtr a, IntPtr lda, IntPtr b, IntPtr ldb);
        void strsm(IntPtr side, IntPtr uplo, IntPtr transa, IntPtr diag, IntPtr m, IntPtr n, IntPtr alpha, IntPtr a, IntPtr lda, IntPtr b, IntPtr ldb);
        void strsm_(IntPtr side, IntPtr uplo, IntPtr transa, IntPtr diag, IntPtr m, IntPtr n, IntPtr alpha, IntPtr a, IntPtr lda, IntPtr b, IntPtr ldb);
        void zgemm(IntPtr transa, IntPtr transb, IntPtr m, IntPtr n, IntPtr k, IntPtr alpha, IntPtr a, IntPtr lda, IntPtr b, IntPtr ldb, IntPtr beta, IntPtr c, IntPtr ldc);
        void zgemm_(IntPtr transa, IntPtr transb, IntPtr m, IntPtr n, IntPtr k, IntPtr alpha, IntPtr a, IntPtr lda, IntPtr b, IntPtr ldb, IntPtr beta, IntPtr c, IntPtr ldc);
        void zhemm(IntPtr side, IntPtr uplo, IntPtr m, IntPtr n, IntPtr alpha, IntPtr a, IntPtr lda, IntPtr b, IntPtr ldb, IntPtr beta, IntPtr c, IntPtr ldc);
        void zhemm_(IntPtr side, IntPtr uplo, IntPtr m, IntPtr n, IntPtr alpha, IntPtr a, IntPtr lda, IntPtr b, IntPtr ldb, IntPtr beta, IntPtr c, IntPtr ldc);
        void zher2k(IntPtr uplo, IntPtr trans, IntPtr n, IntPtr k, IntPtr alpha, IntPtr a, IntPtr lda, IntPtr b, IntPtr ldb, IntPtr beta, IntPtr c, IntPtr ldc);
        void zher2k_(IntPtr uplo, IntPtr trans, IntPtr n, IntPtr k, IntPtr alpha, IntPtr a, IntPtr lda, IntPtr b, IntPtr ldb, IntPtr beta, IntPtr c, IntPtr ldc);
        void zherk(IntPtr uplo, IntPtr trans, IntPtr n, IntPtr k, IntPtr alpha, IntPtr a, IntPtr lda, IntPtr beta, IntPtr c, IntPtr ldc);
        void zherk_(IntPtr uplo, IntPtr trans, IntPtr n, IntPtr k, IntPtr alpha, IntPtr a, IntPtr lda, IntPtr beta, IntPtr c, IntPtr ldc);
        void zsymm(IntPtr side, IntPtr uplo, IntPtr m, IntPtr n, IntPtr alpha, IntPtr a, IntPtr lda, IntPtr b, IntPtr ldb, IntPtr beta, IntPtr c, IntPtr ldc);
        void zsymm_(IntPtr side, IntPtr uplo, IntPtr m, IntPtr n, IntPtr alpha, IntPtr a, IntPtr lda, IntPtr b, IntPtr ldb, IntPtr beta, IntPtr c, IntPtr ldc);
        void zsyr2k(IntPtr uplo, IntPtr trans, IntPtr n, IntPtr k, IntPtr alpha, IntPtr a, IntPtr lda, IntPtr b, IntPtr ldb, IntPtr beta, IntPtr c, IntPtr ldc);
        void zsyr2k_(IntPtr uplo, IntPtr trans, IntPtr n, IntPtr k, IntPtr alpha, IntPtr a, IntPtr lda, IntPtr b, IntPtr ldb, IntPtr beta, IntPtr c, IntPtr ldc);
        void zsyrk(IntPtr uplo, IntPtr trans, IntPtr n, IntPtr k, IntPtr alpha, IntPtr a, IntPtr lda, IntPtr beta, IntPtr c, IntPtr ldc);
        void zsyrk_(IntPtr uplo, IntPtr trans, IntPtr n, IntPtr k, IntPtr alpha, IntPtr a, IntPtr lda, IntPtr beta, IntPtr c, IntPtr ldc);
        void ztrmm(IntPtr side, IntPtr uplo, IntPtr transa, IntPtr diag, IntPtr m, IntPtr n, IntPtr alpha, IntPtr a, IntPtr lda, IntPtr b, IntPtr ldb);
        void ztrmm_(IntPtr side, IntPtr uplo, IntPtr transa, IntPtr diag, IntPtr m, IntPtr n, IntPtr alpha, IntPtr a, IntPtr lda, IntPtr b, IntPtr ldb);
        void ztrsm(IntPtr side, IntPtr uplo, IntPtr transa, IntPtr diag, IntPtr m, IntPtr n, IntPtr alpha, IntPtr a, IntPtr lda, IntPtr b, IntPtr ldb);
        void ztrsm_(IntPtr side, IntPtr uplo, IntPtr transa, IntPtr diag, IntPtr m, IntPtr n, IntPtr alpha, IntPtr a, IntPtr lda, IntPtr b, IntPtr ldb);
#pragma warning restore 1591
    }

    /// <summary>
    /// factory class
    /// </summary>
    public class nvblas
    {
        // TODO : depend on configuration
        private static INVBLAS _instance = null;

        /// <summary>
        /// get current CUDA version
        /// </summary>
        /// <returns></returns>
        public static string GetCudaVersion()
        {
            // If not, get the version configured in app.config
            string cudaVersion = cuda.GetCudaVersion();

            // Otherwise default to latest version
            if (cudaVersion == null) cudaVersion = "75";
            cudaVersion = cudaVersion.Replace(".", ""); // Remove all dots ("7.5" will be understood "75")

            return cudaVersion;
        }

        /// <summary>
        /// returns the an iNVBLAS implementation, matching the current cuda version
        /// </summary>
        public static INVBLAS Instance()
        {
            if (IntPtr.Size != 8)
            {
                throw new ApplicationException("NVBLAS is only available in x64");
            }
            if (_instance == null)
            {
                string cudaVersion = GetCudaVersion();
                switch (cudaVersion)
                {
                    case "90":
                        _instance = new nvblas90();
                        break;
                    case "91":
                        _instance = new nvblas91();
                        break;
                    default:
                        throw new ApplicationException(string.Format("Unsupported version of Cuda {0} for nvblas", cudaVersion));
                }
            }

            return _instance;
        }
    }

    internal class nvblas90_imports
    {
        private const string DLL_NAME = "nvblas64_90.dll";

        [DllImport(DLL_NAME, EntryPoint = "sgemm_", CallingConvention = CallingConvention.Cdecl)]
        public static extern void sgemm_(IntPtr transa, IntPtr transb, IntPtr m, IntPtr n, IntPtr k,
                     IntPtr alpha, IntPtr a, IntPtr lda, IntPtr b, IntPtr ldb,
                     IntPtr beta, IntPtr c, IntPtr ldc);

        [DllImport(DLL_NAME, EntryPoint = "dgemm_", CallingConvention = CallingConvention.Cdecl)]
        public static extern void dgemm_(IntPtr transa, IntPtr transb, IntPtr m, IntPtr n, IntPtr k,
                   IntPtr alpha, IntPtr a, IntPtr lda, IntPtr b, IntPtr ldb,
                   IntPtr beta, IntPtr c, IntPtr ldc);

        [DllImport(DLL_NAME, EntryPoint = "cgemm_", CallingConvention = CallingConvention.Cdecl)]
        public static extern void cgemm_(IntPtr transa, IntPtr transb, IntPtr m, IntPtr n, IntPtr k,
                   IntPtr alpha, IntPtr a, IntPtr lda,
                   IntPtr b, IntPtr ldb, IntPtr beta,
                   IntPtr c, IntPtr ldc);

        [DllImport(DLL_NAME, EntryPoint = "zgemm_", CallingConvention = CallingConvention.Cdecl)]
        public static extern void zgemm_(IntPtr transa, IntPtr transb, IntPtr m, IntPtr n, IntPtr k,
                   IntPtr alpha, IntPtr a, IntPtr lda,
                   IntPtr b, IntPtr ldb, IntPtr beta,
                   IntPtr c, IntPtr ldc);

        [DllImport(DLL_NAME, EntryPoint = "sgemm", CallingConvention = CallingConvention.Cdecl)]
        public static extern void sgemm(IntPtr transa, IntPtr transb, IntPtr m, IntPtr n, IntPtr k,
                   IntPtr alpha, IntPtr a, IntPtr lda, IntPtr b, IntPtr ldb,
                   IntPtr beta, IntPtr c, IntPtr ldc);

        [DllImport(DLL_NAME, EntryPoint = "dgemm", CallingConvention = CallingConvention.Cdecl)]
        public static extern void dgemm(IntPtr transa, IntPtr transb, IntPtr m, IntPtr n, IntPtr k,
                   IntPtr alpha, IntPtr a, IntPtr lda, IntPtr b, IntPtr ldb,
                   IntPtr beta, IntPtr c, IntPtr ldc);

        [DllImport(DLL_NAME, EntryPoint = "cgemm", CallingConvention = CallingConvention.Cdecl)]
        public static extern void cgemm(IntPtr transa, IntPtr transb, IntPtr m, IntPtr n, IntPtr k,
                   IntPtr alpha, IntPtr a, IntPtr lda,
                   IntPtr b, IntPtr ldb, IntPtr beta,
                   IntPtr c, IntPtr ldc);

        [DllImport(DLL_NAME, EntryPoint = "zgemm", CallingConvention = CallingConvention.Cdecl)]
        public static extern void zgemm(IntPtr transa, IntPtr transb, IntPtr m, IntPtr n, IntPtr k,
                   IntPtr alpha, IntPtr a, IntPtr lda,
                   IntPtr b, IntPtr ldb, IntPtr beta,
                   IntPtr c, IntPtr ldc);

        /* SYRK */
        [DllImport(DLL_NAME, EntryPoint = "ssyrk_", CallingConvention = CallingConvention.Cdecl)]
        public static extern void ssyrk_(IntPtr uplo, IntPtr trans, IntPtr n, IntPtr k,
                   IntPtr alpha, IntPtr a, IntPtr lda,
                   IntPtr beta, IntPtr c, IntPtr ldc);

        [DllImport(DLL_NAME, EntryPoint = "dsyrk_", CallingConvention = CallingConvention.Cdecl)]
        public static extern void dsyrk_(IntPtr uplo, IntPtr trans, IntPtr n, IntPtr k,
                   IntPtr alpha, IntPtr a, IntPtr lda, IntPtr beta,
                   IntPtr c, IntPtr ldc);

        [DllImport(DLL_NAME, EntryPoint = "csyrk_", CallingConvention = CallingConvention.Cdecl)]
        public static extern void csyrk_(IntPtr uplo, IntPtr trans, IntPtr n, IntPtr k,
                   IntPtr alpha, IntPtr a, IntPtr lda,
                   IntPtr beta, IntPtr c, IntPtr ldc);

        [DllImport(DLL_NAME, EntryPoint = "zsyrk_", CallingConvention = CallingConvention.Cdecl)]
        public static extern void zsyrk_(IntPtr uplo, IntPtr trans, IntPtr n, IntPtr k,
                   IntPtr alpha, IntPtr a, IntPtr lda,
                   IntPtr beta, IntPtr c, IntPtr ldc);

        [DllImport(DLL_NAME, EntryPoint = "ssyrk", CallingConvention = CallingConvention.Cdecl)]
        public static extern void ssyrk(IntPtr uplo, IntPtr trans, IntPtr n, IntPtr k,
                   IntPtr alpha, IntPtr a, IntPtr lda,
                   IntPtr beta, IntPtr c, IntPtr ldc);

        [DllImport(DLL_NAME, EntryPoint = "dsyrk", CallingConvention = CallingConvention.Cdecl)]
        public static extern void dsyrk(IntPtr uplo, IntPtr trans, IntPtr n, IntPtr k,
                   IntPtr alpha, IntPtr a, IntPtr lda, IntPtr beta,
                   IntPtr c, IntPtr ldc);

        [DllImport(DLL_NAME, EntryPoint = "csyrk", CallingConvention = CallingConvention.Cdecl)]
        public static extern void csyrk(IntPtr uplo, IntPtr trans, IntPtr n, IntPtr k,
                   IntPtr alpha, IntPtr a, IntPtr lda,
                   IntPtr beta, IntPtr c, IntPtr ldc);

        [DllImport(DLL_NAME, EntryPoint = "zsyrk", CallingConvention = CallingConvention.Cdecl)]
        public static extern void zsyrk(IntPtr uplo, IntPtr trans, IntPtr n, IntPtr k,
                   IntPtr alpha, IntPtr a, IntPtr lda,
                   IntPtr beta, IntPtr c, IntPtr ldc);

        /* HERK */
        [DllImport(DLL_NAME, EntryPoint = "cherk_", CallingConvention = CallingConvention.Cdecl)]
        public static extern void cherk_(IntPtr uplo, IntPtr trans, IntPtr n, IntPtr k,
                   IntPtr alpha, IntPtr a, IntPtr lda, IntPtr beta,
                   IntPtr c, IntPtr ldc);

        [DllImport(DLL_NAME, EntryPoint = "zherk_", CallingConvention = CallingConvention.Cdecl)]
        public static extern void zherk_(IntPtr uplo, IntPtr trans, IntPtr n, IntPtr k,
                   IntPtr alpha, IntPtr a, IntPtr lda,
                   IntPtr beta, IntPtr c, IntPtr ldc);

        [DllImport(DLL_NAME, EntryPoint = "cherk", CallingConvention = CallingConvention.Cdecl)]
        public static extern void cherk(IntPtr uplo, IntPtr trans, IntPtr n, IntPtr k,
                   IntPtr alpha, IntPtr a, IntPtr lda, IntPtr beta,
                   IntPtr c, IntPtr ldc);

        [DllImport(DLL_NAME, EntryPoint = "zherk", CallingConvention = CallingConvention.Cdecl)]
        public static extern void zherk(IntPtr uplo, IntPtr trans, IntPtr n, IntPtr k,
                   IntPtr alpha, IntPtr a, IntPtr lda,
                   IntPtr beta, IntPtr c, IntPtr ldc);

        /* TRSM */
        [DllImport(DLL_NAME, EntryPoint = "strsm_", CallingConvention = CallingConvention.Cdecl)]
        public static extern void strsm_(IntPtr side, IntPtr uplo, IntPtr transa, IntPtr diag,
                   IntPtr m, IntPtr n, IntPtr alpha, IntPtr a, IntPtr lda,
                   IntPtr b, IntPtr ldb);

        [DllImport(DLL_NAME, EntryPoint = "dtrsm_", CallingConvention = CallingConvention.Cdecl)]
        public static extern void dtrsm_(IntPtr side, IntPtr uplo, IntPtr transa, IntPtr diag,
                   IntPtr m, IntPtr n, IntPtr alpha, IntPtr a, IntPtr lda,
                   IntPtr b, IntPtr ldb);

        [DllImport(DLL_NAME, EntryPoint = "ctrsm_", CallingConvention = CallingConvention.Cdecl)]
        public static extern void ctrsm_(IntPtr side, IntPtr uplo, IntPtr transa, IntPtr diag,
                   IntPtr m, IntPtr n, IntPtr alpha,
                   IntPtr a, IntPtr lda, IntPtr b, IntPtr ldb);

        [DllImport(DLL_NAME, EntryPoint = "ztrsm_", CallingConvention = CallingConvention.Cdecl)]
        public static extern void ztrsm_(IntPtr side, IntPtr uplo, IntPtr transa, IntPtr diag,
                   IntPtr m, IntPtr n, IntPtr alpha,
                   IntPtr a, IntPtr lda, IntPtr b, IntPtr ldb);

        [DllImport(DLL_NAME, EntryPoint = "strsm", CallingConvention = CallingConvention.Cdecl)]
        public static extern void strsm(IntPtr side, IntPtr uplo, IntPtr transa, IntPtr diag,
                   IntPtr m, IntPtr n, IntPtr alpha, IntPtr a, IntPtr lda,
                   IntPtr b, IntPtr ldb);

        [DllImport(DLL_NAME, EntryPoint = "dtrsm", CallingConvention = CallingConvention.Cdecl)]
        public static extern void dtrsm(IntPtr side, IntPtr uplo, IntPtr transa, IntPtr diag,
                   IntPtr m, IntPtr n, IntPtr alpha, IntPtr a, IntPtr lda,
                   IntPtr b, IntPtr ldb);

        [DllImport(DLL_NAME, EntryPoint = "ctrsm", CallingConvention = CallingConvention.Cdecl)]
        public static extern void ctrsm(IntPtr side, IntPtr uplo, IntPtr transa, IntPtr diag,
                   IntPtr m, IntPtr n, IntPtr alpha,
                   IntPtr a, IntPtr lda, IntPtr b, IntPtr ldb);

        [DllImport(DLL_NAME, EntryPoint = "ztrsm", CallingConvention = CallingConvention.Cdecl)]
        public static extern void ztrsm(IntPtr side, IntPtr uplo, IntPtr transa, IntPtr diag,
                   IntPtr m, IntPtr n, IntPtr alpha,
                   IntPtr a, IntPtr lda, IntPtr b, IntPtr ldb);

        /* SYMM */
        [DllImport(DLL_NAME, EntryPoint = "ssymm_", CallingConvention = CallingConvention.Cdecl)]
        public static extern void ssymm_(IntPtr side, IntPtr uplo, IntPtr m, IntPtr n,
                   IntPtr alpha, IntPtr a, IntPtr lda, IntPtr b, IntPtr ldb,
                   IntPtr beta, IntPtr c, IntPtr ldc);

        [DllImport(DLL_NAME, EntryPoint = "dsymm_", CallingConvention = CallingConvention.Cdecl)]
        public static extern void dsymm_(IntPtr side, IntPtr uplo, IntPtr m, IntPtr n,
                   IntPtr alpha, IntPtr a, IntPtr lda, IntPtr b, IntPtr ldb,
                   IntPtr beta, IntPtr c, IntPtr ldc);

        [DllImport(DLL_NAME, EntryPoint = "csymm_", CallingConvention = CallingConvention.Cdecl)]
        public static extern void csymm_(IntPtr side, IntPtr uplo, IntPtr m, IntPtr n,
                   IntPtr alpha, IntPtr a, IntPtr lda,
                   IntPtr b, IntPtr ldb, IntPtr beta,
                   IntPtr c, IntPtr ldc);

        [DllImport(DLL_NAME, EntryPoint = "zsymm_", CallingConvention = CallingConvention.Cdecl)]
        public static extern void zsymm_(IntPtr side, IntPtr uplo, IntPtr m, IntPtr n,
                   IntPtr alpha, IntPtr a, IntPtr lda,
                   IntPtr b, IntPtr ldb, IntPtr beta,
                   IntPtr c, IntPtr ldc);

        [DllImport(DLL_NAME, EntryPoint = "ssymm", CallingConvention = CallingConvention.Cdecl)]
        public static extern void ssymm(IntPtr side, IntPtr uplo, IntPtr m, IntPtr n,
                   IntPtr alpha, IntPtr a, IntPtr lda, IntPtr b, IntPtr ldb,
                   IntPtr beta, IntPtr c, IntPtr ldc);

        [DllImport(DLL_NAME, EntryPoint = "dsymm", CallingConvention = CallingConvention.Cdecl)]
        public static extern void dsymm(IntPtr side, IntPtr uplo, IntPtr m, IntPtr n,
                   IntPtr alpha, IntPtr a, IntPtr lda, IntPtr b, IntPtr ldb,
                   IntPtr beta, IntPtr c, IntPtr ldc);

        [DllImport(DLL_NAME, EntryPoint = "csymm", CallingConvention = CallingConvention.Cdecl)]
        public static extern void csymm(IntPtr side, IntPtr uplo, IntPtr m, IntPtr n,
                   IntPtr alpha, IntPtr a, IntPtr lda,
                   IntPtr b, IntPtr ldb, IntPtr beta,
                   IntPtr c, IntPtr ldc);

        [DllImport(DLL_NAME, EntryPoint = "zsymm", CallingConvention = CallingConvention.Cdecl)]
        public static extern void zsymm(IntPtr side, IntPtr uplo, IntPtr m, IntPtr n,
                   IntPtr alpha, IntPtr a, IntPtr lda,
                   IntPtr b, IntPtr ldb, IntPtr beta,
                   IntPtr c, IntPtr ldc);

        /* HEMM */
        [DllImport(DLL_NAME, EntryPoint = "chemm_", CallingConvention = CallingConvention.Cdecl)]
        public static extern void chemm_(IntPtr side, IntPtr uplo, IntPtr m, IntPtr n,
                    IntPtr alpha, IntPtr a, IntPtr lda,
                    IntPtr b, IntPtr ldb, IntPtr beta,
                    IntPtr c, IntPtr ldc);

        [DllImport(DLL_NAME, EntryPoint = "zhemm_", CallingConvention = CallingConvention.Cdecl)]
        public static extern void zhemm_(IntPtr side, IntPtr uplo,
                   IntPtr m, IntPtr n, IntPtr alpha, IntPtr a, IntPtr lda,
                   IntPtr b, IntPtr ldb, IntPtr beta, IntPtr c, IntPtr ldc);

        /* HEMM with no underscore*/
        [DllImport(DLL_NAME, EntryPoint = "chemm", CallingConvention = CallingConvention.Cdecl)]
        public static extern void chemm(IntPtr side, IntPtr uplo,
                   IntPtr m, IntPtr n, IntPtr alpha, IntPtr a, IntPtr lda,
                   IntPtr b, IntPtr ldb, IntPtr beta, IntPtr c, IntPtr ldc);

        [DllImport(DLL_NAME, EntryPoint = "zhemm", CallingConvention = CallingConvention.Cdecl)]
        public static extern void zhemm(IntPtr side, IntPtr uplo,
                   IntPtr m, IntPtr n, IntPtr alpha, IntPtr a, IntPtr lda,
                   IntPtr b, IntPtr ldb, IntPtr beta, IntPtr c, IntPtr ldc);

        /* SYR2K */
        [DllImport(DLL_NAME, EntryPoint = "ssyr2k_", CallingConvention = CallingConvention.Cdecl)]
        public static extern void ssyr2k_(IntPtr uplo, IntPtr trans, IntPtr n, IntPtr k,
                   IntPtr alpha,
                   IntPtr a, IntPtr lda,
                   IntPtr b, IntPtr ldb,
                   IntPtr beta, IntPtr c, IntPtr ldc);

        [DllImport(DLL_NAME, EntryPoint = "dsyr2k_", CallingConvention = CallingConvention.Cdecl)]
        public static extern void dsyr2k_(IntPtr uplo, IntPtr trans, IntPtr n, IntPtr k,
                   IntPtr alpha,
                   IntPtr a, IntPtr lda,
                   IntPtr b, IntPtr ldb,
                   IntPtr beta,
                   IntPtr c, IntPtr ldc);

        [DllImport(DLL_NAME, EntryPoint = "csyr2k_", CallingConvention = CallingConvention.Cdecl)]
        public static extern void csyr2k_(IntPtr uplo, IntPtr trans, IntPtr n, IntPtr k,
                   IntPtr alpha,
                   IntPtr a, IntPtr lda,
                   IntPtr b, IntPtr ldb,
                   IntPtr beta, IntPtr c, IntPtr ldc);

        [DllImport(DLL_NAME, EntryPoint = "zsyr2k_", CallingConvention = CallingConvention.Cdecl)]
        public static extern void zsyr2k_(IntPtr uplo, IntPtr trans, IntPtr n, IntPtr k,
                   IntPtr alpha,
                   IntPtr a, IntPtr lda,
                   IntPtr b, IntPtr ldb,
                   IntPtr beta, IntPtr c, IntPtr ldc);

        /* SYR2K no_underscore*/
        [DllImport(DLL_NAME, EntryPoint = "ssyr2k", CallingConvention = CallingConvention.Cdecl)]
        public static extern void ssyr2k(IntPtr uplo, IntPtr trans, IntPtr n, IntPtr k,
                   IntPtr alpha, IntPtr a, IntPtr lda,
                   IntPtr b, IntPtr ldb,
                   IntPtr beta, IntPtr c, IntPtr ldc);

        [DllImport(DLL_NAME, EntryPoint = "dsyr2k", CallingConvention = CallingConvention.Cdecl)]
        public static extern void dsyr2k(IntPtr uplo, IntPtr trans, IntPtr n, IntPtr k,
                   IntPtr alpha,
                   IntPtr a, IntPtr lda,
                   IntPtr b, IntPtr ldb,
                   IntPtr beta,
                   IntPtr c, IntPtr ldc);

        [DllImport(DLL_NAME, EntryPoint = "csyr2k", CallingConvention = CallingConvention.Cdecl)]
        public static extern void csyr2k(IntPtr uplo, IntPtr trans, IntPtr n, IntPtr k,
                   IntPtr alpha,
                   IntPtr a, IntPtr lda,
                   IntPtr b, IntPtr ldb,
                   IntPtr beta, IntPtr c, IntPtr ldc);

        [DllImport(DLL_NAME, EntryPoint = "zsyr2k", CallingConvention = CallingConvention.Cdecl)]
        public static extern void zsyr2k(IntPtr uplo, IntPtr trans, IntPtr n, IntPtr k,
                   IntPtr alpha,
                   IntPtr a, IntPtr lda,
                   IntPtr b, IntPtr ldb,
                   IntPtr beta, IntPtr c, IntPtr ldc);

        /* HERK */
        [DllImport(DLL_NAME, EntryPoint = "cher2k_", CallingConvention = CallingConvention.Cdecl)]
        public static extern void cher2k_(IntPtr uplo, IntPtr trans, IntPtr n, IntPtr k,
                   IntPtr alpha,
                   IntPtr a, IntPtr lda,
                   IntPtr b, IntPtr ldb,
                   IntPtr beta, IntPtr c, IntPtr ldc);

        [DllImport(DLL_NAME, EntryPoint = "zher2k_", CallingConvention = CallingConvention.Cdecl)]
        public static extern void zher2k_(IntPtr uplo, IntPtr trans, IntPtr n, IntPtr k,
                   IntPtr alpha,
                   IntPtr a, IntPtr lda,
                   IntPtr b, IntPtr ldb,
                   IntPtr beta, IntPtr c, IntPtr ldc);

        /* HER2K with no underscore */
        [DllImport(DLL_NAME, EntryPoint = "cher2k", CallingConvention = CallingConvention.Cdecl)]
        public static extern void cher2k(IntPtr uplo, IntPtr trans, IntPtr n, IntPtr k,
                   IntPtr alpha,
                   IntPtr a, IntPtr lda,
                   IntPtr b, IntPtr ldb,
                   IntPtr beta, IntPtr c, IntPtr ldc);

        [DllImport(DLL_NAME, EntryPoint = "zher2k", CallingConvention = CallingConvention.Cdecl)]
        public static extern void zher2k(IntPtr uplo, IntPtr trans, IntPtr n, IntPtr k,
                   IntPtr alpha,
                   IntPtr a, IntPtr lda,
                   IntPtr b, IntPtr ldb,
                   IntPtr beta, IntPtr c, IntPtr ldc);

        /* TRMM */
        [DllImport(DLL_NAME, EntryPoint = "strmm_", CallingConvention = CallingConvention.Cdecl)]
        public static extern void strmm_(IntPtr side, IntPtr uplo, IntPtr transa, IntPtr diag,
                   IntPtr m, IntPtr n, IntPtr alpha, IntPtr a, IntPtr lda,
                   IntPtr b, IntPtr ldb);

        [DllImport(DLL_NAME, EntryPoint = "dtrmm_", CallingConvention = CallingConvention.Cdecl)]
        public static extern void dtrmm_(IntPtr side, IntPtr uplo, IntPtr transa, IntPtr diag,
                   IntPtr m, IntPtr n, IntPtr alpha, IntPtr a, IntPtr lda,
                   IntPtr b, IntPtr ldb);

        [DllImport(DLL_NAME, EntryPoint = "ctrmm_", CallingConvention = CallingConvention.Cdecl)]
        public static extern void ctrmm_(IntPtr side, IntPtr uplo, IntPtr transa, IntPtr diag,
                   IntPtr m, IntPtr n, IntPtr alpha,
                   IntPtr a, IntPtr lda, IntPtr b, IntPtr ldb);

        [DllImport(DLL_NAME, EntryPoint = "ztrmm_", CallingConvention = CallingConvention.Cdecl)]
        public static extern void ztrmm_(IntPtr side, IntPtr uplo, IntPtr transa, IntPtr diag,
                   IntPtr m, IntPtr n, IntPtr alpha,
                   IntPtr a, IntPtr lda, IntPtr b, IntPtr ldb);

        [DllImport(DLL_NAME, EntryPoint = "strmm", CallingConvention = CallingConvention.Cdecl)]
        public static extern void strmm(IntPtr side, IntPtr uplo, IntPtr transa, IntPtr diag,
                   IntPtr m, IntPtr n, IntPtr alpha, IntPtr a, IntPtr lda,
                   IntPtr b, IntPtr ldb);

        [DllImport(DLL_NAME, EntryPoint = "dtrmm", CallingConvention = CallingConvention.Cdecl)]
        public static extern void dtrmm(IntPtr side, IntPtr uplo, IntPtr transa, IntPtr diag,
                   IntPtr m, IntPtr n, IntPtr alpha, IntPtr a, IntPtr lda,
                   IntPtr b, IntPtr ldb);

        [DllImport(DLL_NAME, EntryPoint = "ctrmm", CallingConvention = CallingConvention.Cdecl)]
        public static extern void ctrmm(IntPtr side, IntPtr uplo, IntPtr transa, IntPtr diag,
                   IntPtr m, IntPtr n, IntPtr alpha,
                   IntPtr a, IntPtr lda, IntPtr b, IntPtr ldb);

        [DllImport(DLL_NAME, EntryPoint = "ztrmm", CallingConvention = CallingConvention.Cdecl)]
        public static extern void ztrmm(IntPtr side, IntPtr uplo, IntPtr transa, IntPtr diag,
                   IntPtr m, IntPtr n, IntPtr alpha,
                   IntPtr a, IntPtr lda, IntPtr b, IntPtr ldb);
    }

    internal class nvblas91_imports
    {
        private const string DLL_NAME = "nvblas64_91.dll";

        [DllImport(DLL_NAME, EntryPoint = "sgemm_", CallingConvention = CallingConvention.Cdecl)]
        public static extern void sgemm_(IntPtr transa, IntPtr transb, IntPtr m, IntPtr n, IntPtr k,
                     IntPtr alpha, IntPtr a, IntPtr lda, IntPtr b, IntPtr ldb,
                     IntPtr beta, IntPtr c, IntPtr ldc);

        [DllImport(DLL_NAME, EntryPoint = "dgemm_", CallingConvention = CallingConvention.Cdecl)]
        public static extern void dgemm_(IntPtr transa, IntPtr transb, IntPtr m, IntPtr n, IntPtr k,
                   IntPtr alpha, IntPtr a, IntPtr lda, IntPtr b, IntPtr ldb,
                   IntPtr beta, IntPtr c, IntPtr ldc);

        [DllImport(DLL_NAME, EntryPoint = "cgemm_", CallingConvention = CallingConvention.Cdecl)]
        public static extern void cgemm_(IntPtr transa, IntPtr transb, IntPtr m, IntPtr n, IntPtr k,
                   IntPtr alpha, IntPtr a, IntPtr lda,
                   IntPtr b, IntPtr ldb, IntPtr beta,
                   IntPtr c, IntPtr ldc);

        [DllImport(DLL_NAME, EntryPoint = "zgemm_", CallingConvention = CallingConvention.Cdecl)]
        public static extern void zgemm_(IntPtr transa, IntPtr transb, IntPtr m, IntPtr n, IntPtr k,
                   IntPtr alpha, IntPtr a, IntPtr lda,
                   IntPtr b, IntPtr ldb, IntPtr beta,
                   IntPtr c, IntPtr ldc);

        [DllImport(DLL_NAME, EntryPoint = "sgemm", CallingConvention = CallingConvention.Cdecl)]
        public static extern void sgemm(IntPtr transa, IntPtr transb, IntPtr m, IntPtr n, IntPtr k,
                   IntPtr alpha, IntPtr a, IntPtr lda, IntPtr b, IntPtr ldb,
                   IntPtr beta, IntPtr c, IntPtr ldc);

        [DllImport(DLL_NAME, EntryPoint = "dgemm", CallingConvention = CallingConvention.Cdecl)]
        public static extern void dgemm(IntPtr transa, IntPtr transb, IntPtr m, IntPtr n, IntPtr k,
                   IntPtr alpha, IntPtr a, IntPtr lda, IntPtr b, IntPtr ldb,
                   IntPtr beta, IntPtr c, IntPtr ldc);

        [DllImport(DLL_NAME, EntryPoint = "cgemm", CallingConvention = CallingConvention.Cdecl)]
        public static extern void cgemm(IntPtr transa, IntPtr transb, IntPtr m, IntPtr n, IntPtr k,
                   IntPtr alpha, IntPtr a, IntPtr lda,
                   IntPtr b, IntPtr ldb, IntPtr beta,
                   IntPtr c, IntPtr ldc);

        [DllImport(DLL_NAME, EntryPoint = "zgemm", CallingConvention = CallingConvention.Cdecl)]
        public static extern void zgemm(IntPtr transa, IntPtr transb, IntPtr m, IntPtr n, IntPtr k,
                   IntPtr alpha, IntPtr a, IntPtr lda,
                   IntPtr b, IntPtr ldb, IntPtr beta,
                   IntPtr c, IntPtr ldc);

        /* SYRK */
        [DllImport(DLL_NAME, EntryPoint = "ssyrk_", CallingConvention = CallingConvention.Cdecl)]
        public static extern void ssyrk_(IntPtr uplo, IntPtr trans, IntPtr n, IntPtr k,
                   IntPtr alpha, IntPtr a, IntPtr lda,
                   IntPtr beta, IntPtr c, IntPtr ldc);

        [DllImport(DLL_NAME, EntryPoint = "dsyrk_", CallingConvention = CallingConvention.Cdecl)]
        public static extern void dsyrk_(IntPtr uplo, IntPtr trans, IntPtr n, IntPtr k,
                   IntPtr alpha, IntPtr a, IntPtr lda, IntPtr beta,
                   IntPtr c, IntPtr ldc);

        [DllImport(DLL_NAME, EntryPoint = "csyrk_", CallingConvention = CallingConvention.Cdecl)]
        public static extern void csyrk_(IntPtr uplo, IntPtr trans, IntPtr n, IntPtr k,
                   IntPtr alpha, IntPtr a, IntPtr lda,
                   IntPtr beta, IntPtr c, IntPtr ldc);

        [DllImport(DLL_NAME, EntryPoint = "zsyrk_", CallingConvention = CallingConvention.Cdecl)]
        public static extern void zsyrk_(IntPtr uplo, IntPtr trans, IntPtr n, IntPtr k,
                   IntPtr alpha, IntPtr a, IntPtr lda,
                   IntPtr beta, IntPtr c, IntPtr ldc);

        [DllImport(DLL_NAME, EntryPoint = "ssyrk", CallingConvention = CallingConvention.Cdecl)]
        public static extern void ssyrk(IntPtr uplo, IntPtr trans, IntPtr n, IntPtr k,
                   IntPtr alpha, IntPtr a, IntPtr lda,
                   IntPtr beta, IntPtr c, IntPtr ldc);

        [DllImport(DLL_NAME, EntryPoint = "dsyrk", CallingConvention = CallingConvention.Cdecl)]
        public static extern void dsyrk(IntPtr uplo, IntPtr trans, IntPtr n, IntPtr k,
                   IntPtr alpha, IntPtr a, IntPtr lda, IntPtr beta,
                   IntPtr c, IntPtr ldc);

        [DllImport(DLL_NAME, EntryPoint = "csyrk", CallingConvention = CallingConvention.Cdecl)]
        public static extern void csyrk(IntPtr uplo, IntPtr trans, IntPtr n, IntPtr k,
                   IntPtr alpha, IntPtr a, IntPtr lda,
                   IntPtr beta, IntPtr c, IntPtr ldc);

        [DllImport(DLL_NAME, EntryPoint = "zsyrk", CallingConvention = CallingConvention.Cdecl)]
        public static extern void zsyrk(IntPtr uplo, IntPtr trans, IntPtr n, IntPtr k,
                   IntPtr alpha, IntPtr a, IntPtr lda,
                   IntPtr beta, IntPtr c, IntPtr ldc);

        /* HERK */
        [DllImport(DLL_NAME, EntryPoint = "cherk_", CallingConvention = CallingConvention.Cdecl)]
        public static extern void cherk_(IntPtr uplo, IntPtr trans, IntPtr n, IntPtr k,
                   IntPtr alpha, IntPtr a, IntPtr lda, IntPtr beta,
                   IntPtr c, IntPtr ldc);

        [DllImport(DLL_NAME, EntryPoint = "zherk_", CallingConvention = CallingConvention.Cdecl)]
        public static extern void zherk_(IntPtr uplo, IntPtr trans, IntPtr n, IntPtr k,
                   IntPtr alpha, IntPtr a, IntPtr lda,
                   IntPtr beta, IntPtr c, IntPtr ldc);

        [DllImport(DLL_NAME, EntryPoint = "cherk", CallingConvention = CallingConvention.Cdecl)]
        public static extern void cherk(IntPtr uplo, IntPtr trans, IntPtr n, IntPtr k,
                   IntPtr alpha, IntPtr a, IntPtr lda, IntPtr beta,
                   IntPtr c, IntPtr ldc);

        [DllImport(DLL_NAME, EntryPoint = "zherk", CallingConvention = CallingConvention.Cdecl)]
        public static extern void zherk(IntPtr uplo, IntPtr trans, IntPtr n, IntPtr k,
                   IntPtr alpha, IntPtr a, IntPtr lda,
                   IntPtr beta, IntPtr c, IntPtr ldc);

        /* TRSM */
        [DllImport(DLL_NAME, EntryPoint = "strsm_", CallingConvention = CallingConvention.Cdecl)]
        public static extern void strsm_(IntPtr side, IntPtr uplo, IntPtr transa, IntPtr diag,
                   IntPtr m, IntPtr n, IntPtr alpha, IntPtr a, IntPtr lda,
                   IntPtr b, IntPtr ldb);

        [DllImport(DLL_NAME, EntryPoint = "dtrsm_", CallingConvention = CallingConvention.Cdecl)]
        public static extern void dtrsm_(IntPtr side, IntPtr uplo, IntPtr transa, IntPtr diag,
                   IntPtr m, IntPtr n, IntPtr alpha, IntPtr a, IntPtr lda,
                   IntPtr b, IntPtr ldb);

        [DllImport(DLL_NAME, EntryPoint = "ctrsm_", CallingConvention = CallingConvention.Cdecl)]
        public static extern void ctrsm_(IntPtr side, IntPtr uplo, IntPtr transa, IntPtr diag,
                   IntPtr m, IntPtr n, IntPtr alpha,
                   IntPtr a, IntPtr lda, IntPtr b, IntPtr ldb);

        [DllImport(DLL_NAME, EntryPoint = "ztrsm_", CallingConvention = CallingConvention.Cdecl)]
        public static extern void ztrsm_(IntPtr side, IntPtr uplo, IntPtr transa, IntPtr diag,
                   IntPtr m, IntPtr n, IntPtr alpha,
                   IntPtr a, IntPtr lda, IntPtr b, IntPtr ldb);

        [DllImport(DLL_NAME, EntryPoint = "strsm", CallingConvention = CallingConvention.Cdecl)]
        public static extern void strsm(IntPtr side, IntPtr uplo, IntPtr transa, IntPtr diag,
                   IntPtr m, IntPtr n, IntPtr alpha, IntPtr a, IntPtr lda,
                   IntPtr b, IntPtr ldb);

        [DllImport(DLL_NAME, EntryPoint = "dtrsm", CallingConvention = CallingConvention.Cdecl)]
        public static extern void dtrsm(IntPtr side, IntPtr uplo, IntPtr transa, IntPtr diag,
                   IntPtr m, IntPtr n, IntPtr alpha, IntPtr a, IntPtr lda,
                   IntPtr b, IntPtr ldb);

        [DllImport(DLL_NAME, EntryPoint = "ctrsm", CallingConvention = CallingConvention.Cdecl)]
        public static extern void ctrsm(IntPtr side, IntPtr uplo, IntPtr transa, IntPtr diag,
                   IntPtr m, IntPtr n, IntPtr alpha,
                   IntPtr a, IntPtr lda, IntPtr b, IntPtr ldb);

        [DllImport(DLL_NAME, EntryPoint = "ztrsm", CallingConvention = CallingConvention.Cdecl)]
        public static extern void ztrsm(IntPtr side, IntPtr uplo, IntPtr transa, IntPtr diag,
                   IntPtr m, IntPtr n, IntPtr alpha,
                   IntPtr a, IntPtr lda, IntPtr b, IntPtr ldb);

        /* SYMM */
        [DllImport(DLL_NAME, EntryPoint = "ssymm_", CallingConvention = CallingConvention.Cdecl)]
        public static extern void ssymm_(IntPtr side, IntPtr uplo, IntPtr m, IntPtr n,
                   IntPtr alpha, IntPtr a, IntPtr lda, IntPtr b, IntPtr ldb,
                   IntPtr beta, IntPtr c, IntPtr ldc);

        [DllImport(DLL_NAME, EntryPoint = "dsymm_", CallingConvention = CallingConvention.Cdecl)]
        public static extern void dsymm_(IntPtr side, IntPtr uplo, IntPtr m, IntPtr n,
                   IntPtr alpha, IntPtr a, IntPtr lda, IntPtr b, IntPtr ldb,
                   IntPtr beta, IntPtr c, IntPtr ldc);

        [DllImport(DLL_NAME, EntryPoint = "csymm_", CallingConvention = CallingConvention.Cdecl)]
        public static extern void csymm_(IntPtr side, IntPtr uplo, IntPtr m, IntPtr n,
                   IntPtr alpha, IntPtr a, IntPtr lda,
                   IntPtr b, IntPtr ldb, IntPtr beta,
                   IntPtr c, IntPtr ldc);

        [DllImport(DLL_NAME, EntryPoint = "zsymm_", CallingConvention = CallingConvention.Cdecl)]
        public static extern void zsymm_(IntPtr side, IntPtr uplo, IntPtr m, IntPtr n,
                   IntPtr alpha, IntPtr a, IntPtr lda,
                   IntPtr b, IntPtr ldb, IntPtr beta,
                   IntPtr c, IntPtr ldc);

        [DllImport(DLL_NAME, EntryPoint = "ssymm", CallingConvention = CallingConvention.Cdecl)]
        public static extern void ssymm(IntPtr side, IntPtr uplo, IntPtr m, IntPtr n,
                   IntPtr alpha, IntPtr a, IntPtr lda, IntPtr b, IntPtr ldb,
                   IntPtr beta, IntPtr c, IntPtr ldc);

        [DllImport(DLL_NAME, EntryPoint = "dsymm", CallingConvention = CallingConvention.Cdecl)]
        public static extern void dsymm(IntPtr side, IntPtr uplo, IntPtr m, IntPtr n,
                   IntPtr alpha, IntPtr a, IntPtr lda, IntPtr b, IntPtr ldb,
                   IntPtr beta, IntPtr c, IntPtr ldc);

        [DllImport(DLL_NAME, EntryPoint = "csymm", CallingConvention = CallingConvention.Cdecl)]
        public static extern void csymm(IntPtr side, IntPtr uplo, IntPtr m, IntPtr n,
                   IntPtr alpha, IntPtr a, IntPtr lda,
                   IntPtr b, IntPtr ldb, IntPtr beta,
                   IntPtr c, IntPtr ldc);

        [DllImport(DLL_NAME, EntryPoint = "zsymm", CallingConvention = CallingConvention.Cdecl)]
        public static extern void zsymm(IntPtr side, IntPtr uplo, IntPtr m, IntPtr n,
                   IntPtr alpha, IntPtr a, IntPtr lda,
                   IntPtr b, IntPtr ldb, IntPtr beta,
                   IntPtr c, IntPtr ldc);

        /* HEMM */
        [DllImport(DLL_NAME, EntryPoint = "chemm_", CallingConvention = CallingConvention.Cdecl)]
        public static extern void chemm_(IntPtr side, IntPtr uplo, IntPtr m, IntPtr n,
                    IntPtr alpha, IntPtr a, IntPtr lda,
                    IntPtr b, IntPtr ldb, IntPtr beta,
                    IntPtr c, IntPtr ldc);

        [DllImport(DLL_NAME, EntryPoint = "zhemm_", CallingConvention = CallingConvention.Cdecl)]
        public static extern void zhemm_(IntPtr side, IntPtr uplo,
                   IntPtr m, IntPtr n, IntPtr alpha, IntPtr a, IntPtr lda,
                   IntPtr b, IntPtr ldb, IntPtr beta, IntPtr c, IntPtr ldc);

        /* HEMM with no underscore*/
        [DllImport(DLL_NAME, EntryPoint = "chemm", CallingConvention = CallingConvention.Cdecl)]
        public static extern void chemm(IntPtr side, IntPtr uplo,
                   IntPtr m, IntPtr n, IntPtr alpha, IntPtr a, IntPtr lda,
                   IntPtr b, IntPtr ldb, IntPtr beta, IntPtr c, IntPtr ldc);

        [DllImport(DLL_NAME, EntryPoint = "zhemm", CallingConvention = CallingConvention.Cdecl)]
        public static extern void zhemm(IntPtr side, IntPtr uplo,
                   IntPtr m, IntPtr n, IntPtr alpha, IntPtr a, IntPtr lda,
                   IntPtr b, IntPtr ldb, IntPtr beta, IntPtr c, IntPtr ldc);

        /* SYR2K */
        [DllImport(DLL_NAME, EntryPoint = "ssyr2k_", CallingConvention = CallingConvention.Cdecl)]
        public static extern void ssyr2k_(IntPtr uplo, IntPtr trans, IntPtr n, IntPtr k,
                   IntPtr alpha,
                   IntPtr a, IntPtr lda,
                   IntPtr b, IntPtr ldb,
                   IntPtr beta, IntPtr c, IntPtr ldc);

        [DllImport(DLL_NAME, EntryPoint = "dsyr2k_", CallingConvention = CallingConvention.Cdecl)]
        public static extern void dsyr2k_(IntPtr uplo, IntPtr trans, IntPtr n, IntPtr k,
                   IntPtr alpha,
                   IntPtr a, IntPtr lda,
                   IntPtr b, IntPtr ldb,
                   IntPtr beta,
                   IntPtr c, IntPtr ldc);

        [DllImport(DLL_NAME, EntryPoint = "csyr2k_", CallingConvention = CallingConvention.Cdecl)]
        public static extern void csyr2k_(IntPtr uplo, IntPtr trans, IntPtr n, IntPtr k,
                   IntPtr alpha,
                   IntPtr a, IntPtr lda,
                   IntPtr b, IntPtr ldb,
                   IntPtr beta, IntPtr c, IntPtr ldc);

        [DllImport(DLL_NAME, EntryPoint = "zsyr2k_", CallingConvention = CallingConvention.Cdecl)]
        public static extern void zsyr2k_(IntPtr uplo, IntPtr trans, IntPtr n, IntPtr k,
                   IntPtr alpha,
                   IntPtr a, IntPtr lda,
                   IntPtr b, IntPtr ldb,
                   IntPtr beta, IntPtr c, IntPtr ldc);

        /* SYR2K no_underscore*/
        [DllImport(DLL_NAME, EntryPoint = "ssyr2k", CallingConvention = CallingConvention.Cdecl)]
        public static extern void ssyr2k(IntPtr uplo, IntPtr trans, IntPtr n, IntPtr k,
                   IntPtr alpha, IntPtr a, IntPtr lda,
                   IntPtr b, IntPtr ldb,
                   IntPtr beta, IntPtr c, IntPtr ldc);

        [DllImport(DLL_NAME, EntryPoint = "dsyr2k", CallingConvention = CallingConvention.Cdecl)]
        public static extern void dsyr2k(IntPtr uplo, IntPtr trans, IntPtr n, IntPtr k,
                   IntPtr alpha,
                   IntPtr a, IntPtr lda,
                   IntPtr b, IntPtr ldb,
                   IntPtr beta,
                   IntPtr c, IntPtr ldc);

        [DllImport(DLL_NAME, EntryPoint = "csyr2k", CallingConvention = CallingConvention.Cdecl)]
        public static extern void csyr2k(IntPtr uplo, IntPtr trans, IntPtr n, IntPtr k,
                   IntPtr alpha,
                   IntPtr a, IntPtr lda,
                   IntPtr b, IntPtr ldb,
                   IntPtr beta, IntPtr c, IntPtr ldc);

        [DllImport(DLL_NAME, EntryPoint = "zsyr2k", CallingConvention = CallingConvention.Cdecl)]
        public static extern void zsyr2k(IntPtr uplo, IntPtr trans, IntPtr n, IntPtr k,
                   IntPtr alpha,
                   IntPtr a, IntPtr lda,
                   IntPtr b, IntPtr ldb,
                   IntPtr beta, IntPtr c, IntPtr ldc);

        /* HERK */
        [DllImport(DLL_NAME, EntryPoint = "cher2k_", CallingConvention = CallingConvention.Cdecl)]
        public static extern void cher2k_(IntPtr uplo, IntPtr trans, IntPtr n, IntPtr k,
                   IntPtr alpha,
                   IntPtr a, IntPtr lda,
                   IntPtr b, IntPtr ldb,
                   IntPtr beta, IntPtr c, IntPtr ldc);

        [DllImport(DLL_NAME, EntryPoint = "zher2k_", CallingConvention = CallingConvention.Cdecl)]
        public static extern void zher2k_(IntPtr uplo, IntPtr trans, IntPtr n, IntPtr k,
                   IntPtr alpha,
                   IntPtr a, IntPtr lda,
                   IntPtr b, IntPtr ldb,
                   IntPtr beta, IntPtr c, IntPtr ldc);

        /* HER2K with no underscore */
        [DllImport(DLL_NAME, EntryPoint = "cher2k", CallingConvention = CallingConvention.Cdecl)]
        public static extern void cher2k(IntPtr uplo, IntPtr trans, IntPtr n, IntPtr k,
                   IntPtr alpha,
                   IntPtr a, IntPtr lda,
                   IntPtr b, IntPtr ldb,
                   IntPtr beta, IntPtr c, IntPtr ldc);

        [DllImport(DLL_NAME, EntryPoint = "zher2k", CallingConvention = CallingConvention.Cdecl)]
        public static extern void zher2k(IntPtr uplo, IntPtr trans, IntPtr n, IntPtr k,
                   IntPtr alpha,
                   IntPtr a, IntPtr lda,
                   IntPtr b, IntPtr ldb,
                   IntPtr beta, IntPtr c, IntPtr ldc);

        /* TRMM */
        [DllImport(DLL_NAME, EntryPoint = "strmm_", CallingConvention = CallingConvention.Cdecl)]
        public static extern void strmm_(IntPtr side, IntPtr uplo, IntPtr transa, IntPtr diag,
                   IntPtr m, IntPtr n, IntPtr alpha, IntPtr a, IntPtr lda,
                   IntPtr b, IntPtr ldb);

        [DllImport(DLL_NAME, EntryPoint = "dtrmm_", CallingConvention = CallingConvention.Cdecl)]
        public static extern void dtrmm_(IntPtr side, IntPtr uplo, IntPtr transa, IntPtr diag,
                   IntPtr m, IntPtr n, IntPtr alpha, IntPtr a, IntPtr lda,
                   IntPtr b, IntPtr ldb);

        [DllImport(DLL_NAME, EntryPoint = "ctrmm_", CallingConvention = CallingConvention.Cdecl)]
        public static extern void ctrmm_(IntPtr side, IntPtr uplo, IntPtr transa, IntPtr diag,
                   IntPtr m, IntPtr n, IntPtr alpha,
                   IntPtr a, IntPtr lda, IntPtr b, IntPtr ldb);

        [DllImport(DLL_NAME, EntryPoint = "ztrmm_", CallingConvention = CallingConvention.Cdecl)]
        public static extern void ztrmm_(IntPtr side, IntPtr uplo, IntPtr transa, IntPtr diag,
                   IntPtr m, IntPtr n, IntPtr alpha,
                   IntPtr a, IntPtr lda, IntPtr b, IntPtr ldb);

        [DllImport(DLL_NAME, EntryPoint = "strmm", CallingConvention = CallingConvention.Cdecl)]
        public static extern void strmm(IntPtr side, IntPtr uplo, IntPtr transa, IntPtr diag,
                   IntPtr m, IntPtr n, IntPtr alpha, IntPtr a, IntPtr lda,
                   IntPtr b, IntPtr ldb);

        [DllImport(DLL_NAME, EntryPoint = "dtrmm", CallingConvention = CallingConvention.Cdecl)]
        public static extern void dtrmm(IntPtr side, IntPtr uplo, IntPtr transa, IntPtr diag,
                   IntPtr m, IntPtr n, IntPtr alpha, IntPtr a, IntPtr lda,
                   IntPtr b, IntPtr ldb);

        [DllImport(DLL_NAME, EntryPoint = "ctrmm", CallingConvention = CallingConvention.Cdecl)]
        public static extern void ctrmm(IntPtr side, IntPtr uplo, IntPtr transa, IntPtr diag,
                   IntPtr m, IntPtr n, IntPtr alpha,
                   IntPtr a, IntPtr lda, IntPtr b, IntPtr ldb);

        [DllImport(DLL_NAME, EntryPoint = "ztrmm", CallingConvention = CallingConvention.Cdecl)]
        public static extern void ztrmm(IntPtr side, IntPtr uplo, IntPtr transa, IntPtr diag,
                   IntPtr m, IntPtr n, IntPtr alpha,
                   IntPtr a, IntPtr lda, IntPtr b, IntPtr ldb);
    }

    internal class nvblas90 : INVBLAS
    {
        public void cgemm(IntPtr transa, IntPtr transb, IntPtr m, IntPtr n, IntPtr k, IntPtr alpha, IntPtr a, IntPtr lda, IntPtr b, IntPtr ldb, IntPtr beta, IntPtr c, IntPtr ldc)
        {
            nvblas90_imports.cgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
        }

        public void cgemm_(IntPtr transa, IntPtr transb, IntPtr m, IntPtr n, IntPtr k, IntPtr alpha, IntPtr a, IntPtr lda, IntPtr b, IntPtr ldb, IntPtr beta, IntPtr c, IntPtr ldc)
        {
            nvblas90_imports.cgemm_(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
        }

        public void chemm(IntPtr side, IntPtr uplo, IntPtr m, IntPtr n, IntPtr alpha, IntPtr a, IntPtr lda, IntPtr b, IntPtr ldb, IntPtr beta, IntPtr c, IntPtr ldc)
        {
            nvblas90_imports.chemm(side, uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc);
        }

        public void chemm_(IntPtr side, IntPtr uplo, IntPtr m, IntPtr n, IntPtr alpha, IntPtr a, IntPtr lda, IntPtr b, IntPtr ldb, IntPtr beta, IntPtr c, IntPtr ldc)
        {
            nvblas90_imports.chemm_(side, uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc);
        }

        public void cher2k(IntPtr uplo, IntPtr trans, IntPtr n, IntPtr k, IntPtr alpha, IntPtr a, IntPtr lda, IntPtr b, IntPtr ldb, IntPtr beta, IntPtr c, IntPtr ldc)
        {
            nvblas90_imports.cher2k(uplo, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
        }

        public void cher2k_(IntPtr uplo, IntPtr trans, IntPtr n, IntPtr k, IntPtr alpha, IntPtr a, IntPtr lda, IntPtr b, IntPtr ldb, IntPtr beta, IntPtr c, IntPtr ldc)
        {
            nvblas90_imports.cher2k_(uplo, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
        }

        public void cherk(IntPtr uplo, IntPtr trans, IntPtr n, IntPtr k, IntPtr alpha, IntPtr a, IntPtr lda, IntPtr beta, IntPtr c, IntPtr ldc)
        {
            nvblas90_imports.cherk(uplo, trans, n, k, alpha, a, lda, beta, c, ldc);
        }

        public void cherk_(IntPtr uplo, IntPtr trans, IntPtr n, IntPtr k, IntPtr alpha, IntPtr a, IntPtr lda, IntPtr beta, IntPtr c, IntPtr ldc)
        {
            nvblas90_imports.cherk_(uplo, trans, n, k, alpha, a, lda, beta, c, ldc);
        }

        public void csymm(IntPtr side, IntPtr uplo, IntPtr m, IntPtr n, IntPtr alpha, IntPtr a, IntPtr lda, IntPtr b, IntPtr ldb, IntPtr beta, IntPtr c, IntPtr ldc)
        {
            nvblas90_imports.csymm(side, uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc);
        }

        public void csymm_(IntPtr side, IntPtr uplo, IntPtr m, IntPtr n, IntPtr alpha, IntPtr a, IntPtr lda, IntPtr b, IntPtr ldb, IntPtr beta, IntPtr c, IntPtr ldc)
        {
            nvblas90_imports.csymm_(side, uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc);
        }

        public void csyr2k(IntPtr uplo, IntPtr trans, IntPtr n, IntPtr k, IntPtr alpha, IntPtr a, IntPtr lda, IntPtr b, IntPtr ldb, IntPtr beta, IntPtr c, IntPtr ldc)
        {
            nvblas90_imports.csyr2k(uplo, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
        }

        public void csyr2k_(IntPtr uplo, IntPtr trans, IntPtr n, IntPtr k, IntPtr alpha, IntPtr a, IntPtr lda, IntPtr b, IntPtr ldb, IntPtr beta, IntPtr c, IntPtr ldc)
        {
            nvblas90_imports.csyr2k_(uplo, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
        }

        public void csyrk(IntPtr uplo, IntPtr trans, IntPtr n, IntPtr k, IntPtr alpha, IntPtr a, IntPtr lda, IntPtr beta, IntPtr c, IntPtr ldc)
        {
            nvblas90_imports.csyrk(uplo, trans, n, k, alpha, a, lda, beta, c, ldc);
        }

        public void csyrk_(IntPtr uplo, IntPtr trans, IntPtr n, IntPtr k, IntPtr alpha, IntPtr a, IntPtr lda, IntPtr beta, IntPtr c, IntPtr ldc)
        {
            nvblas90_imports.csyrk_(uplo, trans, n, k, alpha, a, lda, beta, c, ldc);
        }

        public void ctrmm(IntPtr side, IntPtr uplo, IntPtr transa, IntPtr diag, IntPtr m, IntPtr n, IntPtr alpha, IntPtr a, IntPtr lda, IntPtr b, IntPtr ldb)
        {
            nvblas90_imports.ctrmm(side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb);
        }

        public void ctrmm_(IntPtr side, IntPtr uplo, IntPtr transa, IntPtr diag, IntPtr m, IntPtr n, IntPtr alpha, IntPtr a, IntPtr lda, IntPtr b, IntPtr ldb)
        {
            nvblas90_imports.ctrmm_(side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb);
        }

        public void ctrsm(IntPtr side, IntPtr uplo, IntPtr transa, IntPtr diag, IntPtr m, IntPtr n, IntPtr alpha, IntPtr a, IntPtr lda, IntPtr b, IntPtr ldb)
        {
            nvblas90_imports.ctrsm(side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb);
        }

        public void ctrsm_(IntPtr side, IntPtr uplo, IntPtr transa, IntPtr diag, IntPtr m, IntPtr n, IntPtr alpha, IntPtr a, IntPtr lda, IntPtr b, IntPtr ldb)
        {
            nvblas90_imports.ctrsm_(side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb);
        }

        public void dgemm(IntPtr transa, IntPtr transb, IntPtr m, IntPtr n, IntPtr k, IntPtr alpha, IntPtr a, IntPtr lda, IntPtr b, IntPtr ldb, IntPtr beta, IntPtr c, IntPtr ldc)
        {
            nvblas90_imports.dgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
        }

        public void dgemm_(IntPtr transa, IntPtr transb, IntPtr m, IntPtr n, IntPtr k, IntPtr alpha, IntPtr a, IntPtr lda, IntPtr b, IntPtr ldb, IntPtr beta, IntPtr c, IntPtr ldc)
        {
            nvblas90_imports.dgemm_(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
        }

        public void dsymm(IntPtr side, IntPtr uplo, IntPtr m, IntPtr n, IntPtr alpha, IntPtr a, IntPtr lda, IntPtr b, IntPtr ldb, IntPtr beta, IntPtr c, IntPtr ldc)
        {
            nvblas90_imports.dsymm(side, uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc);
        }

        public void dsymm_(IntPtr side, IntPtr uplo, IntPtr m, IntPtr n, IntPtr alpha, IntPtr a, IntPtr lda, IntPtr b, IntPtr ldb, IntPtr beta, IntPtr c, IntPtr ldc)
        {
            nvblas90_imports.dsymm_(side, uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc);
        }

        public void dsyr2k(IntPtr uplo, IntPtr trans, IntPtr n, IntPtr k, IntPtr alpha, IntPtr a, IntPtr lda, IntPtr b, IntPtr ldb, IntPtr beta, IntPtr c, IntPtr ldc)
        {
            nvblas90_imports.dsyr2k(uplo, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
        }

        public void dsyr2k_(IntPtr uplo, IntPtr trans, IntPtr n, IntPtr k, IntPtr alpha, IntPtr a, IntPtr lda, IntPtr b, IntPtr ldb, IntPtr beta, IntPtr c, IntPtr ldc)
        {
            nvblas90_imports.dsyr2k_(uplo, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
        }

        public void dsyrk(IntPtr uplo, IntPtr trans, IntPtr n, IntPtr k, IntPtr alpha, IntPtr a, IntPtr lda, IntPtr beta, IntPtr c, IntPtr ldc)
        {
            nvblas90_imports.dsyrk(uplo, trans, n, k, alpha, a, lda, beta, c, ldc);
        }

        public void dsyrk_(IntPtr uplo, IntPtr trans, IntPtr n, IntPtr k, IntPtr alpha, IntPtr a, IntPtr lda, IntPtr beta, IntPtr c, IntPtr ldc)
        {
            nvblas90_imports.dsyrk_(uplo, trans, n, k, alpha, a, lda, beta, c, ldc);
        }

        public void dtrmm(IntPtr side, IntPtr uplo, IntPtr transa, IntPtr diag, IntPtr m, IntPtr n, IntPtr alpha, IntPtr a, IntPtr lda, IntPtr b, IntPtr ldb)
        {
            nvblas90_imports.dtrmm(side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb);
        }

        public void dtrmm_(IntPtr side, IntPtr uplo, IntPtr transa, IntPtr diag, IntPtr m, IntPtr n, IntPtr alpha, IntPtr a, IntPtr lda, IntPtr b, IntPtr ldb)
        {
            nvblas90_imports.dtrmm_(side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb);
        }

        public void dtrsm(IntPtr side, IntPtr uplo, IntPtr transa, IntPtr diag, IntPtr m, IntPtr n, IntPtr alpha, IntPtr a, IntPtr lda, IntPtr b, IntPtr ldb)
        {
            nvblas90_imports.dtrsm(side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb);
        }

        public void dtrsm_(IntPtr side, IntPtr uplo, IntPtr transa, IntPtr diag, IntPtr m, IntPtr n, IntPtr alpha, IntPtr a, IntPtr lda, IntPtr b, IntPtr ldb)
        {
            nvblas90_imports.dtrsm_(side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb);
        }

        public void sgemm(IntPtr transa, IntPtr transb, IntPtr m, IntPtr n, IntPtr k, IntPtr alpha, IntPtr a, IntPtr lda, IntPtr b, IntPtr ldb, IntPtr beta, IntPtr c, IntPtr ldc)
        {
            nvblas90_imports.sgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
        }

        public void sgemm_(IntPtr transa, IntPtr transb, IntPtr m, IntPtr n, IntPtr k, IntPtr alpha, IntPtr a, IntPtr lda, IntPtr b, IntPtr ldb, IntPtr beta, IntPtr c, IntPtr ldc)
        {
            nvblas90_imports.sgemm_(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
        }

        public void ssymm(IntPtr side, IntPtr uplo, IntPtr m, IntPtr n, IntPtr alpha, IntPtr a, IntPtr lda, IntPtr b, IntPtr ldb, IntPtr beta, IntPtr c, IntPtr ldc)
        {
            nvblas90_imports.ssymm(side, uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc);
        }

        public void ssymm_(IntPtr side, IntPtr uplo, IntPtr m, IntPtr n, IntPtr alpha, IntPtr a, IntPtr lda, IntPtr b, IntPtr ldb, IntPtr beta, IntPtr c, IntPtr ldc)
        {
            nvblas90_imports.ssymm_(side, uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc);
        }

        public void ssyr2k(IntPtr uplo, IntPtr trans, IntPtr n, IntPtr k, IntPtr alpha, IntPtr a, IntPtr lda, IntPtr b, IntPtr ldb, IntPtr beta, IntPtr c, IntPtr ldc)
        {
            nvblas90_imports.ssyr2k(uplo, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
        }

        public void ssyr2k_(IntPtr uplo, IntPtr trans, IntPtr n, IntPtr k, IntPtr alpha, IntPtr a, IntPtr lda, IntPtr b, IntPtr ldb, IntPtr beta, IntPtr c, IntPtr ldc)
        {
            nvblas90_imports.ssyr2k_(uplo, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
        }

        public void ssyrk(IntPtr uplo, IntPtr trans, IntPtr n, IntPtr k, IntPtr alpha, IntPtr a, IntPtr lda, IntPtr beta, IntPtr c, IntPtr ldc)
        {
            nvblas90_imports.ssyrk(uplo, trans, n, k, alpha, a, lda, beta, c, ldc);
        }

        public void ssyrk_(IntPtr uplo, IntPtr trans, IntPtr n, IntPtr k, IntPtr alpha, IntPtr a, IntPtr lda, IntPtr beta, IntPtr c, IntPtr ldc)
        {
            nvblas90_imports.ssyrk_(uplo, trans, n, k, alpha, a, lda, beta, c, ldc);
        }

        public void strmm(IntPtr side, IntPtr uplo, IntPtr transa, IntPtr diag, IntPtr m, IntPtr n, IntPtr alpha, IntPtr a, IntPtr lda, IntPtr b, IntPtr ldb)
        {
            nvblas90_imports.strmm(side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb);
        }

        public void strmm_(IntPtr side, IntPtr uplo, IntPtr transa, IntPtr diag, IntPtr m, IntPtr n, IntPtr alpha, IntPtr a, IntPtr lda, IntPtr b, IntPtr ldb)
        {
            nvblas90_imports.strmm_(side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb);
        }

        public void strsm(IntPtr side, IntPtr uplo, IntPtr transa, IntPtr diag, IntPtr m, IntPtr n, IntPtr alpha, IntPtr a, IntPtr lda, IntPtr b, IntPtr ldb)
        {
            nvblas90_imports.strsm(side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb);
        }

        public void strsm_(IntPtr side, IntPtr uplo, IntPtr transa, IntPtr diag, IntPtr m, IntPtr n, IntPtr alpha, IntPtr a, IntPtr lda, IntPtr b, IntPtr ldb)
        {
            nvblas90_imports.strsm_(side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb);
        }

        public void zgemm(IntPtr transa, IntPtr transb, IntPtr m, IntPtr n, IntPtr k, IntPtr alpha, IntPtr a, IntPtr lda, IntPtr b, IntPtr ldb, IntPtr beta, IntPtr c, IntPtr ldc)
        {
            nvblas90_imports.zgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
        }

        public void zgemm_(IntPtr transa, IntPtr transb, IntPtr m, IntPtr n, IntPtr k, IntPtr alpha, IntPtr a, IntPtr lda, IntPtr b, IntPtr ldb, IntPtr beta, IntPtr c, IntPtr ldc)
        {
            nvblas90_imports.zgemm_(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
        }

        public void zhemm(IntPtr side, IntPtr uplo, IntPtr m, IntPtr n, IntPtr alpha, IntPtr a, IntPtr lda, IntPtr b, IntPtr ldb, IntPtr beta, IntPtr c, IntPtr ldc)
        {
            nvblas90_imports.zhemm(side, uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc);
        }

        public void zhemm_(IntPtr side, IntPtr uplo, IntPtr m, IntPtr n, IntPtr alpha, IntPtr a, IntPtr lda, IntPtr b, IntPtr ldb, IntPtr beta, IntPtr c, IntPtr ldc)
        {
            nvblas90_imports.zhemm_(side, uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc);
        }

        public void zher2k(IntPtr uplo, IntPtr trans, IntPtr n, IntPtr k, IntPtr alpha, IntPtr a, IntPtr lda, IntPtr b, IntPtr ldb, IntPtr beta, IntPtr c, IntPtr ldc)
        {
            nvblas90_imports.zher2k(uplo, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
        }

        public void zher2k_(IntPtr uplo, IntPtr trans, IntPtr n, IntPtr k, IntPtr alpha, IntPtr a, IntPtr lda, IntPtr b, IntPtr ldb, IntPtr beta, IntPtr c, IntPtr ldc)
        {
            nvblas90_imports.zher2k_(uplo, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
        }

        public void zherk(IntPtr uplo, IntPtr trans, IntPtr n, IntPtr k, IntPtr alpha, IntPtr a, IntPtr lda, IntPtr beta, IntPtr c, IntPtr ldc)
        {
            nvblas90_imports.zherk(uplo, trans, n, k, alpha, a, lda, beta, c, ldc);
        }

        public void zherk_(IntPtr uplo, IntPtr trans, IntPtr n, IntPtr k, IntPtr alpha, IntPtr a, IntPtr lda, IntPtr beta, IntPtr c, IntPtr ldc)
        {
            nvblas90_imports.zherk_(uplo, trans, n, k, alpha, a, lda, beta, c, ldc);
        }

        public void zsymm(IntPtr side, IntPtr uplo, IntPtr m, IntPtr n, IntPtr alpha, IntPtr a, IntPtr lda, IntPtr b, IntPtr ldb, IntPtr beta, IntPtr c, IntPtr ldc)
        {
            nvblas90_imports.zsymm(side, uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc);
        }

        public void zsymm_(IntPtr side, IntPtr uplo, IntPtr m, IntPtr n, IntPtr alpha, IntPtr a, IntPtr lda, IntPtr b, IntPtr ldb, IntPtr beta, IntPtr c, IntPtr ldc)
        {
            nvblas90_imports.zsymm_(side, uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc);
        }

        public void zsyr2k(IntPtr uplo, IntPtr trans, IntPtr n, IntPtr k, IntPtr alpha, IntPtr a, IntPtr lda, IntPtr b, IntPtr ldb, IntPtr beta, IntPtr c, IntPtr ldc)
        {
            nvblas90_imports.zsyr2k(uplo, trans, b, k, alpha, a, lda, b, ldb, beta, c, ldc);
        }

        public void zsyr2k_(IntPtr uplo, IntPtr trans, IntPtr n, IntPtr k, IntPtr alpha, IntPtr a, IntPtr lda, IntPtr b, IntPtr ldb, IntPtr beta, IntPtr c, IntPtr ldc)
        {
            nvblas90_imports.zsyr2k_(uplo, trans, b, k, alpha, a, lda, b, ldb, beta, c, ldc);
        }

        public void zsyrk(IntPtr uplo, IntPtr trans, IntPtr n, IntPtr k, IntPtr alpha, IntPtr a, IntPtr lda, IntPtr beta, IntPtr c, IntPtr ldc)
        {
            nvblas90_imports.zsyrk(uplo, trans, n, k, alpha, a, lda, beta, c, ldc);
        }

        public void zsyrk_(IntPtr uplo, IntPtr trans, IntPtr n, IntPtr k, IntPtr alpha, IntPtr a, IntPtr lda, IntPtr beta, IntPtr c, IntPtr ldc)
        {
            nvblas90_imports.zsyrk_(uplo, trans, n, k, alpha, a, lda, beta, c, ldc);
        }

        public void ztrmm(IntPtr side, IntPtr uplo, IntPtr transa, IntPtr diag, IntPtr m, IntPtr n, IntPtr alpha, IntPtr a, IntPtr lda, IntPtr b, IntPtr ldb)
        {
            nvblas90_imports.ztrmm(side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb);
        }

        public void ztrmm_(IntPtr side, IntPtr uplo, IntPtr transa, IntPtr diag, IntPtr m, IntPtr n, IntPtr alpha, IntPtr a, IntPtr lda, IntPtr b, IntPtr ldb)
        {
            nvblas90_imports.ztrmm_(side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb);
        }

        public void ztrsm(IntPtr side, IntPtr uplo, IntPtr transa, IntPtr diag, IntPtr m, IntPtr n, IntPtr alpha, IntPtr a, IntPtr lda, IntPtr b, IntPtr ldb)
        {
            nvblas90_imports.ztrsm(side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb);
        }

        public void ztrsm_(IntPtr side, IntPtr uplo, IntPtr transa, IntPtr diag, IntPtr m, IntPtr n, IntPtr alpha, IntPtr a, IntPtr lda, IntPtr b, IntPtr ldb)
        {
            nvblas90_imports.ztrsm_(side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb);
        }
    }

    internal class nvblas91 : INVBLAS
    {
        public void cgemm(IntPtr transa, IntPtr transb, IntPtr m, IntPtr n, IntPtr k, IntPtr alpha, IntPtr a, IntPtr lda, IntPtr b, IntPtr ldb, IntPtr beta, IntPtr c, IntPtr ldc)
        {
            nvblas91_imports.cgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
        }

        public void cgemm_(IntPtr transa, IntPtr transb, IntPtr m, IntPtr n, IntPtr k, IntPtr alpha, IntPtr a, IntPtr lda, IntPtr b, IntPtr ldb, IntPtr beta, IntPtr c, IntPtr ldc)
        {
            nvblas91_imports.cgemm_(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
        }

        public void chemm(IntPtr side, IntPtr uplo, IntPtr m, IntPtr n, IntPtr alpha, IntPtr a, IntPtr lda, IntPtr b, IntPtr ldb, IntPtr beta, IntPtr c, IntPtr ldc)
        {
            nvblas91_imports.chemm(side, uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc);
        }

        public void chemm_(IntPtr side, IntPtr uplo, IntPtr m, IntPtr n, IntPtr alpha, IntPtr a, IntPtr lda, IntPtr b, IntPtr ldb, IntPtr beta, IntPtr c, IntPtr ldc)
        {
            nvblas91_imports.chemm_(side, uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc);
        }

        public void cher2k(IntPtr uplo, IntPtr trans, IntPtr n, IntPtr k, IntPtr alpha, IntPtr a, IntPtr lda, IntPtr b, IntPtr ldb, IntPtr beta, IntPtr c, IntPtr ldc)
        {
            nvblas91_imports.cher2k(uplo, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
        }

        public void cher2k_(IntPtr uplo, IntPtr trans, IntPtr n, IntPtr k, IntPtr alpha, IntPtr a, IntPtr lda, IntPtr b, IntPtr ldb, IntPtr beta, IntPtr c, IntPtr ldc)
        {
            nvblas91_imports.cher2k_(uplo, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
        }

        public void cherk(IntPtr uplo, IntPtr trans, IntPtr n, IntPtr k, IntPtr alpha, IntPtr a, IntPtr lda, IntPtr beta, IntPtr c, IntPtr ldc)
        {
            nvblas91_imports.cherk(uplo, trans, n, k, alpha, a, lda, beta, c, ldc);
        }

        public void cherk_(IntPtr uplo, IntPtr trans, IntPtr n, IntPtr k, IntPtr alpha, IntPtr a, IntPtr lda, IntPtr beta, IntPtr c, IntPtr ldc)
        {
            nvblas91_imports.cherk_(uplo, trans, n, k, alpha, a, lda, beta, c, ldc);
        }

        public void csymm(IntPtr side, IntPtr uplo, IntPtr m, IntPtr n, IntPtr alpha, IntPtr a, IntPtr lda, IntPtr b, IntPtr ldb, IntPtr beta, IntPtr c, IntPtr ldc)
        {
            nvblas91_imports.csymm(side, uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc);
        }

        public void csymm_(IntPtr side, IntPtr uplo, IntPtr m, IntPtr n, IntPtr alpha, IntPtr a, IntPtr lda, IntPtr b, IntPtr ldb, IntPtr beta, IntPtr c, IntPtr ldc)
        {
            nvblas91_imports.csymm_(side, uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc);
        }

        public void csyr2k(IntPtr uplo, IntPtr trans, IntPtr n, IntPtr k, IntPtr alpha, IntPtr a, IntPtr lda, IntPtr b, IntPtr ldb, IntPtr beta, IntPtr c, IntPtr ldc)
        {
            nvblas91_imports.csyr2k(uplo, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
        }

        public void csyr2k_(IntPtr uplo, IntPtr trans, IntPtr n, IntPtr k, IntPtr alpha, IntPtr a, IntPtr lda, IntPtr b, IntPtr ldb, IntPtr beta, IntPtr c, IntPtr ldc)
        {
            nvblas91_imports.csyr2k_(uplo, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
        }

        public void csyrk(IntPtr uplo, IntPtr trans, IntPtr n, IntPtr k, IntPtr alpha, IntPtr a, IntPtr lda, IntPtr beta, IntPtr c, IntPtr ldc)
        {
            nvblas91_imports.csyrk(uplo, trans, n, k, alpha, a, lda, beta, c, ldc);
        }

        public void csyrk_(IntPtr uplo, IntPtr trans, IntPtr n, IntPtr k, IntPtr alpha, IntPtr a, IntPtr lda, IntPtr beta, IntPtr c, IntPtr ldc)
        {
            nvblas91_imports.csyrk_(uplo, trans, n, k, alpha, a, lda, beta, c, ldc);
        }

        public void ctrmm(IntPtr side, IntPtr uplo, IntPtr transa, IntPtr diag, IntPtr m, IntPtr n, IntPtr alpha, IntPtr a, IntPtr lda, IntPtr b, IntPtr ldb)
        {
            nvblas91_imports.ctrmm(side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb);
        }

        public void ctrmm_(IntPtr side, IntPtr uplo, IntPtr transa, IntPtr diag, IntPtr m, IntPtr n, IntPtr alpha, IntPtr a, IntPtr lda, IntPtr b, IntPtr ldb)
        {
            nvblas91_imports.ctrmm_(side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb);
        }

        public void ctrsm(IntPtr side, IntPtr uplo, IntPtr transa, IntPtr diag, IntPtr m, IntPtr n, IntPtr alpha, IntPtr a, IntPtr lda, IntPtr b, IntPtr ldb)
        {
            nvblas91_imports.ctrsm(side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb);
        }

        public void ctrsm_(IntPtr side, IntPtr uplo, IntPtr transa, IntPtr diag, IntPtr m, IntPtr n, IntPtr alpha, IntPtr a, IntPtr lda, IntPtr b, IntPtr ldb)
        {
            nvblas91_imports.ctrsm_(side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb);
        }

        public void dgemm(IntPtr transa, IntPtr transb, IntPtr m, IntPtr n, IntPtr k, IntPtr alpha, IntPtr a, IntPtr lda, IntPtr b, IntPtr ldb, IntPtr beta, IntPtr c, IntPtr ldc)
        {
            nvblas91_imports.dgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
        }

        public void dgemm_(IntPtr transa, IntPtr transb, IntPtr m, IntPtr n, IntPtr k, IntPtr alpha, IntPtr a, IntPtr lda, IntPtr b, IntPtr ldb, IntPtr beta, IntPtr c, IntPtr ldc)
        {
            nvblas91_imports.dgemm_(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
        }

        public void dsymm(IntPtr side, IntPtr uplo, IntPtr m, IntPtr n, IntPtr alpha, IntPtr a, IntPtr lda, IntPtr b, IntPtr ldb, IntPtr beta, IntPtr c, IntPtr ldc)
        {
            nvblas91_imports.dsymm(side, uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc);
        }

        public void dsymm_(IntPtr side, IntPtr uplo, IntPtr m, IntPtr n, IntPtr alpha, IntPtr a, IntPtr lda, IntPtr b, IntPtr ldb, IntPtr beta, IntPtr c, IntPtr ldc)
        {
            nvblas91_imports.dsymm_(side, uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc);
        }

        public void dsyr2k(IntPtr uplo, IntPtr trans, IntPtr n, IntPtr k, IntPtr alpha, IntPtr a, IntPtr lda, IntPtr b, IntPtr ldb, IntPtr beta, IntPtr c, IntPtr ldc)
        {
            nvblas91_imports.dsyr2k(uplo, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
        }

        public void dsyr2k_(IntPtr uplo, IntPtr trans, IntPtr n, IntPtr k, IntPtr alpha, IntPtr a, IntPtr lda, IntPtr b, IntPtr ldb, IntPtr beta, IntPtr c, IntPtr ldc)
        {
            nvblas91_imports.dsyr2k_(uplo, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
        }

        public void dsyrk(IntPtr uplo, IntPtr trans, IntPtr n, IntPtr k, IntPtr alpha, IntPtr a, IntPtr lda, IntPtr beta, IntPtr c, IntPtr ldc)
        {
            nvblas91_imports.dsyrk(uplo, trans, n, k, alpha, a, lda, beta, c, ldc);
        }

        public void dsyrk_(IntPtr uplo, IntPtr trans, IntPtr n, IntPtr k, IntPtr alpha, IntPtr a, IntPtr lda, IntPtr beta, IntPtr c, IntPtr ldc)
        {
            nvblas91_imports.dsyrk_(uplo, trans, n, k, alpha, a, lda, beta, c, ldc);
        }

        public void dtrmm(IntPtr side, IntPtr uplo, IntPtr transa, IntPtr diag, IntPtr m, IntPtr n, IntPtr alpha, IntPtr a, IntPtr lda, IntPtr b, IntPtr ldb)
        {
            nvblas91_imports.dtrmm(side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb);
        }

        public void dtrmm_(IntPtr side, IntPtr uplo, IntPtr transa, IntPtr diag, IntPtr m, IntPtr n, IntPtr alpha, IntPtr a, IntPtr lda, IntPtr b, IntPtr ldb)
        {
            nvblas91_imports.dtrmm_(side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb);
        }

        public void dtrsm(IntPtr side, IntPtr uplo, IntPtr transa, IntPtr diag, IntPtr m, IntPtr n, IntPtr alpha, IntPtr a, IntPtr lda, IntPtr b, IntPtr ldb)
        {
            nvblas91_imports.dtrsm(side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb);
        }

        public void dtrsm_(IntPtr side, IntPtr uplo, IntPtr transa, IntPtr diag, IntPtr m, IntPtr n, IntPtr alpha, IntPtr a, IntPtr lda, IntPtr b, IntPtr ldb)
        {
            nvblas91_imports.dtrsm_(side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb);
        }

        public void sgemm(IntPtr transa, IntPtr transb, IntPtr m, IntPtr n, IntPtr k, IntPtr alpha, IntPtr a, IntPtr lda, IntPtr b, IntPtr ldb, IntPtr beta, IntPtr c, IntPtr ldc)
        {
            nvblas91_imports.sgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
        }

        public void sgemm_(IntPtr transa, IntPtr transb, IntPtr m, IntPtr n, IntPtr k, IntPtr alpha, IntPtr a, IntPtr lda, IntPtr b, IntPtr ldb, IntPtr beta, IntPtr c, IntPtr ldc)
        {
            nvblas91_imports.sgemm_(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
        }

        public void ssymm(IntPtr side, IntPtr uplo, IntPtr m, IntPtr n, IntPtr alpha, IntPtr a, IntPtr lda, IntPtr b, IntPtr ldb, IntPtr beta, IntPtr c, IntPtr ldc)
        {
            nvblas91_imports.ssymm(side, uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc);
        }

        public void ssymm_(IntPtr side, IntPtr uplo, IntPtr m, IntPtr n, IntPtr alpha, IntPtr a, IntPtr lda, IntPtr b, IntPtr ldb, IntPtr beta, IntPtr c, IntPtr ldc)
        {
            nvblas91_imports.ssymm_(side, uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc);
        }

        public void ssyr2k(IntPtr uplo, IntPtr trans, IntPtr n, IntPtr k, IntPtr alpha, IntPtr a, IntPtr lda, IntPtr b, IntPtr ldb, IntPtr beta, IntPtr c, IntPtr ldc)
        {
            nvblas91_imports.ssyr2k(uplo, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
        }

        public void ssyr2k_(IntPtr uplo, IntPtr trans, IntPtr n, IntPtr k, IntPtr alpha, IntPtr a, IntPtr lda, IntPtr b, IntPtr ldb, IntPtr beta, IntPtr c, IntPtr ldc)
        {
            nvblas91_imports.ssyr2k_(uplo, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
        }

        public void ssyrk(IntPtr uplo, IntPtr trans, IntPtr n, IntPtr k, IntPtr alpha, IntPtr a, IntPtr lda, IntPtr beta, IntPtr c, IntPtr ldc)
        {
            nvblas91_imports.ssyrk(uplo, trans, n, k, alpha, a, lda, beta, c, ldc);
        }

        public void ssyrk_(IntPtr uplo, IntPtr trans, IntPtr n, IntPtr k, IntPtr alpha, IntPtr a, IntPtr lda, IntPtr beta, IntPtr c, IntPtr ldc)
        {
            nvblas91_imports.ssyrk_(uplo, trans, n, k, alpha, a, lda, beta, c, ldc);
        }

        public void strmm(IntPtr side, IntPtr uplo, IntPtr transa, IntPtr diag, IntPtr m, IntPtr n, IntPtr alpha, IntPtr a, IntPtr lda, IntPtr b, IntPtr ldb)
        {
            nvblas91_imports.strmm(side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb);
        }

        public void strmm_(IntPtr side, IntPtr uplo, IntPtr transa, IntPtr diag, IntPtr m, IntPtr n, IntPtr alpha, IntPtr a, IntPtr lda, IntPtr b, IntPtr ldb)
        {
            nvblas91_imports.strmm_(side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb);
        }

        public void strsm(IntPtr side, IntPtr uplo, IntPtr transa, IntPtr diag, IntPtr m, IntPtr n, IntPtr alpha, IntPtr a, IntPtr lda, IntPtr b, IntPtr ldb)
        {
            nvblas91_imports.strsm(side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb);
        }

        public void strsm_(IntPtr side, IntPtr uplo, IntPtr transa, IntPtr diag, IntPtr m, IntPtr n, IntPtr alpha, IntPtr a, IntPtr lda, IntPtr b, IntPtr ldb)
        {
            nvblas91_imports.strsm_(side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb);
        }

        public void zgemm(IntPtr transa, IntPtr transb, IntPtr m, IntPtr n, IntPtr k, IntPtr alpha, IntPtr a, IntPtr lda, IntPtr b, IntPtr ldb, IntPtr beta, IntPtr c, IntPtr ldc)
        {
            nvblas91_imports.zgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
        }

        public void zgemm_(IntPtr transa, IntPtr transb, IntPtr m, IntPtr n, IntPtr k, IntPtr alpha, IntPtr a, IntPtr lda, IntPtr b, IntPtr ldb, IntPtr beta, IntPtr c, IntPtr ldc)
        {
            nvblas91_imports.zgemm_(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
        }

        public void zhemm(IntPtr side, IntPtr uplo, IntPtr m, IntPtr n, IntPtr alpha, IntPtr a, IntPtr lda, IntPtr b, IntPtr ldb, IntPtr beta, IntPtr c, IntPtr ldc)
        {
            nvblas91_imports.zhemm(side, uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc);
        }

        public void zhemm_(IntPtr side, IntPtr uplo, IntPtr m, IntPtr n, IntPtr alpha, IntPtr a, IntPtr lda, IntPtr b, IntPtr ldb, IntPtr beta, IntPtr c, IntPtr ldc)
        {
            nvblas91_imports.zhemm_(side, uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc);
        }

        public void zher2k(IntPtr uplo, IntPtr trans, IntPtr n, IntPtr k, IntPtr alpha, IntPtr a, IntPtr lda, IntPtr b, IntPtr ldb, IntPtr beta, IntPtr c, IntPtr ldc)
        {
            nvblas91_imports.zher2k(uplo, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
        }

        public void zher2k_(IntPtr uplo, IntPtr trans, IntPtr n, IntPtr k, IntPtr alpha, IntPtr a, IntPtr lda, IntPtr b, IntPtr ldb, IntPtr beta, IntPtr c, IntPtr ldc)
        {
            nvblas91_imports.zher2k_(uplo, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
        }

        public void zherk(IntPtr uplo, IntPtr trans, IntPtr n, IntPtr k, IntPtr alpha, IntPtr a, IntPtr lda, IntPtr beta, IntPtr c, IntPtr ldc)
        {
            nvblas91_imports.zherk(uplo, trans, n, k, alpha, a, lda, beta, c, ldc);
        }

        public void zherk_(IntPtr uplo, IntPtr trans, IntPtr n, IntPtr k, IntPtr alpha, IntPtr a, IntPtr lda, IntPtr beta, IntPtr c, IntPtr ldc)
        {
            nvblas91_imports.zherk_(uplo, trans, n, k, alpha, a, lda, beta, c, ldc);
        }

        public void zsymm(IntPtr side, IntPtr uplo, IntPtr m, IntPtr n, IntPtr alpha, IntPtr a, IntPtr lda, IntPtr b, IntPtr ldb, IntPtr beta, IntPtr c, IntPtr ldc)
        {
            nvblas91_imports.zsymm(side, uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc);
        }

        public void zsymm_(IntPtr side, IntPtr uplo, IntPtr m, IntPtr n, IntPtr alpha, IntPtr a, IntPtr lda, IntPtr b, IntPtr ldb, IntPtr beta, IntPtr c, IntPtr ldc)
        {
            nvblas91_imports.zsymm_(side, uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc);
        }

        public void zsyr2k(IntPtr uplo, IntPtr trans, IntPtr n, IntPtr k, IntPtr alpha, IntPtr a, IntPtr lda, IntPtr b, IntPtr ldb, IntPtr beta, IntPtr c, IntPtr ldc)
        {
            nvblas91_imports.zsyr2k(uplo, trans, b, k, alpha, a, lda, b, ldb, beta, c, ldc);
        }

        public void zsyr2k_(IntPtr uplo, IntPtr trans, IntPtr n, IntPtr k, IntPtr alpha, IntPtr a, IntPtr lda, IntPtr b, IntPtr ldb, IntPtr beta, IntPtr c, IntPtr ldc)
        {
            nvblas91_imports.zsyr2k_(uplo, trans, b, k, alpha, a, lda, b, ldb, beta, c, ldc);
        }

        public void zsyrk(IntPtr uplo, IntPtr trans, IntPtr n, IntPtr k, IntPtr alpha, IntPtr a, IntPtr lda, IntPtr beta, IntPtr c, IntPtr ldc)
        {
            nvblas91_imports.zsyrk(uplo, trans, n, k, alpha, a, lda, beta, c, ldc);
        }

        public void zsyrk_(IntPtr uplo, IntPtr trans, IntPtr n, IntPtr k, IntPtr alpha, IntPtr a, IntPtr lda, IntPtr beta, IntPtr c, IntPtr ldc)
        {
            nvblas91_imports.zsyrk_(uplo, trans, n, k, alpha, a, lda, beta, c, ldc);
        }

        public void ztrmm(IntPtr side, IntPtr uplo, IntPtr transa, IntPtr diag, IntPtr m, IntPtr n, IntPtr alpha, IntPtr a, IntPtr lda, IntPtr b, IntPtr ldb)
        {
            nvblas91_imports.ztrmm(side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb);
        }

        public void ztrmm_(IntPtr side, IntPtr uplo, IntPtr transa, IntPtr diag, IntPtr m, IntPtr n, IntPtr alpha, IntPtr a, IntPtr lda, IntPtr b, IntPtr ldb)
        {
            nvblas91_imports.ztrmm_(side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb);
        }

        public void ztrsm(IntPtr side, IntPtr uplo, IntPtr transa, IntPtr diag, IntPtr m, IntPtr n, IntPtr alpha, IntPtr a, IntPtr lda, IntPtr b, IntPtr ldb)
        {
            nvblas91_imports.ztrsm(side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb);
        }

        public void ztrsm_(IntPtr side, IntPtr uplo, IntPtr transa, IntPtr diag, IntPtr m, IntPtr n, IntPtr alpha, IntPtr a, IntPtr lda, IntPtr b, IntPtr ldb)
        {
            nvblas91_imports.ztrsm_(side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb);
        }
    }
}

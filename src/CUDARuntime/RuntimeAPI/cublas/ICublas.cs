/* (c) ALTIMESH 2018 -- all rights reserved */
using System;

namespace Hybridizer.Runtime.CUDAImports
{
#pragma warning disable 1591
    public interface ICUBLAS
    {
        cublasStatus_t Create(out cublasHandle_t handle);
        cublasStatus_t Destroy(cublasHandle_t handle);
        cublasStatus_t GetVersion(out int version);
        cublasStatus_t GetVersion(cublasHandle_t handle, out int version);
        cublasStatus_t GetProperty(libraryPropertyType_t type, out int property);
        cublasStatus_t SetStream(cublasHandle_t handle, cudaStream_t streamId);
        cublasStatus_t GetStream(cublasHandle_t handle, out cudaStream_t streamId);
        cublasStatus_t SetPointerMode(cublasHandle_t handle, cublasPointerMode_t mode);
        cublasStatus_t GetPointerMode(cublasHandle_t handle, out cublasPointerMode_t mode);
        cublasStatus_t SetVector(int n, int elemSize, IntPtr x, int incx, IntPtr y, int incy);
        cublasStatus_t GetVector(int n, int elemSize, IntPtr x, int incx, IntPtr y, int incy);
        cublasStatus_t GetMatrix(int rows, int cols, int elemSize, IntPtr A, int lda, IntPtr B, int ldb);
        cublasStatus_t SetMatrix(int rows, int cols, int elemSize, IntPtr A, int lda, IntPtr B, int ldb);
        cublasStatus_t SetVectorAsync(int n, int elemSize, IntPtr hostPtr, int incx, IntPtr devicePtr, int incy, cudaStream_t stream);
        cublasStatus_t GetVectorAsync(int n, int elemSize, IntPtr devicePtr, int incx, IntPtr hostPtr, int incy, cudaStream_t stream);
        cublasStatus_t SetMatrixAsync(int rows, int cols, int elemSize, IntPtr A, int lda, IntPtr B, int ldb, cudaStream_t stream);
        cublasStatus_t GetMatrixAsync(int rows, int cols, int elemSize, IntPtr A, int lda, IntPtr B, int ldb, cudaStream_t stream);
        cublasStatus_t SetAtomicsMode(cublasHandle_t handle, cublasAtomicsMode_t mode);
        cublasStatus_t GetAtomicsMode(cublasHandle_t handle, out cublasAtomicsMode_t mode);
        cublasStatus_t GetMathMode(cublasHandle_t handle, cublasMath_t mode);
        cublasStatus_t SetMathMode(cublasHandle_t handle, out cublasMath_t mode);


        #region Level-1 Functions

        cublasStatus_t Isamax(cublasHandle_t handle, int n, IntPtr x, int incx, IntPtr result);
        cublasStatus_t Idamax(cublasHandle_t handle, int n, IntPtr x, int incx, IntPtr result);
        cublasStatus_t Icamax(cublasHandle_t handle, int n, IntPtr x, int incx, IntPtr result);
        cublasStatus_t Izamax(cublasHandle_t handle, int n, IntPtr x, int incx, IntPtr result);

        cublasStatus_t Isamin(cublasHandle_t handle, int n, IntPtr x, int incx, IntPtr result);
        cublasStatus_t Idamin(cublasHandle_t handle, int n, IntPtr x, int incx, IntPtr result);
        cublasStatus_t Icamin(cublasHandle_t handle, int n, IntPtr x, int incx, IntPtr result);
        cublasStatus_t Izamin(cublasHandle_t handle, int n, IntPtr x, int incx, IntPtr result);

        cublasStatus_t Sasum(cublasHandle_t handle, int n, IntPtr x, int incx, IntPtr result);
        cublasStatus_t Dasum(cublasHandle_t handle, int n, IntPtr x, int incx, IntPtr result);
        cublasStatus_t Scasum(cublasHandle_t handle, int n, IntPtr x, int incx, IntPtr result);
        cublasStatus_t Dzasum(cublasHandle_t handle, int n, IntPtr x, int incx, IntPtr result);

        cublasStatus_t Saxpy(cublasHandle_t handle, int n, IntPtr alpha, IntPtr x, int incx, IntPtr y, int incy);
        cublasStatus_t Daxpy(cublasHandle_t handle, int n, IntPtr alpha, IntPtr x, int incx, IntPtr y, int incy);
        cublasStatus_t Caxpy(cublasHandle_t handle, int n, IntPtr alpha, IntPtr x, int incx, IntPtr y, int incy);
        cublasStatus_t Zaxpy(cublasHandle_t handle, int n, IntPtr alpha, IntPtr x, int incx, IntPtr y, int incy);

        cublasStatus_t Scopy(cublasHandle_t handle, int n, IntPtr x, int incx, IntPtr y, int incy);
        cublasStatus_t Dcopy(cublasHandle_t handle, int n, IntPtr x, int incx, IntPtr y, int incy);
        cublasStatus_t Ccopy(cublasHandle_t handle, int n, IntPtr x, int incx, IntPtr y, int incy);
        cublasStatus_t Zcopy(cublasHandle_t handle, int n, IntPtr x, int incx, IntPtr y, int incy);

        cublasStatus_t Sdot(cublasHandle_t handle, int n, IntPtr x, int incx, IntPtr y, int incy, IntPtr result);
        cublasStatus_t Ddot(cublasHandle_t handle, int n, IntPtr x, int incx, IntPtr y, int incy, IntPtr result);
        cublasStatus_t Cdotu(cublasHandle_t handle, int n, IntPtr x, int incx, IntPtr y, int incy, IntPtr result);
        cublasStatus_t Cdotc(cublasHandle_t handle, int n, IntPtr x, int incx, IntPtr y, int incy, IntPtr result);
        cublasStatus_t Zdotu(cublasHandle_t handle, int n, IntPtr x, int incx, IntPtr y, int incy, IntPtr result);
        cublasStatus_t Zdotc(cublasHandle_t handle, int n, IntPtr x, int incx, IntPtr y, int incy, IntPtr result);

        cublasStatus_t Snrm2(cublasHandle_t handle, int n, IntPtr x, int incx, IntPtr result);
        cublasStatus_t Dnrm2(cublasHandle_t handle, int n, IntPtr x, int incx, IntPtr result);
        cublasStatus_t Scnrm2(cublasHandle_t handle, int n, IntPtr x, int incx, IntPtr result);
        cublasStatus_t Dznrm2(cublasHandle_t handle, int n, IntPtr x, int incx, IntPtr result);

        cublasStatus_t Srot(cublasHandle_t handle, int n, IntPtr x, int incx, IntPtr y, int incy, IntPtr c, IntPtr s);
        cublasStatus_t Drot(cublasHandle_t handle, int n, IntPtr x, int incx, IntPtr y, int incy, IntPtr c, IntPtr s);
        cublasStatus_t Crot(cublasHandle_t handle, int n, IntPtr x, int incx, IntPtr y, int incy, IntPtr c, IntPtr s);
        cublasStatus_t Csrot(cublasHandle_t handle, int n, IntPtr x, int incx, IntPtr y, int incy, IntPtr c, IntPtr s);
        cublasStatus_t Zrot(cublasHandle_t handle, int n, IntPtr x, int incx, IntPtr y, int incy, IntPtr c, IntPtr s);
        cublasStatus_t Zdrot(cublasHandle_t handle, int n, IntPtr x, int incx, IntPtr y, int incy, IntPtr c, IntPtr s);

        cublasStatus_t Srotg(cublasHandle_t handle, IntPtr a, IntPtr b, IntPtr c, IntPtr s);
        cublasStatus_t Drotg(cublasHandle_t handle, IntPtr a, IntPtr b, IntPtr c, IntPtr s);
        cublasStatus_t Crotg(cublasHandle_t handle, IntPtr a, IntPtr b, IntPtr c, IntPtr s);
        cublasStatus_t Zrotg(cublasHandle_t handle, IntPtr a, IntPtr b, IntPtr c, IntPtr s);

        cublasStatus_t Srotm(cublasHandle_t handle, int n, IntPtr x, int incx, IntPtr y, int incy, IntPtr param);
        cublasStatus_t Drotm(cublasHandle_t handle, int n, IntPtr x, int incx, IntPtr y, int incy, IntPtr param);

        cublasStatus_t Srotmg(cublasHandle_t handle, IntPtr d1, IntPtr d2, IntPtr x1, IntPtr y1, IntPtr param);
        cublasStatus_t Drotmg(cublasHandle_t handle, IntPtr d1, IntPtr d2, IntPtr x1, IntPtr y1, IntPtr param);

        cublasStatus_t Sscal(cublasHandle_t handle, int n, IntPtr alpha, IntPtr x, int incx);
        cublasStatus_t Dscal(cublasHandle_t handle, int n, IntPtr alpha, IntPtr x, int incx);
        cublasStatus_t Cscal(cublasHandle_t handle, int n, IntPtr alpha, IntPtr x, int incx);
        cublasStatus_t Csscal(cublasHandle_t handle, int n, IntPtr alpha, IntPtr x, int incx);
        cublasStatus_t Zscal(cublasHandle_t handle, int n, IntPtr alpha, IntPtr x, int incx);
        cublasStatus_t Zdscal(cublasHandle_t handle, int n, IntPtr alpha, IntPtr x, int incx);

        cublasStatus_t Sswap(cublasHandle_t handle, int n, IntPtr x, int incx, IntPtr y, int incy);
        cublasStatus_t Dswap(cublasHandle_t handle, int n, IntPtr x, int incx, IntPtr y, int incy);
        cublasStatus_t Cswap(cublasHandle_t handle, int n, IntPtr x, int incx, IntPtr y, int incy);
        cublasStatus_t Zswap(cublasHandle_t handle, int n, IntPtr x, int incx, IntPtr y, int incy);

        #endregion

        #region Level-2 Functions

        cublasStatus_t Sgbmv(cublasHandle_t handle, cublasOperation_t trans, int m, int n, int kl, int ku, IntPtr alpha, IntPtr A, int lda, IntPtr x, int incx, IntPtr beta, IntPtr y, int incy);
        cublasStatus_t Dgbmv(cublasHandle_t handle, cublasOperation_t trans, int m, int n, int kl, int ku, IntPtr alpha, IntPtr A, int lda, IntPtr x, int incx, IntPtr beta, IntPtr y, int incy);
        cublasStatus_t Cgbmv(cublasHandle_t handle, cublasOperation_t trans, int m, int n, int kl, int ku, IntPtr alpha, IntPtr A, int lda, IntPtr x, int incx, IntPtr beta, IntPtr y, int incy);
        cublasStatus_t Zgbmv(cublasHandle_t handle, cublasOperation_t trans, int m, int n, int kl, int ku, IntPtr alpha, IntPtr A, int lda, IntPtr x, int incx, IntPtr beta, IntPtr y, int incy);

        cublasStatus_t Sgemv(cublasHandle_t handle, cublasOperation_t trans, int m, int n, IntPtr alpha, IntPtr A, int lda, IntPtr x, int incx, IntPtr beta, IntPtr y, int incy);
        cublasStatus_t Dgemv(cublasHandle_t handle, cublasOperation_t trans, int m, int n, IntPtr alpha, IntPtr A, int lda, IntPtr x, int incx, IntPtr beta, IntPtr y, int incy);
        cublasStatus_t Cgemv(cublasHandle_t handle, cublasOperation_t trans, int m, int n, IntPtr alpha, IntPtr A, int lda, IntPtr x, int incx, IntPtr beta, IntPtr y, int incy);
        cublasStatus_t Zgemv(cublasHandle_t handle, cublasOperation_t trans, int m, int n, IntPtr alpha, IntPtr A, int lda, IntPtr x, int incx, IntPtr beta, IntPtr y, int incy);

        cublasStatus_t Sger(cublasHandle_t handle, int m, int n, IntPtr alpha, IntPtr x, int incx, IntPtr y, int incy, IntPtr A, int lda);
        cublasStatus_t Dger(cublasHandle_t handle, int m, int n, IntPtr alpha, IntPtr x, int incx, IntPtr y, int incy, IntPtr A, int lda);
        cublasStatus_t Cgeru(cublasHandle_t handle, int m, int n, IntPtr alpha, IntPtr x, int incx, IntPtr y, int incy, IntPtr A, int lda);
        cublasStatus_t Cgerc(cublasHandle_t handle, int m, int n, IntPtr alpha, IntPtr x, int incx, IntPtr y, int incy, IntPtr A, int lda);
        cublasStatus_t Zgeru(cublasHandle_t handle, int m, int n, IntPtr alpha, IntPtr x, int incx, IntPtr y, int incy, IntPtr A, int lda);
        cublasStatus_t Zgerc(cublasHandle_t handle, int m, int n, IntPtr alpha, IntPtr x, int incx, IntPtr y, int incy, IntPtr A, int lda);

        cublasStatus_t Ssbmv(cublasHandle_t handle, cublasFillMode_t uplo, int n, int k, IntPtr alpha, IntPtr A, int lda, IntPtr x, int incx, IntPtr beta, IntPtr y, int incy);
        cublasStatus_t Dsbmv(cublasHandle_t handle, cublasFillMode_t uplo, int n, int k, IntPtr alpha, IntPtr A, int lda, IntPtr x, int incx, IntPtr beta, IntPtr y, int incy);

        cublasStatus_t Sspmv(cublasHandle_t handle, cublasFillMode_t uplo, int n, IntPtr alpha, IntPtr AP, IntPtr x, int incx, IntPtr beta, IntPtr y, int incy);
        cublasStatus_t Dspmv(cublasHandle_t handle, cublasFillMode_t uplo, int n, IntPtr alpha, IntPtr AP, IntPtr x, int incx, IntPtr beta, IntPtr y, int incy);

        cublasStatus_t Sspr(cublasHandle_t handle, cublasFillMode_t uplo, int n, IntPtr alpha, IntPtr x, int incx, IntPtr AP);
        cublasStatus_t Dspr(cublasHandle_t handle, cublasFillMode_t uplo, int n, IntPtr alpha, IntPtr x, int incx, IntPtr AP);

        cublasStatus_t Sspr2(cublasHandle_t handle, cublasFillMode_t uplo, int n, IntPtr alpha, IntPtr x, int incx, IntPtr y, int incy, IntPtr AP);
        cublasStatus_t Dspr2(cublasHandle_t handle, cublasFillMode_t uplo, int n, IntPtr alpha, IntPtr x, int incx, IntPtr y, int incy, IntPtr AP);

        cublasStatus_t Ssymv(cublasHandle_t handle, cublasFillMode_t uplo, int n, IntPtr alpha, IntPtr A, int lda, IntPtr x, int incx, IntPtr beta, IntPtr y, int incy);
        cublasStatus_t Dsymv(cublasHandle_t handle, cublasFillMode_t uplo, int n, IntPtr alpha, IntPtr A, int lda, IntPtr x, int incx, IntPtr beta, IntPtr y, int incy);
        cublasStatus_t Csymv(cublasHandle_t handle, cublasFillMode_t uplo, int n, IntPtr alpha, IntPtr A, int lda, IntPtr x, int incx, IntPtr beta, IntPtr y, int incy);
        cublasStatus_t Zsymv(cublasHandle_t handle, cublasFillMode_t uplo, int n, IntPtr alpha, IntPtr A, int lda, IntPtr x, int incx, IntPtr beta, IntPtr y, int incy);

        cublasStatus_t Ssyr(cublasHandle_t handle, cublasFillMode_t uplo, int n, IntPtr alpha, IntPtr x, int incx, IntPtr A, int lda);
        cublasStatus_t Dsyr(cublasHandle_t handle, cublasFillMode_t uplo, int n, IntPtr alpha, IntPtr x, int incx, IntPtr A, int lda);
        cublasStatus_t Csyr(cublasHandle_t handle, cublasFillMode_t uplo, int n, IntPtr alpha, IntPtr x, int incx, IntPtr A, int lda);
        cublasStatus_t Zsyr(cublasHandle_t handle, cublasFillMode_t uplo, int n, IntPtr alpha, IntPtr x, int incx, IntPtr A, int lda);

        cublasStatus_t Ssyr2(cublasHandle_t handle, cublasFillMode_t uplo, int n, IntPtr alpha, IntPtr x, int incx, IntPtr y, int incy, IntPtr A, int lda);
        cublasStatus_t Dsyr2(cublasHandle_t handle, cublasFillMode_t uplo, int n, IntPtr alpha, IntPtr x, int incx, IntPtr y, int incy, IntPtr A, int lda);
        cublasStatus_t Csyr2(cublasHandle_t handle, cublasFillMode_t uplo, int n, IntPtr alpha, IntPtr x, int incx, IntPtr y, int incy, IntPtr A, int lda);
        cublasStatus_t Zsyr2(cublasHandle_t handle, cublasFillMode_t uplo, int n, IntPtr alpha, IntPtr x, int incx, IntPtr y, int incy, IntPtr A, int lda);

        cublasStatus_t Stbmv(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, int k, IntPtr A, int lda, IntPtr x, int incx);
        cublasStatus_t Dtbmv(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, int k, IntPtr A, int lda, IntPtr x, int incx);
        cublasStatus_t Ctbmv(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, int k, IntPtr A, int lda, IntPtr x, int incx);
        cublasStatus_t Ztbmv(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, int k, IntPtr A, int lda, IntPtr x, int incx);

        cublasStatus_t Stbsv(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, int k, IntPtr A, int lda, IntPtr x, int incx);
        cublasStatus_t Dtbsv(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, int k, IntPtr A, int lda, IntPtr x, int incx);
        cublasStatus_t Ctbsv(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, int k, IntPtr A, int lda, IntPtr x, int incx);
        cublasStatus_t Ztbsv(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, int k, IntPtr A, int lda, IntPtr x, int incx);

        cublasStatus_t Stpmv(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, IntPtr AP, IntPtr x, int incx);
        cublasStatus_t Dtpmv(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, IntPtr AP, IntPtr x, int incx);
        cublasStatus_t Ctpmv(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, IntPtr AP, IntPtr x, int incx);
        cublasStatus_t Ztpmv(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, IntPtr AP, IntPtr x, int incx);

        cublasStatus_t Stpsv(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, IntPtr AP, IntPtr x, int incx);
        cublasStatus_t Dtpsv(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, IntPtr AP, IntPtr x, int incx);
        cublasStatus_t Ctpsv(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, IntPtr AP, IntPtr x, int incx);
        cublasStatus_t Ztpsv(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, IntPtr AP, IntPtr x, int incx);

        cublasStatus_t Strmv(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, IntPtr A, int lda, IntPtr x, int incx);
        cublasStatus_t Dtrmv(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, IntPtr A, int lda, IntPtr x, int incx);
        cublasStatus_t Ctrmv(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, IntPtr A, int lda, IntPtr x, int incx);
        cublasStatus_t Ztrmv(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, IntPtr A, int lda, IntPtr x, int incx);

        cublasStatus_t Strsv(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, IntPtr A, int lda, IntPtr x, int incx);
        cublasStatus_t Dtrsv(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, IntPtr A, int lda, IntPtr x, int incx);
        cublasStatus_t Ctrsv(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, IntPtr A, int lda, IntPtr x, int incx);
        cublasStatus_t Ztrsv(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, IntPtr A, int lda, IntPtr x, int incx);

        cublasStatus_t Chemv(cublasHandle_t handle, cublasFillMode_t uplo, int n, IntPtr alpha, IntPtr A, int lda, IntPtr x, int incx, IntPtr beta, IntPtr y, int incy);
        cublasStatus_t Zhemv(cublasHandle_t handle, cublasFillMode_t uplo, int n, IntPtr alpha, IntPtr A, int lda, IntPtr x, int incx, IntPtr beta, IntPtr y, int incy);

        cublasStatus_t Chbmv(cublasHandle_t handle, cublasFillMode_t uplo, int n, int k, IntPtr alpha, IntPtr A, int lda, IntPtr x, int incx, IntPtr beta, IntPtr y, int incy);
        cublasStatus_t Zhbmv(cublasHandle_t handle, cublasFillMode_t uplo, int n, int k, IntPtr alpha, IntPtr A, int lda, IntPtr x, int incx, IntPtr beta, IntPtr y, int incy);

        cublasStatus_t Chpmv(cublasHandle_t handle, cublasFillMode_t uplo, int n, IntPtr alpha, IntPtr AP, IntPtr x, int incx, IntPtr beta, IntPtr y, int incy);
        cublasStatus_t Zhpmv(cublasHandle_t handle, cublasFillMode_t uplo, int n, IntPtr alpha, IntPtr AP, IntPtr x, int incx, IntPtr beta, IntPtr y, int incy);

        cublasStatus_t Cher(cublasHandle_t handle, cublasFillMode_t uplo, int n, IntPtr alpha, IntPtr x, int incx, IntPtr A, int lda);
        cublasStatus_t Zher(cublasHandle_t handle, cublasFillMode_t uplo, int n, IntPtr alpha, IntPtr x, int incx, IntPtr A, int lda);

        cublasStatus_t Cher2(cublasHandle_t handle, cublasFillMode_t uplo, int n, IntPtr alpha, IntPtr x, int incx, IntPtr y, int incy, IntPtr A, int lda);
        cublasStatus_t Zher2(cublasHandle_t handle, cublasFillMode_t uplo, int n, IntPtr alpha, IntPtr x, int incx, IntPtr y, int incy, IntPtr A, int lda);

        cublasStatus_t Chpr(cublasHandle_t handle, cublasFillMode_t uplo, int n, IntPtr alpha, IntPtr x, int incx, IntPtr AP);
        cublasStatus_t Zhpr(cublasHandle_t handle, cublasFillMode_t uplo, int n, IntPtr alpha, IntPtr x, int incx, IntPtr AP);

        cublasStatus_t Chpr2(cublasHandle_t handle, cublasFillMode_t uplo, int n, IntPtr alpha, IntPtr x, int incx, IntPtr y, int incy, IntPtr AP);
        cublasStatus_t Zhpr2(cublasHandle_t handle, cublasFillMode_t uplo, int n, IntPtr alpha, IntPtr x, int incx, IntPtr y, int incy, IntPtr AP);

        #endregion

        #region Level-3 Functions

        cublasStatus_t Sgemm(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, IntPtr alpha, IntPtr A, int lda, IntPtr B, int ldb, IntPtr beta, IntPtr C, int ldc);
        cublasStatus_t Dgemm(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, IntPtr alpha, IntPtr A, int lda, IntPtr B, int ldb, IntPtr beta, IntPtr C, int ldc);
        cublasStatus_t Cgemm(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, IntPtr alpha, IntPtr A, int lda, IntPtr B, int ldb, IntPtr beta, IntPtr C, int ldc);
        cublasStatus_t Zgemm(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, IntPtr alpha, IntPtr A, int lda, IntPtr B, int ldb, IntPtr beta, IntPtr C, int ldc);

        cublasStatus_t Ssymm(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, int m, int n, IntPtr alpha, IntPtr A, int lda, IntPtr B, int ldb, IntPtr beta, IntPtr C, int ldc);
        cublasStatus_t Dsymm(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, int m, int n, IntPtr alpha, IntPtr A, int lda, IntPtr B, int ldb, IntPtr beta, IntPtr C, int ldc);
        cublasStatus_t Csymm(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, int m, int n, IntPtr alpha, IntPtr A, int lda, IntPtr B, int ldb, IntPtr beta, IntPtr C, int ldc);
        cublasStatus_t Zsymm(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, int m, int n, IntPtr alpha, IntPtr A, int lda, IntPtr B, int ldb, IntPtr beta, IntPtr C, int ldc);

        cublasStatus_t Ssyrk(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, IntPtr alpha, IntPtr A, int lda, IntPtr beta, IntPtr C, int ldc);
        cublasStatus_t Dsyrk(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, IntPtr alpha, IntPtr A, int lda, IntPtr beta, IntPtr C, int ldc);
        cublasStatus_t Csyrk(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, IntPtr alpha, IntPtr A, int lda, IntPtr beta, IntPtr C, int ldc);
        cublasStatus_t Zsyrk(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, IntPtr alpha, IntPtr A, int lda, IntPtr beta, IntPtr C, int ldc);

        cublasStatus_t Ssyr2k(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, IntPtr alpha, IntPtr A, int lda, IntPtr B, int ldb, IntPtr beta, IntPtr C, int ldc);
        cublasStatus_t Dsyr2k(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, IntPtr alpha, IntPtr A, int lda, IntPtr B, int ldb, IntPtr beta, IntPtr C, int ldc);
        cublasStatus_t Csyr2k(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, IntPtr alpha, IntPtr A, int lda, IntPtr B, int ldb, IntPtr beta, IntPtr C, int ldc);
        cublasStatus_t Zsyr2k(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, IntPtr alpha, IntPtr A, int lda, IntPtr B, int ldb, IntPtr beta, IntPtr C, int ldc);

        cublasStatus_t Strmm(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, IntPtr alpha, IntPtr A, int lda, IntPtr B, int ldb, IntPtr C, int ldc);
        cublasStatus_t Dtrmm(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, IntPtr alpha, IntPtr A, int lda, IntPtr B, int ldb, IntPtr C, int ldc);
        cublasStatus_t Ctrmm(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, IntPtr alpha, IntPtr A, int lda, IntPtr B, int ldb, IntPtr C, int ldc);
        cublasStatus_t Ztrmm(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, IntPtr alpha, IntPtr A, int lda, IntPtr B, int ldb, IntPtr C, int ldc);

        cublasStatus_t Strsm(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, IntPtr alpha, IntPtr A, int lda, IntPtr B, int ldb);
        cublasStatus_t Dtrsm(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, IntPtr alpha, IntPtr A, int lda, IntPtr B, int ldb);
        cublasStatus_t Ctrsm(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, IntPtr alpha, IntPtr A, int lda, IntPtr B, int ldb);
        cublasStatus_t Ztrsm(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, IntPtr alpha, IntPtr A, int lda, IntPtr B, int ldb);

        cublasStatus_t Chemm(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, int m, int n, IntPtr alpha, IntPtr A, int lda, IntPtr B, int ldb, IntPtr beta, IntPtr C, int ldc);
        cublasStatus_t Zhemm(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, int m, int n, IntPtr alpha, IntPtr A, int lda, IntPtr B, int ldb, IntPtr beta, IntPtr C, int ldc);

        cublasStatus_t Cherk(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, IntPtr alpha, IntPtr A, int lda, IntPtr beta, IntPtr C, int ldc);
        cublasStatus_t Zherk(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, IntPtr alpha, IntPtr A, int lda, IntPtr beta, IntPtr C, int ldc);

        cublasStatus_t Cher2k(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, IntPtr alpha, IntPtr A, int lda, IntPtr B, int ldb, IntPtr beta, IntPtr C, int ldc);
        cublasStatus_t Zher2k(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, IntPtr alpha, IntPtr A, int lda, IntPtr B, int ldb, IntPtr beta, IntPtr C, int ldc);

        #endregion
    }

#pragma warning restore 1591
}
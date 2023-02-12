/* (c) ALTIMESH 2018 -- all rights reserved */
using System;
using System.Runtime.InteropServices;
#pragma warning disable 0169
namespace Hybridizer.Runtime.CUDAImports
{
    /// <summary>
    /// This is a pointer type to an opaque cuSPARSE context, which the user must initialize by calling prior to calling cusparseCreate() any other library function. The handle created and returned by cusparseCreate() must be passed to every cuSPARSE function
    /// </summary>
    [IntrinsicType("cusparseHandle_t")]
    public struct cusparseHandle_t { IntPtr _inner; }

    /// <summary>
    /// This structure is used to describe the shape and properties of a matrix.
    /// </summary>
    [IntrinsicType("cusparseMatDescr_t")]
    public struct cusparseMatDescr_t { IntPtr _inner; }

    /// <summary>
    /// This is a pointer type to an opaque structure holding the information collected in the analysis phase of the solution of the sparse triangular linear system.It is expected to be passed unchanged to the solution phase of the sparse triangular linear system.
    /// </summary>
    [IntrinsicType("cusparseSolveAnalysisInfo_t")]
    public struct cusparseSolveAnalysisInfo_t { IntPtr _inner; }

    /// <summary>
    /// This is a pointer type to an opaque structure holding the information used in csrsv2_bufferSize(), csrsv2_analysis(), and csrsv2_solve().
    /// </summary>
    [IntrinsicType("csrsv2Info_t")]
    public struct csrsv2Info_t { IntPtr _inner; }

    /// <summary>
    /// This is a pointer type to an opaque structure holding the information used in csrsm2_bufferSize(), csrsm2_analysis(), and csrsm2_solve().
    /// </summary>
    [IntrinsicType("csrsm2Info_t")]
    public struct csrsm2Info_t { IntPtr _inner; }

    /// <summary>
    /// This is a pointer type to an opaque structure holding the information used in csric02_bufferSize(), csric02_analysis(), and csric02(). 
    /// </summary>
    [IntrinsicType("csric02Info_t")]
    public struct csric02Info_t { IntPtr _inner; }

    /// <summary>
    /// This is a pointer type to an opaque structure holding the information used in bsric02_bufferSize(), bsric02_analysis(), and bsric02(). 
    /// </summary>
    [IntrinsicType("bsric02Info_t")]
    public struct bsric02Info_t { IntPtr _inner; }

    /// <summary>
    /// This is a pointer type to an opaque structure holding the information used in csrilu02_bufferSize(), csrilu02_analysis(), and csrilu02(). 
    /// </summary>
    [IntrinsicType("csrilu02Info_t")]
    public struct csrilu02Info_t { IntPtr _inner; }

    /// <summary>
    /// This is a pointer type to an opaque structure holding the information used in bsrilu02_bufferSize(), bsrilu02_analysis(), and bsrilu02()
    /// </summary>
    [IntrinsicType("bsrilu02Info_t")]
    public struct bsrilu02Info_t { IntPtr _inner; }


    /// <summary>
    /// This is a pointer type to an opaque structure holding the information used in bsrsv2_bufferSize(), bsrsv2_analysis(), and bsrsv2_solve(). 
    /// </summary>
    [IntrinsicType("bsrsv2Info_t")]
    public struct bsrsv2Info_t { IntPtr _inner; }

    /// <summary>
    /// This is a pointer type to an opaque structure holding the information used in bsrsm2_bufferSize(), bsrsm2_analysis(), and bsrsm2_solve(). 
    /// </summary>
    [IntrinsicType("bsrsm2Info_t")]
    public struct bsrsm2Info_t { IntPtr _inner; } 

    /// <summary>
    /// internal
    /// </summary>
    [IntrinsicType("cusparseHybMat_t")]
    public struct cusparseHybMat_t { IntPtr _inner; }

    /// <summary>
    ///  Opaque structure holding the sorting information
    /// </summary>
    [IntrinsicType("csru2csrInfo_t")]
    public struct csru2csrInfo_t { IntPtr _inner; }

    /// <summary>
    /// Opaque structure holding the coloring information
    /// </summary>
    [IntrinsicType("cusparseColorInfo_t")]
    public struct cusparseColorInfo_t { IntPtr _inner; }

    /// <summary>
    /// Opaque structures holding sparse gemm information
    /// </summary>
    [IntrinsicType("csrgemm2Info_t")]
    public struct csrgemm2Info_t { IntPtr _inner; }
}
#pragma warning restore 0169
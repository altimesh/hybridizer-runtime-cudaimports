/* (c) ALTIMESH 2018 -- all rights reserved */
using System;
using System.Runtime.InteropServices;
#pragma warning disable 0169
namespace Hybridizer.Runtime.CUDAImports
{
    /// <summary>
    /// CUSPARSE status type returns
    /// </summary>
    [IntrinsicType("cusparseStatus_t")]
    public enum cusparseStatus_t : int
    {
        /// <summary></summary>
        CUSPARSE_STATUS_SUCCESS=0,
        /// <summary></summary>
        CUSPARSE_STATUS_NOT_INITIALIZED = 1,
        /// <summary></summary>
        CUSPARSE_STATUS_ALLOC_FAILED = 2,
        /// <summary></summary>
        CUSPARSE_STATUS_INVALID_VALUE = 3,
        /// <summary></summary>
        CUSPARSE_STATUS_ARCH_MISMATCH = 4,
        /// <summary></summary>
        CUSPARSE_STATUS_MAPPING_ERROR = 5,
        /// <summary></summary>
        CUSPARSE_STATUS_EXECUTION_FAILED = 6,
        /// <summary></summary>
        CUSPARSE_STATUS_INTERNAL_ERROR = 7,
        /// <summary></summary>
        CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED = 8,
        /// <summary></summary>
        CUSPARSE_STATUS_ZERO_PIVOT = 9
    } ;

    /// <summary>
    /// Types definitions
    /// </summary>
    [IntrinsicType("cusparsePointerMode_t")]
    public enum cusparsePointerMode_t : int
    {
        /// <summary></summary>
        CUSPARSE_POINTER_MODE_HOST = 0,
        /// <summary></summary>
        CUSPARSE_POINTER_MODE_DEVICE = 1 
    }

    /// <summary>
    /// Types definitions
    /// </summary>
    [IntrinsicType(" cusparseAlgMode_t")]
    public enum cusparseAlgMode_t : int
    {
        /// <summary></summary>
        CUSPARSE_ALG_NAIVE = 0,
        /// <summary></summary>
        CUSPARSE_ALG_MERGE_PATH = 1
    }

    /// <summary></summary>
    [IntrinsicType("cusparseAction_t")]
    public enum cusparseAction_t : int
    {
        /// <summary></summary>
        CUSPARSE_ACTION_SYMBOLIC = 0,
        /// <summary></summary>
        CUSPARSE_ACTION_NUMERIC = 1   
    }

    /// <summary></summary>
    [IntrinsicType("cusparseMatrixType_t")]
    public enum cusparseMatrixType_t : int
    {
        /// <summary></summary>
        CUSPARSE_MATRIX_TYPE_GENERAL = 0,
        /// <summary></summary>
        CUSPARSE_MATRIX_TYPE_SYMMETRIC = 1,
        /// <summary></summary>
        CUSPARSE_MATRIX_TYPE_HERMITIAN = 2,
        /// <summary></summary>
        CUSPARSE_MATRIX_TYPE_TRIANGULAR = 3 
    }

    /// <summary></summary>
    [IntrinsicType("cusparseFillMode_t")]
    public enum cusparseFillMode_t : int
    {
        /// <summary></summary>
        CUSPARSE_FILL_MODE_LOWER = 0,
        /// <summary></summary>
        CUSPARSE_FILL_MODE_UPPER = 1
    }
#pragma warning disable 1591
    [IntrinsicType("cusparseDiagType_t")]
    public enum cusparseDiagType_t : int
    {
        CUSPARSE_DIAG_TYPE_NON_UNIT = 0, 
        CUSPARSE_DIAG_TYPE_UNIT = 1
    }

    [IntrinsicType("cusparseIndexBase_t")]
    public enum cusparseIndexBase_t : int
    {
        CUSPARSE_INDEX_BASE_ZERO = 0, 
        CUSPARSE_INDEX_BASE_ONE = 1
    }

    [IntrinsicType("cusparseOperation_t")]
    public enum cusparseOperation_t : int
    {
        CUSPARSE_OPERATION_NON_TRANSPOSE = 0,  
        CUSPARSE_OPERATION_TRANSPOSE = 1,  
        CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE = 2  
    }

    [IntrinsicType("cusparseDirection_t")]
    public enum cusparseDirection_t : int
    {
        CUSPARSE_DIRECTION_ROW = 0,  
        CUSPARSE_DIRECTION_COLUMN = 1  
    }

    [IntrinsicType("cusparseHybPartition_t")]
    public enum cusparseHybPartition_t : int
    {
        CUSPARSE_HYB_PARTITION_AUTO = 0,  // automatically decide how to split the data into regular/irregular part
        CUSPARSE_HYB_PARTITION_USER = 1,  // store data into regular part up to a user specified treshhold
        CUSPARSE_HYB_PARTITION_MAX = 2    // store all data in the regular part
    }

    [IntrinsicType("cusparseSolvePolicy_t")]
    public enum cusparseSolvePolicy_t : int
    {
        CUSPARSE_SOLVE_POLICY_NO_LEVEL = 0, // no level information is generated, only reports structural zero.
        CUSPARSE_SOLVE_POLICY_USE_LEVEL = 1  
    }

    [IntrinsicType("cusparseSideMode_t")]
    public enum cusparseSideMode_t : int
    {
        CUSPARSE_SIDE_LEFT =0,
        CUSPARSE_SIDE_RIGHT=1
    }
#pragma warning restore 1591
}
#pragma warning restore 0169
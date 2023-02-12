using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;

namespace Hybridizer.Runtime.CUDAImports
{

#pragma warning disable CS1591 // Missing XML comment for publicly visible type or member

    /// <summary>
    /// Error Status Codes.
    /// Almost all NPP function return error-status information using these return codes.Negative return codes indicate errors, positive return codes indicate warnings, a return code of 0 indicates success.
    /// </summary>
    [IntrinsicInclude("npp.h")]
    [IntrinsicType("NppStatus")]
    public enum NppStatus
    {
        /// <summary>
        /// negative return-codes indicate errors
        /// </summary>
        NOT_SUPPORTED_MODE_ERROR = -9999,

        INVALID_HOST_POINTER_ERROR = -1032,
        INVALID_DEVICE_POINTER_ERROR = -1031,
        LUT_PALETTE_BITSIZE_ERROR = -1030,
        /// <summary>
        /// ZeroCrossing mode not supported
        /// </summary>
        ZC_MODE_NOT_SUPPORTED_ERROR = -1028,
        NOT_SUFFICIENT_COMPUTE_CAPABILITY = -1027,
        TEXTURE_BIND_ERROR = -1024,
        WRONG_INTERSECTION_ROI_ERROR = -1020,
        HAAR_CLASSIFIER_PIXEL_MATCH_ERROR = -1006,
        MEMFREE_ERROR = -1005,
        MEMSET_ERROR = -1004,
        MEMCPY_ERROR = -1003,
        ALIGNMENT_ERROR = -1002,
        CUDA_KERNEL_EXECUTION_ERROR = -1000,

        /// <summary>
        /// Unsupported round mode
        /// </summary>
        ROUND_MODE_NOT_SUPPORTED_ERROR = -213,

        /// <summary>
        /// Image pixels are constant for quality index
        /// </summary>
        QUALITY_INDEX_ERROR = -210,

        /// <summary>
        /// One of the output image dimensions is less than 1 pixel
        /// </summary>
        RESIZE_NO_OPERATION_ERROR = -201,

        /// <summary>
        /// Number overflows the upper or lower limit of the data type
        /// </summary>
        OVERFLOW_ERROR = -109,
        /// <summary>
        /// Step value is not pixel multiple
        /// </summary>
        NOT_EVEN_STEP_ERROR = -108,
        /// <summary>
        /// Number of levels for histogram is less than 2
        /// </summary>
        HISTOGRAM_NUMBER_OF_LEVELS_ERROR = -107,
        /// <summary>
        /// Number of levels for LUT is less than 2
        /// </summary>
        LUT_NUMBER_OF_LEVELS_ERROR = -106,

        /// <summary>
        /// Processed data is corrupted
        /// </summary>
        CORRUPTED_DATA_ERROR = -61,
        /// <summary>
        /// Wrong order of the destination channels
        /// </summary>
        CHANNEL_ORDER_ERROR = -60,
        /// <summary>
        /// All values of the mask are zero
        /// </summary>
        ZERO_MASK_VALUE_ERROR = -59,
        /// <summary>
        /// The quadrangle is nonconvex or degenerates into triangle, line or point
        /// </summary>
        QUADRANGLE_ERROR = -58,
        /// <summary>
        ///  Size of the rectangle region is less than or equal to 1
        /// </summary>
        RECTANGLE_ERROR = -57,
        /// <summary>
        /// Unallowable values of the transformation coefficients
        /// </summary>
        COEFFICIENT_ERROR = -56,
        /// <summary>
        /// Bad or unsupported number of channels
        /// </summary>
        NUMBER_OF_CHANNELS_ERROR = -53,
        /// <summary>
        /// Channel of interest is not 1, 2, or 3
        /// </summary>
        COI_ERROR = -52,
        /// <summary>
        /// Divisor is equal to zero
        /// </summary>
        DIVISOR_ERROR = -51,

        /// <summary>
        /// Illegal channel index
        /// </summary>
        CHANNEL_ERROR = -47,
        /// <summary>
        /// Stride is less than the row length
        /// </summary>
        STRIDE_ERROR = -37,

        /// <summary>
        /// Anchor point is outside mask
        /// </summary>
        ANCHOR_ERROR = -34,
        /// <summary>
        /// Lower bound is larger than upper bound
        /// </summary>
        MASK_SIZE_ERROR = -33,

        RESIZE_FACTOR_ERROR = -23,
        INTERPOLATION_ERROR = -22,
        MIRROR_FLIP_ERROR = -21,
        MOMENT_00_ZERO_ERROR = -20,
        THRESHOLD_NEGATIVE_LEVEL_ERROR = -19,
        THRESHOLD_ERROR = -18,
        CONTEXT_MATCH_ERROR = -17,
        FFT_FLAG_ERROR = -16,
        FFT_ORDER_ERROR = -15,
        /// <summary>
        /// Step is less or equal zero
        /// </summary>
        STEP_ERROR = -14,
        SCALE_RANGE_ERROR = -13,
        DATA_TYPE_ERROR = -12,
        OUT_OFF_RANGE_ERROR = -11,
        DIVIDE_BY_ZERO_ERROR = -10,
        MEMORY_ALLOCATION_ERR = -9,
        NULL_POINTER_ERROR = -8,
        RANGE_ERROR = -7,
        SIZE_ERROR = -6,
        BAD_ARGUMENT_ERROR = -5,
        NO_MEMORY_ERROR = -4,
        NOT_IMPLEMENTED_ERROR = -3,
        ERROR = -2,
        ERROR_RESERVED = -1,

        /// <summary>
        /// Error free operation
        /// </summary>
        NO_ERROR = 0,
        /// <summary>
        /// Successful operation (same as NO_ERROR)
        /// </summary>
        SUCCESS = NO_ERROR,

        /* positive return-codes indicate warnings */
        /// <summary>
        ///  Indicates that no operation was performed
        /// </summary>
        NO_OPERATION_WARNING = 1,
        /// <summary>
        /// Divisor is zero however does not terminate the execution
        /// </summary>
        DIVIDE_BY_ZERO_WARNING = 6,
        /// <summary>
        /// Indicates that the quadrangle passed to one of affine warping functions doesn't have necessary properties. First 3 vertices are used, the fourth vertex discarded.
        /// </summary>
        AFFINE_QUAD_INCORRECT_WARNING = 28,
        /// <summary>
        /// The given ROI has no interestion with either the source or destination ROI. Thus no operation was performed.
        /// </summary>
        WRONG_INTERSECTION_ROI_WARNING = 29,
        /// <summary>
        /// The given quadrangle has no intersection with either the source or destination ROI. Thus no operation was performed.
        /// </summary>
        WRONG_INTERSECTION_QUAD_WARNING = 30,
        /// <summary>
        /// Image size isn't multiple of two. Indicates that in case of 422/411/420 sampling the ROI width/height was modified for proper processing.
        /// </summary>
        DOUBLE_SIZE_WARNING = 35,

        /// <summary>
        ///  Speed reduction due to uncoalesced memory accesses warning.
        /// </summary>
        MISALIGNED_DST_ROI_WARNING = 10000,
    }

#pragma warning restore CS1591 // Missing XML comment for publicly visible type or member
}

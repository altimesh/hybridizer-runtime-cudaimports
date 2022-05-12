namespace Hybridizer.Runtime.CUDAImports
{
    /// <summary>
    ///  CUDA texture resource view formats
    /// </summary>
    [IntrinsicType("cudaResourceViewFormat")]
    public enum cudaResourceViewFormat
    {
        /// <summary>
        /// No resource view format (use underlying resource format)
        /// </summary>
        cudaResViewFormatNone = 0x00,
        /// <summary>
        /// 1 channel unsigned 8-bit integers
        /// </summary>
        cudaResViewFormatUnsignedChar1 = 0x01,
        /// <summary>
        /// 2 channel unsigned 8-bit integers
        /// </summary>
        cudaResViewFormatUnsignedChar2 = 0x02,
        /// <summary>
        /// 4 channel unsigned 8-bit integers
        /// </summary>
        cudaResViewFormatUnsignedChar4 = 0x03,
        /// <summary>
        /// 1 channel signed 8-bit integers
        /// </summary>
        cudaResViewFormatSignedChar1 = 0x04,
        /// <summary>
        /// 2 channel signed 8-bit integers
        /// </summary>
        cudaResViewFormatSignedChar2 = 0x05,
        /// <summary>
        /// 4 channel signed 8-bit integers
        /// </summary>
        cudaResViewFormatSignedChar4 = 0x06,
        /// <summary>
        /// 1 channel unsigned 16-bit integers
        /// </summary>
        cudaResViewFormatUnsignedShort1 = 0x07,
        /// <summary>
        /// 2 channel unsigned 16-bit integers
        /// </summary>
        cudaResViewFormatUnsignedShort2 = 0x08,
        /// <summary>
        /// 4 channel unsigned 16-bit integers
        /// </summary>
        cudaResViewFormatUnsignedShort4 = 0x09,
        /// <summary>
        /// 1 channel signed 16-bit integers
        /// </summary>
        cudaResViewFormatSignedShort1 = 0x0a,
        /// <summary>
        /// 2 channel signed 16-bit integers
        /// </summary>
        cudaResViewFormatSignedShort2 = 0x0b,
        /// <summary>
        /// 4 channel signed 16-bit integers
        /// </summary>
        cudaResViewFormatSignedShort4 = 0x0c,
        /// <summary>
        /// 1 channel unsigned 32-bit integers
        /// </summary>
        cudaResViewFormatUnsignedInt1 = 0x0d,
        /// <summary>
        /// 2 channel unsigned 32-bit integers
        /// </summary>
        cudaResViewFormatUnsignedInt2 = 0x0e,
        /// <summary>
        /// 4 channel unsigned 32-bit integers
        /// </summary>
        cudaResViewFormatUnsignedInt4 = 0x0f,
        /// <summary>
        /// 1 channel signed 32-bit integers
        /// </summary>
        cudaResViewFormatSignedInt1 = 0x10,
        /// <summary>
        /// 2 channel signed 32-bit integers
        /// </summary>
        cudaResViewFormatSignedInt2 = 0x11,
        /// <summary>
        /// 4 channel signed 32-bit integers
        /// </summary>
        cudaResViewFormatSignedInt4 = 0x12,
        /// <summary>
        /// 1 channel 16-bit floating point
        /// </summary>
        cudaResViewFormatHalf1 = 0x13,
        /// <summary>
        /// 2 channel 16-bit floating point
        /// </summary>
        cudaResViewFormatHalf2 = 0x14,
        /// <summary>
        /// 4 channel 16-bit floating point
        /// </summary>
        cudaResViewFormatHalf4 = 0x15,
        /// <summary>
        /// 1 channel 32-bit floating point
        /// </summary>
        cudaResViewFormatFloat1 = 0x16,
        /// <summary>
        /// 2 channel 32-bit floating point
        /// </summary>
        cudaResViewFormatFloat2 = 0x17,
        /// <summary>
        /// 4 channel 32-bit floating point
        /// </summary>
        cudaResViewFormatFloat4 = 0x18,
        /// <summary>
        /// Block compressed 1
        /// </summary>
        cudaResViewFormatUnsignedBlockCompressed1 = 0x19,
        /// <summary>
        /// Block compressed 2
        /// </summary>
        cudaResViewFormatUnsignedBlockCompressed2 = 0x1a,
        /// <summary>
        /// Block compressed 3
        /// </summary>
        cudaResViewFormatUnsignedBlockCompressed3 = 0x1b,
        /// <summary>
        /// Block compressed 4 unsigned
        /// </summary>
        cudaResViewFormatUnsignedBlockCompressed4 = 0x1c,
        /// <summary>
        /// Block compressed 4 signed
        /// </summary>
        cudaResViewFormatSignedBlockCompressed4 = 0x1d,
        /// <summary>
        /// Block compressed 5 unsigned
        /// </summary>
        cudaResViewFormatUnsignedBlockCompressed5 = 0x1e,
        /// <summary>
        /// Block compressed 5 signed
        /// </summary>
        cudaResViewFormatSignedBlockCompressed5 = 0x1f,
        /// <summary>
        /// Block compressed 6 unsigned half-float
        /// </summary>
        cudaResViewFormatUnsignedBlockCompressed6H = 0x20,
        /// <summary>
        /// Block compressed 6 signed half-float
        /// </summary>
        cudaResViewFormatSignedBlockCompressed6H = 0x21,
        /// <summary>
        /// Block compressed 7
        /// </summary>
        cudaResViewFormatUnsignedBlockCompressed7 = 0x22
    }
}
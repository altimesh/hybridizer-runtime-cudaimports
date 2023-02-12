namespace Hybridizer.Runtime.CUDAImports
{
    /// <summary>
    /// complex double-precision
    /// </summary>
    [IntrinsicInclude("cublas.h")]
    [IntrinsicType("cuDoubleComplex")]
    public struct cuDoubleComplex
    {
        /// <summary>
        /// real part
        /// </summary>
        public double re;
        /// <summary>
        /// imaginary part
        /// </summary>
        public double im;
    }
}
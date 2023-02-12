namespace Hybridizer.Runtime.CUDAImports
{
    /// <summary>
    /// complex single-precision
    /// </summary>
    [IntrinsicInclude("<cublas.h>")]
    [IntrinsicType("cuComplex")]
    public struct cuComplex
    {
        /// <summary>
        /// real part
        /// </summary>
        /// 
        [IntrinsicRename("x")]
        public float re;
        /// <summary>
        /// imaginary part
        /// </summary>
        [IntrinsicRename("y")]
        public float im;
    }
}
namespace Hybridizer.Runtime.CUDAImports
{
    /// <summary>
    /// 8 booleans
    /// </summary>
    [IntrinsicType("bool8")] // mask?
    public struct bool8
    {
        /// <summary>
        /// component
        /// </summary>
        public bool x, y, z, w, x2, y2, z2, w2;
    }
}
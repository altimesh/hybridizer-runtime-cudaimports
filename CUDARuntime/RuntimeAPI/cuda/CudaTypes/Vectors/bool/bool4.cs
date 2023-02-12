namespace Hybridizer.Runtime.CUDAImports
{
    /// <summary>
    /// 4 booleans
    /// </summary>
    [IntrinsicType("bool4")] // mask?
    public struct bool4
    {
        /// <summary>
        /// component
        /// </summary>
        public bool x, y, z, w;
    }
}
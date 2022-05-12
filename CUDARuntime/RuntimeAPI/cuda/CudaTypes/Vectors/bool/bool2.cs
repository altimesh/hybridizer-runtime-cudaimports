namespace Hybridizer.Runtime.CUDAImports
{
    /// <summary>
    /// 2 booleans
    /// </summary>
    [IntrinsicType("bool2")] // mask?
    public struct bool2
    {
        /// <summary>
        /// component
        /// </summary>
        public bool x, y;
    }
}
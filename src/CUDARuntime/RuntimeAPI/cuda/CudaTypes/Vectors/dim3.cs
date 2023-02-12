using System.Runtime.InteropServices;

namespace Hybridizer.Runtime.CUDAImports
{ 
    /// <summary>
    /// dimension structure.
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    [IntrinsicType("dim3")]
    public struct dim3
    {
        /// <summary>
        /// components
        /// </summary>
        public int x, y, z;

        /// <summary>
        /// Assignment constructor
        /// </summary>
        /// <param name="x"></param>
        /// <param name="y"></param>
        /// <param name="z"></param>
        public dim3(int x, int y, int z)
        {
            this.x = x;
            this.y = y;
            this.z = z;
        }
    }
}
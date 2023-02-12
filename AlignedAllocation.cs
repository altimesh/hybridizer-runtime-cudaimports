using System;
using System.Runtime.InteropServices;

namespace Hybridizer.Runtime.CUDAImports
{
    /// <summary>
    /// Aligned memory allocation helper
    /// </summary>
    public unsafe struct AlignedAllocation
    {
        void* _allocated;
        IntPtr _aligned;
        private long _allocatedSize;

        /// <summary>
        /// The aligned pointer
        /// </summary>
        public IntPtr Aligned
        {
            get { return _aligned; }
        }

        /// <summary>
        /// Total allocated size
        /// </summary>
        public long AllocatedSize
        {
            get { return _allocatedSize; }
        }

        /// <summary>
        /// Allocates aligned memory
        /// </summary>
        /// <param name="size">Amount of bytes to allocate</param>
        /// <param name="alignment">Alignement</param>
        /// <returns></returns>
        public static AlignedAllocation Alloc(long size, int alignment)
        {
            AlignedAllocation res;
            var sizeP = new IntPtr(size + alignment - 1);
            res._allocated = (void*) Marshal.AllocHGlobal(sizeP);
            long all = (long)res._allocated + (alignment - 1 - ((long)res._allocated + alignment - 1) % alignment);
            res._aligned = new IntPtr(all);
            res._allocatedSize = size + alignment - 1;
            GC.AddMemoryPressure(res.AllocatedSize);

            return res;
        }

        /// <summary>
        /// Releases allocated memory
        /// </summary>
        public void Free()
        {
            if (_allocated != (void*) 0)
            {
                GC.RemoveMemoryPressure(AllocatedSize);
                Marshal.FreeHGlobal(new IntPtr(_allocated));
                _allocated = (void*)0;
                _aligned = IntPtr.Zero;
            }
        }
    }
}
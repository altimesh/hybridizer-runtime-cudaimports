using System;
using System.Runtime.InteropServices;

namespace Hybridizer.Runtime.CUDAImports
{
    /// <summary>
    /// $size\_t$ type has different bit-size storage depending on architecture.
    /// </summary>
    [IntrinsicType("size_t")]
    [Guid("0F4E0F1A-A925-4A6B-9378-0F2AEBB3073B")]
    public struct size_t
    {
        IntPtr _inner;

        /// <summary>
        /// constructor from 32 bits signed integer
        /// </summary>
        public size_t(int val) { _inner = new IntPtr(val); }
        /// <summary>
        /// constructor from 32 bits sunigned integer
        /// </summary>
        public size_t(uint val) { _inner = new IntPtr((long)val); }
        /// <summary>
        /// constructor from 64 bits signed integer
        /// </summary>
        public size_t(long val) { _inner = new IntPtr(val); }

        /// <summary>
        /// implicit conversion operator
        /// </summary>
        public static implicit operator size_t(int val) { return new size_t(val); }
        /// <summary>
        /// implicit conversion operator
        /// </summary>
        public static implicit operator size_t(uint val) { return new size_t(val); }
        /// <summary>
        /// implicit conversion operator
        /// </summary>
        public static implicit operator size_t(long val) { return new size_t(val); }

        /// <summary>
        /// explicit conversion operator
        /// </summary>
        public static explicit operator int(size_t val) { return unchecked((int)val._inner.ToInt64()); }
        /// <summary>
        /// explicit conversion operator
        /// </summary>
        public static explicit operator uint(size_t val) { return unchecked((uint)val._inner.ToInt64()); }
        /// <summary>
        /// explicit conversion operator
        /// </summary>
        public static explicit operator long(size_t val) { return val._inner.ToInt64(); }
        /// <summary>
        /// explicit conversion operator
        /// </summary>
        public static explicit operator ulong(size_t val) { return (ulong)val._inner.ToInt64(); }
        /// <summary>
        /// Print contents of size\_t as a 64 bits integer
        /// </summary>
        /// <returns></returns>
        public override string ToString()
        {
            return _inner.ToInt64().ToString();
        }
    }
}
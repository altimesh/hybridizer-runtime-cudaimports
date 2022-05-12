using System.Runtime.InteropServices;

namespace Hybridizer.Runtime.CUDAImports
{
    /// <summary>
    /// 2 64 bits integers
    /// </summary>
    [IntrinsicType("long2")]
    [StructLayout(LayoutKind.Explicit)]
    public unsafe struct long2
    {
        /// <summary>
        /// x
        /// </summary>
        [FieldOffset(0)]
        public long x;
        /// <summary>
        /// y
        /// </summary>
        [FieldOffset(4)]
        public long y;

        /// <summary>
        /// constructor from components
        /// </summary>
        public long2(long xx, long yy)
        {
            x = xx;
            y = yy;
        }

        /// <summary>
        /// copy constructor
        /// </summary>
        public long2(long2 a)
        {
            x = a.x;
            y = a.y;
        }

        /// <summary>
        /// constructor from single component
        /// </summary>
        public long2(long val)
        {
            x = val;
            y = val;
        }

        /// <summary>
        /// addition operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator+")]
        public static long2 operator +(long2 a, long2 b)
        {
            long2 res;
            res.x = a.x + b.x;
            res.y = a.y + b.y;
            return res;
        }

        /// <summary>
        /// addition operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator+")]
        public static long2 operator +(int a, long2 b)
        {
            long2 res;
            res.x = a + b.x;
            res.y = a + b.y;
            return res;
        }

        /// <summary>
        /// addition operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator+")]
        public static long2 operator +(long2 a, int b)
        {
            long2 res;
            res.x = a.x + b;
            res.y = a.y + b;
            return res;
        }

        /// <summary>
        /// substraction operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator-")]
        public static long2 operator -(long2 a, long2 b)
        {
            long2 res;
            res.x = a.x - b.x;
            res.y = a.y - b.y;
            return res;
        }

        /// <summary>
        /// substraction operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator-")]
        public static long2 operator -(int a, long2 b)
        {
            long2 res;
            res.x = a - b.x;
            res.y = a - b.y;
            return res;
        }

        /// <summary>
        /// substraction operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator-")]
        public static long2 operator -(long2 a, int b)
        {
            long2 res;
            res.x = a.x - b;
            res.y = a.y - b;
            return res;
        }

        /// <summary>
        /// multiplication operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator*")]
        public static long2 operator *(long2 a, long2 b)
        {
            long2 res;
            res.x = a.x * b.x;
            res.y = a.y * b.y;
            return res;
        }

        /// <summary>
        /// multiplication operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator*")]
        public static long2 operator *(int a, long2 b)
        {
            long2 res;
            res.x = a * b.x;
            res.y = a * b.y;
            return res;
        }

        /// <summary>
        /// multiplication operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator*")]
        public static long2 operator *(long2 a, int b)
        {
            long2 res;
            res.x = a.x * b;
            res.y = a.y * b;
            return res;
        }

        /// <summary>
        /// division operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator/")]
        public static long2 operator /(long2 a, long2 b)
        {
            long2 res;
            res.x = a.x / b.x;
            res.y = a.y / b.y;
            return res;
        }

        /// <summary>
        /// division operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator/")]
        public static long2 operator /(int a, long2 b)
        {
            long2 res;
            res.x = a / b.x;
            res.y = a / b.y;
            return res;
        }

        /// <summary>
        /// division operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator/")]
        public static long2 operator /(long2 a, int b)
        {
            long2 res;
            res.x = a.x / b;
            res.y = a.y / b;
            return res;
        }

        /// <summary>
        /// bitwise AND operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator&")]
        public static long2 operator &(long2 a, long2 b)
        {
            long2 res;
            res.x = a.x & b.x;
            res.y = a.y & b.y;
            return res;
        }

        /// <summary>
        /// bitwise AND operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator&")]
        public static long2 operator &(int a, long2 b)
        {
            long2 res;
            res.x = a & b.x;
            res.y = a & b.y;
            return res;
        }

        /// <summary>
        /// bitwise AND operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator&")]
        public static long2 operator &(long2 a, int b)
        {
            long2 res;
            res.x = a.x & b;
            res.y = a.y & b;
            return res;
        }

        /// <summary>
        /// bitwise OR operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator|")]
        public static long2 operator |(long2 a, long2 b)
        {
            long2 res;
            res.x = a.x | b.x;
            res.y = a.y | b.y;
            return res;
        }

        /// <summary>
        /// bitwise OR operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator|")]
        public static long2 operator |(long a, long2 b)
        {
            long2 res;
            res.x = a | b.x;
            res.y = a | b.y;
            return res;
        }

        /// <summary>
        /// bitwise OR operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator|")]
        public static long2 operator |(long2 a, long b)
        {
            long2 res;
            res.x = a.x | b;
            res.y = a.y | b;
            return res;
        }

        /// <summary>
        /// bitwise XOR operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator^")]
        public static long2 operator ^(long2 a, long2 b)
        {
            long2 res;
            res.x = a.x ^ b.x;
            res.y = a.y ^ b.y;
            return res;
        }

        /// <summary>
        /// bitwise XOR operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator^")]
        public static long2 operator ^(int a, long2 b)
        {
            long2 res;
            res.x = a ^ b.x;
            res.y = a ^ b.y;
            return res;
        }

        /// <summary>
        /// bitwise XOR operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator^")]
        public static long2 operator ^(long2 a, int b)
        {
            long2 res;
            res.x = a.x ^ b;
            res.y = a.y ^ b;
            return res;
        }

        /// <summary>
        /// greater or equal operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator>=")]
        public static bool2 operator >=(long2 l, long2 r)
        {
            bool2 res;
            res.x = l.x >= r.x;
            res.y = l.y >= r.y;
            return res;
        }

        /// <summary>
        /// strict greater operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator>")]
        public static bool2 operator >(long2 l, long2 r)
        {
            bool2 res;
            res.x = l.x > r.x;
            res.y = l.y > r.y;
            return res;
        }

        /// <summary>
        /// less or equal operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator<=")]
        public static bool2 operator <=(long2 l, long2 r)
        {
            bool2 res;
            res.x = l.x <= r.x;
            res.y = l.y <= r.y;
            return res;
        }

        /// <summary>
        /// strict less operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator>=")]
        public static bool2 operator <(long2 l, long2 r)
        {
            bool2 res;
            res.x = l.x < r.x;
            res.y = l.y < r.y;
            return res;
        }

        /// <summary>
        /// left shift operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator<<")]
        public static long2 LeftShift(long2 a, long2 shift)
        {
            long2 res;
            res.x = a.x << (int)(shift.x);
            res.y = a.y << (int)(shift.y);
            return res;
        }

        /// <summary>
        /// right shift operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator>>")]
        public static long2 RightShift(long2 a, long2 shift)
        {
            long2 res;
            res.x = a.x >> (int)(shift.x);
            res.y = a.y >> (int)(shift.y);
            return res;
        }

        /// <summary>
        /// Stores in memory
        /// </summary>
        /// <param name="ptr"></param>
        /// <param name="val"></param>
        /// <param name="alignment">has to be a compile time constant</param>
        [IntrinsicFunction(IsNaked = true, Name = "__hybridizer_store_int2")]
        public unsafe static void Store(long2* ptr, long2 val, int alignment)
        {
            *ptr = val;
        }

        /// <summary>
        /// stores in memory
        /// </summary>
        /// <param name="ptr">destination pointer</param>
        /// <param name="val">value to store</param>
        /// <param name="alignment">has to be a compile time constant</param>
        [IntrinsicFunction(IsNaked = true, Name = "__hybridizer_store_splat_int2")]
        public unsafe static void Store(long2* ptr, int val, int alignment)
        {
            *ptr = new long2(val);
        }

        /// <summary>
        /// loads from memory
        /// </summary>
        /// <param name="ptr"></param>
        /// <param name="alignment">has to be a compile time constant</param>
        [IntrinsicFunction(IsNaked = true, Name = "__hybridizer_load_int2")]
        public unsafe static long2 Load(long2* ptr, int alignment)
        {
            return *ptr;
        }

        /// <summary>
        /// selects components from l or r, depending on mask value
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "hybridizer::select<double2>")]
        public static long2 Select(bool2 mask, long2 l, long2 r)
        {
            long2 res;
            res.x = mask.x ? l.x : r.x;
            res.y = mask.y ? l.y : r.y;
            return res;
        }
    }
}
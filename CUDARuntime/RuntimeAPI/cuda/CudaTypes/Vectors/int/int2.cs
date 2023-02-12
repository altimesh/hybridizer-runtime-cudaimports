using System.Runtime.InteropServices;

namespace Hybridizer.Runtime.CUDAImports
{
    /// <summary>
    /// 2 32 bits integers
    /// </summary>
    [IntrinsicType("int2")]
    [StructLayout(LayoutKind.Explicit)]
    public unsafe struct int2
    {
        /// <summary>
        /// x
        /// </summary>
        [FieldOffset(0)]
        public int x;
        /// <summary>
        /// y
        /// </summary>
        [FieldOffset(4)]
        public int y;

        /// <summary>
        /// constructor from components
        /// </summary>
        private int2(int xx, int yy)
        {
            x = xx;
            y = yy;
        }

        /// <summary>
        /// make an int2 from 2 ints
        /// </summary>
        /// <param name="xx"></param>
        /// <param name="yy"></param>
        /// <returns></returns>
        [IntrinsicFunction("make_int2")]
        public static int2 make_int2(int xx, int yy)
        {
            return new int2(xx, yy);
        }

        /// <summary>
        /// constructor from float2
        /// </summary>
        public int2(float2 val)
        {
            x = (int)val.x;
            y = (int)val.y;
        }

        /// <summary>
        /// constructor from 64 bits integer
        /// </summary>
        /// <param name="val">lower part goes to x, high part to y</param>
        public int2(long val)
        {
            x = (int)(val & 0xFFFFFFFFL);
            y = (int)((val >> 32) & 0xFFFFFFFFL);
        }

        /// <summary>
        /// conversion to 64 bits integer
        /// </summary>
        public static explicit operator long(int2 res)
        {
            long* ptr = (long*)&res;
            return *ptr;
        }

        /// <summary>
        /// conversion to float2
        /// </summary>
        public static explicit operator float2(int2 res)
        {
            float2* ptr = (float2*)&res;
            return *ptr;
        }

        /// <summary>
        /// conversion to 64 bits floating point
        /// </summary>
        public static explicit operator double(int2 res)
        {
            double* ptr = (double*)&res;
            return *ptr;
        }

        /// <summary>
        /// conversion to short4
        /// </summary>
        public static explicit operator short4(int2 res)
        {
            short4* ptr = (short4*)&res;
            return *ptr;
        }

        /// <summary>
        /// conversion to char8
        /// </summary>
        public static explicit operator char8(int2 res)
        {
            char8* ptr = (char8*)&res;
            return *ptr;
        }

        /// <summary>
        /// addition operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator+")]
        public static int2 operator +(int2 a, int2 b)
        {
            int2 res;
            res.x = a.x + b.x;
            res.y = a.y + b.y;
            return res;
        }

        /// <summary>
        /// addition operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator+")]
        public static int2 operator +(int a, int2 b)
        {
            int2 res;
            res.x = a + b.x;
            res.y = a + b.y;
            return res;
        }

        /// <summary>
        /// addition operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator+")]
        public static int2 operator +(int2 a, int b)
        {
            int2 res;
            res.x = a.x + b;
            res.y = a.y + b;
            return res;
        }

        /// <summary>
        /// substraction operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator-")]
        public static int2 operator -(int2 a, int2 b)
        {
            int2 res;
            res.x = a.x - b.x;
            res.y = a.y - b.y;
            return res;
        }

        /// <summary>
        /// substraction operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator-")]
        public static int2 operator -(int a, int2 b)
        {
            int2 res;
            res.x = a - b.x;
            res.y = a - b.y;
            return res;
        }

        /// <summary>
        /// substraction operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator-")]
        public static int2 operator -(int2 a, int b)
        {
            int2 res;
            res.x = a.x - b;
            res.y = a.y - b;
            return res;
        }

        /// <summary>
        /// multiplication operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator*")]
        public static int2 operator *(int2 a, int2 b)
        {
            int2 res;
            res.x = a.x * b.x;
            res.y = a.y * b.y;
            return res;
        }

        /// <summary>
        /// multiplication operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator*")]
        public static int2 operator *(int a, int2 b)
        {
            int2 res;
            res.x = a * b.x;
            res.y = a * b.y;
            return res;
        }

        /// <summary>
        /// multiplication operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator*")]
        public static int2 operator *(int2 a, int b)
        {
            int2 res;
            res.x = a.x * b;
            res.y = a.y * b;
            return res;
        }

        /// <summary>
        /// division operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator/")]
        public static int2 operator /(int2 a, int2 b)
        {
            int2 res;
            res.x = a.x / b.x;
            res.y = a.y / b.y;
            return res;
        }

        /// <summary>
        /// division operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator/")]
        public static int2 operator /(int a, int2 b)
        {
            int2 res;
            res.x = a / b.x;
            res.y = a / b.y;
            return res;
        }

        /// <summary>
        /// division operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator/")]
        public static int2 operator /(int2 a, int b)
        {
            int2 res;
            res.x = a.x / b;
            res.y = a.y / b;
            return res;
        }

        /// <summary>
        /// bitwise AND operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator&")]
        public static int2 operator &(int2 a, int2 b)
        {
            int2 res;
            res.x = a.x & b.x;
            res.y = a.y & b.y;
            return res;
        }

        /// <summary>
        /// bitwise AND operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator&")]
        public static int2 operator &(int a, int2 b)
        {
            int2 res;
            res.x = a & b.x;
            res.y = a & b.y;
            return res;
        }

        /// <summary>
        /// bitwise AND operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator&")]
        public static int2 operator &(int2 a, int b)
        {
            int2 res;
            res.x = a.x & b;
            res.y = a.y & b;
            return res;
        }

        /// <summary>
        /// bitwise OR operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator|")]
        public static int2 operator |(int2 a, int2 b)
        {
            int2 res;
            res.x = a.x | b.x;
            res.y = a.y | b.y;
            return res;
        }

        /// <summary>
        /// bitwise OR operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator|")]
        public static int2 operator |(int a, int2 b)
        {
            int2 res;
            res.x = a | b.x;
            res.y = a | b.y;
            return res;
        }

        /// <summary>
        /// bitwise OR operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator|")]
        public static int2 operator |(int2 a, int b)
        {
            int2 res;
            res.x = a.x | b;
            res.y = a.y | b;
            return res;
        }

        /// <summary>
        /// bitwise XOR operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator^")]
        public static int2 operator ^(int2 a, int2 b)
        {
            int2 res;
            res.x = a.x ^ b.x;
            res.y = a.y ^ b.y;
            return res;
        }

        /// <summary>
        /// bitwise XOR operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator^")]
        public static int2 operator ^(int a, int2 b)
        {
            int2 res;
            res.x = a ^ b.x;
            res.y = a ^ b.y;
            return res;
        }

        /// <summary>
        /// bitwise XOR operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator^")]
        public static int2 operator ^(int2 a, int b)
        {
            int2 res;
            res.x = a.x ^ b;
            res.y = a.y ^ b;
            return res;
        }

        /// <summary>
        /// Stores in memory
        /// </summary>
        /// <param name="ptr"></param>
        /// <param name="val"></param>
        /// <param name="alignment">has to be a compile time constant</param>
        [IntrinsicFunction(IsNaked = true, Name = "__hybridizer_store_int2")]
        public unsafe static void Store(int2* ptr, int2 val, int alignment)
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
        public unsafe static void Store(int2* ptr, int val, int alignment)
        {
            *ptr = new int2(val);
        }

        /// <summary>
        /// loads from memory
        /// </summary>
        /// <param name="ptr"></param>
        /// <param name="alignment">has to be a compile time constant</param>
        [IntrinsicFunction(IsNaked = true, Name = "__hybridizer_load_int2")]
        public unsafe static int2 Load(int2* ptr, int alignment)
        {
            return *ptr;
        }
    }
}
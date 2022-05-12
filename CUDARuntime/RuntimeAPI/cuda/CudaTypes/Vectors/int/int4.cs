using System.Runtime.InteropServices;

namespace Hybridizer.Runtime.CUDAImports
{
    /// <summary>
    /// 4 integers, packed
    /// </summary>
    [IntrinsicType("int4")]
    [StructLayout(LayoutKind.Explicit)]
    public struct int4
    {
        /// <summary>
        /// first component
        /// </summary>
        [FieldOffset(0)]
        public int x;
        /// <summary>
        /// second component
        /// </summary>
        [FieldOffset(4)]
        public int y;
        /// <summary>
        /// third component
        /// </summary>
        [FieldOffset(8)]
        public int z;
        /// <summary>
        /// fourth component
        /// </summary>
        [FieldOffset(12)]
        public int w;

        /// <summary>
        /// constructor from 4 distinc integers
        /// </summary>
        private int4(int xx, int yy, int zz, int ww)
        {
            x = xx;
            y = yy;
            z = zz;
            w = ww;
        }

        /// <summary>
        /// make an int4 from 4 ints
        /// </summary>
        /// <param name="xx"></param>
        /// <param name="yy"></param>
        /// <param name="zz"></param>
        /// <param name="ww"></param>
        /// <returns></returns>
        [IntrinsicFunction("make_int4")]
        public static int4 make_int4(int xx, int yy, int zz, int ww)
        {
            return new int4(xx, yy, zz, ww);
        }

        /// <summary>
        /// addition operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator+")]
        public static int4 operator +(int4 a, int4 b)
        {
            int4 res;
            res.x = a.x + b.x;
            res.y = a.y + b.y;
            res.z = a.z + b.z;
            res.w = a.w + b.w;
            return res;
        }

        /// <summary>
        /// addition operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator+")]
        public static int4 operator +(int a, int4 b)
        {
            int4 res;
            res.x = a + b.x;
            res.y = a + b.y;
            res.z = a + b.z;
            res.w = a + b.w;
            return res;
        }

        /// <summary>
        /// addition operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator+")]
        public static int4 operator +(int4 a, int b)
        {
            int4 res;
            res.x = a.x + b;
            res.y = a.y + b;
            res.z = a.z + b;
            res.w = a.w + b;
            return res;
        }

        /// <summary>
        /// substraction operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator-")]
        public static int4 operator -(int4 a, int4 b)
        {
            int4 res;
            res.x = a.x - b.x;
            res.y = a.y - b.y;
            res.z = a.z - b.z;
            res.w = a.w - b.w;
            return res;
        }

        /// <summary>
        /// substraction operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator-")]
        public static int4 operator -(int a, int4 b)
        {
            int4 res;
            res.x = a - b.x;
            res.y = a - b.y;
            res.z = a - b.z;
            res.w = a - b.w;
            return res;
        }

        /// <summary>
        /// substraction operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator-")]
        public static int4 operator -(int4 a, int b)
        {
            int4 res;
            res.x = a.x - b;
            res.y = a.y - b;
            res.z = a.z - b;
            res.w = a.w - b;
            return res;
        }

        /// <summary>
        /// multiplication operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator*")]
        public static int4 operator *(int4 a, int4 b)
        {
            int4 res;
            res.x = a.x * b.x;
            res.y = a.y * b.y;
            res.z = a.z * b.z;
            res.w = a.w * b.w;
            return res;
        }

        /// <summary>
        /// multiplication operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator*")]
        public static int4 operator *(int a, int4 b)
        {
            int4 res;
            res.x = a * b.x;
            res.y = a * b.y;
            res.z = a * b.z;
            res.w = a * b.w;
            return res;
        }

        /// <summary>
        /// multiplication operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator*")]
        public static int4 operator *(int4 a, int b)
        {
            int4 res;
            res.x = a.x * b;
            res.y = a.y * b;
            res.z = a.z * b;
            res.w = a.w * b;
            return res;
        }

        /// <summary>
        /// division operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator/")]
        public static int4 operator /(int4 a, int4 b)
        {
            int4 res;
            res.x = a.x / b.x;
            res.y = a.y / b.y;
            res.z = a.z / b.z;
            res.w = a.w / b.w;
            return res;
        }

        /// <summary>
        /// division operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator/")]
        public static int4 operator /(int a, int4 b)
        {
            int4 res;
            res.x = a / b.x;
            res.y = a / b.y;
            res.z = a / b.z;
            res.w = a / b.w;
            return res;
        }

        /// <summary>
        /// division operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator/")]
        public static int4 operator /(int4 a, int b)
        {
            int4 res;
            res.x = a.x / b;
            res.y = a.y / b;
            res.z = a.z / b;
            res.w = a.w / b;
            return res;
        }

        /// <summary>
        /// bitwise AND operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator&")]
        public static int4 operator &(int4 a, int4 b)
        {
            int4 res;
            res.x = a.x & b.x;
            res.y = a.y & b.y;
            res.z = a.z & b.z;
            res.w = a.w & b.w;
            return res;
        }

        /// <summary>
        /// bitwise AND operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator&")]
        public static int4 operator &(int a, int4 b)
        {
            int4 res;
            res.x = a & b.x;
            res.y = a & b.y;
            res.z = a & b.z;
            res.w = a & b.w;
            return res;
        }

        /// <summary>
        /// bitwise AND operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator&")]
        public static int4 operator &(int4 a, int b)
        {
            int4 res;
            res.x = a.x & b;
            res.y = a.y & b;
            res.z = a.z & b;
            res.w = a.w & b;
            return res;
        }



        /// <summary>
        /// bitwise OR operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator|")]
        public static int4 operator |(int4 a, int4 b)
        {
            int4 res;
            res.x = a.x | b.x;
            res.y = a.y | b.y;
            res.z = a.z | b.z;
            res.w = a.w | b.w;
            return res;
        }

        /// <summary>
        /// bitwise OR operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator|")]
        public static int4 operator |(int a, int4 b)
        {
            int4 res;
            res.x = a | b.x;
            res.y = a | b.y;
            res.z = a | b.z;
            res.w = a | b.w;
            return res;
        }

        /// <summary>
        /// bitwise OR operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator|")]
        public static int4 operator |(int4 a, int b)
        {
            int4 res;
            res.x = a.x | b;
            res.y = a.y | b;
            res.z = a.z | b;
            res.w = a.w | b;
            return res;
        }

        /// <summary>
        /// bitwise XOR operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator^")]
        public static int4 operator ^(int4 a, int4 b)
        {
            int4 res;
            res.x = a.x ^ b.x;
            res.y = a.y ^ b.y;
            res.z = a.z ^ b.z;
            res.w = a.w ^ b.w;
            return res;
        }

        /// <summary>
        /// bitwise XOR operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator^")]
        public static int4 operator ^(int a, int4 b)
        {
            int4 res;
            res.x = a ^ b.x;
            res.y = a ^ b.y;
            res.z = a ^ b.z;
            res.w = a ^ b.w;
            return res;
        }

        /// <summary>
        /// bitwise XOR operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator^")]
        public static int4 operator ^(int4 a, int b)
        {
            int4 res;
            res.x = a.x ^ b;
            res.y = a.y ^ b;
            res.z = a.z ^ b;
            res.w = a.w ^ b;
            return res;
        }
        /// <summary>
        /// stores in memory
        /// </summary>
        /// <param name="ptr">destination pointer</param>
        /// <param name="val">value to store</param>
        /// <param name="alignment">has to be a compile time constant</param>
        [IntrinsicFunction(IsNaked = true, Name = "__hybridizer_store_int4")]
        public unsafe static void Store(int4* ptr, int4 val, int alignment)
        {
            *ptr = val;
        }
    }
}
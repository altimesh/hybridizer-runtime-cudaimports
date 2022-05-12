using System.Runtime.InteropServices;

namespace Hybridizer.Runtime.CUDAImports
{
    /// <summary>
    /// 8 32 bits integers
    /// </summary>
    [IntrinsicType("int8")]
    [StructLayout(LayoutKind.Explicit)]
    public struct int8
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
        /// z
        /// </summary>
        [FieldOffset(8)]
        public int z;
        /// <summary>
        /// w
        /// </summary>
        [FieldOffset(12)]
        public int w;
        /// <summary>
        /// x2
        /// </summary>
        [FieldOffset(16)]
        public int x2;
        /// <summary>
        /// y2
        /// </summary>
        [FieldOffset(20)]
        public int y2;
        /// <summary>
        /// z2
        /// </summary>
        [FieldOffset(24)]
        public int z2;
        /// <summary>
        /// w2
        /// </summary>
        [FieldOffset(28)]
        public int w2;

        /// <summary>
        /// constructor from single component
        /// </summary>
        public int8(int xx)
        {
            x = xx;
            y = xx;
            z = xx;
            w = xx;
            x2 = xx;
            y2 = xx;
            z2 = xx;
            w2 = xx;
        }

        /// <summary>
        /// constructor from components
        /// </summary>
        public int8(int xx, int yy, int zz, int ww, int xx2, int yy2, int zz2, int ww2)
        {
            x = xx;
            y = yy;
            z = zz;
            w = ww;
            x2 = xx2;
            y2 = yy2;
            z2 = zz2;
            w2 = ww2;
        }

        /// <summary>
        /// copy constructor
        /// </summary>
        public int8(int8 res)
        {
            x = res.x;
            y = res.y;
            z = res.z;
            w = res.w;
            x2 = res.x2;
            y2 = res.y2;
            z2 = res.z2;
            w2 = res.w2;
        }

        /// <summary>
        /// addition operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator+")]
        public static int8 operator +(int8 a, int8 b)
        {
            int8 res;
            res.x = a.x + b.x;
            res.y = a.y + b.y;
            res.z = a.z + b.z;
            res.w = a.w + b.w;
            res.x2 = a.x2 + b.x2;
            res.y2 = a.y2 + b.y2;
            res.z2 = a.z2 + b.z2;
            res.w2 = a.w2 + b.w2;
            return res;
        }

        /// <summary>
        /// bitwise XOR operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator^")]
        public static int8 operator ^(int8 a, int8 b)
        {
            int8 res;
            res.x = a.x ^ b.x;
            res.y = a.y ^ b.y;
            res.z = a.z ^ b.z;
            res.w = a.w ^ b.w;
            res.x2 = a.x2 ^ b.x2;
            res.y2 = a.y2 ^ b.y2;
            res.z2 = a.z2 ^ b.z2;
            res.w2 = a.w2 ^ b.w2;
            return res;
        }

        /// <summary>
        /// bitwise XOR operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator^")]
        public static int8 operator ^(int a, int8 b)
        {
            int8 res;
            res.x = a ^ b.x;
            res.y = a ^ b.y;
            res.z = a ^ b.z;
            res.w = a ^ b.w;
            res.x2 = a ^ b.x2;
            res.y2 = a ^ b.y2;
            res.z2 = a ^ b.z2;
            res.w2 = a ^ b.w2;
            return res;
        }

        /// <summary>
        /// bitwise XOR operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator^")]
        public static int8 operator ^(int8 a, int b)
        {
            int8 res;
            res.x = a.x ^ b;
            res.y = a.y ^ b;
            res.z = a.z ^ b;
            res.w = a.w ^ b;
            res.x2 = a.x2 ^ b;
            res.y2 = a.y2 ^ b;
            res.z2 = a.z2 ^ b;
            res.w2 = a.w2 ^ b;
            return res;
        }

        /// <summary>
        /// addition operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator+")]
        public static int8 operator +(int a, int8 b)
        {
            int8 res;
            res.x = a + b.x;
            res.y = a + b.y;
            res.z = a + b.z;
            res.w = a + b.w;
            res.x2 = a + b.x2;
            res.y2 = a + b.y2;
            res.z2 = a + b.z2;
            res.w2 = a + b.w2;
            return res;
        }

        /// <summary>
        /// addition operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator+")]
        public static int8 operator +(int8 a, int b)
        {
            int8 res;
            res.x = a.x + b;
            res.y = a.y + b;
            res.z = a.z + b;
            res.w = a.w + b;
            res.x2 = a.x2 + b;
            res.y2 = a.y2 + b;
            res.z2 = a.z2 + b;
            res.w2 = a.w2 + b;
            return res;
        }

        /// <summary>
        /// substraction operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator-")]
        public static int8 operator -(int8 a, int8 b)
        {
            int8 res;
            res.x = a.x - b.x;
            res.y = a.y - b.y;
            res.z = a.z - b.z;
            res.w = a.w - b.w;
            res.x2 = a.x2 - b.x2;
            res.y2 = a.y2 - b.y2;
            res.z2 = a.z2 - b.z2;
            res.w2 = a.w2 - b.w2;
            return res;
        }

        /// <summary>
        /// substraction operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator-")]
        public static int8 operator -(int a, int8 b)
        {
            int8 res;
            res.x = a - b.x;
            res.y = a - b.y;
            res.z = a - b.z;
            res.w = a - b.w;
            res.x2 = a - b.x2;
            res.y2 = a - b.y2;
            res.z2 = a - b.z2;
            res.w2 = a - b.w2;
            return res;
        }

        /// <summary>
        /// substraction operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator-")]
        public static int8 operator -(int8 a, int b)
        {
            int8 res;
            res.x = a.x - b;
            res.y = a.y - b;
            res.z = a.z - b;
            res.w = a.w - b;
            res.x2 = a.x2 - b;
            res.y2 = a.y2 - b;
            res.z2 = a.z2 - b;
            res.w2 = a.w2 - b;
            return res;
        }

        /// <summary>
        /// multiplication operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator*")]
        public static int8 operator *(int8 a, int8 b)
        {
            int8 res;
            res.x = a.x * b.x;
            res.y = a.y * b.y;
            res.z = a.z * b.z;
            res.w = a.w * b.w;
            res.x2 = a.x2 * b.x2;
            res.y2 = a.y2 * b.y2;
            res.z2 = a.z2 * b.z2;
            res.w2 = a.w2 * b.w2;
            return res;
        }

        /// <summary>
        /// multiplication operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator*")]
        public static int8 operator *(int a, int8 b)
        {
            int8 res;
            res.x = a * b.x;
            res.y = a * b.y;
            res.z = a * b.z;
            res.w = a * b.w;
            res.x2 = a * b.x2;
            res.y2 = a * b.y2;
            res.z2 = a * b.z2;
            res.w2 = a * b.w2;
            return res;
        }

        /// <summary>
        /// multiplication operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator*")]
        public static int8 operator *(int8 a, int b)
        {
            int8 res;
            res.x = a.x * b;
            res.y = a.y * b;
            res.z = a.z * b;
            res.w = a.w * b;
            res.x2 = a.x2 * b;
            res.y2 = a.y2 * b;
            res.z2 = a.z2 * b;
            res.w2 = a.w2 * b;
            return res;
        }

        /// <summary>
        /// division operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator/")]
        public static int8 operator /(int8 a, int8 b)
        {
            int8 res;
            res.x = a.x / b.x;
            res.y = a.y / b.y;
            res.z = a.z / b.z;
            res.w = a.w / b.w;
            res.x2 = a.x2 / b.x2;
            res.y2 = a.y2 / b.y2;
            res.z2 = a.z2 / b.z2;
            res.w2 = a.w2 / b.w2;
            return res;
        }

        /// <summary>
        /// division operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator/")]
        public static int8 operator /(int a, int8 b)
        {
            int8 res;
            res.x = a / b.x;
            res.y = a / b.y;
            res.z = a / b.z;
            res.w = a / b.w;
            res.x2 = a / b.x2;
            res.y2 = a / b.y2;
            res.z2 = a / b.z2;
            res.w2 = a / b.w2;
            return res;
        }

        /// <summary>
        /// division operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator/")]
        public static int8 operator /(int8 a, int b)
        {
            int8 res;
            res.x = a.x / b;
            res.y = a.y / b;
            res.z = a.z / b;
            res.w = a.w / b;
            res.x2 = a.x2 / b;
            res.y2 = a.y2 / b;
            res.z2 = a.z2 / b;
            res.w2 = a.w2 / b;
            return res;
        }

        /// <summary>
        /// stores in memory
        /// </summary>
        /// <param name="ptr">destination pointer</param>
        /// <param name="val">value to store</param>
        /// <param name="alignment">has to be a compile time constant</param>
        [IntrinsicFunction(IsNaked = true, Name = "__hybridizer_store_int8")]
        public unsafe static void Store(int8* ptr, int8 val, int alignment)
        {
            *ptr = val;
        }

        /// <summary>
        /// stores in memory
        /// </summary>
        /// <param name="ptr">destination pointer</param>
        /// <param name="val">value to store</param>
        /// <param name="alignment">has to be a compile time constant</param>
        [IntrinsicFunction(IsNaked = true, Name = "__hybridizer_store_splat_int8")]
        public unsafe static void Store(int8* ptr, int val, int alignment)
        {
            *ptr = new int8(val);
        }

        /// <summary>
        /// loads from memory
        /// </summary>
        /// <param name="ptr"></param>
        /// <param name="alignment">has to be a compile time constant</param>
        [IntrinsicFunction(IsNaked = true, Name = "__hybridizer_load_int8")]
        public unsafe static int8 Load(int8* ptr, int alignment)
        {
            return *ptr;
        }
    }
}
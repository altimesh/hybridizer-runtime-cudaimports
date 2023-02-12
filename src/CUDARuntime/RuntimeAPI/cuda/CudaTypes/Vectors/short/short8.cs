using System.Runtime.InteropServices;

namespace Hybridizer.Runtime.CUDAImports
{
    /// <summary>
    /// 8 16 bits integers
    /// </summary>
    [IntrinsicType("short8")]
    [StructLayout(LayoutKind.Explicit)]
    public struct short8
    {
        /// <summary>
        /// x
        /// </summary>
        [FieldOffset(0)]
        public short x;
        /// <summary>
        /// y
        /// </summary>
        [FieldOffset(2)]
        public short y;
        /// <summary>
        /// z
        /// </summary>
        [FieldOffset(4)]
        public short z;
        /// <summary>
        /// w
        /// </summary>
        [FieldOffset(6)]
        public short w;
        /// <summary>
        /// x2
        /// </summary>
        [FieldOffset(8)]
        public short x2;
        /// <summary>
        /// y2
        /// </summary>
        [FieldOffset(10)]
        public short y2;
        /// <summary>
        /// z2
        /// </summary>
        [FieldOffset(12)]
        public short z2;
        /// <summary>
        /// w2
        /// </summary>
        [FieldOffset(14)]
        public short w2;

        /// <summary>
        /// constructor from components
        /// </summary>
        public short8(short xx, short yy, short zz, short ww, short xx2, short yy2, short zz2, short ww2)
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
        /// constructor from single component
        /// </summary>
        public short8(short val)
        {
            x = val;
            y = val;
            z = val;
            w = val;
            x2 = val;
            y2 = val;
            z2 = val;
            w2 = val;
        }

        /// <summary>
        /// copy constructor
        /// </summary>
        public short8(short8 res)
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
        public static short8 operator +(short8 a, short8 b)
        {
            short8 res;
            res.x = (short)(a.x + b.x);
            res.y = (short)(a.y + b.y);
            res.z = (short)(a.z + b.z);
            res.w = (short)(a.w + b.w);
            res.x2 = (short)(a.x2 + b.x2);
            res.y2 = (short)(a.y2 + b.y2);
            res.z2 = (short)(a.z2 + b.z2);
            res.w2 = (short)(a.w2 + b.w2);
            return res;
        }

        /// <summary>
        /// addition operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator+")]
        public static short8 operator +(short a, short8 b)
        {
            short8 res;
            res.x = (short)(a + b.x);
            res.y = (short)(a + b.y);
            res.z = (short)(a + b.z);
            res.w = (short)(a + b.w);
            res.x2 = (short)(a + b.x2);
            res.y2 = (short)(a + b.y2);
            res.z2 = (short)(a + b.z2);
            res.w2 = (short)(a + b.w2);
            return res;
        }

        /// <summary>
        /// addition operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator+")]
        public static short8 operator +(short8 a, short b)
        {
            short8 res;
            res.x = (short)(a.x + b);
            res.y = (short)(a.y + b);
            res.z = (short)(a.z + b);
            res.w = (short)(a.w + b);
            res.x2 = (short)(a.x2 + b);
            res.y2 = (short)(a.y2 + b);
            res.z2 = (short)(a.z2 + b);
            res.w2 = (short)(a.w2 + b);
            return res;
        }

        /// <summary>
        /// substraction operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator-")]
        public static short8 operator -(short8 a, short8 b)
        {
            short8 res;
            res.x = (short)(a.x - b.x);
            res.y = (short)(a.y - b.y);
            res.z = (short)(a.z - b.z);
            res.w = (short)(a.w - b.w);
            res.x2 = (short)(a.x2 - b.x2);
            res.y2 = (short)(a.y2 - b.y2);
            res.z2 = (short)(a.z2 - b.z2);
            res.w2 = (short)(a.w2 - b.w2);
            return res;
        }

        /// <summary>
        /// substraction operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator-")]
        public static short8 operator -(short a, short8 b)
        {
            short8 res;
            res.x = (short)(a - b.x);
            res.y = (short)(a - b.y);
            res.z = (short)(a - b.z);
            res.w = (short)(a - b.w);
            res.x2 = (short)(a - b.x2);
            res.y2 = (short)(a - b.y2);
            res.z2 = (short)(a - b.z2);
            res.w2 = (short)(a - b.w2);
            return res;
        }

        /// <summary>
        /// substraction operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator-")]
        public static short8 operator -(short8 a, short b)
        {
            short8 res;
            res.x = (short)(a.x - b);
            res.y = (short)(a.y - b);
            res.z = (short)(a.z - b);
            res.w = (short)(a.w - b);
            res.x2 = (short)(a.x2 - b);
            res.y2 = (short)(a.y2 - b);
            res.z2 = (short)(a.z2 - b);
            res.w2 = (short)(a.w2 - b);
            return res;
        }

        /// <summary>
        /// multiplication operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator*")]
        public static short8 operator *(short8 a, short8 b)
        {
            short8 res;
            res.x = (short)(a.x * b.x);
            res.y = (short)(a.y * b.y);
            res.z = (short)(a.z * b.z);
            res.w = (short)(a.w * b.w);
            res.x2 = (short)(a.x2 * b.x2);
            res.y2 = (short)(a.y2 * b.y2);
            res.z2 = (short)(a.z2 * b.z2);
            res.w2 = (short)(a.w2 * b.w2);
            return res;
        }

        /// <summary>
        /// multiplication operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator*")]
        public static short8 operator *(short a, short8 b)
        {
            short8 res;
            res.x = (short)(a * b.x);
            res.y = (short)(a * b.y);
            res.z = (short)(a * b.z);
            res.w = (short)(a * b.w);
            res.x2 = (short)(a * b.x2);
            res.y2 = (short)(a * b.y2);
            res.z2 = (short)(a * b.z2);
            res.w2 = (short)(a * b.w2);
            return res;
        }

        /// <summary>
        /// multiplication operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator*")]
        public static short8 operator *(short8 a, short b)
        {
            short8 res;
            res.x = (short)(a.x * b);
            res.y = (short)(a.y * b);
            res.z = (short)(a.z * b);
            res.w = (short)(a.w * b);
            res.x2 = (short)(a.x2 * b);
            res.y2 = (short)(a.y2 * b);
            res.z2 = (short)(a.z2 * b);
            res.w2 = (short)(a.w2 * b);
            return res;
        }

        /// <summary>
        /// division operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator/")]
        public static short8 operator /(short8 a, short8 b)
        {
            short8 res;
            res.x = (short)(a.x / b.x);
            res.y = (short)(a.y / b.y);
            res.z = (short)(a.z / b.z);
            res.w = (short)(a.w / b.w);
            res.x2 = (short)(a.x2 / b.x2);
            res.y2 = (short)(a.y2 / b.y2);
            res.z2 = (short)(a.z2 / b.z2);
            res.w2 = (short)(a.w2 / b.w2);
            return res;
        }

        /// <summary>
        /// division operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator/")]
        public static short8 operator /(short a, short8 b)
        {
            short8 res;
            res.x = (short)(a / b.x);
            res.y = (short)(a / b.y);
            res.z = (short)(a / b.z);
            res.w = (short)(a / b.w);
            res.x2 = (short)(a / b.x2);
            res.y2 = (short)(a / b.y2);
            res.z2 = (short)(a / b.z2);
            res.w2 = (short)(a / b.w2);
            return res;
        }

        /// <summary>
        /// division operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator/")]
        public static short8 operator /(short8 a, short b)
        {
            short8 res;
            res.x = (short)(a.x / b);
            res.y = (short)(a.y / b);
            res.z = (short)(a.z / b);
            res.w = (short)(a.w / b);
            res.x2 = (short)(a.x2 / b);
            res.y2 = (short)(a.y2 / b);
            res.z2 = (short)(a.z2 / b);
            res.w2 = (short)(a.w2 / b);
            return res;
        }

        /// <summary>
        /// bitwise AND operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator&")]
        public static short8 operator &(short8 a, short8 b)
        {
            short8 res;
            res.x = (short)(a.x & b.x);
            res.y = (short)(a.y & b.y);
            res.z = (short)(a.z & b.z);
            res.w = (short)(a.w & b.w);
            res.x2 = (short)(a.x2 & b.x2);
            res.y2 = (short)(a.y2 & b.y2);
            res.z2 = (short)(a.z2 & b.z2);
            res.w2 = (short)(a.w2 & b.w2);
            return res;
        }

        /// <summary>
        /// bitwise AND operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator&")]
        public static short8 operator &(short a, short8 b)
        {
            short8 res;
            res.x = (short)(a & b.x);
            res.y = (short)(a & b.y);
            res.z = (short)(a & b.z);
            res.w = (short)(a & b.w);
            res.x2 = (short)(a & b.x2);
            res.y2 = (short)(a & b.y2);
            res.z2 = (short)(a & b.z2);
            res.w2 = (short)(a & b.w2);
            return res;
        }

        /// <summary>
        /// bitwise AND operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator&")]
        public static short8 operator &(short8 a, short b)
        {
            short8 res;
            res.x = (short)(a.x & b);
            res.y = (short)(a.y & b);
            res.z = (short)(a.z & b);
            res.w = (short)(a.w & b);
            res.x2 = (short)(a.x2 & b);
            res.y2 = (short)(a.y2 & b);
            res.z2 = (short)(a.z2 & b);
            res.w2 = (short)(a.w2 & b);
            return res;
        }

        /// <summary>
        /// bitwise OR operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator|")]
        public static short8 operator |(short8 a, short8 b)
        {
            short8 res;
            res.x = (short)(a.x | b.x);
            res.y = (short)(a.y | b.y);
            res.z = (short)(a.z | b.z);
            res.w = (short)(a.w | b.w);
            res.x2 = (short)(a.x2 | b.x2);
            res.y2 = (short)(a.y2 | b.y2);
            res.z2 = (short)(a.z2 | b.z2);
            res.w2 = (short)(a.w2 | b.w2);
            return res;
        }

        /// <summary>
        /// bitwise OR operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator|")]
        public static short8 operator |(short a, short8 b)
        {
            short8 res;
            res.x = (short)(a | b.x);
            res.y = (short)(a | b.y);
            res.z = (short)(a | b.z);
            res.w = (short)(a | b.w);
            res.x2 = (short)(a | b.x2);
            res.y2 = (short)(a | b.y2);
            res.z2 = (short)(a | b.z2);
            res.w2 = (short)(a | b.w2);
            return res;
        }

        /// <summary>
        /// bitwise OR operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator|")]
        public static short8 operator |(short8 a, short b)
        {
            short8 res;
            res.x = (short)(a.x | b);
            res.y = (short)(a.y | b);
            res.z = (short)(a.z | b);
            res.w = (short)(a.w | b);
            res.x2 = (short)(a.x2 | b);
            res.y2 = (short)(a.y2 | b);
            res.z2 = (short)(a.z2 | b);
            res.w2 = (short)(a.w2 | b);
            return res;
        }

        /// <summary>
        /// bitwise XOR operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator^")]
        public static short8 operator ^(short8 a, short8 b)
        {
            short8 res;
            res.x = (short)(a.x ^ b.x);
            res.y = (short)(a.y ^ b.y);
            res.z = (short)(a.z ^ b.z);
            res.w = (short)(a.w ^ b.w);
            res.x2 = (short)(a.x2 ^ b.x2);
            res.y2 = (short)(a.y2 ^ b.y2);
            res.z2 = (short)(a.z2 ^ b.z2);
            res.w2 = (short)(a.w2 ^ b.w2);
            return res;
        }

        /// <summary>
        /// bitwise XOR operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator^")]
        public static short8 operator ^(short a, short8 b)
        {
            short8 res;
            res.x = (short)(a ^ b.x);
            res.y = (short)(a ^ b.y);
            res.z = (short)(a ^ b.z);
            res.w = (short)(a ^ b.w);
            res.x2 = (short)(a ^ b.x2);
            res.y2 = (short)(a ^ b.y2);
            res.z2 = (short)(a ^ b.z2);
            res.w2 = (short)(a ^ b.w2);
            return res;
        }

        /// <summary>
        /// bitwise XOR operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator^")]
        public static short8 operator ^(short8 a, short b)
        {
            short8 res;
            res.x = (short)(a.x ^ b);
            res.y = (short)(a.y ^ b);
            res.z = (short)(a.z ^ b);
            res.w = (short)(a.w ^ b);
            res.x2 = (short)(a.x2 ^ b);
            res.y2 = (short)(a.y2 ^ b);
            res.z2 = (short)(a.z2 ^ b);
            res.w2 = (short)(a.w2 ^ b);
            return res;
        }

        /// <summary>
        /// loads from memory
        /// </summary>
        /// <param name="ptr"></param>
        /// <param name="alignment">has to be a compile time constant</param>
        [IntrinsicFunction(IsNaked = true, Name = "__hybridizer_load_short8")]
        public unsafe static short8 Load(short8* ptr, int alignment)
        {
            return *ptr;
        }

        /// <summary>
        /// stores in memory
        /// </summary>
        /// <param name="ptr">destination pointer</param>
        /// <param name="val">value to store</param>
        /// <param name="alignment">has to be a compile time constant</param>
        [IntrinsicFunction(IsNaked = true, Name = "__hybridizer_store_short8")]
        public unsafe static void Store(short8* ptr, short8 val, int alignment)
        {
            *ptr = val;
        }
    }
}
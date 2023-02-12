using System.Runtime.InteropServices;

namespace Hybridizer.Runtime.CUDAImports
{
    /// <summary>
    /// 4 16 bits integers
    /// </summary>
    [IntrinsicType("short4")]
    [StructLayout(LayoutKind.Explicit)]
    public struct short4
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
        /// constructor from components
        /// </summary>
        public short4(short xx, short yy, short zz, short ww)
        {
            x = xx;
            y = yy;
            z = zz;
            w = ww;
        }

        /// <summary>
        /// constructor from single component
        /// </summary>
        public short4(short val)
        {
            x = val;
            y = val;
            z = val;
            w = val;
        }

        /// <summary>
        /// copy constructor
        /// </summary>
        /// <param name="res"></param>
        public short4(short4 res)
        {
            x = res.x;
            y = res.y;
            z = res.z;
            w = res.w;
        }

        /// <summary>
        /// addition operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator+")]
        public static short4 operator +(short4 a, short4 b)
        {
            short4 res;
            res.x = (short)(a.x + b.x);
            res.y = (short)(a.y + b.y);
            res.z = (short)(a.z + b.z);
            res.w = (short)(a.w + b.w);
            return res;
        }

        /// <summary>
        /// addition operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator+")]
        public static short4 operator +(short a, short4 b)
        {
            short4 res;
            res.x = (short)(a + b.x);
            res.y = (short)(a + b.y);
            res.z = (short)(a + b.z);
            res.w = (short)(a + b.w);
            return res;
        }

        /// <summary>
        /// addition operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator+")]
        public static short4 operator +(short4 a, short b)
        {
            short4 res;
            res.x = (short)(a.x + b);
            res.y = (short)(a.y + b);
            res.z = (short)(a.z + b);
            res.w = (short)(a.w + b);
            return res;
        }

        /// <summary>
        /// substraction operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator-")]
        public static short4 operator -(short4 a, short4 b)
        {
            short4 res;
            res.x = (short)(a.x - b.x);
            res.y = (short)(a.y - b.y);
            res.z = (short)(a.z - b.z);
            res.w = (short)(a.w - b.w);
            return res;
        }

        /// <summary>
        /// substraction operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator-")]
        public static short4 operator -(short a, short4 b)
        {
            short4 res;
            res.x = (short)(a - b.x);
            res.y = (short)(a - b.y);
            res.z = (short)(a - b.z);
            res.w = (short)(a - b.w);
            return res;
        }

        /// <summary>
        /// substraction operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator-")]
        public static short4 operator -(short4 a, short b)
        {
            short4 res;
            res.x = (short)(a.x - b);
            res.y = (short)(a.y - b);
            res.z = (short)(a.z - b);
            res.w = (short)(a.w - b);
            return res;
        }

        /// <summary>
        /// multiplication operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator*")]
        public static short4 operator *(short4 a, short4 b)
        {
            short4 res;
            res.x = (short)(a.x * b.x);
            res.y = (short)(a.y * b.y);
            res.z = (short)(a.z * b.z);
            res.w = (short)(a.w * b.w);
            return res;
        }

        /// <summary>
        /// multiplication operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator*")]
        public static short4 operator *(short a, short4 b)
        {
            short4 res;
            res.x = (short)(a * b.x);
            res.y = (short)(a * b.y);
            res.z = (short)(a * b.z);
            res.w = (short)(a * b.w);
            return res;
        }

        /// <summary>
        /// multiplication operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator*")]
        public static short4 operator *(short4 a, short b)
        {
            short4 res;
            res.x = (short)(a.x * b);
            res.y = (short)(a.y * b);
            res.z = (short)(a.z * b);
            res.w = (short)(a.w * b);
            return res;
        }

        /// <summary>
        /// division operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator/")]
        public static short4 operator /(short4 a, short4 b)
        {
            short4 res;
            res.x = (short)(a.x / b.x);
            res.y = (short)(a.y / b.y);
            res.z = (short)(a.z / b.z);
            res.w = (short)(a.w / b.w);
            return res;
        }

        /// <summary>
        /// division operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator/")]
        public static short4 operator /(short a, short4 b)
        {
            short4 res;
            res.x = (short)(a / b.x);
            res.y = (short)(a / b.y);
            res.z = (short)(a / b.z);
            res.w = (short)(a / b.w);
            return res;
        }

        /// <summary>
        /// division operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator/")]
        public static short4 operator /(short4 a, short b)
        {
            short4 res;
            res.x = (short)(a.x / b);
            res.y = (short)(a.y / b);
            res.z = (short)(a.z / b);
            res.w = (short)(a.w / b);
            return res;
        }

        /// <summary>
        /// loads from memory
        /// </summary>
        /// <param name="ptr"></param>
        /// <param name="alignment">has to be a compile time constant</param>
        [IntrinsicFunction(IsNaked = true, Name = "__hybridizer_load_short4")]
        public unsafe static short4 Load(short4* ptr, int alignment)
        {
            return *ptr;
        }

        /// <summary>
        /// stores in memory
        /// </summary>
        /// <param name="ptr">destination pointer</param>
        /// <param name="val">value to store</param>
        /// <param name="alignment">has to be a compile time constant</param>
        [IntrinsicFunction(IsNaked = true, Name = "__hybridizer_store_short4")]
        public unsafe static void Store(short4* ptr, short4 val, int alignment)
        {
            *ptr = val;
        }
    }

}
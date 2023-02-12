using System.Runtime.InteropServices;

namespace Hybridizer.Runtime.CUDAImports
{
    /// <summary>
    /// 2 26 bits integers
    /// </summary>
    [IntrinsicType("short2")]
    [StructLayout(LayoutKind.Explicit)]
    public unsafe struct short2
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
        /// constructor from 32 bits integer
        /// </summary>
        public short2(int val)
        {
            x = (short)(val & 0xFFFF);
            y = (short)((val >> 16) & 0xFFFF);
        }

        /// <summary>
        /// constructor from components
        /// </summary>
        public short2(short xx, short yy)
        {
            x = xx;
            y = yy;
        }

        /// <summary>
        /// constructor from single component
        /// </summary>
        public short2(short val)
        {
            x = val;
            y = val;
        }

        /// <summary>
        /// copy constructor
        /// </summary>
        public short2(short2 res)
        {
            x = res.x;
            y = res.y;
        }

        /// <summary>
        /// conversion to 32 bits integer
        /// </summary>
        /// <param name="res"></param>
        public static explicit operator int(short2 res)
        {
            int* ptr = (int*)&res;
            return *ptr;
        }

        /// <summary>
        /// addition operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator+")]
        public static short2 operator +(short2 a, short2 b)
        {
            short2 res;
            res.x = (short)(a.x + b.x);
            res.y = (short)(a.y + b.y);
            return res;
        }

        /// <summary>
        /// addition operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator+")]
        public static short2 operator +(short a, short2 b)
        {
            short2 res;
            res.x = (short)(a + b.x);
            res.y = (short)(a + b.y);
            return res;
        }

        /// <summary>
        /// addition operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator+")]
        public static short2 operator +(short2 a, short b)
        {
            short2 res;
            res.x = (short)(a.x + b);
            res.y = (short)(a.y + b);
            return res;
        }

        /// <summary>
        /// substraction operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator-")]
        public static short2 operator -(short2 a, short2 b)
        {
            short2 res;
            res.x = (short)(a.x - b.x);
            res.y = (short)(a.y - b.y);
            return res;
        }

        /// <summary>
        /// substraction operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator-")]
        public static short2 operator -(short a, short2 b)
        {
            short2 res;
            res.x = (short)(a - b.x);
            res.y = (short)(a - b.y);
            return res;
        }

        /// <summary>
        /// substraction operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator-")]
        public static short2 operator -(short2 a, short b)
        {
            short2 res;
            res.x = (short)(a.x - b);
            res.y = (short)(a.y - b);
            return res;
        }

        /// <summary>
        /// multiplication operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator*")]
        public static short2 operator *(short2 a, short2 b)
        {
            short2 res;
            res.x = (short)(a.x * b.x);
            res.y = (short)(a.y * b.y);
            return res;
        }

        /// <summary>
        /// multiplication operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator*")]
        public static short2 operator *(short a, short2 b)
        {
            short2 res;
            res.x = (short)(a * b.x);
            res.y = (short)(a * b.y);
            return res;
        }

        /// <summary>
        /// multiplication operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator*")]
        public static short2 operator *(short2 a, short b)
        {
            short2 res;
            res.x = (short)(a.x * b);
            res.y = (short)(a.y * b);
            return res;
        }

        /// <summary>
        /// division operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator/")]
        public static short2 operator /(short2 a, short2 b)
        {
            short2 res;
            res.x = (short)(a.x / b.x);
            res.y = (short)(a.y / b.y);
            return res;
        }

        /// <summary>
        /// division operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator/")]
        public static short2 operator /(short a, short2 b)
        {
            short2 res;
            res.x = (short)(a / b.x);
            res.y = (short)(a / b.y);
            return res;
        }

        /// <summary>
        /// division operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator/")]
        public static short2 operator /(short2 a, short b)
        {
            short2 res;
            res.x = (short)(a.x / b);
            res.y = (short)(a.y / b);
            return res;
        }

        /// <summary>
        /// stores in memory
        /// </summary>
        /// <param name="ptr">destination pointer</param>
        /// <param name="val">value to store</param>
        /// <param name="alignment">has to be a compile time constant</param>
        [IntrinsicFunction(IsNaked = true, Name = "__hybridizer_store_short2")]
        public unsafe static void Store(short2* ptr, short2 val, int alignment)
        {
            *ptr = val;
        }

        /// <summary>
        /// stores in memory
        /// </summary>
        /// <param name="ptr">destination pointer</param>
        /// <param name="val">value to store</param>
        /// <param name="alignment">has to be a compile time constant</param>
        [IntrinsicFunction(IsNaked = true, Name = "__hybridizer_store_splat_short2")]
        public unsafe static void Store(short2* ptr, sbyte val, int alignment)
        {
            *ptr = new short2(val);
        }

        /// <summary>
        /// loads from memory
        /// </summary>
        /// <param name="ptr"></param>
        /// <param name="alignment">has to be a compile time constant</param>
        [IntrinsicFunction(IsNaked = true, Name = "__hybridizer_load_short2")]
        public unsafe static short2 Load(short2* ptr, int alignment)
        {
            return *ptr;
        }
    }
}
using System.Runtime.InteropServices;

namespace Hybridizer.Runtime.CUDAImports
{
    /// <summary>
    /// four unsigned signed bytes
    /// </summary>
    [IntrinsicType("uchar4")]
    [IntrinsicPrimitive("uchar4")]
    [StructLayout(LayoutKind.Explicit)]
    public struct uchar4
    {
        /// <summary>
        /// x
        /// </summary>
        [FieldOffset(0)]
        public byte x;
        /// <summary>
        /// y
        /// </summary>
        [FieldOffset(1)]
        public byte y;
        /// <summary>
        /// z
        /// </summary>
        [FieldOffset(2)]
        public byte z;
        /// <summary>
        /// w
        /// </summary>
        [FieldOffset(3)]
        public byte w;

        /// <summary>
        /// constructor from components
        /// </summary>
        /// <param name="xx"></param>
        /// <param name="yy"></param>
        /// <param name="zz"></param>
        /// <param name="ww"></param>
        public uchar4(byte xx, byte yy, byte zz, byte ww)
        {
            x = xx;
            y = yy;
            z = zz;
            w = ww;
        }

        /// <summary>
        ///  constructor from signed 32 bits integer
        /// </summary>
        /// <param name="val"></param>
        public uchar4(int val)
        {
            // TODO: is that correct?? from pr60960 it looks like, but from logic it doesn't
            x = (byte)(val & 0xFF);
            y = (byte)((val >> 8) & 0xFF);
            z = (byte)((val >> 16) & 0xFF);
            w = (byte)((val >> 24) & 0xFF);
        }

        /// <summary>
        /// constructor from single component
        /// </summary>
        public uchar4(byte val)
        {
            x = val;
            y = val;
            z = val;
            w = val;
        }

        /// <summary>
        ///  copy constructor
        /// </summary>
        public uchar4(uchar4 res)
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
        public static uchar4 operator +(uchar4 a, uchar4 b)
        {
            uchar4 res;
            res.x = (byte)(a.x + b.x);
            res.y = (byte)(a.y + b.y);
            res.z = (byte)(a.z + b.z);
            res.w = (byte)(a.w + b.w);
            return res;
        }

        /// <summary>
        /// addition operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator+")]
        public static uchar4 operator +(byte a, uchar4 b)
        {
            uchar4 res;
            res.x = (byte)(a + b.x);
            res.y = (byte)(a + b.y);
            res.z = (byte)(a + b.z);
            res.w = (byte)(a + b.w);
            return res;
        }

        /// <summary>
        /// addition operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator+")]
        public static uchar4 operator +(uchar4 a, sbyte b)
        {
            uchar4 res;
            res.x = (byte)(a.x + b);
            res.y = (byte)(a.y + b);
            res.z = (byte)(a.z + b);
            res.w = (byte)(a.w + b);
            return res;
        }

        /// <summary>
        /// substraction operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator-")]
        public static uchar4 operator -(uchar4 a, uchar4 b)
        {
            uchar4 res;
            res.x = (byte)(a.x - b.x);
            res.y = (byte)(a.y - b.y);
            res.z = (byte)(a.z - b.z);
            res.w = (byte)(a.w - b.w);
            return res;
        }

        /// <summary>
        /// substraction operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator-")]
        public static uchar4 operator -(byte a, uchar4 b)
        {
            uchar4 res;
            res.x = (byte)(a - b.x);
            res.y = (byte)(a - b.y);
            res.z = (byte)(a - b.z);
            res.w = (byte)(a - b.w);
            return res;
        }

        /// <summary>
        /// substraction operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator-")]
        public static uchar4 operator -(uchar4 a, byte b)
        {
            uchar4 res;
            res.x = (byte)(a.x - b);
            res.y = (byte)(a.y - b);
            res.z = (byte)(a.z - b);
            res.w = (byte)(a.w - b);
            return res;
        }

        /// <summary>
        /// multiplication operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator*")]
        public static uchar4 operator *(uchar4 a, uchar4 b)
        {
            uchar4 res;
            res.x = (byte)(a.x * b.x);
            res.y = (byte)(a.y * b.y);
            res.z = (byte)(a.z * b.z);
            res.w = (byte)(a.w * b.w);
            return res;
        }

        /// <summary>
        /// multiplication operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator*")]
        public static uchar4 operator *(byte a, uchar4 b)
        {
            uchar4 res;
            res.x = (byte)(a * b.x);
            res.y = (byte)(a * b.y);
            res.z = (byte)(a * b.z);
            res.w = (byte)(a * b.w);
            return res;
        }

        /// <summary>
        /// multiplication operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator*")]
        public static uchar4 operator *(uchar4 a, byte b)
        {
            uchar4 res;
            res.x = (byte)(a.x * b);
            res.y = (byte)(a.y * b);
            res.z = (byte)(a.z * b);
            res.w = (byte)(a.w * b);
            return res;
        }

        /// <summary>
        /// division operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator/")]
        public static uchar4 operator /(uchar4 a, uchar4 b)
        {
            uchar4 res;
            res.x = (byte)(a.x / b.x);
            res.y = (byte)(a.y / b.y);
            res.z = (byte)(a.z / b.z);
            res.w = (byte)(a.w / b.w);
            return res;
        }

        /// <summary>
        /// division operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator/")]
        public static uchar4 operator /(byte a, uchar4 b)
        {
            uchar4 res;
            res.x = (byte)(a / b.x);
            res.y = (byte)(a / b.y);
            res.z = (byte)(a / b.z);
            res.w = (byte)(a / b.w);
            return res;
        }

        /// <summary>
        /// division operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator/")]
        public static uchar4 operator /(uchar4 a, byte b)
        {
            uchar4 res;
            res.x = (byte)(a.x / b);
            res.y = (byte)(a.y / b);
            res.z = (byte)(a.z / b);
            res.w = (byte)(a.w / b);
            return res;
        }

        /// <summary>
        /// stores in memory
        /// </summary>
        /// <param name="ptr">destination pointer</param>
        /// <param name="val">value to store</param>
        /// <param name="alignment">has to be a compile time constant</param>
        [IntrinsicFunction(IsNaked = true, Name = "__hybridizer_store_uchar4")]
        public unsafe static void Store(uchar4* ptr, uchar4 val, int alignment)
        {
            *ptr = val;
        }

        /// <summary>
        /// stores in memory
        /// </summary>
        /// <param name="ptr">destination pointer</param>
        /// <param name="val">value to store</param>
        /// <param name="alignment">has to be a compile time constant</param>
        [IntrinsicFunction(IsNaked = true, Name = "__hybridizer_store_splat_uchar4")]
        public unsafe static void Store(uchar4* ptr, byte val, int alignment)
        {
            *ptr = new uchar4(val);
        }

        /// <summary>
        /// loads from memory
        /// </summary>
        /// <param name="ptr"></param>
        /// <param name="alignment">has to be a compile time constant</param>
        [IntrinsicFunction(IsNaked = true, Name = "__hybridizer_load_uchar4")]
        public unsafe static uchar4 Load(uchar4* ptr, int alignment)
        {
            return *ptr;
        }
    }
}
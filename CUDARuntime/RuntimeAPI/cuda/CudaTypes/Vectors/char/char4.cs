using System.Runtime.InteropServices;

namespace Hybridizer.Runtime.CUDAImports
{
    /// <summary>
    /// four signed bytes
    /// </summary>
    [IntrinsicType("char4")]
    [StructLayout(LayoutKind.Explicit)]
    public struct char4
    {
        /// <summary>
        /// x
        /// </summary>
        [FieldOffset(0)]
        public sbyte x;
        /// <summary>
        /// y
        /// </summary>
        [FieldOffset(1)]
        public sbyte y;
        /// <summary>
        /// z
        /// </summary>
        [FieldOffset(2)]
        public sbyte z;
        /// <summary>
        /// w
        /// </summary>
        [FieldOffset(3)]
        public sbyte w;

        /// <summary>
        /// constructor from components
        /// </summary>
        /// <param name="xx"></param>
        /// <param name="yy"></param>
        /// <param name="zz"></param>
        /// <param name="ww"></param>
        public char4(sbyte xx, sbyte yy, sbyte zz, sbyte ww)
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
        public char4(int val)
        {
            // TODO: is that correct?? from pr60960 it looks like, but from logic it doesn't
            x = (sbyte)(val & 0xFF);
            y = (sbyte)((val >> 8) & 0xFF);
            z = (sbyte)((val >> 16) & 0xFF);
            w = (sbyte)((val >> 24) & 0xFF);
        }

        /// <summary>
        /// constructor from single component
        /// </summary>
        public char4(sbyte val)
        {
            x = val;
            y = val;
            z = val;
            w = val;
        }

        /// <summary>
        ///  copy constructor
        /// </summary>
        public char4(char4 res)
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
        public static char4 operator +(char4 a, char4 b)
        {
            char4 res;
            res.x = (sbyte)(a.x + b.x);
            res.y = (sbyte)(a.y + b.y);
            res.z = (sbyte)(a.z + b.z);
            res.w = (sbyte)(a.w + b.w);
            return res;
        }

        /// <summary>
        /// addition operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator+")]
        public static char4 operator +(sbyte a, char4 b)
        {
            char4 res;
            res.x = (sbyte)(a + b.x);
            res.y = (sbyte)(a + b.y);
            res.z = (sbyte)(a + b.z);
            res.w = (sbyte)(a + b.w);
            return res;
        }

        /// <summary>
        /// addition operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator+")]
        public static char4 operator +(char4 a, sbyte b)
        {
            char4 res;
            res.x = (sbyte)(a.x + b);
            res.y = (sbyte)(a.y + b);
            res.z = (sbyte)(a.z + b);
            res.w = (sbyte)(a.w + b);
            return res;
        }

        /// <summary>
        /// substraction operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator-")]
        public static char4 operator -(char4 a, char4 b)
        {
            char4 res;
            res.x = (sbyte)(a.x - b.x);
            res.y = (sbyte)(a.y - b.y);
            res.z = (sbyte)(a.z - b.z);
            res.w = (sbyte)(a.w - b.w);
            return res;
        }

        /// <summary>
        /// substraction operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator-")]
        public static char4 operator -(sbyte a, char4 b)
        {
            char4 res;
            res.x = (sbyte)(a - b.x);
            res.y = (sbyte)(a - b.y);
            res.z = (sbyte)(a - b.z);
            res.w = (sbyte)(a - b.w);
            return res;
        }

        /// <summary>
        /// substraction operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator-")]
        public static char4 operator -(char4 a, sbyte b)
        {
            char4 res;
            res.x = (sbyte)(a.x - b);
            res.y = (sbyte)(a.y - b);
            res.z = (sbyte)(a.z - b);
            res.w = (sbyte)(a.w - b);
            return res;
        }

        /// <summary>
        /// multiplication operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator*")]
        public static char4 operator *(char4 a, char4 b)
        {
            char4 res;
            res.x = (sbyte)(a.x * b.x);
            res.y = (sbyte)(a.y * b.y);
            res.z = (sbyte)(a.z * b.z);
            res.w = (sbyte)(a.w * b.w);
            return res;
        }

        /// <summary>
        /// multiplication operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator*")]
        public static char4 operator *(sbyte a, char4 b)
        {
            char4 res;
            res.x = (sbyte)(a * b.x);
            res.y = (sbyte)(a * b.y);
            res.z = (sbyte)(a * b.z);
            res.w = (sbyte)(a * b.w);
            return res;
        }

        /// <summary>
        /// multiplication operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator*")]
        public static char4 operator *(char4 a, sbyte b)
        {
            char4 res;
            res.x = (sbyte)(a.x * b);
            res.y = (sbyte)(a.y * b);
            res.z = (sbyte)(a.z * b);
            res.w = (sbyte)(a.w * b);
            return res;
        }

        /// <summary>
        /// division operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator/")]
        public static char4 operator /(char4 a, char4 b)
        {
            char4 res;
            res.x = (sbyte)(a.x / b.x);
            res.y = (sbyte)(a.y / b.y);
            res.z = (sbyte)(a.z / b.z);
            res.w = (sbyte)(a.w / b.w);
            return res;
        }

        /// <summary>
        /// division operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator/")]
        public static char4 operator /(sbyte a, char4 b)
        {
            char4 res;
            res.x = (sbyte)(a / b.x);
            res.y = (sbyte)(a / b.y);
            res.z = (sbyte)(a / b.z);
            res.w = (sbyte)(a / b.w);
            return res;
        }

        /// <summary>
        /// division operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator/")]
        public static char4 operator /(char4 a, sbyte b)
        {
            char4 res;
            res.x = (sbyte)(a.x / b);
            res.y = (sbyte)(a.y / b);
            res.z = (sbyte)(a.z / b);
            res.w = (sbyte)(a.w / b);
            return res;
        }

        /// <summary>
        /// stores in memory
        /// </summary>
        /// <param name="ptr">destination pointer</param>
        /// <param name="val">value to store</param>
        /// <param name="alignment">has to be a compile time constant</param>
        [IntrinsicFunction(IsNaked = true, Name = "__hybridizer_store_char4")]
        public unsafe static void Store(char4* ptr, char4 val, int alignment)
        {
            *ptr = val;
        }

        /// <summary>
        /// stores in memory
        /// </summary>
        /// <param name="ptr">destination pointer</param>
        /// <param name="val">value to store</param>
        /// <param name="alignment">has to be a compile time constant</param>
        [IntrinsicFunction(IsNaked = true, Name = "__hybridizer_store_splat_char4")]
        public unsafe static void Store(char4* ptr, sbyte val, int alignment)
        {
            *ptr = new char4(val);
        }

        /// <summary>
        /// loads from memory
        /// </summary>
        /// <param name="ptr"></param>
        /// <param name="alignment">has to be a compile time constant</param>
        [IntrinsicFunction(IsNaked = true, Name = "__hybridizer_load_char4")]
        public unsafe static char4 Load(char4* ptr, int alignment)
        {
            return *ptr;
        }
    }
}
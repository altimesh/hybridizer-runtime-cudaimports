using System.Runtime.InteropServices;

namespace Hybridizer.Runtime.CUDAImports
{
    /// <summary>
    /// 8 signed bytes
    /// </summary>
    [IntrinsicType("char8")]
    [StructLayout(LayoutKind.Explicit)]
    public struct char8
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
        /// x2
        /// </summary>
        [FieldOffset(4)]
        public sbyte x2;
        /// <summary>
        /// y2
        /// </summary>
        [FieldOffset(5)]
        public sbyte y2;
        /// <summary>
        /// z2
        /// </summary>
        [FieldOffset(6)]
        public sbyte z2;
        /// <summary>
        /// w2
        /// </summary>
        [FieldOffset(7)]
        public sbyte w2;

        /// <summary>
        /// constructor from components
        /// </summary>
        public char8(sbyte xx, sbyte yy, sbyte zz, sbyte ww, sbyte xx2, sbyte yy2, sbyte zz2, sbyte ww2)
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
        public char8(sbyte val)
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
        public char8(char8 res)
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
        public static char8 operator +(char8 a, char8 b)
        {
            char8 res;
            res.x = (sbyte)(a.x + b.x);
            res.y = (sbyte)(a.y + b.y);
            res.z = (sbyte)(a.z + b.z);
            res.w = (sbyte)(a.w + b.w);
            res.x2 = (sbyte)(a.x2 + b.x2);
            res.y2 = (sbyte)(a.y2 + b.y2);
            res.z2 = (sbyte)(a.z2 + b.z2);
            res.w2 = (sbyte)(a.w2 + b.w2);
            return res;
        }

        /// <summary>
        /// addition operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator+")]
        public static char8 operator +(sbyte a, char8 b)
        {
            char8 res;
            res.x = (sbyte)(a + b.x);
            res.y = (sbyte)(a + b.y);
            res.z = (sbyte)(a + b.z);
            res.w = (sbyte)(a + b.w);
            res.x2 = (sbyte)(a + b.x2);
            res.y2 = (sbyte)(a + b.y2);
            res.z2 = (sbyte)(a + b.z2);
            res.w2 = (sbyte)(a + b.w2);
            return res;
        }

        /// <summary>
        /// addition operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator+")]
        public static char8 operator +(char8 a, sbyte b)
        {
            char8 res;
            res.x = (sbyte)(a.x + b);
            res.y = (sbyte)(a.y + b);
            res.z = (sbyte)(a.z + b);
            res.w = (sbyte)(a.w + b);
            res.x2 = (sbyte)(a.x2 + b);
            res.y2 = (sbyte)(a.y2 + b);
            res.z2 = (sbyte)(a.z2 + b);
            res.w2 = (sbyte)(a.w2 + b);
            return res;
        }

        /// <summary>
        /// substraction operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator-")]
        public static char8 operator -(char8 a, char8 b)
        {
            char8 res;
            res.x = (sbyte)(a.x - b.x);
            res.y = (sbyte)(a.y - b.y);
            res.z = (sbyte)(a.z - b.z);
            res.w = (sbyte)(a.w - b.w);
            res.x2 = (sbyte)(a.x2 - b.x2);
            res.y2 = (sbyte)(a.y2 - b.y2);
            res.z2 = (sbyte)(a.z2 - b.z2);
            res.w2 = (sbyte)(a.w2 - b.w2);
            return res;
        }

        /// <summary>
        /// substraction operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator-")]
        public static char8 operator -(sbyte a, char8 b)
        {
            char8 res;
            res.x = (sbyte)(a - b.x);
            res.y = (sbyte)(a - b.y);
            res.z = (sbyte)(a - b.z);
            res.w = (sbyte)(a - b.w);
            res.x2 = (sbyte)(a - b.x2);
            res.y2 = (sbyte)(a - b.y2);
            res.z2 = (sbyte)(a - b.z2);
            res.w2 = (sbyte)(a - b.w2);
            return res;
        }

        /// <summary>
        /// substraction operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator-")]
        public static char8 operator -(char8 a, sbyte b)
        {
            char8 res;
            res.x = (sbyte)(a.x - b);
            res.y = (sbyte)(a.y - b);
            res.z = (sbyte)(a.z - b);
            res.w = (sbyte)(a.w - b);
            res.x2 = (sbyte)(a.x2 - b);
            res.y2 = (sbyte)(a.y2 - b);
            res.z2 = (sbyte)(a.z2 - b);
            res.w2 = (sbyte)(a.w2 - b);
            return res;
        }

        /// <summary>
        /// multiplication operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator*")]
        public static char8 operator *(char8 a, char8 b)
        {
            char8 res;
            res.x = (sbyte)(a.x * b.x);
            res.y = (sbyte)(a.y * b.y);
            res.z = (sbyte)(a.z * b.z);
            res.w = (sbyte)(a.w * b.w);
            res.x2 = (sbyte)(a.x2 * b.x2);
            res.y2 = (sbyte)(a.y2 * b.y2);
            res.z2 = (sbyte)(a.z2 * b.z2);
            res.w2 = (sbyte)(a.w2 * b.w2);
            return res;
        }

        /// <summary>
        /// multiplication operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator*")]
        public static char8 operator *(sbyte a, char8 b)
        {
            char8 res;
            res.x = (sbyte)(a * b.x);
            res.y = (sbyte)(a * b.y);
            res.z = (sbyte)(a * b.z);
            res.w = (sbyte)(a * b.w);
            res.x2 = (sbyte)(a * b.x2);
            res.y2 = (sbyte)(a * b.y2);
            res.z2 = (sbyte)(a * b.z2);
            res.w2 = (sbyte)(a * b.w2);
            return res;
        }

        /// <summary>
        /// multiplication operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator*")]
        public static char8 operator *(char8 a, sbyte b)
        {
            char8 res;
            res.x = (sbyte)(a.x * b);
            res.y = (sbyte)(a.y * b);
            res.z = (sbyte)(a.z * b);
            res.w = (sbyte)(a.w * b);
            res.x2 = (sbyte)(a.x2 * b);
            res.y2 = (sbyte)(a.y2 * b);
            res.z2 = (sbyte)(a.z2 * b);
            res.w2 = (sbyte)(a.w2 * b);
            return res;
        }

        /// <summary>
        /// division operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator/")]
        public static char8 operator /(char8 a, char8 b)
        {
            char8 res;
            res.x = (sbyte)(a.x / b.x);
            res.y = (sbyte)(a.y / b.y);
            res.z = (sbyte)(a.z / b.z);
            res.w = (sbyte)(a.w / b.w);
            res.x2 = (sbyte)(a.x2 / b.x2);
            res.y2 = (sbyte)(a.y2 / b.y2);
            res.z2 = (sbyte)(a.z2 / b.z2);
            res.w2 = (sbyte)(a.w2 / b.w2);
            return res;
        }

        /// <summary>
        /// division operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator/")]
        public static char8 operator /(sbyte a, char8 b)
        {
            char8 res;
            res.x = (sbyte)(a / b.x);
            res.y = (sbyte)(a / b.y);
            res.z = (sbyte)(a / b.z);
            res.w = (sbyte)(a / b.w);
            res.x2 = (sbyte)(a / b.x2);
            res.y2 = (sbyte)(a / b.y2);
            res.z2 = (sbyte)(a / b.z2);
            res.w2 = (sbyte)(a / b.w2);
            return res;
        }

        /// <summary>
        /// division operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator/")]
        public static char8 operator /(char8 a, sbyte b)
        {
            char8 res;
            res.x = (sbyte)(a.x / b);
            res.y = (sbyte)(a.y / b);
            res.z = (sbyte)(a.z / b);
            res.w = (sbyte)(a.w / b);
            res.x2 = (sbyte)(a.x2 / b);
            res.y2 = (sbyte)(a.y2 / b);
            res.z2 = (sbyte)(a.z2 / b);
            res.w2 = (sbyte)(a.w2 / b);
            return res;
        }

        /// <summary>
        /// stores in memory
        /// </summary>
        /// <param name="ptr">destination pointer</param>
        /// <param name="val">value to store</param>
        /// <param name="alignment">has to be a compile time constant</param>
        [IntrinsicFunction(IsNaked = true, Name = "__hybridizer_store_char8")]
        public unsafe static void Store(char8* ptr, char8 val, int alignment)
        {
            *ptr = val;
        }

        /// <summary>
        /// stores in memory
        /// </summary>
        /// <param name="ptr">destination pointer</param>
        /// <param name="val">value to store</param>
        /// <param name="alignment">has to be a compile time constant</param>
        [IntrinsicFunction(IsNaked = true, Name = "__hybridizer_store_splat_char8")]
        public unsafe static void Store(char8* ptr, sbyte val, int alignment)
        {
            *ptr = new char8(val);
        }

        /// <summary>
        /// loads from memory
        /// </summary>
        /// <param name="ptr"></param>
        /// <param name="alignment">has to be a compile time constant</param>
        [IntrinsicFunction(IsNaked = true, Name = "__hybridizer_load_char8")]
        public unsafe static char8 Load(char8* ptr, int alignment)
        {
            return *ptr;
        }
    }
}
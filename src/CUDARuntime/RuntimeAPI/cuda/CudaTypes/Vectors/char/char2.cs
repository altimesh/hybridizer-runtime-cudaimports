using System.Runtime.InteropServices;

namespace Hybridizer.Runtime.CUDAImports
{
    /// <summary>
    /// two signed bytes
    /// </summary>
    [IntrinsicType("char2")]
    [StructLayout(LayoutKind.Explicit)]
    public unsafe struct char2
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
        /// constructor from 32 bits integer
        /// </summary>
        public char2(int val)
        {
            x = (sbyte)(val & 0xFF);
            y = (sbyte)((val >> 8) & 0xFF);
        }
        /// <summary>
        /// constructor from components
        /// </summary>
        public char2(sbyte xx, sbyte yy)
        {
            x = xx;
            y = yy;
        }

        /// <summary>
        /// constructor from single component
        /// </summary>
        /// <param name="val"></param>
        public char2(sbyte val)
        {
            x = val;
            y = val;
        }

        /// <summary>
        /// copy constructor
        /// </summary>
        public char2(char2 res)
        {
            x = res.x;
            y = res.y;
        }

        /// <summary>
        /// conversion to short
        /// </summary>
        public static explicit operator short(char2 res)
        {
            short* ptr = (short*)&res;
            return *ptr;
        }

        /// <summary>
        /// addition operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator+")]
        public static char2 operator +(char2 a, char2 b)
        {
            char2 res;
            res.x = (sbyte)(a.x + b.x);
            res.y = (sbyte)(a.y + b.y);
            return res;
        }

        /// <summary>
        /// addition operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator+")]
        public static char2 operator +(sbyte a, char2 b)
        {
            char2 res;
            res.x = (sbyte)(a + b.x);
            res.y = (sbyte)(a + b.y);
            return res;
        }

        /// <summary>
        /// addition operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator+")]
        public static char2 operator +(char2 a, sbyte b)
        {
            char2 res;
            res.x = (sbyte)(a.x + b);
            res.y = (sbyte)(a.y + b);
            return res;
        }

        /// <summary>
        /// substraction operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator-")]
        public static char2 operator -(char2 a, char2 b)
        {
            char2 res;
            res.x = (sbyte)(a.x - b.x);
            res.y = (sbyte)(a.y - b.y);
            return res;
        }

        /// <summary>
        /// substraction operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator-")]
        public static char2 operator -(sbyte a, char2 b)
        {
            char2 res;
            res.x = (sbyte)(a - b.x);
            res.y = (sbyte)(a - b.y);
            return res;
        }

        /// <summary>
        /// substraction operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator-")]
        public static char2 operator -(char2 a, sbyte b)
        {
            char2 res;
            res.x = (sbyte)(a.x - b);
            res.y = (sbyte)(a.y - b);
            return res;
        }

        /// <summary>
        /// multiplication operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator*")]
        public static char2 operator *(char2 a, char2 b)
        {
            char2 res;
            res.x = (sbyte)(a.x * b.x);
            res.y = (sbyte)(a.y * b.y);
            return res;
        }

        /// <summary>
        /// multiplication operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator*")]
        public static char2 operator *(sbyte a, char2 b)
        {
            char2 res;
            res.x = (sbyte)(a * b.x);
            res.y = (sbyte)(a * b.y);
            return res;
        }

        /// <summary>
        /// multiplication operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator*")]
        public static char2 operator *(char2 a, sbyte b)
        {
            char2 res;
            res.x = (sbyte)(a.x * b);
            res.y = (sbyte)(a.y * b);
            return res;
        }

        /// <summary>
        /// division operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator/")]
        public static char2 operator /(char2 a, char2 b)
        {
            char2 res;
            res.x = (sbyte)(a.x / b.x);
            res.y = (sbyte)(a.y / b.y);
            return res;
        }

        /// <summary>
        /// division operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator/")]
        public static char2 operator /(sbyte a, char2 b)
        {
            char2 res;
            res.x = (sbyte)(a / b.x);
            res.y = (sbyte)(a / b.y);
            return res;
        }

        /// <summary>
        /// division operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator/")]
        public static char2 operator /(char2 a, sbyte b)
        {
            char2 res;
            res.x = (sbyte)(a.x / b);
            res.y = (sbyte)(a.y / b);
            return res;
        }
    }
}
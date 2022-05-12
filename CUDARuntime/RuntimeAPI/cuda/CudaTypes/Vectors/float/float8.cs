using System.Runtime.InteropServices;

namespace Hybridizer.Runtime.CUDAImports
{
    /// <summary>
    /// 8 32 bits floats
    /// </summary>
    [IntrinsicType("float8")]
    [StructLayout(LayoutKind.Explicit)]
    public struct float8
    {
        /// <summary>
        /// x
        /// </summary>
        [FieldOffset(0)]
        public float x;
        /// <summary>
        /// y
        /// </summary>
        [FieldOffset(4)]
        public float y;
        /// <summary>
        /// z
        /// </summary>
        [FieldOffset(8)]
        public float z;
        /// <summary>
        /// w
        /// </summary>
        [FieldOffset(12)]
        public float w;
        /// <summary>
        /// x2
        /// </summary>
        [FieldOffset(16)]
        public float x2;
        /// <summary>
        /// y2
        /// </summary>
        [FieldOffset(20)]
        public float y2;
        /// <summary>
        /// z2
        /// </summary>
        [FieldOffset(24)]
        public float z2;
        /// <summary>
        /// w2
        /// </summary>
        [FieldOffset(28)]
        public float w2;

        /// <summary>
        /// copy constructor
        /// </summary>
        public float8(float8 res)
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
        /// constructor from components
        /// </summary>
        public float8(float xx, float yy, float zz, float ww, float xx2, float yy2, float zz2, float ww2)
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
        /// selects components from l or r, depending on mask value
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "hybridizer::select<float8>")]
        public static float8 Select(bool8 mask, float8 l, float8 r)
        {
            float8 res;
            res.x = mask.x ? l.x : r.x;
            res.y = mask.y ? l.y : r.y;
            res.z = mask.z ? l.z : r.z;
            res.w = mask.w ? l.w : r.w;
            res.x2 = mask.x2 ? l.x2 : r.x2;
            res.y2 = mask.y2 ? l.y2 : r.y2;
            res.z2 = mask.z2 ? l.z2 : r.z2;
            res.w2 = mask.w2 ? l.w2 : r.w2;
            return res;
        }

        /// <summary>
        /// loads from memory
        /// </summary>
        /// <param name="ptr"></param>
        /// <param name="alignment">has to be a compile time constant</param>
        [IntrinsicFunction(IsNaked = true, Name = "__hybridizer_load_float8")]
        public unsafe static float8 Load(float8* ptr, int alignment)
        {
            return *ptr;
        }

        /// <summary>
        /// stores in memory
        /// </summary>
        /// <param name="ptr">destination pointer</param>
        /// <param name="val">value to store</param>
        /// <param name="alignment">has to be a compile time constant</param>
        [IntrinsicFunction(IsNaked = true, Name = "__hybridizer_store_float8")]
        public unsafe static void Store(float8* ptr, float8 val, int alignment)
        {
            *ptr = val;
        }

        /// <summary>
        /// addition operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator+")]
        public static float8 operator +(float8 a, float8 b)
        {
            float8 res;
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
        /// addition operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator+")]
        public static float8 operator +(float a, float8 b)
        {
            float8 res;
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
        /// strict less operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator<")]
        public static bool8 operator <(float8 l, float8 r)
        {
            bool8 res;
            res.x = l.x < r.x;
            res.y = l.y < r.y;
            res.z = l.z < r.z;
            res.w = l.w < r.w;
            res.x2 = l.x2 < r.x2;
            res.y2 = l.y2 < r.y2;
            res.z2 = l.z2 < r.z2;
            res.w2 = l.w2 < r.w2;
            return res;
        }

        /// <summary>
        /// less or equal operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator<=")]
        public static bool8 operator <=(float8 l, float8 r)
        {
            bool8 res;
            res.x = l.x <= r.x;
            res.y = l.y <= r.y;
            res.z = l.z <= r.z;
            res.w = l.w <= r.w;
            res.x2 = l.x2 <= r.x2;
            res.y2 = l.y2 <= r.y2;
            res.z2 = l.z2 <= r.z2;
            res.w2 = l.w2 <= r.w2;
            return res;
        }

        /// <summary>
        /// greater or equal operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator>=")]
        public static bool8 operator >=(float8 l, float8 r)
        {
            bool8 res;
            res.x = l.x >= r.x;
            res.y = l.y >= r.y;
            res.z = l.z >= r.z;
            res.w = l.w >= r.w;
            res.x2 = l.x2 >= r.x2;
            res.y2 = l.y2 >= r.y2;
            res.z2 = l.z2 >= r.z2;
            res.w2 = l.w2 >= r.w2;
            return res;
        }

        /// <summary>
        /// strict greater operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator>")]
        public static bool8 operator >(float8 l, float8 r)
        {
            bool8 res;
            res.x = l.x > r.x;
            res.y = l.y > r.y;
            res.z = l.z > r.z;
            res.w = l.w > r.w;
            res.x2 = l.x2 > r.x2;
            res.y2 = l.y2 > r.y2;
            res.z2 = l.z2 > r.z2;
            res.w2 = l.w2 > r.w2;
            return res;
        }

        /// <summary>
        /// strict greater operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator>")]
        public static bool8 operator >(float8 l, float r)
        {
            bool8 res;
            res.x = l.x > r;
            res.y = l.y > r;
            res.z = l.z > r;
            res.w = l.w > r;
            res.x2 = l.x2 > r;
            res.y2 = l.y2 > r;
            res.z2 = l.z2 > r;
            res.w2 = l.w2 > r;
            return res;
        }

        /// <summary>
        /// strict less operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator>")]
        public static bool8 operator <(float8 l, float r)
        {
            bool8 res;
            res.x = l.x < r;
            res.y = l.y < r;
            res.z = l.z < r;
            res.w = l.w < r;
            res.x2 = l.x2 < r;
            res.y2 = l.y2 < r;
            res.z2 = l.z2 < r;
            res.w2 = l.w2 < r;
            return res;
        }

        /// <summary>
        /// addition operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator+")]
        public static float8 operator +(float8 a, float b)
        {
            float8 res;
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
        public static float8 operator -(float8 a, float8 b)
        {
            float8 res;
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
        public static float8 operator -(float a, float8 b)
        {
            float8 res;
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
        public static float8 operator -(float8 a, float b)
        {
            float8 res;
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
        public static float8 operator *(float8 a, float8 b)
        {
            float8 res;
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
        public static float8 operator *(float a, float8 b)
        {
            float8 res;
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
        public static float8 operator *(float8 a, float b)
        {
            float8 res;
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
        public static float8 operator /(float8 a, float8 b)
        {
            float8 res;
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
        public static float8 operator /(float a, float8 b)
        {
            float8 res;
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
        public static float8 operator /(float8 a, float b)
        {
            float8 res;
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
    }
}
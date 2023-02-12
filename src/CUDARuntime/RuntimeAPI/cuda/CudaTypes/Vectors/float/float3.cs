using System.Runtime.InteropServices;

namespace Hybridizer.Runtime.CUDAImports
{
    /// <summary>
    /// 3 32 bits floating points elements, packed
    /// </summary>
    [IntrinsicType("float3")]
    [StructLayout(LayoutKind.Explicit)]
    public unsafe struct float3
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
        /// conversion to signed 64 bits integer
        /// </summary>
        public static explicit operator long(float3 res)
        {
            long* tmp = (long*)(&res);
            return *tmp;
        }

        /// <summary>
        /// constructor from 3 float
        /// </summary>
        /// <param name="xx"></param>
        /// <param name="yy"></param>
        /// <param name="zz"></param>
        private float3(float xx, float yy, float zz)
        {
            x = xx;
            y = yy;
            z = zz;
        }

        /// <summary>
        /// make a float3 from 3 floats
        /// </summary>
        /// <param name="xx"></param>
        /// <param name="yy"></param>
        /// <param name="zz"></param>
        /// <returns></returns>
        [IntrinsicFunction("make_float3")]
        public static float3 make_float3(float xx, float yy, float zz)
        {
            return new float3(xx, yy, zz);
        }

        /// <summary>
        /// addition operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator+")]
        public static float3 operator +(float3 a, float3 b)
        {
            float3 res;
            res.x = a.x + b.x;
            res.y = a.y + b.y;
            res.z = a.z + b.z;
            return res;
        }

        /// <summary>
        /// addition operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator+")]
        public static float3 operator +(float a, float3 b)
        {
            float3 res;
            res.x = a + b.x;
            res.y = a + b.y;
            res.z = a + b.z;
            return res;
        }

        /// <summary>
        /// addition operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator+")]
        public static float3 operator +(float3 a, float b)
        {
            float3 res;
            res.x = a.x + b;
            res.y = a.y + b;
            res.z = a.z + b;
            return res;
        }

        /// <summary>
        /// substraction operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator-")]
        public static float3 operator -(float3 a, float3 b)
        {
            float3 res;
            res.x = a.x - b.x;
            res.y = a.y - b.y;
            res.z = a.z - b.z;
            return res;
        }

        /// <summary>
        /// substraction operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator-")]
        public static float3 operator -(float a, float3 b)
        {
            float3 res;
            res.x = a - b.x;
            res.y = a - b.y;
            res.z = a - b.z;
            return res;
        }

        /// <summary>
        /// substraction operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator-")]
        public static float3 operator -(float3 a, float b)
        {
            float3 res;
            res.x = a.x - b;
            res.y = a.y - b;
            res.z = a.z - b;
            return res;
        }

        /// <summary>
        /// multiplication operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator*")]
        public static float3 operator *(float3 a, float3 b)
        {
            float3 res;
            res.x = a.x * b.x;
            res.y = a.y * b.y;
            res.z = a.z * b.y;
            return res;
        }

        /// <summary>
        /// multiplication operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator*")]
        public static float3 operator *(float a, float3 b)
        {
            float3 res;
            res.x = a * b.x;
            res.y = a * b.y;
            res.z = a * b.z;
            return res;
        }

        /// <summary>
        /// multiplication operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator*")]
        public static float3 operator *(float3 a, float b)
        {
            float3 res;
            res.x = a.x * b;
            res.y = a.y * b;
            res.z = a.z * b;
            return res;
        }

        /// <summary>
        /// division operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator/")]
        public static float3 operator /(float3 a, float3 b)
        {
            float3 res;
            res.x = a.x / b.x;
            res.y = a.y / b.y;
            res.z = a.z / b.y;
            return res;
        }

        /// <summary>
        /// division operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator/")]
        public static float3 operator /(float a, float3 b)
        {
            float3 res;
            res.x = a / b.x;
            res.y = a / b.y;
            res.z = a / b.z;
            return res;
        }

        /// <summary>
        /// division operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator/")]
        public static float3 operator /(float3 a, float b)
        {
            float3 res;
            res.x = a.x / b;
            res.y = a.y / b;
            res.z = a.z / b;
            return res;
        }

        /// <summary>
        /// stores in memory
        /// </summary>
        /// <param name="ptr">destination pointer</param>
        /// <param name="val">value to store</param>
        /// <param name="alignment">has to be a compile time constant</param>
        [IntrinsicFunction(IsNaked = true, Name = "__hybridizer_store_float3")]
        public unsafe static void Store(float3* ptr, float3 val, int alignment)
        {
            *ptr = val;
        }

        /// <summary>
        /// loads from memory
        /// </summary>
        /// <param name="ptr"></param>
        /// <param name="alignment">has to be a compile time constant</param>
        [IntrinsicFunction(IsNaked = true, Name = "__hybridizer_load_float3")]
        public unsafe static float3 Load(float3* ptr, int alignment)
        {
            return *ptr;
        }
    }
}
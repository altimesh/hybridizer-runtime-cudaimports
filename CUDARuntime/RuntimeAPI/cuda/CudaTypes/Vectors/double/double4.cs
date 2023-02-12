using System.Runtime.InteropServices;

namespace Hybridizer.Runtime.CUDAImports
{
    /// <summary>
    /// 4 64 bits floating points elements
    /// </summary>
    [IntrinsicType("double4")]
    [StructLayout(LayoutKind.Explicit)]
    public struct double4
    {
        /// <summary>
        /// x
        /// </summary>
        [FieldOffset(0)]
        public double x;
        /// <summary>
        /// y
        /// </summary>
        [FieldOffset(8)]
        public double y;
        /// <summary>
        /// z
        /// </summary>
        [FieldOffset(16)]
        public double z;
        /// <summary>
        /// w
        /// </summary>
        [FieldOffset(24)]
        public double w;

        private double4(double xx, double yy, double zz, double ww)
        {
            x = xx;
            y = yy;
            z = zz;
            w = ww;
        }

        /// <summary>
        /// make a double4 from 4 doubles
        /// </summary>
        /// <param name="xx"></param>
        /// <param name="yy"></param>
        /// <param name="zz"></param>
        /// <param name="ww"></param>
        /// <returns></returns>
        [IntrinsicFunction("make_double4")]
        public static double4 make_double4(double xx, double yy, double zz, double ww)
        {
            return new double4(xx, yy, zz, ww);
        }

        /// <summary>
        /// addition operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator+")]
        public static double4 operator +(double4 a, double4 b)
        {
            double4 res;
            res.x = a.x + b.x;
            res.y = a.y + b.y;
            res.z = a.z + b.z;
            res.w = a.w + b.w;
            return res;
        }

        /// <summary>
        /// addition operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator+")]
        public static double4 operator +(double a, double4 b)
        {
            double4 res;
            res.x = a + b.x;
            res.y = a + b.y;
            res.z = a + b.z;
            res.w = a + b.w;
            return res;
        }

        /// <summary>
        /// addition operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator+")]
        public static double4 operator +(double4 a, double b)
        {
            double4 res;
            res.x = a.x + b;
            res.y = a.y + b;
            res.z = a.z + b;
            res.w = a.w + b;
            return res;
        }



        /// <summary>
        /// substraction operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator-")]
        public static double4 operator -(double4 a, double4 b)
        {
            double4 res;
            res.x = a.x - b.x;
            res.y = a.y - b.y;
            res.z = a.z - b.z;
            res.w = a.w - b.w;
            return res;
        }

        /// <summary>
        /// substraction operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator-")]
        public static double4 operator -(double a, double4 b)
        {
            double4 res;
            res.x = a - b.x;
            res.y = a - b.y;
            res.z = a - b.z;
            res.w = a - b.w;
            return res;
        }

        /// <summary>
        /// substraction operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator-")]
        public static double4 operator -(double4 a, double b)
        {
            double4 res;
            res.x = a.x - b;
            res.y = a.y - b;
            res.z = a.z - b;
            res.w = a.w - b;
            return res;
        }

        /// <summary>
        /// multiplication operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator*")]
        public static double4 operator *(double4 a, double4 b)
        {
            double4 res;
            res.x = a.x * b.x;
            res.y = a.y * b.y;
            res.z = a.z * b.z;
            res.w = a.w * b.w;
            return res;
        }

        /// <summary>
        /// multiplication operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator*")]
        public static double4 operator *(double a, double4 b)
        {
            double4 res;
            res.x = a * b.x;
            res.y = a * b.y;
            res.z = a * b.z;
            res.w = a * b.w;
            return res;
        }

        /// <summary>
        /// multiplication operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator*")]
        public static double4 operator *(double4 a, double b)
        {
            double4 res;
            res.x = a.x * b;
            res.y = a.y * b;
            res.z = a.z * b;
            res.w = a.w * b;
            return res;
        }

        /// <summary>
        /// division operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator/")]
        public static double4 operator /(double4 a, double4 b)
        {
            double4 res;
            res.x = a.x / b.x;
            res.y = a.y / b.y;
            res.z = a.z / b.z;
            res.w = a.w / b.w;
            return res;
        }

        /// <summary>
        /// division operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator/")]
        public static double4 operator /(double a, double4 b)
        {
            double4 res;
            res.x = a / b.x;
            res.y = a / b.y;
            res.z = a / b.z;
            res.w = a / b.w;
            return res;
        }

        /// <summary>
        /// division operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator/")]
        public static double4 operator /(double4 a, double b)
        {
            double4 res;
            res.x = a.x / b;
            res.y = a.y / b;
            res.z = a.z / b;
            res.w = a.w / b;
            return res;
        }
    }
}
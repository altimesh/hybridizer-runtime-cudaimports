using System.Runtime.InteropServices;

namespace Hybridizer.Runtime.CUDAImports
{
    /// <summary>
    /// 2 64 bits floating point elements, packed
    /// </summary>
    [IntrinsicType("double2")]
    [StructLayout(LayoutKind.Explicit)]
    public struct double2
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
        /// constructor from 2 64 bits float
        /// </summary>
        /// <param name="a"></param>
        /// <param name="b"></param>
        private double2(double a, double b)
        {
            x = a;
            y = b;
        }

        /// <summary>
        /// make a double2 from 2 doubles
        /// </summary>
        /// <param name="xx"></param>
        /// <param name="yy"></param>
        /// <returns></returns>
        [IntrinsicFunction("make_double2")]
        public static double2 make_double2(double xx, double yy)
        {
            return new double2(xx, yy);
        }

        /// <summary>
        /// copy constructor
        /// </summary>
        public double2(double2 a)
        {
            x = a.x;
            y = a.y;
        }

        /// <summary>
        /// addition operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator+")]
        public static double2 operator +(double2 a, double2 b)
        {
            double2 res;
            res.x = a.x + b.x;
            res.y = a.y + b.y;
            return res;
        }

        /// <summary>
        /// addition operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator+")]
        public static double2 operator +(double a, double2 b)
        {
            double2 res;
            res.x = a + b.x;
            res.y = a + b.y;
            return res;
        }

        /// <summary>
        /// addition operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator+")]
        public static double2 operator +(double2 a, double b)
        {
            double2 res;
            res.x = a.x + b;
            res.y = a.y + b;
            return res;
        }

        /// <summary>
        /// substraction operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator-")]
        public static double2 operator -(double2 a, double2 b)
        {
            double2 res;
            res.x = a.x - b.x;
            res.y = a.y - b.y;
            return res;
        }

        /// <summary>
        /// substraction operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator-")]
        public static double2 operator -(double a, double2 b)
        {
            double2 res;
            res.x = a - b.x;
            res.y = a - b.y;
            return res;
        }

        /// <summary>
        /// substraction operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator-")]
        public static double2 operator -(double2 a, double b)
        {
            double2 res;
            res.x = a.x - b;
            res.y = a.y - b;
            return res;
        }

        /// <summary>
        /// multiplication operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator*")]
        public static double2 operator *(double2 a, double2 b)
        {
            double2 res;
            res.x = a.x * b.x;
            res.y = a.y * b.y;
            return res;
        }

        /// <summary>
        /// multiplication operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator*")]
        public static double2 operator *(double a, double2 b)
        {
            double2 res;
            res.x = a * b.x;
            res.y = a * b.y;
            return res;
        }

        /// <summary>
        /// multiplication operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator*")]
        public static double2 operator *(double2 a, double b)
        {
            double2 res;
            res.x = a.x * b;
            res.y = a.y * b;
            return res;
        }

        /// <summary>
        /// division operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator/")]
        public static double2 operator /(double2 a, double2 b)
        {
            double2 res;
            res.x = a.x / b.x;
            res.y = a.y / b.y;
            return res;
        }

        /// <summary>
        /// division operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator/")]
        public static double2 operator /(double a, double2 b)
        {
            double2 res;
            res.x = a / b.x;
            res.y = a / b.y;
            return res;
        }

        /// <summary>
        /// division operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator/")]
        public static double2 operator /(double2 a, double b)
        {
            double2 res;
            res.x = a.x / b;
            res.y = a.y / b;
            return res;
        }

        /// <summary>
        /// greater or equal operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator>=")]
        public static bool2 operator >=(double2 l, double2 r)
        {
            bool2 res;
            res.x = l.x >= r.x;
            res.y = l.y >= r.y;
            return res;
        }

        /// <summary>
        /// strict greater operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator>")]
        public static bool2 operator >(double2 l, double2 r)
        {
            bool2 res;
            res.x = l.x > r.x;
            res.y = l.y > r.y;
            return res;
        }

        /// <summary>
        /// less or equal operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator<=")]
        public static bool2 operator <=(double2 l, double2 r)
        {
            bool2 res;
            res.x = l.x <= r.x;
            res.y = l.y <= r.y;
            return res;
        }

        /// <summary>
        /// strict less operator
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "operator>=")]
        public static bool2 operator <(double2 l, double2 r)
        {
            bool2 res;
            res.x = l.x < r.x;
            res.y = l.y < r.y;
            return res;
        }

        /// <summary>
        /// selects components from l or r, depending on mask value
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "hybridizer::select<double2>")]
        public static double2 Select(bool2 mask, double2 l, double2 r)
        {
            double2 res;
            res.x = mask.x ? l.x : r.x;
            res.y = mask.y ? l.y : r.y;
            return res;
        }

        /// <summary>
        /// stores in memory
        /// </summary>
        /// <param name="ptr">destination pointer</param>
        /// <param name="val">value to store</param>
        /// <param name="alignment">has to be a compile time constant</param>
        [IntrinsicFunction(IsNaked = true, Name = "__hybridizer_store_double2")]
        public unsafe static void Store(double2* ptr, double2 val, int alignment)
        {
            *ptr = val;
        }

        /// <summary>
        /// loads from memory
        /// </summary>
        /// <param name="ptr"></param>
        /// <param name="alignment">has to be a compile time constant</param>
        [IntrinsicFunction(IsNaked = true, Name = "__hybridizer_load_double2")]
        public unsafe static double2 Load(double2* ptr, int alignment)
        {
            return *ptr;
        }
    }
}
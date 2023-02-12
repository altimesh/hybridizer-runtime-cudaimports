/* (c) ALTIMESH 2018 -- all rights reserved */
namespace Hybridizer.Runtime.CUDAImports
{
    /// <summary>
    /// helper functions
    /// </summary>
    public class JavaRuntime
    {
        /// <summary>
        /// compares two double precision real number
        /// </summary>
        /// <returns>-1 if d2 > d1, 1 if d1 > d2, 0 otherwise</returns>
        [ReturnTypeInference(Return=VectorizerIntrinsicReturn.VectorTransitive, Index = 0)]
        public static int Dcmpl(double d1, double d2)
        {
            if (d2 > d1) return -1;
            if (d1 > d2) return 1;
            return 0;
        }

        /// <summary>
        /// compares two float 32 precision real number
        /// </summary>
        /// <returns>-1 if d2 > d1, 1 if d1 > d2, 0 otherwise</returns>
        [ReturnTypeInference(Return = VectorizerIntrinsicReturn.VectorTransitive, Index = 0)]
        public static int Fcmpl(float d1, float d2)
        {
            if (d2 > d1) return -1;
            if (d1 > d2) return 1;
            return 0;
        }
    }
}

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
#pragma warning disable 1591
namespace Hybridizer.Runtime.CUDAImports
{
    public static class VectorUnit
    {
        public static alignedindex ID { [IntrinsicConstant("__hybridizer_threadIdxX")] get { return alignedindex.VectorUnitID; } }
        public static int Count { [IntrinsicConstant("__hybridizer_blockDimX")] get { return 1; } }
        [IntrinsicFunction("__syncthreads")]
        public static void Barrier() { }
        public static int WarpSize { [IntrinsicConstant("warpSize")] get { return 1; } }
        [IntrinsicFunction(Name = "hybridizer::reduce_add", Flavor = (int) HybridizerFlavor.AVX), ReturnTypeInference(VectorizerIntrinsicReturn.Unchanged), HybridNakedFunction]
        public static double ReduceAdd(double input) { return input ; }
    }
}
#pragma warning disable 1591

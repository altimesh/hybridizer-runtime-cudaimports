using System;
using Hybridizer.Runtime.CUDAImports.curand_device;

namespace Hybridizer.Runtime.CUDAImports.Tests
{
    /// <summary>
    /// Description résumée pour UnitTest1
    /// </summary>
    public class UnitTest_CurandInterface
    {
        public interface ICurandGenerator
        {
            float curand_normal();
        }

        public struct Generator_curandStateMRG32k3a_t : ICurandGenerator
        {
            curandStateMRG32k3a_t gen;

            public float curand_normal()
            {
                return gen.curand_normal();
            }
        }

        public class GenericCurandGenerator
        {
            ICurandGenerator generator;

            public GenericCurandGenerator(int nbGen, Type t)
            {
                if (t == typeof(curandStateMRG32k3a_t))
                {
                    generator = new Generator_curandStateMRG32k3a_t();
                }
                else
                    throw new NotImplementedException();
            }

            [Kernel()]
            public float GenerateNormals()
            {
                return generator.curand_normal();
            }
        }


    }
}

/* (c) ALTIMESH 2018 -- all rights reserved */
using System;
using System.Runtime.InteropServices;

using size_t = System.UIntPtr ;

namespace Hybridizer.Runtime.CUDAImports
{
#pragma warning disable CS1591 // Missing XML comment for publicly visible type or member

	public interface ICurand
	{
		curandStatus_t curandCreateGenerator(out curandGenerator_t generator, curandRngType_t type);
		curandStatus_t curandGenerate(curandGenerator_t generator, IntPtr outputPtr, size_t num);
		curandStatus_t curandGenerateUniform(curandGenerator_t generator, IntPtr outputPtr, size_t num);
		curandStatus_t curandGenerateNormal(curandGenerator_t generator, IntPtr outputPtr, size_t n, float mean, float stddev);
		curandStatus_t curandGenerateLogNormal(curandGenerator_t generator, IntPtr outputPtr, size_t n, float mean, float stddev);
		curandStatus_t curandGeneratePoisson(curandGenerator_t generator, IntPtr outputPtr, size_t n, double lambda);
		curandStatus_t curandGenerateUniformDouble(curandGenerator_t generator, IntPtr outputPtr, size_t num);
		curandStatus_t curandGenerateNormalDouble(curandGenerator_t generator, IntPtr outputPtr, size_t n, double mean, double stddev);
		curandStatus_t curandGenerateLogNormalDouble(curandGenerator_t generator, IntPtr outputPtr, size_t n, double mean, double stddev);
		curandStatus_t curandSetPseudoRandomGeneratorSeed(curandGenerator_t prngGPU, UInt64 seed);
		curandStatus_t curandSetGeneratorOrdering(curandGenerator_t prngGPU, curandOrdering_t order);
		curandStatus_t curandSetGeneratorOffset(curandGenerator_t generator, UInt64 offset);
	}
#pragma warning restore CS1591 // Missing XML comment for publicly visible type or member

}

/* (c) ALTIMESH 2018 -- all rights reserved */
using System;
using System.Runtime.InteropServices;

using size_t = System.UIntPtr ;

namespace Hybridizer.Runtime.CUDAImports
{
	/// <summary>
	/// types of curand rng algo
	/// </summary>
	[IntrinsicType("curandRngType_t")]
	public enum curandRngType_t : int
	{
		/// <summary>
		/// test -- unused
		/// </summary>
		CURAND_RNG_TEST = 0,
		/// <summary>
		/// Default pseudorandom generator
		/// </summary>
		CURAND_RNG_PSEUDO_DEFAULT = 100, 
		/// <summary>
		/// XORWOW pseudorandom generator
		/// </summary>
		CURAND_RNG_PSEUDO_XORWOW = 101, 
		/// <summary>
		/// MRG32k3a pseudorandom generator
		/// </summary>
		CURAND_RNG_PSEUDO_MRG32K3A = 121, 
		/// <summary>
		/// Mersenne Twister pseudorandom generator
		/// </summary>
		CURAND_RNG_PSEUDO_MTGP32 = 141, 
		/// <summary>
		/// Default pseudorandom generator
		/// </summary>
		CURAND_RNG_PSEUDO_PHILOX4_32_10 = 161, 
		/// <summary>
		/// Default quasirandom generator
		/// </summary>
		CURAND_RNG_QUASI_DEFAULT = 200, 
		/// <summary>
		/// Sobol32 quasirandom generator
		/// </summary>
		CURAND_RNG_QUASI_SOBOL32 = 201, 
		/// <summary>
		/// Scrambled Sobol32 quasirandom generator
		/// </summary>
		CURAND_RNG_QUASI_SCRAMBLED_SOBOL32 = 202,  
		/// <summary>
		/// Sobol64 quasirandom generator
		/// </summary>
		CURAND_RNG_QUASI_SOBOL64 = 203, 
		/// <summary>
		/// Scrambled Sobol64 quasirandom generator
		/// </summary>
		CURAND_RNG_QUASI_SCRAMBLED_SOBOL64 = 204  
	}
}

/* (c) ALTIMESH 2018 -- all rights reserved */
using System;
using System.Runtime.InteropServices;

using size_t = System.UIntPtr ;

namespace Hybridizer.Runtime.CUDAImports
{
	[IntrinsicType("curandOrdering_t")]
	public enum curandOrdering_t
	{
		/// <summary>
		/// Best ordering for pseudorandom results
		/// </summary>
		CURAND_ORDERING_PSEUDO_BEST = 100, 
		/// <summary>
		/// Specific default 4096 thread sequence for pseudorandom results
		/// </summary>
		CURAND_ORDERING_PSEUDO_DEFAULT = 101, 
		/// <summary>
		/// Specific seeding pattern for fast lower quality pseudorandom results
		/// </summary>
		CURAND_ORDERING_PSEUDO_SEEDED = 102, 
		/// <summary>
		/// Specific n-dimensional ordering for quasirandom results
		/// </summary>
		CURAND_ORDERING_QUASI_DEFAULT = 201 
	}
}

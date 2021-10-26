/* (c) ALTIMESH 2018 -- all rights reserved */
using System;
using size_t = System.UIntPtr ;

namespace Hybridizer.Runtime.CUDAImports
{
	// TODO : check that
	[IntrinsicType("curandGenerator_t")]
	public struct curandGenerator_t
	{
		#pragma warning disable 0169
		public IntPtr _inner ;
		#pragma warning restore 0169
	}
}

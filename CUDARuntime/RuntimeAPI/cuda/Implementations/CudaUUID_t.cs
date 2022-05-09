using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Runtime.InteropServices;

namespace Hybridizer.Runtime.CUDAImports
{
	/// <summary>
	/// CUDA runtime API wrapper
	/// </summary>
	public unsafe partial class cuda
	{
		[StructLayout(LayoutKind.Sequential, Pack = 4, Size = 16)]
		internal struct cudaUUID_t
		{
			[MarshalAs(UnmanagedType.ByValArray, SizeConst = 16)]
			public char[] bytes;
		}
	}
}

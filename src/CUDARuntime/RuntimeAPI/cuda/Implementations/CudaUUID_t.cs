using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Runtime.InteropServices;

namespace Hybridizer.Runtime.CUDAImports
{
	/// <summary>
	/// struct wrapping a cuda uuid_type
	/// </summary>
	[StructLayout(LayoutKind.Sequential, Pack = 4, Size = 16)]
	public struct cudaUUID_t
	{
		/// <summary>
		///  bytes of uuid
		/// </summary>
		[MarshalAs(UnmanagedType.ByValArray, SizeConst = 16)]
		public char[] bytes;
	}
}

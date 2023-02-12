using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Hybridizer.Runtime.CUDAImports
{
    /// <summary>
    /// Simple random class (host)
    /// </summary>
    public class JavaRandom
    {
		ulong seed;

        /// <summary>
        /// constructor
        /// </summary>
        /// <param name="s">seed</param>
		public JavaRandom(ulong s)
		{
			seed = (s ^ 0x5DEECE66DUL) & ((1UL << 48) - 1UL);
		}

        /// <summary>
        /// generates a random float32
        /// </summary>
        /// <returns></returns>
        public float nextFloat()
        {
            return Next(24) / ((float)(1 << 24));
        }

        /// <summary>
        /// generates a random unsigned int 32
        /// </summary>
        public uint Next(int bits)
        {
            seed = (seed * 0x5DEECE66DL + 0xBL) & ((1L << 48) - 1);
		    uint res =  (uint) (seed >> (48 - bits));
		    return res;
        }
    }
}

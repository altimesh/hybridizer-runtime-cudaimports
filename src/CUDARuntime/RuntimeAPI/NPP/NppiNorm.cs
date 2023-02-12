using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;

namespace Hybridizer.Runtime.CUDAImports
{
    /// <summary>
    /// type of norms -- enumerator
    /// </summary>
    [IntrinsicInclude("npp.h")]
    [IntrinsicType("NppiNorm")]
    public enum NppiNorm
    {
        /// <summary>
        /// maximum
        /// </summary>
        nppiNormInf = 0,
        /// <summary>
        /// sum
        /// </summary>
        nppiNormL1 = 1,
        /// <summary>
        /// square root of sum of squares
        /// </summary>
        nppiNormL2 = 2
    }
}

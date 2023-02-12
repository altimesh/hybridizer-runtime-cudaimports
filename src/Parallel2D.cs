using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Hybridizer.Runtime.CUDAImports
{

    /// <summary>
    /// Similar to Parallel class in <see cref="System.Threading.Tasks.Parallel"/>
    /// </summary>
    public class Parallel2D
    {
        /// <summary>
        /// Work is distributed on J -- on I we assume a pragma simd
        /// Assume that memory access is j*N + i (coalesced in i)
        /// Still need to be improved -- both in C# and in AVX
        /// </summary>
        public static void For(int fromJInclusive, int toJExclusive, int fromIInclusive, int toIExclusive, Action<int, int> action)
        {
            Parallel.For(fromJInclusive, toJExclusive, j =>
            {
                for (int i = fromIInclusive; i < toIExclusive; ++i)
                {
                    action(i, j);
                }
            });
        }
    }
}

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;

namespace Hybridizer.Runtime.CUDAImports
{
	public struct AtomicExpr
	{
		unsafe class Reinterpret
		{
			public static ulong Cast(double input)
			{
				return *((ulong*)(&input));
			}

			public static uint Cast(float input)
			{
				return *((uint*)(&input));
			}
		}

		[IntrinsicFunction("hybridizer::atomicExpr<double>::apply")]
		public static double apply(ref double address, double val, Func<double, double, double> func)
        {
            double initialValue, computedValue;
            do
            {
                initialValue = address;
                computedValue = func(initialValue, val);
            } while (initialValue != Interlocked.CompareExchange(ref address, computedValue, initialValue));
            return address;
        }

		[IntrinsicFunction("hybridizer::atomicExpr<float>::apply")]
		public static float apply(ref float address, float val, Func<float, float, float> func)
        {
            float initialValue, computedValue;
            do
            {
                initialValue = address;
                computedValue = func(initialValue, val);
            } while (initialValue != Interlocked.CompareExchange(ref address, computedValue, initialValue));
            return address;
        }

		[IntrinsicFunction("hybridizer::atomicExpr<int>::apply")]
		public static int apply(ref int address, int val, Func<int, int, int> func)
        {
            int initialValue, computedValue;
            do
            {
                initialValue = address;
                computedValue = func(initialValue, val);
            } while (initialValue != Interlocked.CompareExchange(ref address, computedValue, initialValue));
            return address;
        }

		[IntrinsicFunction("hybridizer::atomicExpr<long long>::apply")]
		public static long apply(ref long address, long val, Func<long, long, long> func)
        {
            long initialValue, computedValue;
            do
            {
                initialValue = address;
                computedValue = func(initialValue, val);
            } while (initialValue != Interlocked.CompareExchange(ref address, computedValue, initialValue));
            return address;
        }
	}
}

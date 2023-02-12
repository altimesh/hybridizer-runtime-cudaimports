/* (c) ALTIMESH 2018 -- all rights reserved */
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Hybridizer.Runtime.CUDAImports
{
#pragma warning disable 1591
	/// <summary>
	///  math functions
	/// </summary>
	[IntrinsicInclude("hybridizer.math.cuh", Flavor = (int) HybridizerFlavor.CUDA)]
	public class HybMath
	{
		
		[IntrinsicFunction("hybridizer::exp"), HybridNakedFunction]
		public static double Exp(double x)
		{
			return Math.Exp(x);
		}

		[IntrinsicFunction("hybridizer::exp"), HybridNakedFunction]
		public static float Exp(float x)
		{
			return (float)Math.Exp(x);
		}

		[IntrinsicFunction("hybridizer::expm1"), HybridNakedFunction]
		public static double Expm1(double x)
		{
			return (Math.Exp(x) - 1.0);
		}

		[IntrinsicFunction("hybridizer::expm1"), HybridNakedFunction]
		public static float Expm1(float x)
		{
			return (float)(Math.Exp(x) - 1.0);
		}

		[IntrinsicFunction("hybridizer::floor"), HybridNakedFunction]
		public static double Floor(double x)
		{
			return Math.Floor(x);
		}

		[IntrinsicFunction("hybridizer::floor"), HybridNakedFunction]
		public static float Floor(float x)
		{
			return (float) Math.Floor(x);
		}

		[IntrinsicFunction("hybridizer::ceil"), HybridNakedFunction]
		public static double Ceil(double x)
		{
			return Math.Ceiling(x);
		}

		[IntrinsicFunction("hybridizer::ceil"), HybridNakedFunction]
		public static float Ceil(float x)
		{
			return (float)Math.Ceiling(x);
		}

		[IntrinsicFunction("hybridizer::log"), HybridNakedFunction]
		public static double Log(double x)
		{
			return Math.Log(x);
		}

		[IntrinsicFunction("hybridizer::log"), HybridNakedFunction]
		public static float Log(float x)
		{
			return (float)Math.Log(x);
		}

		[IntrinsicFunction("hybridizer::cos"), HybridNakedFunction]
		public static double Cos(double x)
		{
			return Math.Cos(x);
		}

		[IntrinsicFunction("hybridizer::cos"), HybridNakedFunction]
		public static float Cos(float x)
		{
			return (float)Math.Cos(x);
		}

		[IntrinsicFunction("hybridizer::sin"), HybridNakedFunction]
		public static double Sin(double x)
		{
			return Math.Sin(x);
		}

		[IntrinsicFunction("hybridizer::sin"), HybridNakedFunction]
		public static float Sin(float x)
		{
			return (float)Math.Sin(x);
		}

		[IntrinsicFunction("hybridizer::sqrt"), HybridNakedFunction]
		public static double Sqrt(double x)
		{
			return Math.Sqrt(x);
		}

		[IntrinsicFunction("hybridizer::sqrt"), HybridNakedFunction]
		public static float Sqrt(float x)
		{
			return (float)Math.Sqrt(x);
		}

		[IntrinsicFunction("hybridizer::rsqrt"), HybridNakedFunction]
		public static double Rsqrt(double x)
		{
			return 1.0 / Math.Sqrt(x);
		}

		[IntrinsicFunction("hybridizer::rsqrt"), HybridNakedFunction]
		public static float Rsqrt(float x)
		{
			return 1.0F / (float)Math.Sqrt(x);
		}

		[IntrinsicFunction("hybridizer::fabs"), HybridNakedFunction]
		public static double Fabs(double x)
		{
			return Math.Abs(x);
		}

		[IntrinsicFunction("hybridizer::fabs"), HybridNakedFunction]
		public static float Abs(float x)
		{
			return (float)Math.Abs(x);
		}

		[IntrinsicFunction("hybridizer::mod"), HybridNakedFunction]
		public static float Mod(float x, float y)
		{
			return (float)Math.IEEERemainder(x, y);
		}

		[IntrinsicFunction("hybridizer::mod"), HybridNakedFunction]
		public static double Mod(double x, double y)
		{
			return Math.IEEERemainder(x, y);
		}

		[IntrinsicFunction("hybridizer::pow"), HybridNakedFunction]
		public static double Pow(double x, double a)
		{
			return Math.Pow(x, a);
		}

		[IntrinsicFunction("hybridizer::pow"), HybridNakedFunction]
		public static float Pow(float x, float a)
		{
			return (float) Math.Pow(x, a);
		}
		
		[HybridArithmeticFunction]
		static double RationalApproximation(double t)
		{
			return t - ((0.010328 * t + 0.802853) * t + 2.515517) / (((0.001308 * t + 0.189269) * t + 1.432788) * t + 1.0);
		}
		
		[IntrinsicFunction("::erfcinv", Flavor = (int) HybridizerFlavor.CUDA), HybridNakedFunction]
		public static double InvErfC(double p)
		{
			if (p < 0.5)
			{
				return -RationalApproximation(HybMath.Sqrt(-2.0 * HybMath.Log(p)));
			}
			else
			{
				return RationalApproximation(HybMath.Sqrt(-2.0 * HybMath.Log(1.0 - p)));
			}
		}

		[HybridArithmeticFunction]
		static float RationalApproximation(float t)
		{
			return t - ((0.010328F * t + 0.802853F) * t + 2.515517F) / (((0.001308F * t + 0.189269F) * t + 1.432788F) * t + 1.0F);
		}
		
		[IntrinsicFunction("::erfcinvf", Flavor = (int)HybridizerFlavor.CUDA), HybridNakedFunction]
		public static float InvErfC(float p)
		{
			if (p < 0.5F)
			{
				return -RationalApproximation(HybMath.Sqrt(-2.0F * HybMath.Log(p)));
			}
			else
			{
				return RationalApproximation(HybMath.Sqrt(-2.0F * HybMath.Log(1.0F - p)));
			}
		}
	}

#pragma warning restore 1591
}

/* (c) ALTIMESH 2018 -- all rights reserved */
using System;
using System.Runtime.InteropServices;

#pragma warning disable 1591 // TODO: documentation
namespace Hybridizer.Runtime.CUDAImports
{
    /// <summary>
    /// Half (16 bits) precision floating point type
    /// </summary>
    [IntrinsicIncludeCUDA("<cuda_fp16.h>")]
    [IntrinsicType("half")]
    [IntrinsicPrimitive("half")]
    [Guid("506315CE-E8F4-46A3-AE9F-A1A950A8FD4C")]
    public struct half
    {
        internal ushort x;

        #region Intrinsic functions from cuda_fp16.h

        /// <summary>
        /// Converts float number to half precision in round-to-nearest mode and
        /// returns \p half with converted value.
        /// </summary>
        /// <param name="a"></param>
        /// <returns></returns>
        [IntrinsicFunction(IsNaked=true, Name="__float2half")]
        private static ushort __float2half(float a) { throw new NotImplementedException(); }

        /// <summary>
        /// converts 
        /// </summary>
        /// <param name="a"></param>
        /// <returns></returns>
        [IntrinsicFunction(IsNaked=true, Name="__half2float")]
        private static float __half2float(ushort a) { throw new NotImplementedException(); }

        [IntrinsicFunction(IsNaked=true, Name="__hisinf")]
        private static int __hisinf(ushort a) { throw new NotImplementedException(); }
        #endregion

        /// <summary>
        /// constructor from float32
        /// </summary>
        /// <param name="a"></param>
        public half (float a) 
        { 
            x = __float2half(a);
        }

        /// <summary>
        /// mapping of intrinsic function float2half
        /// </summary>
        /// <param name="f"></param>
        /// <returns></returns>
        [IntrinsicFunction(IsNaked=true, Name="__float2half")]
        public static half float2half(float f)
        {
            half res;
            res.x = __float2half(f);
            return res;
        }

        /// <summary>
        /// conversion back to float32
        /// </summary>
        public static implicit operator float (half h)
        {
            return __half2float (h.x);
        }

        /// <summary>
        /// conversion back to ushort
        /// </summary>
        public static implicit operator ushort(half h)
        {
            return h.x;
        }

        /// <summary>
        /// conversion from ushort
        /// </summary>
        public static implicit operator half(ushort h)
        {
            half res = new half(); res.x = h; return res;
        }

        /// <summary>
        /// Returns -1 iff \p a is equal to negative infinity, 1 iff \p a is
        /// equal to positive infinity and 0 otherwise.
        /// </summary>
        public int IsInfinite
        {
            get
            {
                return __hisinf(x) ;
            }
        }

        /// <summary>
        /// NaN detection
        /// </summary>
        public bool IsNan
        {
            [IntrinsicFunction(IsNaked=true, Name="__hisnan")]
            get { throw new NotImplementedException(); }
        }
        
        [IntrinsicFunction(IsNaked=true, Name="__hadd")]
        public static half operator +(half a, half b) { throw new NotImplementedException(); }
        [IntrinsicFunction(IsNaked=true, Name="__hsub")]
        public static half operator -(half a, half b) { throw new NotImplementedException(); }
        [IntrinsicFunction(IsNaked=true, Name="__hmul")]
        public static half operator *(half a, half b) { throw new NotImplementedException(); }

        [IntrinsicFunction(IsNaked=true, Name="__hadd_sat")]
        public static half __hadd_sat(half a, half b) { throw new NotImplementedException(); }
        [IntrinsicFunction(IsNaked=true, Name="__hsub_sat")]
        public static half __hsub_sat(half a, half b) { throw new NotImplementedException(); }
        [IntrinsicFunction(IsNaked=true, Name="__hmul_sat")]
        public static half __hmul_sat(half a, half b) { throw new NotImplementedException(); }

        [IntrinsicFunction(IsNaked=true, Name="__hfma")]
        public static half __hfma(half a, half b, half c) { throw new NotImplementedException(); }
        [IntrinsicFunction(IsNaked=true, Name="__hfma_sat")]
        public static half __hfma_sat(half a, half b, half c) { throw new NotImplementedException(); }

        [IntrinsicFunction(IsNaked=true, Name="__hneg")]
        public static half __hneg(half a) { throw new NotImplementedException(); }

        [IntrinsicFunction(IsNaked=true, Name="__heq")]
        public static bool operator ==(half a, half b) { throw new NotImplementedException(); }
        [IntrinsicFunction(IsNaked=true, Name="__hne")]
        public static bool operator !=(half a, half b) { throw new NotImplementedException(); }
        [IntrinsicFunction(IsNaked=true, Name="__hle")]
        public static bool operator <=(half a, half b) { throw new NotImplementedException(); }
        [IntrinsicFunction(IsNaked=true, Name="__hge")]
        public static bool operator >=(half a, half b) { throw new NotImplementedException(); }
        [IntrinsicFunction(IsNaked=true, Name="__hlt")]
        public static bool operator <(half a, half b) { throw new NotImplementedException(); }
        [IntrinsicFunction(IsNaked=true, Name="__hgt")]
        public static bool operator >(half a, half b) { throw new NotImplementedException(); }

        [IntrinsicFunction(IsNaked=true, Name="__hequ")]
        public static bool __hequ(half a, half b) { throw new NotImplementedException(); }
        [IntrinsicFunction(IsNaked=true, Name="__hneu")]
        public static bool __hneu(half a, half b) { throw new NotImplementedException(); }
        [IntrinsicFunction(IsNaked=true, Name="__hleu")]
        public static bool __hleu(half a, half b) { throw new NotImplementedException(); }
        [IntrinsicFunction(IsNaked=true, Name="__hgeu")]
        public static bool __hgeu(half a, half b) { throw new NotImplementedException(); }
        [IntrinsicFunction(IsNaked=true, Name="__hltu")]
        public static bool __hltu(half a, half b) { throw new NotImplementedException(); }
        [IntrinsicFunction(IsNaked=true, Name="__hgtu")]
        public static bool __hgtu(half a, half b) { throw new NotImplementedException(); }

        public bool Equals(half other)
        {
            return x == other.x;
        }

        public override bool Equals(object obj)
        {
            if (ReferenceEquals(null, obj)) return false;
            return obj is half && Equals((half) obj);
        }

        public override int GetHashCode()
        {
            return x.GetHashCode();
        }
    }
    
    [IntrinsicIncludeCUDA("<cuda_fp16.h>")]
    [IntrinsicType("half2")]
    [IntrinsicPrimitive("half2")]
    [Guid("FB1994C2-5E2A-4748-B745-C09CD7AD9C06")]
    public struct half2
    {
        internal uint x;

        #region Intrinsic functions from cuda_fp16.h

        /// <summary>
        /// Converts input to half precision in round-to-nearest mode and
        /// populates both halves of \p half2 with converted value.
        /// </summary>
        /// <param name="f"></param>
        /// <returns></returns>
        [IntrinsicFunction(IsNaked=true, Name="__float2half2_rn")]
        private static uint __float2half2_rn (float f) { throw new NotImplementedException(); }

        /// <summary>
        /// Converts both input floats to half precision in round-to-nearest mode and
        /// combines the results into one \p half2 number. Low 16 bits of the return
        /// value correspond to the input \p a, high 16 bits correspond to the input \p
        /// b.
        /// </summary>
        /// <param name="a"></param>
        /// <param name="b"></param>
        /// <returns></returns>
        [IntrinsicFunction(IsNaked=true, Name="__floats2half2_rn")]
        private static uint __floats2half2_rn(float a, float b) { throw new NotImplementedException(); }

        /// <summary>
        /// Converts both components of float2 to half precision in round-to-nearest mode
        /// and combines the results into one \p half2 number. Low 16 bits of the return
        /// value correspond to \p a.x and high 16 bits of the return value correspond to
        /// \p a.y.
        /// </summary>
        /// <param name="a"></param>
        /// <returns></returns>
        [IntrinsicFunction(IsNaked=true, Name="__float22half2_rn")]
        private static uint __float22half2_rn(float2 a) { throw new NotImplementedException(); }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="a"></param>
        /// <returns></returns>
        [IntrinsicFunction(IsNaked=true, Name="__half22float2")]
        private static float2 __half22float2(uint a) { throw new NotImplementedException(); }

        [IntrinsicFunction(IsNaked=true, Name="__low2float")]
        private static float __low2float(uint a) { throw new NotImplementedException(); }

        [IntrinsicFunction(IsNaked=true, Name="__high2float")]
        private static float __high2float(uint a) { throw new NotImplementedException(); }

        [IntrinsicFunction(IsNaked=true, Name="__low2half")]
        private static ushort __low2half(uint a) { throw new NotImplementedException(); }

        [IntrinsicFunction(IsNaked=true, Name="__high2half")]
        private static ushort __high2half(uint a) { throw new NotImplementedException(); }

        /// <summary>
        /// Returns \p half2 number with both halves equal to the input \p a \p half
        /// number.
        /// </summary>
        /// <param name="a"></param>
        /// <returns></returns>
        [IntrinsicFunction(IsNaked=true, Name="__half2half2")]
        private static uint __half2half2(ushort a) { throw new NotImplementedException(); }

        /// <summary>
        /// Swaps both halves of the \p half2 input and returns a new \p half2 number
        /// with swapped halves.
        /// </summary>
        /// <param name="a"></param>
        [IntrinsicFunction(IsNaked=true, Name="__lowhigh2highlow")]
        private static uint __lowhigh2highlow(uint a) { throw new NotImplementedException(); }

        /// <summary>
        /// Combines two input \p half number \p a and \p b into one \p half2 number.
        /// Input \p a is stored in low 16 bits of the return value, input \p b is stored
        /// in high 16 bits of the return value.
        /// </summary>
        /// <param name="a"></param>
        /// <param name="b"></param>
        /// <returns></returns>
        [IntrinsicFunction(IsNaked=true, Name="__halves2half2")]
        private static uint __halves2half2(ushort a, ushort b) { throw new NotImplementedException(); }

        #endregion

        public half2(float f)
        {
            x = __float2half2_rn(f);
        }

        public half2(float a, float b)
        {
            x = __floats2half2_rn(a, b);
        }

        public half2(float2 a)
        {
            x = __float22half2_rn(a);
        }

        public half2(half a)
        {
            x = __half2half2(a.x);
        }

        public half2(half a, half b)
        {
            x = __halves2half2(a.x, b.x);
        }

        public static implicit operator float2(half2 h)
        {
            return __half22float2(h.x);
        }

        public float lo_float { get { return __low2float(x); } }
        public float hi_float { get { return __high2float(x); } }
        public half lo { get { half res; res.x = __low2half(x); return res; } set { x = __halves2half2(value.x, __high2half(x)); } }
        public half hi { get { half res; res.x = __high2half(x); return res; } set { x = __halves2half2(__low2half(x), value.x); } }

        public half2 swap() { half2 res; res.x = __lowhigh2highlow(x); return res; }

        /// <summary>
        /// Extracts low 16 bits from each of the two \p half2 inputs and combines into
        /// one \p half2 number. Low 16 bits from input \p a is stored in low 16 bits of
        /// the return value, low 16 bits from input \p b is stored in high 16 bits of
        /// the return value.
        /// </summary>
        /// <param name="a"></param>
        /// <param name="b"></param>
        /// <returns></returns>
        [IntrinsicFunction(IsNaked=true, Name="__lows2half2")]
        public static half2 __lows2half2(half2 a, half2 b) { throw new NotImplementedException(); }

        /// <summary>
        /// Extracts high 16 bits from each of the two \p half2 inputs and combines into
        /// one \p half2 number. High 16 bits from input \p a is stored in low 16 bits of
        /// the return value, high 16 bits from input \p b is stored in high 16 bits of
        /// the return value.
        /// </summary>
        /// <param name="a"></param>
        /// <param name="b"></param>
        /// <returns></returns>
        [IntrinsicFunction(IsNaked=true, Name="__highs2half2")]
        public static half2 __highs2half2(half2 a, half2 b) { throw new NotImplementedException(); }

        /// <summary>
        /// Extracts low 16 bits from \p half2 input \p a and returns a new \p half2
        /// number which has both halves equal to the extracted bits.
        /// </summary>
        /// <param name="a"></param>
        /// <returns></returns>
        [IntrinsicFunction(IsNaked=true, Name="__low2half2")]
        public static half2 __low2half2(half2 a) { throw new NotImplementedException(); }

        /// <summary>
        ///  Extracts high 16 bits from \p half2 input \p a and returns a new \p half2
        /// number which has both halves equal to the extracted bits.
        /// </summary>
        /// <param name="a"></param>
        /// <returns></returns>
        [IntrinsicFunction(IsNaked=true, Name="__high2half2")]
        public static half2 __high2half2(half2 a) { throw new NotImplementedException(); }

        /// <summary>
        /// Performs \p half2 vector if-equal comparison of inputs \p a and \p b.
        /// The corresponding \p half results are set to 1.0 for true, or 0.0 for false.
        /// NaN inputs generate false results.
        /// </summary>
        /// <param name="a"></param>
        /// <param name="b"></param>
        /// <returns></returns>
        [IntrinsicFunction(IsNaked=true, Name="__heq2")]
        public static half2 __heq2(half2 a, half2 b) { throw new NotImplementedException(); }

        /// <summary>
        /// Performs \p half2 vector not-equal comparison of inputs \p a and \p b.
        /// The corresponding \p half results are set to 1.0 for true, or 0.0 for false.
        /// NaN inputs generate false results.
        /// </summary>
        /// <param name="a"></param>
        /// <param name="b"></param>
        /// <returns></returns>
        [IntrinsicFunction(IsNaked=true, Name="__hne2")]
        public static half2 __hne2(half2 a, half2 b) { throw new NotImplementedException(); }

        /// <summary>
        /// Performs \p half2 vector less-equal comparison of inputs \p a and \p b.
        /// The corresponding \p half results are set to 1.0 for true, or 0.0 for false.
        /// NaN inputs generate false results.
        /// </summary>
        /// <param name="a"></param>
        /// <param name="b"></param>
        /// <returns></returns>
        [IntrinsicFunction(IsNaked=true, Name="__hle2")]
        public static half2 __hle2(half2 a, half2 b) { throw new NotImplementedException(); }

        /// <summary>
        /// Performs \p half2 vector greater-equal comparison of inputs \p a and \p b.
        /// The corresponding \p half results are set to 1.0 for true, or 0.0 for false.
        /// NaN inputs generate false results.
        /// </summary>
        /// <param name="a"></param>
        /// <param name="b"></param>
        /// <returns></returns>
        [IntrinsicFunction(IsNaked=true, Name="__hge2")]
        public static half2 __hge2(half2 a, half2 b) { throw new NotImplementedException(); }

        /// <summary>
        /// Performs \p half2 vector less-than comparison of inputs \p a and \p b.
        /// The corresponding \p half results are set to 1.0 for true, or 0.0 for false.
        /// NaN inputs generate false results.
        /// </summary>
        /// <param name="a"></param>
        /// <param name="b"></param>
        /// <returns></returns>
        [IntrinsicFunction(IsNaked=true, Name="__hlt2")]
        public static half2 __hlt2(half2 a, half2 b) { throw new NotImplementedException(); }

        /// <summary>
        /// Performs \p half2 vector greater-than comparison of inputs \p a and \p b.
        /// The corresponding \p half results are set to 1.0 for true, or 0.0 for false.
        /// NaN inputs generate false results.
        /// </summary>
        /// <param name="a"></param>
        /// <param name="b"></param>
        /// <returns></returns>
        [IntrinsicFunction(IsNaked=true, Name="__hgt2")]
        public static half2 __hgt2(half2 a, half2 b) { throw new NotImplementedException(); }

        /// <summary>
        /// Performs \p half2 vector if-equal comparison of inputs \p a and \p b.
        /// The corresponding \p half results are set to 1.0 for true, or 0.0 for false.
        /// NaN inputs generate true results.
        /// </summary>
        /// <param name="a"></param>
        /// <param name="b"></param>
        /// <returns></returns>
        [IntrinsicFunction(IsNaked=true, Name="__hequ2")]
        public static half2 __hequ2(half2 a, half2 b) { throw new NotImplementedException(); }

        /// <summary>
        /// Performs \p half2 vector not-equal comparison of inputs \p a and \p b.
        /// The corresponding \p half results are set to 1.0 for true, or 0.0 for false.
        /// NaN inputs generate true results.
        /// </summary>
        /// <param name="a"></param>
        /// <param name="b"></param>
        /// <returns></returns>
        [IntrinsicFunction(IsNaked=true, Name="__hneu2")]
        public static half2 __hneu2(half2 a, half2 b) { throw new NotImplementedException(); }

        /// <summary>
        /// Performs \p half2 vector less-equal comparison of inputs \p a and \p b.
        /// The corresponding \p half results are set to 1.0 for true, or 0.0 for false.
        /// NaN inputs generate true results.
        /// </summary>
        /// <param name="a"></param>
        /// <param name="b"></param>
        /// <returns></returns>
        [IntrinsicFunction(IsNaked=true, Name="__hleu2")]
        public static half2 __hleu2(half2 a, half2 b) { throw new NotImplementedException(); }

        /// <summary>
        /// Performs \p half2 vector greater-equal comparison of inputs \p a and \p b.
        /// The corresponding \p half results are set to 1.0 for true, or 0.0 for false.
        /// NaN inputs generate true results.
        /// </summary>
        /// <param name="a"></param>
        /// <param name="b"></param>
        /// <returns></returns>
        [IntrinsicFunction(IsNaked=true, Name="__hgeu2")]
        public static half2 __hgeu2(half2 a, half2 b) { throw new NotImplementedException(); }

        /// <summary>
        /// Performs \p half2 vector less-than comparison of inputs \p a and \p b.
        /// The corresponding \p half results are set to 1.0 for true, or 0.0 for false.
        /// NaN inputs generate true results.
        /// </summary>
        /// <param name="a"></param>
        /// <param name="b"></param>
        /// <returns></returns>
        [IntrinsicFunction(IsNaked=true, Name="__hltu2")]
        public static half2 __hltu2(half2 a, half2 b) { throw new NotImplementedException(); }

        /// <summary>
        /// Performs \p half2 vector greater-than comparison of inputs \p a and \p b.
        /// The corresponding \p half results are set to 1.0 for true, or 0.0 for false.
        /// NaN inputs generate true results.
        /// </summary>
        /// <param name="a"></param>
        /// <param name="b"></param>
        /// <returns></returns>
        [IntrinsicFunction(IsNaked=true, Name="__hgtu2")]
        public static half2 __hgtu2(half2 a, half2 b) { throw new NotImplementedException(); }

        /// <summary>
        /// Determine whether each half of input \p half2 number \p a is a NaN.
        /// </summary>
        /// <param name="a"></param>
        /// <param name="b"></param>
        /// <returns>Returns \p half2 which has the corresponding \p half results set to
        /// 1.0 for true, or 0.0 for false.</returns>
        [IntrinsicFunction(IsNaked=true, Name="__hisnan2")]
        public static half2 __hisnan2(half2 a, half2 b) { throw new NotImplementedException(); }

        [IntrinsicFunction(IsNaked=true, Name="__hadd2")]
        public static half2 operator +(half2 a, half2 b) { throw new NotImplementedException(); }
        [IntrinsicFunction(IsNaked=true, Name="__hsub2")]
        public static half2 operator -(half2 a, half2 b) { throw new NotImplementedException(); }
        [IntrinsicFunction(IsNaked=true, Name="__hmul2")]
        public static half2 operator *(half2 a, half2 b) { throw new NotImplementedException(); }

        /// <summary>
        /// Performs \p half2 vector add of inputs \p a and \p b, in round-to-nearest mode,
        /// and clamps the results to range [0.0, 1.0]. NaN results are flushed to +0.0.
        /// </summary>
        /// <param name="a"></param>
        /// <param name="b"></param>
        /// <returns></returns>
        [IntrinsicFunction(IsNaked=true, Name="__hadd2_sat")]
        public static half2 __hadd2_sat(half2 a, half2 b) { throw new NotImplementedException(); }
        /// <summary>
        /// Subtracts \p half2 input vector \p b from input vector \p a in round-to-nearest
        /// mode,
        /// and clamps the results to range [0.0, 1.0]. NaN results are flushed to +0.0.
        /// </summary>
        /// <param name="a"></param>
        /// <param name="b"></param>
        /// <returns></returns>
        [IntrinsicFunction(IsNaked=true, Name="__hsub2_sat")]
        public static half2 __hsub2_sat(half2 a, half2 b) { throw new NotImplementedException(); }
        /// <summary>
        /// Performs \p half2 vector multiplication of inputs \p a and \p b, in
        /// round-to-nearest mode, and clamps the results to range [0.0, 1.0]. NaN
        /// results are flushed to +0.0.
        /// </summary>
        /// <param name="a"></param>
        /// <param name="b"></param>
        /// <returns></returns>
        [IntrinsicFunction(IsNaked=true, Name="__hmul2_sat")]
        public static half2 __hmul2_sat(half2 a, half2 b) { throw new NotImplementedException(); }

        /// <summary>
        /// Performs \p half2 vector multiply on inputs \p a and \p b,
        /// then performs a \p half2 vector add of the result with \p c,
        /// rounding the result once in round-to-nearest mode.
        /// </summary>
        /// <param name="a"></param>
        /// <param name="b"></param>
        /// <param name="c"></param>
        /// <returns></returns>
        [IntrinsicFunction(IsNaked=true, Name="__hfma2")]
        public static half2 __hfma2(half2 a, half2 b, half2 c) { throw new NotImplementedException(); }

        /// <summary>
        /// Performs \p half2 vector multiply on inputs \p a and \p b,
        /// then performs a \p half2 vector add of the result with \p c,
        /// rounding the result once in round-to-nearest mode, and clamps the results to
        /// range [0.0, 1.0]. NaN results are flushed to +0.0.
        /// </summary>
        /// <param name="a"></param>
        /// <param name="b"></param>
        /// <param name="c"></param>
        /// <returns></returns>
        [IntrinsicFunction(IsNaked=true, Name="__hfma2_sat")]
        public static half2 __hfma2_sat(half2 a, half2 b, half2 c) { throw new NotImplementedException(); }

        /// <summary>
        /// Negates both halves of the input \p half2 number \p a and returns the result.
        /// </summary>
        /// <param name="a"></param>
        /// <returns></returns>
        [IntrinsicFunction(IsNaked=true, Name="__hneg2")]
        public static half2 __hneg2(half2 a) { throw new NotImplementedException(); }


        [IntrinsicFunction(IsNaked=true, Name="__hbeq2")]
        public static bool operator ==(half2 a, half2 b) { throw new NotImplementedException(); }
        [IntrinsicFunction(IsNaked=true, Name="__hbne2")]
        public static bool operator !=(half2 a, half2 b) { throw new NotImplementedException(); }
        [IntrinsicFunction(IsNaked=true, Name="__hble2")]
        public static bool operator <=(half2 a, half2 b) { throw new NotImplementedException(); }
        [IntrinsicFunction(IsNaked=true, Name="__hbge2")]
        public static bool operator >=(half2 a, half2 b) { throw new NotImplementedException(); }
        [IntrinsicFunction(IsNaked=true, Name="__hblt2")]
        public static bool operator <(half2 a, half2 b) { throw new NotImplementedException(); }
        [IntrinsicFunction(IsNaked=true, Name="__hbgt2")]
        public static bool operator >(half2 a, half2 b) { throw new NotImplementedException(); }


        [IntrinsicFunction(IsNaked=true, Name="__hbequ2")]
        public static half2 __hbequ2(half2 a, half2 b) { throw new NotImplementedException(); }
        [IntrinsicFunction(IsNaked=true, Name="__hbneu2")]
        public static half2 __hbneu2(half2 a, half2 b) { throw new NotImplementedException(); }
        [IntrinsicFunction(IsNaked=true, Name="__hbleu2")]
        public static half2 __hbleu2(half2 a, half2 b) { throw new NotImplementedException(); }
        [IntrinsicFunction(IsNaked=true, Name="__hbgeu2")]
        public static half2 __hbgeu2(half2 a, half2 b) { throw new NotImplementedException(); }
        [IntrinsicFunction(IsNaked=true, Name="__hbltu2")]
        public static half2 __hbltu2(half2 a, half2 b) { throw new NotImplementedException(); }
        [IntrinsicFunction(IsNaked=true, Name="__hbgtu2")]
        public static half2 __hbgtu2(half2 a, half2 b) { throw new NotImplementedException(); }

        public bool Equals(half2 other)
        {
            return x == other.x;
        }

        public override bool Equals(object obj)
        {
            if (ReferenceEquals(null, obj)) return false;
            return obj is half2 && Equals((half2) obj);
        }

        public override int GetHashCode()
        {
            return (int) x;
        }
    }

    [IntrinsicIncludeCUDA("<cuda_fp16.h>")]
    [IntrinsicType("half8")]
    [Guid("FB1994C2-5E2A-4748-B745-C09CD7AD9C06")]
    public struct half8 
    {
        [IntrinsicFunction(IsNaked=true, Name="hybridizer::select<half8>")]
        public static half8 Select(bool8 mask, half8 l, half8 r)
        {
            throw new NotImplementedException();
        }

        [IntrinsicFunction(IsNaked=true, Name="__hybridizer_load_half8")]
        public unsafe static half8 Load(half8* ptr, int alignment)
        {
            throw new NotImplementedException();
        }

        [IntrinsicFunction(IsNaked=true, Name="__hybridizer_store_half8")]
        public unsafe static void Store(half8* ptr, half8 val, int alignment)
        {
            throw new NotImplementedException();
        }

        public half8(half8 res)
        {
            throw new NotImplementedException();
        }

        public half8(half xx, half yy, half zz, half ww, half xx2, half yy2, half zz2, half ww2)
        {
            throw new NotImplementedException();
        }

        [IntrinsicFunction(IsNaked=true, Name="operator+")]
        public static half8 operator +(half8 a, half8 b)
        {
            throw new NotImplementedException();
        }

        [IntrinsicFunction(IsNaked=true, Name="operator+")]
        public static half8 operator +(half a, half8 b)
        {
            throw new NotImplementedException();
        }

        [IntrinsicFunction(IsNaked=true, Name="operator<")]
        public static bool8 operator <(half8 l, half8 r)
        {
            throw new NotImplementedException();
        }

        [IntrinsicFunction(IsNaked=true, Name="operator<=")]
        public static bool8 operator <=(half8 l, half8 r)
        {
            throw new NotImplementedException();
        }

        [IntrinsicFunction(IsNaked=true, Name="operator>=")]
        public static bool8 operator >=(half8 l, half8 r)
        {
            throw new NotImplementedException();
        }

        [IntrinsicFunction(IsNaked=true, Name="operator>")]
        public static bool8 operator >(half8 l, half8 r)
        {
            throw new NotImplementedException();
        }

        [IntrinsicFunction(IsNaked=true, Name="operator+")]
        public static half8 operator +(half8 a, half b)
        {
            throw new NotImplementedException();
        }

        [IntrinsicFunction(IsNaked=true, Name="operator-")]
        public static half8 operator -(half8 a, half8 b)
        {
            throw new NotImplementedException();
        }

        [IntrinsicFunction(IsNaked=true, Name="operator-")]
        public static half8 operator -(half a, half8 b)
        {
            throw new NotImplementedException();
        }

        [IntrinsicFunction(IsNaked=true, Name="operator-")]
        public static half8 operator -(half8 a, half b)
        {
            throw new NotImplementedException();
        }

        [IntrinsicFunction(IsNaked=true, Name="operator*")]
        public static half8 operator *(half8 a, half8 b)
        {
            throw new NotImplementedException();
        }

        [IntrinsicFunction(IsNaked=true, Name="operator*")]
        public static half8 operator *(half a, half8 b)
        {
            throw new NotImplementedException();
        }

        [IntrinsicFunction(IsNaked=true, Name="operator*")]
        public static half8 operator *(half8 a, half b)
        {
            throw new NotImplementedException();
        }

        [IntrinsicFunction(IsNaked=true, Name="operator/")]
        public static half8 operator /(half8 a, half8 b)
        {
            throw new NotImplementedException();
        }

        [IntrinsicFunction(IsNaked=true, Name="operator/")]
        public static half8 operator /(half a, half8 b)
        {
            throw new NotImplementedException();
        }

        [IntrinsicFunction(IsNaked=true, Name="operator/")]
        public static half8 operator /(half8 a, half b)
        {
            throw new NotImplementedException();
        }
    }
}
#pragma warning restore 1591
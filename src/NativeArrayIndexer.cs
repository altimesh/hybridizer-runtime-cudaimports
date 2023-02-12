/* (c) ALTIMESH 2018 -- all rights reserved */
using System;
using System.Runtime.InteropServices;

#pragma warning disable 1591
namespace Hybridizer.Runtime.CUDAImports
{
    public static class IntPtrExtension
    {
        public static long ToInt64(this IntPtr ptr, bool trycatch)
        {
            try
            {
                return ptr.ToInt64();
            }
            catch (NullReferenceException)
            {
                return 0;
            }
        }
    }

    [Guid("EA8CF552-F1D0-437E-9288-F2F8D9D28F15")]
    [IntrinsicType("hybridizer::nativearrayindexer<>")]
    [IntrinsicType("__hybridizer_nativearrayindexer", 8)]
    public unsafe struct NativeArrayIndexer<T> where T : struct
    {
        IntPtr _ptr;
        IntPtr _index; // in bytes

        /// <summary>
        /// 
        /// </summary>
        /// <param name="ptr"></param>
        /// <param name="index">in bytes</param>
        private NativeArrayIndexer(IntPtr ptr, IntPtr index)
        {
            if (ptr == null || index == null)
            {
                throw new ArgumentNullException("can't build a native array indexer with null object");
            }

            _ptr = ptr;
            _index = index;
        }

        [IntrinsicFunction("hybridizer::nativearrayindexer<>::build")]
        [IntrinsicFunction("__hybridizer_nativearrayindexer_build", 8)]
        [ReturnTypeInference(VectorizerIntrinsicReturn.VectorTransitive)]
        public static NativeArrayIndexer<T> Build(IntPtr ptr, IntPtr index)
        {
            return new NativeArrayIndexer<T>(ptr, index);
        }

        public static NativeArrayIndexer<T> operator +(NativeArrayIndexer<T> p, int displacement)
        {
            return new NativeArrayIndexer<T>(p._ptr, p._index + displacement);
        }
        public static NativeArrayIndexer<T> operator -(NativeArrayIndexer<T> p, int displacement)
        {
            return new NativeArrayIndexer<T>(p._ptr, p._index - displacement);
        }
        public static NativeArrayIndexer<T> operator +(NativeArrayIndexer<T> p, long displacement)
        {
            return new NativeArrayIndexer<T>(p._ptr, new IntPtr(p._index.ToInt64() + displacement));
        }
        public static NativeArrayIndexer<T> operator -(NativeArrayIndexer<T> p, long displacement)
        {
            return new NativeArrayIndexer<T>(p._ptr, new IntPtr(p._index.ToInt64() - displacement));
        }
        public static NativeArrayIndexer<T> operator +(NativeArrayIndexer<T> p, IntPtr displacement)
        {
            return new NativeArrayIndexer<T>(p._ptr, new IntPtr(p._index.ToInt64() + displacement.ToInt64()));
        }
        public static NativeArrayIndexer<T> operator -(NativeArrayIndexer<T> p, IntPtr displacement)
        {
            return new NativeArrayIndexer<T>(p._ptr, new IntPtr(p._index.ToInt64() - displacement.ToInt64()));
        }
        public static IntPtr operator -(NativeArrayIndexer<T> p, NativeArrayIndexer<T> q)
        {
            var result = new IntPtr(p._ptr.ToInt64() - q._ptr.ToInt64());
            return result;
        }

        [IntrinsicFunction("hybridizer::nativearrayindexer<>::getpointer")]
        [IntrinsicFunction("__hybridizer_nativearrayindexer_getpointer", 8)]
        public static IntPtr getpointer(NativeArrayIndexer<T> ptr)
        {
            return new IntPtr(ptr._ptr.ToInt64(true) + ptr._index.ToInt64(true));
        }

        #region load // NOTE : cannot take pointer of a managed type T...


        [IntrinsicFunction("hybridizer::nativearrayindexer<>::load<char>")]
        [IntrinsicFunction("__hybridizer_nativearrayindexer_load_char", 8)]
        [ReturnTypeInference(Return = VectorizerIntrinsicReturn.VectorTransitive, Index = 0)]
        public static sbyte load_char(ref NativeArrayIndexer<T> p)
        {
            return *((sbyte*)new IntPtr(p._ptr.ToInt64() + p._index.ToInt64()).ToPointer());
        }

        [IntrinsicFunction("hybridizer::nativearrayindexer<>::load<short>")]
        [IntrinsicFunction("__hybridizer_nativearrayindexer_load_short", 8)]
        [ReturnTypeInference(Return = VectorizerIntrinsicReturn.VectorTransitive, Index = 0)]
        public static short load_short(ref NativeArrayIndexer<T> p)
        {
            return *((short*)new IntPtr(p._ptr.ToInt64() + p._index.ToInt64()).ToPointer());
        }

        [IntrinsicFunction("hybridizer::nativearrayindexer<>::load<int>")]
        [IntrinsicFunction("__hybridizer_nativearrayindexer_load_int", 8)]
        [ReturnTypeInference(Return = VectorizerIntrinsicReturn.VectorTransitive, Index = 0)]
        public static int load_int(ref NativeArrayIndexer<T> p)
        {
            return *((int*)new IntPtr(p._ptr.ToInt64() + p._index.ToInt64()).ToPointer());
        }

        [IntrinsicFunction("hybridizer::nativearrayindexer<>::load<long>")]
        [IntrinsicFunction("__hybridizer_nativearrayindexer_load_long", 8)]
        [ReturnTypeInference(Return = VectorizerIntrinsicReturn.VectorTransitive, Index = 0)]
        public static long load_long(ref NativeArrayIndexer<T> p)
        {
            return *((long*)new IntPtr(p._ptr.ToInt64() + p._index.ToInt64()).ToPointer());
        }

        [IntrinsicFunction("hybridizer::nativearrayindexer<>::load<float>")]
        [IntrinsicFunction("__hybridizer_nativearrayindexer_load_float", 8)]
        [ReturnTypeInference(Return = VectorizerIntrinsicReturn.VectorTransitive, Index = 0)]
        public static float load_float(ref NativeArrayIndexer<T> p)
        {
            return *((float*)new IntPtr(p._ptr.ToInt64() + p._index.ToInt64()).ToPointer());
        }

        [IntrinsicFunction("hybridizer::nativearrayindexer<>::load<double>")]
        [IntrinsicFunction("__hybridizer_nativearrayindexer_load_double", 8)]
        [ReturnTypeInference(Return = VectorizerIntrinsicReturn.VectorTransitive, Index = 0)]
        public static double load_double (ref NativeArrayIndexer<T> p)
        {
            return *((double*)new IntPtr(p._ptr.ToInt64() + p._index.ToInt64()).ToPointer());
        }

        [IntrinsicFunction("hybridizer::nativearrayindexer<>::load<void*>")]
        [IntrinsicFunction("__hybridizer_nativearrayindexer_load_ptr", 8)]
        [ReturnTypeInference(Return = VectorizerIntrinsicReturn.VectorTransitive, Index = 0)]
        public static IntPtr load_IntPtr(ref NativeArrayIndexer<T> p)
        {
            return *((IntPtr*)new IntPtr(p._ptr.ToInt64() + p._index.ToInt64()).ToPointer());
        }

        #endregion

        #region store

        [IntrinsicFunction("hybridizer::nativearrayindexer<>::store")]
        [IntrinsicFunction("__hybridizer_nativearrayindexer_store_double", 8)]
        public void store(double value) 
        { 
            *((double*) new IntPtr(_ptr.ToInt64() + _index.ToInt64()).ToPointer()) = value; 
        }

        [IntrinsicFunction("hybridizer::nativearrayindexer<>::store")]
        [IntrinsicFunction("__hybridizer_nativearrayindexer_store_float", 8)]
        public void store(float value)
        { 
            *((float*)new IntPtr(_ptr.ToInt64() + _index.ToInt64()).ToPointer()) = value; 
        }

        [IntrinsicFunction("hybridizer::nativearrayindexer<>::store")]
        [IntrinsicFunction("__hybridizer_nativearrayindexer_store_int", 8)]
        public void store(int value) 
        { 
            *((int*)new IntPtr(_ptr.ToInt64() + _index.ToInt64()).ToPointer()) = value; 
        }

        [IntrinsicFunction("hybridizer::nativearrayindexer<>::store")]
        [IntrinsicFunction("__hybridizer_nativearrayindexer_store_uint", 8)]
        public void store(uint value) 
        { 
            *((uint*)new IntPtr(_ptr.ToInt64() + _index.ToInt64()).ToPointer()) = value; 
        }

        [IntrinsicFunction("hybridizer::nativearrayindexer<>::store")]
        [IntrinsicFunction("__hybridizer_nativearrayindexer_store_long", 8)]
        public void store(long value) 
        { 
            *((long*)new IntPtr(_ptr.ToInt64() + _index.ToInt64()).ToPointer()) = value; 
        }

        [IntrinsicFunction("hybridizer::nativearrayindexer<>::store")]
        [IntrinsicFunction("__hybridizer_nativearrayindexer_store_ulong", 8)]
        public void store(ulong value) 
        { 
            *((ulong*)new IntPtr(_ptr.ToInt64() + _index.ToInt64()).ToPointer()) = value; 
        }

        [IntrinsicFunction("hybridizer::nativearrayindexer<>::store")]
        [IntrinsicFunction("__hybridizer_nativearrayindexer_store_short", 8)]
        public void store(short value) 
        { 
            *((short*)new IntPtr(_ptr.ToInt64() + _index.ToInt64()).ToPointer()) = value; 
        }

        [IntrinsicFunction("hybridizer::nativearrayindexer<>::store")]
        [IntrinsicFunction("__hybridizer_nativearrayindexer_store_ushort", 8)]
        public void store(ushort value) 
        { 
            *((ushort*)new IntPtr(_ptr.ToInt64() + _index.ToInt64()).ToPointer()) = value; 
        }

        [IntrinsicFunction("hybridizer::nativearrayindexer<>::store")]
        [IntrinsicFunction("__hybridizer_nativearrayindexer_store_sbyte", 8)]
        public void store(sbyte value) 
        { 
            *((sbyte*)new IntPtr(_ptr.ToInt64() + _index.ToInt64()).ToPointer()) = value;
        }

        [IntrinsicFunction("hybridizer::nativearrayindexer<>::store")]
        public void store(byte value) 
        { 
            *((byte*)new IntPtr(_ptr.ToInt64() + _index.ToInt64()).ToPointer()) = value; 
        }

        [IntrinsicFunction("hybridizer::nativearrayindexer<>::store")]
        [IntrinsicFunction("__hybridizer_nativearrayindexer_store_ptr", 8)]
        public void store(IntPtr value) 
        {
            if (_ptr == null || _index == null)
            {
                throw new NullReferenceException("can't store in null");
            }

            *((IntPtr*)new IntPtr(_ptr.ToInt64() + _index.ToInt64()).ToPointer()) = value; 
        }
        public void store<U>(NativeArrayIndexer<U> value) where U : struct 
        { 
            store(value._ptr); 
        }

        #endregion
    }
}
#pragma warning restore 1591
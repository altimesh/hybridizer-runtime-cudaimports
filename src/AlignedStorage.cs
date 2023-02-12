using System;
using System.IO;
using System.Runtime.InteropServices;
using System.Text;

namespace Hybridizer.Runtime.CUDAImports
{
    /// <summary>
    /// internal
    /// </summary>
    /// <typeparam name="T">Usually an int</typeparam>
    [Guid("B31BAC1A-9EA4-4DC0-80BB-7BAF74BF19B3")]
    public struct Coalesced<T> { }

    /// <summary>
    /// An index, aligned to 32 -- also representing the next 32 indices
    /// <example>
    /// 0, 1, 2, ... 31
    /// 64, 65, 66, ... 95
    /// </example>
    /// Allows memory load/store optimization
    /// </summary>
    [Guid("34ED120C-EE03-45E7-B4F0-A589D1BBA1B6")]
    // [IntrinsicType("hybridizer::alignedindex<int>", Flavor = (int)HybridizerFlavor.AVX)]
    [IntrinsicType("hybridizer::alignedindex<int>", VectorizedType=typeof(Coalesced<int>))]
    // [IntrinsicType("int")]
    public struct alignedindex
    {
        int _inner;

        /// <summary>
        /// the underlying int
        /// </summary>
        public int Inner { get { return _inner; } }

        /// <summary></summary>
        public static alignedindex VectorUnitID { get { return new alignedindex(0); } }
        /// <summary></summary>
        public static int VectorUnitSize { get { return 1; } }

        /// <summary>
        /// addition operator
        /// </summary>
        public static alignedindex operator +(alignedindex a, int b)
        {
            return new alignedindex(a._inner + b);
        }

        /// <summary></summary>
        public static bool operator <(alignedindex y, int x) { return y._inner < x; }
        /// <summary></summary>
        public static bool operator <=(alignedindex y, int x) { return y._inner <= x; }
        /// <summary></summary>
        public static bool operator <(alignedindex y, alignedindex x) { return y._inner < x._inner; }
        /// <summary></summary>
        public static bool operator <=(alignedindex y, alignedindex x) { return y._inner <= x._inner; }
        /// <summary></summary>
        public static bool operator >(alignedindex y, int x) { return y._inner > x; }
        /// <summary></summary>
        public static bool operator >=(alignedindex y, int x) { return y._inner >= x; }
        /// <summary></summary>
        public static bool operator >(alignedindex y, alignedindex x) { return y._inner > x._inner; }
        /// <summary></summary>
        public static bool operator >=(alignedindex y, alignedindex x) { return y._inner >= x._inner; }


        private alignedindex(int inner) { _inner = inner; }

        /// <summary>
        /// conversion to int32
        /// </summary>
        // TODO : check/enforce for alignement somehow...
        [ReturnTypeInference(Return = VectorizerIntrinsicReturn.VectorTransitive, Index = 0)]
        public static implicit operator alignedindex(int t) { return new alignedindex(t); }
        /// <summary>
        /// conversion to an int32
        /// </summary>
        [ReturnTypeInference(Return = VectorizerIntrinsicReturn.VectorTransitive, Index = 0)]
        public static implicit operator int(alignedindex t) { return t.Inner; }
    }

    /// <summary>
    /// A <see cref="DoubleResidentArray"/> with an underlying pointer aligned to 32
    /// Allows memory load/store optimization
    /// </summary>
    [IntrinsicType("hybridizer::alignedstorage<double>")]
    [ICustomMarshalledSize(8)]
    public unsafe struct alignedstorage_double : ICustomMarshalled
    {
        private DoubleResidentArray _data;

        /// <summary>
        /// Host Pointer
        /// </summary>
        [HybridizerIgnore]
        public IntPtr HostPointer { get { return _data.HostPointer; } }

        [HybridizerIgnore]
        int _size;

        /// <summary>
        /// Size in bytes
        /// </summary>
        public int Size
        {
            [HybridizerIgnore]
            get { return _size; }
        }

        /// <summary>
        ///  indexer with aligned index
        /// </summary>
        public double this[alignedindex idx]
        {
            [IntrinsicFunction("get_Item", IsNaked = true, IsMember = true)]
            get { return _data[idx.Inner]; }

            set { _data[idx.Inner] = value; }
        }

        /// <summary>
        /// indexer with raw int--should not be used
        /// </summary>
        /// <param name="idx"></param>
        /// <returns></returns>
        public double this[int idx]
        {
            get { return _data[idx]; }

            set { _data[idx] = value; }
        }

        /// <summary>
        /// constructor
        /// </summary>
        public alignedstorage_double(int size, object source = null)
        {
            _size = size;
            _data = new DoubleResidentArray(size, source);
        }

        /// <summary>
        /// release memory
        /// </summary>
        public void destroy()
        {
            if (_data != null)
                _data.Dispose();
            _data = null;
        }

        /// <summary>
        /// Copy memory from src
        /// </summary>
        public void CopyFrom(alignedstorage_double src, int size)
        {
            for (int i = 0; i < size; ++i)
                _data[i] = src[i];
        }

        /// <summary>
        /// Marshals to native memory
        /// </summary>
        public void MarshalTo(BinaryWriter bw, HybridizerFlavor flavor)
        {
            if (flavor == HybridizerFlavor.CUDA || flavor == HybridizerFlavor.KEPLER)
            {
                _data.RefreshDevice();
                bw.Write(_data.DevicePointer.ToInt64());
                _data.Status = ResidentArrayStatus.HostNeedsRefresh;
            }
            else
            {
                bw.Write(_data.HostPointer.ToInt64());
            }
        }

        /// <summary>
        /// Unmarshals from native memory
        /// </summary>
        public void UnmarshalFrom(BinaryReader br, HybridizerFlavor flavor)
        {
            //br.ReadBytes(8);
        }

        /// <summary>
        /// A string representation
        /// </summary>
        public override string ToString()
        {
            var res = new StringBuilder();
            for (int i = 0; i < _size; ++i)
                res.AppendFormat("{0} ", this[i]);
            return res.ToString();
        }
    }

    /// <summary>
    /// A <see cref="alignedstorage_double"/> using <see href="https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#zero-copy">zero-copy</see>
    /// </summary>
    [IntrinsicType("hybridizer::alignedstorage<double>")]
    [ICustomMarshalledSize(8)]
    public unsafe struct alignedstorage_double_zerocopy : ICustomMarshalled
    {
        private DoubleZeroCopyResidentArray _data;

        /// <summary>
        /// Host Pointer
        /// </summary>
        [HybridizerIgnore]
        public IntPtr HostPointer { get { return _data.HostPointer; } }

        [HybridizerIgnore]
        int _size;

        /// <summary>
        /// Size in bytes
        /// </summary>
        public int Size
        {
            [HybridizerIgnore]
            get { return _size; }
        }

        /// <summary>
        /// indexer using <see cref="alignedindex"/>
        /// </summary>
        public double this[alignedindex idx]
        {
            [IntrinsicFunction("get_Item", IsNaked = true, IsMember = true)]
            get { return _data[idx.Inner]; }

            set { _data[idx.Inner] = value; }
        }

        /// <summary>
        /// indexer using raw int -- should not be used
        /// </summary>
        /// <param name="idx"></param>
        /// <returns></returns>
        public double this[int idx]
        {
            get { return _data[idx]; }

            set { _data[idx] = value; }
        }

        /// <summary>
        /// constructor
        /// </summary>
        public alignedstorage_double_zerocopy(int size, object source = null)
        {
            _size = size;
            _data = new DoubleZeroCopyResidentArray(size, source);
        }

        /// <summary>
        /// releases memory
        /// </summary>
        public void destroy()
        {
            if (_data != null)
                _data.Dispose();
            _data = null;
        }

        /// <summary>
        /// copies memory from src
        /// </summary>
        public void CopyFrom(alignedstorage_double src, int size)
        {
            for (int i = 0; i < size; ++i)
                _data[i] = src[i];
        }

        /// <summary>
        /// Marshals to Native
        /// </summary>
        public void MarshalTo(BinaryWriter bw, HybridizerFlavor flavor)
        {
            if (flavor == HybridizerFlavor.CUDA || flavor == HybridizerFlavor.KEPLER)
            {
                _data.RefreshDevice();
                bw.Write(_data.DevicePointer.ToInt64());
                _data.Status = ResidentArrayStatus.HostNeedsRefresh;
            }
            else
            {
                bw.Write(_data.HostPointer.ToInt64());
            }
        }

        /// <summary>
        /// Unmarshals from
        /// </summary>
        public void UnmarshalFrom(BinaryReader br, HybridizerFlavor flavor)
        {
            //br.ReadBytes(8);
        }

        /// <summary>
        /// A string representation
        /// </summary>
        public override string ToString()
        {
            var res = new StringBuilder();
            for (int i = 0; i < _size; ++i)
                res.AppendFormat("{0} ", this[i]);
            return res.ToString();
        }
    }

    /// <summary>
    /// A <see cref="IntResidentArray"/> with memory aligned to 32
    /// </summary>
    [IntrinsicType("hybridizer::alignedstorage<int>")]
    [ICustomMarshalledSize(8)]
    public unsafe struct alignedstorage_int : ICustomMarshalled
    {
        private IntResidentArray _data;

        /// <summary>
        /// Host Pointer
        /// </summary>
        [HybridizerIgnore]
        public IntPtr HostPointer { get { return _data.HostPointer; } }

        [HybridizerIgnore]
        int _size;

        /// <summary>
        /// Size in bytes
        /// </summary>
        public int Size
        {
            [HybridizerIgnore]
            get { return _size; }
        }

        /// <summary>
        /// indexer using aligned index
        /// </summary>
        public int this[alignedindex idx]
        {
            [IntrinsicFunction("get_Item", IsNaked = true, IsMember = true)]
            get { return _data[idx.Inner]; }

            set { _data[idx.Inner] = value; }
        }

        /// <summary>
        /// indexer using raw int -- should not be used
        /// </summary>
        public int this[int idx]
        {
            get { return _data[idx]; }

            set { _data[idx] = value; }
        }

        /// <summary>
        /// constructor
        /// </summary>
        /// <param name="size"></param>
        public alignedstorage_int(int size)
        {
            _size = size;
            _data = new IntResidentArray(size);
        }

        /// <summary>
        /// releases memory
        /// </summary>
        public void destroy()
        {
            if (_data != null)
                _data.Dispose();
            _data = null;
        }

        /// <summary>
        /// copy memory from src
        /// </summary>
        public void CopyFrom(alignedstorage_int src, int size)
        {
            for (int i = 0; i < size; ++i)
                _data[i] = src[i];
        }
        
        /// <summary>
        /// Marshal to native
        /// </summary>
        public void MarshalTo(BinaryWriter bw, HybridizerFlavor flavor)
        {
            if (flavor == HybridizerFlavor.CUDA || flavor == HybridizerFlavor.KEPLER)
            {
                _data.RefreshDevice();
                bw.Write(_data.DevicePointer.ToInt64());
                _data.Status = ResidentArrayStatus.HostNeedsRefresh;
            }
            else
            {
                bw.Write(_data.HostPointer.ToInt64());
            }
        }

        /// <summary>
        /// Unmarshal from native
        /// </summary>
        public void UnmarshalFrom(BinaryReader br, HybridizerFlavor flavor)
        {
            //br.ReadBytes(8);
        }

        /// <summary>
        /// A string representation
        /// </summary>
        public override string ToString()
        {
            var res = new StringBuilder();
            for (int i = 0; i < _size; ++i)
                res.AppendFormat("{0} ", this[i]);
            return res.ToString();
        }
    }

    /// <summary>
    /// A <see cref="FloatResidentArray"/> with underlying memory aligned to 32
    /// </summary>
    [IntrinsicType("hybridizer::alignedstorage<float>")]
    [ICustomMarshalledSize(8)]
    public unsafe struct alignedstorage_float : ICustomMarshalled
    {
        private FloatResidentArray _data;

        /// <summary>
        /// Host Pointer
        /// </summary>
        [HybridizerIgnore]
        public IntPtr HostPointer { get { return _data.HostPointer; } }

        [HybridizerIgnore]
        int _size;

        /// <summary>
        /// Size in bytes
        /// </summary>
        public int Size
        {
            [HybridizerIgnore]
            get { return _size; }
        }

        /// <summary>
        /// indexer using alignedindex
        /// </summary>
        public float this[alignedindex idx]
        {
            [IntrinsicFunction("get_Item", IsNaked = true, IsMember = true)]
            get { return _data[idx.Inner]; }

            set { _data[idx.Inner] = value; }
        }

        /// <summary>
        /// index using raw int -- should not be used
        /// </summary>
        public float this[int idx]
        {
            get { return _data[idx]; }

            set { _data[idx] = value; }
        }

        /// <summary>
        /// constructor
        /// </summary>
        public alignedstorage_float(int size)
        {
            _size = size;
            _data = new FloatResidentArray(size);
        }

        /// <summary>
        /// releases memory
        /// </summary>
        public void destroy()
        {
            if (_data != null)
                _data.Dispose();
            _data = null;
        }

        /// <summary>
        /// copy memory from src
        /// </summary>
        public void CopyFrom(alignedstorage_float src, int size)
        {
            for (int i = 0; i < size; ++i)
                _data[i] = src[i];
        }

        /// <summary>
        ///  marshals to native
        /// </summary>
        public void MarshalTo(BinaryWriter bw, HybridizerFlavor flavor)
        {
            if (flavor == HybridizerFlavor.CUDA || flavor == HybridizerFlavor.KEPLER)
            {
                _data.RefreshDevice();
                bw.Write(_data.DevicePointer.ToInt64());
                _data.Status = ResidentArrayStatus.HostNeedsRefresh;
            }
            else
            {
                bw.Write(_data.HostPointer.ToInt64());
            }
        }

        /// <summary>
        /// unmarshals from native
        /// </summary>
        public void UnmarshalFrom(BinaryReader br, HybridizerFlavor flavor)
        {
            //br.ReadBytes(8);
        }

        /// <summary>
        /// A string representation
        /// </summary>
        public override string ToString()
        {
            var res = new StringBuilder();
            for (int i = 0; i < _size; ++i)
                res.AppendFormat("{0} ", this[i]);
            return res.ToString();
        }
    }
}

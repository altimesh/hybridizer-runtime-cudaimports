/* (c) ALTIMESH 2018 -- all rights reserved */
using System;
using System.Diagnostics;
using System.Runtime.InteropServices;

namespace Hybridizer.Runtime.CUDAImports
{
    /// <summary>
    /// A resident array of double precision real numbers, allocated using zero-copy
    /// Complete documentation <see href="https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#zero-copy">here </see>
    /// </summary>
    [HybridMappedJavaTypeAttribute("com.altimesh.hybridizer.runtime.api.IResidentArray")]
    public unsafe class DoubleZeroCopyResidentArray : IResidentData, IDisposable
    {
#pragma warning disable 1591
        [ResidentArrayHostAttribute]
        protected double* tab = (double*)0;
#pragma warning restore 1591

        ResidentArrayStatus _status = ResidentArrayStatus.NoAction;

        long _count;

        [HybridizerIgnore]
        long _size;

        [HybridizerIgnore]
        IntPtr _dPtr = IntPtr.Zero;

        [HybridizerIgnore]
        private bool _hostAllocLocal;

        [HybridizerIgnore]
        private AlignedAllocation _allocation;

        [HybridizerIgnore]
        private object _source;

        /// <summary>
        /// constructor
        /// </summary>
        public DoubleZeroCopyResidentArray(long count, object source = null)
        {
            _count = count;
            _size = count * sizeof(double);
            Status = ResidentArrayStatus.NoAction;
            _source = source;
        }

        /// <summary>
        /// indexer
        /// </summary>
        public double this[int index]
        {
            get
            {
                CheckRefreshHost();
                return GetResidentArray(tab, index);
            }
            set
            {
                CheckRefreshHost();
                SetResidentArray(tab, index, value);
            }
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="tab"></param>
        /// <param name="index"></param>
        /// <returns></returns>
        [IntrinsicFunction("hybridizer::getarraydouble")]
        [ReturnTypeInference(VectorizerIntrinsicReturn.VectorTransitive, Index = 1)]
        private double GetResidentArray(double* tab, int index)
        {
            return tab[index];
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="tab"></param>
        /// <param name="index"></param>
        /// <param name="value"></param>
        [IntrinsicFunction("hybridizer::setarraydouble")]
        private void SetResidentArray(double* tab, int index, double value)
        {
            tab[index] = value;
            Status = ResidentArrayStatus.DeviceNeedsRefresh;
        }

        /// <summary>
        /// 
        /// </summary>
        [IntrinsicConstant("")]
        private void CheckRefreshHost()
        {
            if ((IntPtr)tab == IntPtr.Zero)
                AllocHost();
            if (_status == ResidentArrayStatus.HostNeedsRefresh)
                RefreshHost();
        }

        #region IResidentArray

        /// <summary>
        /// Memory status
        /// </summary>
        public ResidentArrayStatus Status
        {
            get
            {
                return _status;
            }
            set
            {
                _status = value;
            }
        }

        /// <summary>
        /// Device pointer
        /// </summary>
        public IntPtr DevicePointer
        {
            get
            {
                if (_dPtr == IntPtr.Zero)
                    AllocDevice();
                return _dPtr; 
            }
            set
            {
                _dPtr = value ;
            }
        }

        /// <summary>
        /// Host pointer
        /// </summary>
        public IntPtr HostPointer
        {
            get
            {
                if ((IntPtr)tab == IntPtr.Zero)
                    AllocHost();
                return (IntPtr) tab;
            }
        }

        /// <summary>
        /// Size in bytes
        /// </summary>
        public long Size
        {
            get
            {
                return _size;
            }
        }

        /// <summary>
        /// Moves memory from device to host
        /// </summary>
        public void RefreshHost()
        {
            Status = ResidentArrayStatus.NoAction;
        }

        /// <summary>
        /// Moves memory from Host to Device
        /// </summary>
        public void RefreshDevice()
        {
            Status = ResidentArrayStatus.NoAction;
        }

        #endregion

        /// <summary>
        /// number of elements
        /// </summary>
        public long Count
        {
            get { return _count; }
        }

        private unsafe void Unlock(IntPtr hPtr)
        {
            cuerr = cuda.HostUnregister(hPtr);
        }

        private void Lock(IntPtr hPtr)
        {
            cuerr = cuda.HostRegister(hPtr, _size, 1);
        }

        private void AllocHost()
        {
            _hostAllocLocal = true;
            _allocation = AlignedAllocation.Alloc(_size, 32);
            tab = (double*) _allocation.Aligned;
            if (tab == null)
            {
                Process currentProcess = System.Diagnostics.Process.GetCurrentProcess();
                long totalBytesOfMemoryUsed = currentProcess.WorkingSet64;
                throw new ApplicationException("Cannot allocate resident array host - requested size: " + _size + " - WS: " + totalBytesOfMemoryUsed);
            }
        }

        cudaError_t cuerr
        {
            set
            {
                if (value != cudaError_t.cudaSuccess)
                    throw new ApplicationException(string.Format("CUDA EXCEPTION {0} : {1}", (int)value, cuda.GetErrorString(value)));
            }
        }

        private void AllocDevice()
        {
            cuerr = cuda.HostRegister(HostPointer, _size, 1);
            cuerr = cuda.HostGetDevicePointer(out _dPtr, HostPointer, 0);
        }

        #region IDisposable
#pragma warning disable 1591
        public void Dispose()
        {
            Dispose(true);
        }

        private void Dispose(bool disposing)
        {
            if (disposing)
                GC.SuppressFinalize(this);
            if (_hostAllocLocal)
            {
                _allocation.Free();
            }

            if (_dPtr != IntPtr.Zero)
            {
                if (cuda.HostUnregister(HostPointer) != cudaError_t.cudaSuccess)
                    throw new ApplicationException(string.Format("Error in cuda HostUnregister {0:X} - source: {1}", HostPointer.ToInt64(), _source));
                _dPtr = IntPtr.Zero;
            }
        }

        /// <summary>
        /// 
        /// </summary>
        ~DoubleZeroCopyResidentArray()
        {
            Dispose(false);
        }
#pragma warning restore 1591
        #endregion
    }

    /// <summary>
    /// A resident array of float 32 elements, allocated using zero-copy
    /// Zero-copy documentation <see href="https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#zero-copy">here</see>
    /// </summary>
    public unsafe class FloatZeroCopyResidentArray : IResidentData, IDisposable
    {
#pragma warning disable 1591
        [ResidentArrayHostAttribute]
        protected float* tab = (float*)0;
#pragma warning restore 1591

        ResidentArrayStatus _status = ResidentArrayStatus.NoAction;

        long _count;

        [HybridizerIgnore]
        long _size;

        [HybridizerIgnore]
        IntPtr _dPtr = IntPtr.Zero;

        [HybridizerIgnore]
        private bool _hostAllocLocal;

        [HybridizerIgnore]
        private AlignedAllocation _allocation;

        [HybridizerIgnore]
        private object _source;

        /// <summary>
        /// 
        /// </summary>
        public FloatZeroCopyResidentArray(long count, object source = null)
        {
            _count = count;
            _size = count * sizeof(float);
            Status = ResidentArrayStatus.NoAction;
            _source = source;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="index"></param>
        /// <returns></returns>
        public float this[int index]
        {
            get
            {
                CheckRefreshHost();
                return GetResidentArray(tab, index);
            }
            set
            {
                CheckRefreshHost();
                SetResidentArray(tab, index, value);
            }
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="tab"></param>
        /// <param name="index"></param>
        /// <returns></returns>
        [IntrinsicFunction("hybridizer::getarrayfloat")]
        [ReturnTypeInference(VectorizerIntrinsicReturn.VectorTransitive, Index = 1)]
        private float GetResidentArray(float* tab, int index)
        {
            return tab[index];
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="tab"></param>
        /// <param name="index"></param>
        /// <param name="value"></param>
        [IntrinsicFunction("hybridizer::setarrayfloat")]
        private void SetResidentArray(float* tab, int index, float value)
        {
            tab[index] = value;
            Status = ResidentArrayStatus.DeviceNeedsRefresh;
        }

        /// <summary>
        /// 
        /// </summary>
        [IntrinsicConstant("")]
        private void CheckRefreshHost()
        {
            if ((IntPtr)tab == IntPtr.Zero)
                AllocHost();
            if (_status == ResidentArrayStatus.HostNeedsRefresh)
                RefreshHost();
        }

        #region IResidentArray

        /// <summary>
        /// Memory status
        /// </summary>
        public ResidentArrayStatus Status
        {
            get
            {
                return _status;
            }
            set
            {
                _status = value;
            }
        }

        /// <summary>
        /// Device Pointer
        /// </summary>
        public IntPtr DevicePointer
        {
            get
            {
                if (_dPtr == IntPtr.Zero)
                    AllocDevice();
                return _dPtr;
            }
            set
            {
                _dPtr = value;
            }
        }

        /// <summary>
        /// Host Pointer
        /// </summary>
        public IntPtr HostPointer
        {
            get
            {
                if ((IntPtr)tab == IntPtr.Zero)
                    AllocHost();
                return (IntPtr)tab;
            }
        }

        /// <summary>
        /// Size in Bytes
        /// </summary>
        public long Size
        {
            get
            {
                return _size;
            }
        }

        /// <summary>
        /// Moves memory from Device To Host
        /// </summary>
        public void RefreshHost()
        {
            Status = ResidentArrayStatus.NoAction;
        }

        /// <summary>
        /// Moves memory from Host To Device
        /// </summary>
        public void RefreshDevice()
        {
            Status = ResidentArrayStatus.NoAction;
        }

        #endregion

        /// <summary>
        /// Number of elements
        /// </summary>
        public long Count
        {
            get { return _count; }
        }

        private unsafe void Unlock(IntPtr hPtr)
        {
            cuerr = cuda.HostUnregister(hPtr);
        }

        private void Lock(IntPtr hPtr)
        {
            cuerr = cuda.HostRegister(hPtr, _size, 1);
        }

        private void AllocHost()
        {
            _hostAllocLocal = true;
            _allocation = AlignedAllocation.Alloc(_size, 32);
            tab = (float*)_allocation.Aligned;
            if (tab == null)
            {
                Process currentProcess = System.Diagnostics.Process.GetCurrentProcess();
                long totalBytesOfMemoryUsed = currentProcess.WorkingSet64;
                throw new ApplicationException("Cannot allocate resident array host - requested size: " + _size + " - WS: " + totalBytesOfMemoryUsed);
            }
        }

        cudaError_t cuerr
        {
            set
            {
                if (value != cudaError_t.cudaSuccess)
                    throw new ApplicationException(string.Format("CUDA EXCEPTION {0} : {1}", (int)value, cuda.GetErrorString(value)));
            }
        }

        private void AllocDevice()
        {
            cuerr = cuda.HostRegister(HostPointer, _size, 1);
            cuerr = cuda.HostGetDevicePointer(out _dPtr, HostPointer, 0);
        }

        #region IDisposable

        /// <summary>
        /// 
        /// </summary>
        public void Dispose()
        {
            Dispose(true);
        }

        private void Dispose(bool disposing)
        {
            if (disposing)
                GC.SuppressFinalize(this);
            if (_hostAllocLocal)
            {
                _allocation.Free();
            }

            if (_dPtr != IntPtr.Zero)
            {
                if (cuda.HostUnregister(HostPointer) != cudaError_t.cudaSuccess)
                    throw new ApplicationException(string.Format("Error in cuda HostUnregister {0:X} - source: {1}", HostPointer.ToInt64(), _source));
                _dPtr = IntPtr.Zero;
            }
        }

        /// <summary>
        /// 
        /// </summary>
        ~FloatZeroCopyResidentArray()
        {
            Dispose(false);
        }

        #endregion
    }
}

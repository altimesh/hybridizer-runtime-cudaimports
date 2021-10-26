/* (c) ALTIMESH 2018 -- all rights reserved */
using System;
using System.Diagnostics;
using System.Runtime.InteropServices;

namespace Hybridizer.Runtime.CUDAImports
{
    /// <summary>
    /// A resident array of double precision real number
    /// </summary>
    [HybridMappedJavaTypeAttribute("com.altimesh.hybridizer.runtime.api.IResidentArray")]
    public unsafe class DoubleResidentArray : IResidentData, IDisposable
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
        public DoubleResidentArray(long count, object source = null)
        {
            _count = count;
            _size = count * sizeof(double);
            Status = ResidentArrayStatus.NoAction;
            _source = source;
        }

        /// <summary>
        /// indexer
        /// </summary>
        /// <param name="index"></param>
        /// <returns></returns>
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
        /// Memory Status
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
                _dPtr = value ;
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
                return (IntPtr) tab;
            }
        }

        /// <summary>
        /// Size
        /// </summary>
        public long Size
        {
            get
            {
                return _size;
            }
        }

        /// <summary>
        /// moves memory from device to host
        /// </summary>
        public void RefreshHost()
        {
            if ((IntPtr)tab == IntPtr.Zero)
                AllocHost();
            if (_dPtr != IntPtr.Zero)
            {
                cuda.HostRegister((IntPtr)tab, _size, 0);
                if (cuda.Memcpy((IntPtr) tab, _dPtr, _size, cudaMemcpyKind.cudaMemcpyDeviceToHost) != cudaError_t.cudaSuccess)
                    throw new ApplicationException(string.Format("Error in cuda {0} ({1}) memcopy to {2:X}", cuda.GetLastError(), cuda.GetErrorString(cuda.GetLastError()), _dPtr.ToInt64()));
                cuda.HostUnregister((IntPtr)tab);
                Status = ResidentArrayStatus.NoAction;
            }
            if (Status == ResidentArrayStatus.HostNeedsRefresh)
                Status = ResidentArrayStatus.NoAction;
        }

        /// <summary>
        /// moves memory from host to device
        /// </summary>
        public void RefreshDevice()
        {
            if (_dPtr == IntPtr.Zero)
                AllocDevice();
            if ((IntPtr) tab != IntPtr.Zero)
            {
                cuda.HostRegister((IntPtr) tab, _size, 0);
                if (cuda.Memcpy(_dPtr, (IntPtr)tab, _size, cudaMemcpyKind.cudaMemcpyHostToDevice) != cudaError_t.cudaSuccess)
                    throw new ApplicationException("Error while copying resident array");
                cuda.HostUnregister((IntPtr)tab);
                Status = ResidentArrayStatus.NoAction;
            }
            if (Status == ResidentArrayStatus.DeviceNeedsRefresh)
                Status = ResidentArrayStatus.NoAction;
        }

        #endregion

        /// <summary>
        ///  Count
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
            cuerr = cuda.Malloc(out _dPtr, _size);
            //Console.WriteLine("Allocated on device " + _source);
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
                if (cuda.Free(_dPtr) != cudaError_t.cudaSuccess)
                    throw new ApplicationException(string.Format("Error in cuda Free {0:X} - source: {1}", _dPtr.ToInt64(), _source));
                _dPtr = IntPtr.Zero;
            }
        }
        
        ~DoubleResidentArray()
        {
            Dispose(false);
        }
#pragma warning restore 1591
        #endregion

    }
	
	/// <summary>
	/// A resident array of int32 elements
	/// </summary>
	public unsafe class IntResidentArray : IResidentData, IDisposable
    {
#pragma warning disable 1591
        [ResidentArrayHostAttribute]
        protected int* tab = (int*)0;
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

        /// <summary>
        /// 
        /// </summary>
        /// <param name="count"></param>
        public IntResidentArray(long count)
        {
            _count = count;
            _size = count * sizeof(int);
            Status = ResidentArrayStatus.NoAction; ;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="index"></param>
        /// <returns></returns>
        public int this[int index]
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
        [IntrinsicFunction("hybridizer::getarrayint")]
        [ReturnTypeInference(VectorizerIntrinsicReturn.VectorTransitive, Index = 1)]
        private int GetResidentArray(int* tab, int index)
        {
            return tab[index];
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="tab"></param>
        /// <param name="index"></param>
        /// <param name="value"></param>
        [IntrinsicFunction("hybridizer::setarrayint")]
        private void SetResidentArray(int* tab, int index, int value)
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
        /// memory status
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
        /// device pointer
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
        /// host pointer
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
        /// Size
        /// </summary>
        public long Size
        {
            get
            {
                return _size;
            }
        }

        /// <summary>
        /// moves memory from device to host
        /// </summary>
        public void RefreshHost()
        {
            if ((IntPtr)tab == IntPtr.Zero)
                AllocHost();
            if (_dPtr != IntPtr.Zero)
            {
                if (cuda.Memcpy((IntPtr)tab, _dPtr, _size, cudaMemcpyKind.cudaMemcpyDeviceToHost) != cudaError_t.cudaSuccess)
                    throw new ApplicationException(string.Format("Error in cuda {0} ({1}) memcopy to {2:X}", cuda.GetLastError(), cuda.GetErrorString(cuda.GetLastError()), _dPtr.ToInt64()));
                Status = ResidentArrayStatus.NoAction;
            }
            if (Status == ResidentArrayStatus.HostNeedsRefresh)
                Status = ResidentArrayStatus.NoAction;
        }

        /// <summary>
        /// moves memory from host to device
        /// </summary>
        public void RefreshDevice()
        {
            if (_dPtr == IntPtr.Zero)
                AllocDevice();
            if ((IntPtr)tab != IntPtr.Zero)
            {
                if (cuda.Memcpy(_dPtr, (IntPtr)tab, _size, cudaMemcpyKind.cudaMemcpyHostToDevice) != cudaError_t.cudaSuccess)
                    throw new ApplicationException("Error while copying resident array");
                Status = ResidentArrayStatus.NoAction;
            }
            if (Status == ResidentArrayStatus.DeviceNeedsRefresh)
                Status = ResidentArrayStatus.NoAction;
        }

        #endregion

        /// <summary>
        /// Count
        /// </summary>
        public long Count
        {
            get { return _count; }
        }

        private void AllocHost()
        {
            _hostAllocLocal = true;
            _allocation = AlignedAllocation.Alloc(_size, 32);
            tab = (int*)_allocation.Aligned;
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
            cuerr = cuda.Malloc(out _dPtr, _size);
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
                _allocation.Free();

            if (_dPtr != IntPtr.Zero)
            {
                if (cuda.Free(_dPtr) != cudaError_t.cudaSuccess)
                    throw new ApplicationException(string.Format("Error in cuda Free {0:X}", _dPtr.ToInt64()));
                _dPtr = IntPtr.Zero;
            }
        }

        /// <summary>
        /// 
        /// </summary>
        ~IntResidentArray()
        {
            Dispose(false);
        }

        #endregion

    }

    /// <summary>
    /// A resident Array of float 32 elements
    /// </summary>
    public unsafe class FloatResidentArray : IResidentData, IDisposable
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

        /// <summary>
        /// 
        /// </summary>
        /// <param name="count"></param>
        public FloatResidentArray(long count)
        {
            _count = count;
            _size = count * sizeof(float);
            Status = ResidentArrayStatus.NoAction; ;
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
        /// Size
        /// </summary>
        public long Size
        {
            get
            {
                return _size;
            }
        }

        /// <summary>
        /// Moves memory from Device to Host
        /// </summary>
        public void RefreshHost()
        {
            if ((IntPtr)tab == IntPtr.Zero)
                AllocHost();
            if (_dPtr != IntPtr.Zero)
            {
                if (cuda.Memcpy((IntPtr)tab, _dPtr, _size, cudaMemcpyKind.cudaMemcpyDeviceToHost) != cudaError_t.cudaSuccess)
                    throw new ApplicationException(string.Format("Error in cuda {0} ({1}) memcopy to {2:X}", cuda.GetLastError(), cuda.GetErrorString(cuda.GetLastError()), _dPtr.ToInt64()));
                Status = ResidentArrayStatus.NoAction;
            }
            if (Status == ResidentArrayStatus.HostNeedsRefresh)
                Status = ResidentArrayStatus.NoAction;
        }

        /// <summary>
        /// Moves memory from Host to Device
        /// </summary>
        public void RefreshDevice()
        {
            if (_dPtr == IntPtr.Zero)
                AllocDevice();
            if ((IntPtr)tab != IntPtr.Zero)
            {
                if (cuda.Memcpy(_dPtr, (IntPtr)tab, _size, cudaMemcpyKind.cudaMemcpyHostToDevice) != cudaError_t.cudaSuccess)
                    throw new ApplicationException("Error while copying resident array");
                Status = ResidentArrayStatus.NoAction;
            }
            if (Status == ResidentArrayStatus.DeviceNeedsRefresh)
                Status = ResidentArrayStatus.NoAction;
        }

        #endregion

        /// <summary>
        /// Count
        /// </summary>
        public long Count
        {
            get { return _count; }
        }

        private void AllocHost()
        {
            _hostAllocLocal = true;
            _allocation = AlignedAllocation.Alloc(_size, 32);
            tab = (float*)_allocation.Aligned;
            if (tab == null)
            {
                Process currentProcess = Process.GetCurrentProcess();
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
            cuerr = cuda.Malloc(out _dPtr, _size);
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
                _allocation.Free();

            if (_dPtr != IntPtr.Zero)
            {
                if (cuda.Free(_dPtr) != cudaError_t.cudaSuccess)
                    throw new ApplicationException(string.Format("Error in cuda Free {0:X}", _dPtr.ToInt64()));
                _dPtr = IntPtr.Zero;
            }
        }

        /// <summary>
        /// 
        /// </summary>
        ~FloatResidentArray()
        {
            Dispose(false);
        }

        #endregion

    }

	public unsafe class ResidentArrayGeneric<T> : IResidentData, IDisposable where T: struct
	{
#pragma warning disable 1591
		[ResidentArrayHostAttribute]
		protected IntPtr tab = IntPtr.Zero;
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

		/// <summary>
		/// 
		/// </summary>
		/// <param name="count"></param>
		public ResidentArrayGeneric(long count)
		{
			_count = count;
			_size = count * Marshal.SizeOf(default(T));
			Status = ResidentArrayStatus.NoAction; ;
		}

		/// <summary>
		/// 
		/// </summary>
		/// <param name="index"></param>
		/// <returns></returns>
		public T this[int index]
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
		/// assuming memory is already allocated by a call to constructor
		/// </summary>
		/// <param name="data"></param>
		[HybridizerIgnore]
		public void Import(T[] data)
		{
			for(int i = 0; i < data.Length; ++i)
			{
				this[i] = data[i];
			}
		}
		
		/// <summary>
		/// 
		/// </summary>
		/// <param name="tab"></param>
		/// <param name="index"></param>
		/// <returns></returns>
		[IntrinsicFunction("hybridizer::getarray<>")]
		[ReturnTypeInference(VectorizerIntrinsicReturn.VectorTransitive, Index = 1)]
		private T GetResidentArray(IntPtr tab, int index)
		{
			int elementSize = Marshal.SizeOf(default(T));
			byte* ptr = ((byte*)tab) + index * elementSize;
			T[] result = new T[1];
			var handle = GCHandle.Alloc(result, GCHandleType.Pinned);
			byte* result_ptr = (byte*) handle.AddrOfPinnedObject();
			for(int k = 0; k < elementSize; ++k)
			{
				result_ptr[k] = ptr[k];
			}

			handle.Free();
			return result[0];
		}

		/// <summary>
		/// 
		/// </summary>
		/// <param name="tab"></param>
		/// <param name="index"></param>
		/// <param name="value"></param>
		[IntrinsicFunction("hybridizer::setarray<>")]
		private void SetResidentArray(IntPtr tab, int index, T value)
		{
			int elementSize = Marshal.SizeOf(default(T));
			byte* ptr = ((byte*)tab) + index * elementSize;
			var handle = GCHandle.Alloc(value, GCHandleType.Pinned);
			byte* value_ptr = (byte*)handle.AddrOfPinnedObject();
			for (int k = 0; k < elementSize; ++k)
			{
				ptr[k] = value_ptr[k];
			}

			handle.Free();
			Status = ResidentArrayStatus.DeviceNeedsRefresh;
		}

		//public ref T GetElement(int index)
		//{
		//	int elementSize = Marshal.SizeOf(default(T));
		//	byte* ptr = ((byte*)tab) + index * elementSize;
		//	ref T result;
		//	var handle = GCHandle.Alloc(result, GCHandleType.Pinned);
		//	byte* result_ptr = (byte*)handle.AddrOfPinnedObject();
		//	for (int k = 0; k < elementSize; ++k)
		//	{
		//		result_ptr[k] = ptr[k];
		//	}
		//	handle.Free();
		//	return ref result;
		//}

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
		/// memory status
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
		/// device pointer
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
		/// host pointer
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
		/// Size
		/// </summary>
		public long Size
		{
			get
			{
				return _size;
			}
		}

		/// <summary>
		/// moves memory from device to host
		/// </summary>
		public void RefreshHost()
		{
			if ((IntPtr)tab == IntPtr.Zero)
				AllocHost();
			if (_dPtr != IntPtr.Zero)
			{
				if (cuda.Memcpy((IntPtr)tab, _dPtr, _size, cudaMemcpyKind.cudaMemcpyDeviceToHost) != cudaError_t.cudaSuccess)
					throw new ApplicationException(string.Format("Error in cuda {0} ({1}) memcopy to {2:X}", cuda.GetLastError(), cuda.GetErrorString(cuda.GetLastError()), _dPtr.ToInt64()));
				Status = ResidentArrayStatus.NoAction;
			}
			if (Status == ResidentArrayStatus.HostNeedsRefresh)
				Status = ResidentArrayStatus.NoAction;
		}

		/// <summary>
		/// moves memory from host to device
		/// </summary>
		public void RefreshDevice()
		{
			if (_dPtr == IntPtr.Zero)
				AllocDevice();
			if ((IntPtr)tab != IntPtr.Zero)
			{
				if (cuda.Memcpy(_dPtr, (IntPtr)tab, _size, cudaMemcpyKind.cudaMemcpyHostToDevice) != cudaError_t.cudaSuccess)
					throw new ApplicationException("Error while copying resident array");
				Status = ResidentArrayStatus.NoAction;
			}
			if (Status == ResidentArrayStatus.DeviceNeedsRefresh)
				Status = ResidentArrayStatus.NoAction;
		}

		#endregion

		/// <summary>
		/// Count
		/// </summary>
		public long Count
		{
			get { return _count; }
		}

		private void AllocHost()
		{
			_hostAllocLocal = true;
			_allocation = AlignedAllocation.Alloc(_size, 32);
			tab = _allocation.Aligned;
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
			cuerr = cuda.Malloc(out _dPtr, _size);
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
				_allocation.Free();

			if (_dPtr != IntPtr.Zero)
			{
				if (cuda.Free(_dPtr) != cudaError_t.cudaSuccess)
					throw new ApplicationException(string.Format("Error in cuda Free {0:X}", _dPtr.ToInt64()));
				_dPtr = IntPtr.Zero;
			}
		}

		/// <summary>
		/// 
		/// </summary>
		~ResidentArrayGeneric()
		{
			Dispose(false);
		}

		#endregion

	}

}

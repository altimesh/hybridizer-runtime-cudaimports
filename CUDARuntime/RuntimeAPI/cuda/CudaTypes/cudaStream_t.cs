using System;

namespace Hybridizer.Runtime.CUDAImports
{
    /// <summary>
    /// CUDA stream
    /// </summary>
    [IntrinsicType("cudaStream_t")]
    public struct cudaStream_t
    {
        /// <summary>
        /// inner opaque pointer
        /// </summary>
        public IntPtr _inner;

        /// <summary>
        /// constructor from native pointer
        /// </summary>
        public cudaStream_t(IntPtr ptr)
        {
            this._inner = ptr;
        }

        /// <summary>
        /// void stream
        /// </summary>
        public static cudaStream_t NO_STREAM = new cudaStream_t(IntPtr.Zero);

        /// <summary>
        /// string representation
        /// </summary>
        public override string ToString()
        {
            return string.Format("Stream {0}", _inner.ToInt64());
        }

        /// <summary>
        /// convert a cudastream to a custream
        /// </summary>
        /// <param name="stream"></param>
        public static explicit operator CUstream(cudaStream_t stream)
        {
            return new CUstream(stream._inner);
        }

        /// <summary>
        /// returns the internal pointer
        /// </summary>
        /// <param name="stream"></param>
        public static implicit operator IntPtr(cudaStream_t stream)
        {
            return stream._inner;

        }

        /// <summary>
        /// equals operator (with other stream)
        /// </summary>
        public bool Equals(cudaStream_t other)
        {
            return _inner.Equals(other._inner);
        }

        /// <summary>
        /// equals operator (with object)
        /// </summary>
        public override bool Equals(object obj)
        {
            if (ReferenceEquals(null, obj)) return false;
            return obj is cudaStream_t && Equals((cudaStream_t)obj);
        }

        /// <summary>
        /// </summary>
        public override int GetHashCode()
        {
            return _inner.GetHashCode();
        }
    }
}
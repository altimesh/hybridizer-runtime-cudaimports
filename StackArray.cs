/* (c) ALTIMESH 2018 -- all rights reserved */
using System.Runtime.InteropServices;

namespace Hybridizer.Runtime.CUDAImports
{
    /// <summary>
    /// An array on stack
    /// </summary>
    /// <typeparam name="T"></typeparam>
    [HybridizerIgnore]
    [Guid("8897A065-A055-48BE-8488-D5AF4A3BCB53")]
    [IntrinsicType("::hybridizer::stackarray<>")]
    public struct StackArray<T> where T:struct
    {
        private T[] _data;

        /// <summary>
        /// constructor
        /// </summary>
        /// <param name="count">elements count</param>
        public StackArray(int count)
        {
            _data = new T[count];
        }

        /// <summary>
        /// get the underlying array -- allowing operations on it
        /// </summary>
        public T[] data {
            get { return _data; }
            private set { _data = value; }
        }

        /// <summary>
        /// get or set value in the array
        /// </summary>
        public T this[int idx]
        {
            get { return _data[idx]; }
            set { _data[idx] = value; }
        }
    }
}

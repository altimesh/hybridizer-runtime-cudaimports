/* (c) ALTIMESH 2018 -- all rights reserved */
namespace Hybridizer.Runtime.CUDAImports
{
    /// <summary>
    /// structure to allocate shared memory in device code
    /// </summary>
    /// <example>
    /// <code>
    /// float[] a = new SharedMemoryAllocator&lt;float&gt;.allocate(32)
    /// </code>
    /// </example>
    [IntrinsicType("hybridizer::sharedmemoryallocator<>", NotVectorizable = true)]
    public struct   SharedMemoryAllocator<T> where T : struct
    {   
        /// <summary>
        /// Performs the allocation
        /// </summary>
        /// <param name="count">number of elements in the result array</param>
        /// <returns></returns>
        [ReturnTypeInference(VectorizerIntrinsicReturn.Unchanged)]
        public T[] allocate(int count) { return new T[count]; }
    }
}

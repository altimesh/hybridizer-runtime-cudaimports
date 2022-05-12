using System.Runtime.InteropServices;

namespace Hybridizer.Runtime.CUDAImports
{
    /// <summary>
    /// CUDA 3D memory copying parameters
    /// </summary>
    [IntrinsicType("cudaMemcpy3DParms")]
    [StructLayout(LayoutKind.Sequential)]
    public struct cudaMemcpy3DParms
    {
        /// <summary>
        /// Source memory address
        /// </summary>
        cudaArray_t srcArray;
        /// <summary>
        /// Source position offset
        /// </summary>
        cudaPos srcPos;
        /// <summary>
        /// Pitched source memory address
        /// </summary>
        cudaPitchedPtr srcPtr;

        /// <summary>
        /// Destination memory address
        /// </summary>
        cudaArray_t dstArray;
        /// <summary>
        /// Destination position offset
        /// </summary>
        cudaPos dstPos;
        /// <summary>
        /// Pitched destination memory address
        /// </summary>
        cudaPitchedPtr dstPtr;

        /// <summary>
        /// Requested memory copy size
        /// </summary>
        cudaFuncAttributes extent;
        /// <summary>
        /// Type of transfer
        /// </summary>
        cudaMemcpyKind kind;
    }
}
using System;

namespace Hybridizer.Runtime.CUDAImports
{
    /// <summary>
    /// CUDA 3D cross-device memory copying parameters
    /// </summary>
    [IntrinsicType("cudaMemcpy3DPeerParms")]
    public struct cudaMemcpy3DPeerParms
    {
#pragma warning disable 0169
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
        /// Source device
        /// </summary>
        int srcDevice;

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
        /// Destination device
        /// </summary>
        int dstDevice;

        /// <summary>
        /// Requested memory copy size
        /// </summary>
        cudaFuncAttributes extent;
#pragma warning restore 0169
    }
}
/* (c) ALTIMESH 2018 -- all rights reserved */
using System;
using System.Collections.Generic;
using System.Configuration;
using System.Diagnostics;
using System.Text;
using System.Runtime.InteropServices;
using System.Reflection;

namespace Hybridizer.Runtime.CUDAImports
{
    /// <summary>
    /// CUDA device attributes
    /// </summary>
    public enum cudaDeviceAttr : int
    {
        /// <summary>
        ///  Maximum number of threads per block 
        /// </summary>
        cudaDevAttrMaxThreadsPerBlock = 1,
        /// <summary>
        ///  Maximum block dimension X 
        /// </summary>
        cudaDevAttrMaxBlockDimX = 2,
        /// <summary>
        ///  Maximum block dimension Y 
        /// </summary>
        cudaDevAttrMaxBlockDimY = 3,
        /// <summary>
        ///  Maximum block dimension Z 
        /// </summary>
        cudaDevAttrMaxBlockDimZ = 4,
        /// <summary>
        ///  Maximum grid dimension X 
        /// </summary>
        cudaDevAttrMaxGridDimX = 5,
        /// <summary>
        ///  Maximum grid dimension Y 
        /// </summary>
        cudaDevAttrMaxGridDimY = 6,
        /// <summary>
        ///          Maximum grid dimension Z 
        /// </summary>
        cudaDevAttrMaxGridDimZ = 7,
        /// <summary>
        ///          Maximum shared memory available per block in bytes 
        /// </summary>
        cudaDevAttrMaxSharedMemoryPerBlock = 8,
        /// <summary>
        ///          Memory available on device for __constant__ variables in a CUDA C kernel in bytes 
        /// </summary>
        cudaDevAttrTotalConstantMemory = 9,
        /// <summary>
        ///          Warp size in threads 
        /// </summary>
        cudaDevAttrWarpSize = 10,
        /// <summary>
        ///          Maximum pitch in bytes allowed by memory copies 
        /// </summary>
        cudaDevAttrMaxPitch = 11,
        /// <summary>
        ///          Maximum number of 32-bit registers available per block 
        /// </summary>
        cudaDevAttrMaxRegistersPerBlock = 12,
        /// <summary>
        ///          Peak clock frequency in kilohertz 
        /// </summary>
        cudaDevAttrClockRate = 13,
        /// <summary>
        ///          Alignment requirement for textures 
        /// </summary>
        cudaDevAttrTextureAlignment = 14,
        /// <summary>
        ///          Device can possibly copy memory and execute a kernel concurrently 
        /// </summary>
        cudaDevAttrGpuOverlap = 15,
        /// <summary>
        ///          Number of multiprocessors on device 
        /// </summary>
        cudaDevAttrMultiProcessorCount = 16,
        /// <summary>
        ///          Specifies whether there is a run time limit on kernels 
        /// </summary>
        cudaDevAttrKernelExecTimeout = 17,
        /// <summary>
        ///          Device is integrated with host memory 
        /// </summary>
        cudaDevAttrIntegrated = 18,
        /// <summary>
        ///          Device can map host memory into CUDA address space 
        /// </summary>
        cudaDevAttrCanMapHostMemory = 19,
        /// <summary>
        ///          Compute mode (See cudaComputeMode for details) 
        /// </summary>
        cudaDevAttrComputeMode = 20,
        /// <summary>
        ///          Maximum 1D texture width 
        /// </summary>
        cudaDevAttrMaxTexture1DWidth = 21,
        /// <summary>
        ///          Maximum 2D texture width 
        /// </summary>
        cudaDevAttrMaxTexture2DWidth = 22,
        /// <summary>
        ///          Maximum 2D texture height 
        /// </summary>
        cudaDevAttrMaxTexture2DHeight = 23,
        /// <summary>
        ///          Maximum 3D texture width 
        /// </summary>
        cudaDevAttrMaxTexture3DWidth = 24,
        /// <summary>
        ///         Maximum 3D texture height 
        /// </summary>
        cudaDevAttrMaxTexture3DHeight = 25,
        /// <summary>
        ///          Maximum 3D texture depth 
        /// </summary>
        cudaDevAttrMaxTexture3DDepth = 26,
        /// <summary>
        ///          Maximum 2D layered texture width 
        /// </summary>
        cudaDevAttrMaxTexture2DLayeredWidth = 27,
        /// <summary>
        ///          Maximum 2D layered texture height 
        /// </summary>
        cudaDevAttrMaxTexture2DLayeredHeight = 28,
        /// <summary>
        ///          Maximum layers in a 2D layered texture 
        /// </summary>
        cudaDevAttrMaxTexture2DLayeredLayers = 29,
        /// <summary>
        ///          Alignment requirement for surfaces 
        /// </summary>
        cudaDevAttrSurfaceAlignment = 30,
        /// <summary>
        ///          Device can possibly execute multiple kernels concurrently 
        /// </summary>
        cudaDevAttrConcurrentKernels = 31,
        /// <summary>
        ///          Device has ECC support enabled 
        /// </summary>
        cudaDevAttrEccEnabled = 32,
        /// <summary>
        ///          PCI bus ID of the device 
        /// </summary>
        cudaDevAttrPciBusId = 33,
        /// <summary>
        ///          PCI device ID of the device 
        /// </summary>
        cudaDevAttrPciDeviceId = 34,
        /// <summary>
        ///          Device is using TCC driver model 
        /// </summary>
        cudaDevAttrTccDriver = 35,
        /// <summary>
        ///          Peak memory clock frequency in kilohertz 
        /// </summary>
        cudaDevAttrMemoryClockRate = 36,
        /// <summary>
        ///          Global memory bus width in bits 
        /// </summary>
        cudaDevAttrGlobalMemoryBusWidth = 37,
        /// <summary>
        ///          Size of L2 cache in bytes 
        /// </summary>
        cudaDevAttrL2CacheSize = 38,
        /// <summary>
        ///          Maximum resident threads per multiprocessor 
        /// </summary>
        cudaDevAttrMaxThreadsPerMultiProcessor = 39,
        /// <summary>
        ///          Number of asynchronous engines 
        /// </summary>
        cudaDevAttrAsyncEngineCount = 40,
        /// <summary>
        ///          Device shares a unified address space with the host 
        /// </summary>
        cudaDevAttrUnifiedAddressing = 41,
        /// <summary>
        ///          Maximum 1D layered texture width 
        /// </summary>
        cudaDevAttrMaxTexture1DLayeredWidth = 42,
        /// <summary>
        ///          Maximum layers in a 1D layered texture 
        /// </summary>
        cudaDevAttrMaxTexture1DLayeredLayers = 43,
        /// <summary>
        ///          Maximum 2D texture width if cudaArrayTextureGather is set 
        /// </summary>
        cudaDevAttrMaxTexture2DGatherWidth = 45,
        /// <summary>
        ///          Maximum 2D texture height if cudaArrayTextureGather is set 
        /// </summary>
        cudaDevAttrMaxTexture2DGatherHeight = 46,
        /// <summary>
        ///          Alternate maximum 3D texture width 
        /// </summary>
        cudaDevAttrMaxTexture3DWidthAlt = 47,
        /// <summary>
        ///         Alternate maximum 3D texture height 
        /// </summary>
        cudaDevAttrMaxTexture3DHeightAlt = 48,
        /// <summary>
        ///          Alternate maximum 3D texture depth 
        /// </summary>
        cudaDevAttrMaxTexture3DDepthAlt = 49,
        /// <summary>
        ///          PCI domain ID of the device 
        /// </summary>
        cudaDevAttrPciDomainId = 50,
        /// <summary>
        ///          Pitch alignment requirement for textures 
        /// </summary>
        cudaDevAttrTexturePitchAlignment = 51,
        /// <summary>
        ///          Maximum cubemap texture width/height 
        /// </summary>
        cudaDevAttrMaxTextureCubemapWidth = 52,
        /// <summary>
        ///          Maximum cubemap layered texture width/height 
        /// </summary>
        cudaDevAttrMaxTextureCubemapLayeredWidth = 53,
        /// <summary>
        ///          Maximum layers in a cubemap layered texture 
        /// </summary>
        cudaDevAttrMaxTextureCubemapLayeredLayers = 54,
        /// <summary>
        ///         Maximum 1D surface width 
        /// </summary>
        cudaDevAttrMaxSurface1DWidth = 55,
        /// <summary>
        ///          Maximum 2D surface width 
        /// </summary>
        cudaDevAttrMaxSurface2DWidth = 56,
        /// <summary>
        ///          Maximum 2D surface height 
        /// </summary>
        cudaDevAttrMaxSurface2DHeight = 57,
        /// <summary>
        ///          Maximum 3D surface width 
        /// </summary>
        cudaDevAttrMaxSurface3DWidth = 58,
        /// <summary>
        ///          Maximum 3D surface height 
        /// </summary>
        cudaDevAttrMaxSurface3DHeight = 59,
        /// <summary>
        ///          Maximum 3D surface depth 
        /// </summary>
        cudaDevAttrMaxSurface3DDepth = 60,
        /// <summary>
        ///          Maximum 1D layered surface width 
        /// </summary>
        cudaDevAttrMaxSurface1DLayeredWidth = 61,
        /// <summary>
        ///         Maximum layers in a 1D layered surface 
        /// </summary>
        cudaDevAttrMaxSurface1DLayeredLayers = 62,
        /// <summary>
        ///          Maximum 2D layered surface width 
        /// </summary>
        cudaDevAttrMaxSurface2DLayeredWidth = 63,
        /// <summary>
        ///          Maximum 2D layered surface height 
        /// </summary>
        cudaDevAttrMaxSurface2DLayeredHeight = 64,
        /// <summary>
        ///          Maximum layers in a 2D layered surface 
        /// </summary>
        cudaDevAttrMaxSurface2DLayeredLayers = 65,
        /// <summary>
        ///          Maximum cubemap surface width 
        /// </summary>
        cudaDevAttrMaxSurfaceCubemapWidth = 66,
        /// <summary>
        ///          Maximum cubemap layered surface width 
        /// </summary>
        cudaDevAttrMaxSurfaceCubemapLayeredWidth = 67,
        /// <summary>
        ///          Maximum layers in a cubemap layered surface 
        /// </summary>
        cudaDevAttrMaxSurfaceCubemapLayeredLayers = 68,
        /// <summary>
        ///          Maximum 1D linear texture width 
        /// </summary>
        cudaDevAttrMaxTexture1DLinearWidth = 69,
        /// <summary>
        ///          Maximum 2D linear texture width 
        /// </summary>
        cudaDevAttrMaxTexture2DLinearWidth = 70,
        /// <summary>
        ///         Maximum 2D linear texture height 
        /// </summary>
        cudaDevAttrMaxTexture2DLinearHeight = 71,
        /// <summary>
        ///          Maximum 2D linear texture pitch in bytes 
        /// </summary>
        cudaDevAttrMaxTexture2DLinearPitch = 72,
        /// <summary>
        ///          Maximum mipmapped 2D texture width 
        /// </summary>
        cudaDevAttrMaxTexture2DMipmappedWidth = 73,
        /// <summary>
        ///          Maximum mipmapped 2D texture height 
        /// </summary>
        cudaDevAttrMaxTexture2DMipmappedHeight = 74,
        /// <summary>
        ///          Major compute capability version number 
        /// </summary>
        cudaDevAttrComputeCapabilityMajor = 75,
        /// <summary>
        ///          Minor compute capability version number 
        /// </summary>
        cudaDevAttrComputeCapabilityMinor = 76,
        /// <summary>
        ///          Maximum mipmapped 1D texture width 
        /// </summary>
        cudaDevAttrMaxTexture1DMipmappedWidth = 77,
    }
}
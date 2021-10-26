using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;

namespace Hybridizer.Runtime.CUDAImports
{
    interface INPPI
    {
        NppStatus CompressMarkerLabelsGetBufferSize_32u8u_C1R(int maxLabel, out int bufferSize);

        NppStatus CompressMarkerLabelsGetBufferSize_16u_C1R(int maxLabel, out int bufferSize);

        NppStatus CompressMarkerLabels_32u_C1IR_Ctx(IntPtr pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nStartingNumber, out int pNewNumber, IntPtr pBuffer, NppStreamContext ctx);

        NppStatus CompressMarkerLabels_32u_C1IR(IntPtr pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nStartingNumber, out int pNewNumber, IntPtr pBuffer);

        NppStatus LabelMarkersGetBufferSize_16u_C1R(NppiSize oSizeROI, out int nBufferSize);

        NppStatus LabelMarkersUFGetBufferSize_32u_C1R(NppiSize oSizeROI, out int nBufferSize);

        NppStatus CompressMarkerLabelsGetBufferSize_32u_C1R(int nStartingNumber, out int nBufferSize);

        NppStatus LabelMarkersGetBufferSize_8u32u_C1R(NppiSize oSizeROI, out int nBufferSize);

        NppStatus LabelMarkers_8u32u_C1R(IntPtr pSrc, int nSrcStep, IntPtr pDst, int nDstStep, NppiSize oSizeROI, byte nMinVal, NppiNorm eNorm, out int pNumber, IntPtr pBuffer);

        NppStatus LabelMarkers_8u32u_C1R_Ctx(IntPtr pSrc, int nSrcStep, IntPtr pDst, int nDstStep, NppiSize oSizeROI, byte nMinVal, NppiNorm eNorm, out int pNumber, IntPtr pBuffer, NppStreamContext ctx);


        NppStatus LabelMarkersUF_16u32u_C1R(IntPtr pSrcDst, int nSrcDstStep, NppiSize oSizeROI, ushort nMinVal, NppiNorm eNorm, out int pNumber, IntPtr pBuffer);

        NppStatus LabelMarkersUF_16u32u_C1R_Ctx(IntPtr pSrcDst, int nSrcDstStep, NppiSize oSizeROI, ushort nMinVal, NppiNorm eNorm, out int pNumber, IntPtr pBuffer, NppStreamContext ctx);

        NppStatus LabelMarkersUF_32u_C1R_Ctx(IntPtr pSrc, int nSrcStep, IntPtr pDst, int nDstStep, NppiSize oSizeROI, NppiNorm eNorm,  IntPtr pBuffer, NppStreamContext ctx);

        NppStatus LabelMarkersUF_32u_C1R(IntPtr pSrc, int nSrcStep, IntPtr pDst, int nDstStep, NppiSize oSizeROI, NppiNorm eNorm, IntPtr pBuffer);

        NppStatus LabelMarkers_16u_C1IR(IntPtr pSrcDst, int nSrcDstStep, NppiSize oSizeROI, ushort nMinVal, NppiNorm eNorm, out int pNumber, IntPtr pBuffer);

        NppStatus LabelMarkers_16u_C1IR_Ctx(IntPtr pSrcDst, int nSrcDstStep, NppiSize oSizeROI, ushort nMinVal, NppiNorm eNorm, out int pNumber, IntPtr pBuffer, NppStreamContext ctx);


        NppStatus CompressMarkerLabels_32u8u_C1R(IntPtr pSrc, int nSrcStep, IntPtr pDst, int nDstStep, NppiSize oSizeROI, int nStartingNumber, out int pNewNumber, IntPtr pBuffer);

        NppStatus CompressMarkerLabels_32u8u_C1R_Ctx(IntPtr pSrc, int nSrcStep, IntPtr pDst, int nDstStep, NppiSize oSizeROI, int nStartingNumber, out int pNewNumber, IntPtr pBuffer, NppStreamContext ctx);


        NppStatus CompressMarkerLabels_16u_C1IR(IntPtr pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nStartingNumber, out int pNewNumber, IntPtr pBuffer);

        NppStatus CompressMarkerLabels_16u_C1IR_Ctx(IntPtr pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nStartingNumber, out int pNewNumber, IntPtr pBuffer, NppStreamContext ctx);


        NppStatus BoundSegments_8u_C1IR(IntPtr pSrcDst, int nSrcDstStep, NppiSize oSizeROI, byte nBorderVal);

        NppStatus BoundSegments_8u_C1IR_Ctx(IntPtr pSrcDst, int nSrcDstStep, NppiSize oSizeROI, byte nBorderVal, NppStreamContext ctx);


        NppStatus BoundSegments_16u_C1IR(IntPtr pSrcDst, int nSrcDstStep, NppiSize oSizeROI, ushort nBorderVal);

        NppStatus BoundSegments_16u_C1IR_Ctx(IntPtr pSrcDst, int nSrcDstStep, NppiSize oSizeROI, ushort nBorderVal, NppStreamContext ctx);
    }
}

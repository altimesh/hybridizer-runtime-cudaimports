/* (c) ALTIMESH 2018 -- all rights reserved */
//#define DEBUG
//#define DEBUG_ALLOC
//#define DEBUG_MEMCPY

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Diagnostics;
using System.Security;
using System.Text;
using System.Threading;
using NamingTools = Altimesh.Hybridizer.Runtime.NamingTools;

namespace Hybridizer.Runtime.CUDAImports
{
    internal partial class CudaAggregatedSerializationState : CudaAbstractSerializationState
    {
        public static int MAX_SIZE_FOR_AGGREGATION = 100000; // in bytes, objects bigger than this will be serialized independently
        private ICollection<AggregatedAllocator> allAllocators = new HashSet<AggregatedAllocator>();
        internal AggregatedAllocator _currentAllocator;

        public override bool IsClean()
        {
            return base.IsClean() && allAllocators.Count == 0;
        }

        internal AggregatedAllocator createAllocator(object o, long size)
        {
            return new AggregatedAllocator(this, size);
        }

        internal long DeviceMemoryPressure
        {
            get
            {
                return (from a in allAllocators select a.DeviceMemoryPressure).Sum();
            }
        }

        internal CudaAggregatedSerializationState(NativePtrConverter ptrConverter, ICudaMarshalling cuda, cudaStream_t stream)
            : base(ptrConverter, cuda, stream)
        {
            serializer = new CUDAAggregatedSerializer(this);
            deserializer = new CUDAAggregatedDeserializer(this);
        }

        internal CudaAggregatedSerializationState(NativePtrConverter ptrConverter, ICudaMarshalling cuda)
            : base(ptrConverter, cuda)
        {
            serializer = new CUDAAggregatedSerializer(this);
            deserializer = new CUDAAggregatedDeserializer(this);
        }

        internal override void RemoveObject(object p)
        {
            base.RemoveObject(p);
            AggregatedAllocator.RemoveObject(p);
        }
    }
}
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;

namespace Hybridizer.Runtime.CUDAImports
{

#pragma warning disable CS1591 // Missing XML comment for publicly visible type or member

    /// <summary>
    /// Python function -- requires python feature in license
    /// </summary>
    [ICustomMarshalledSize(8)]
    [Guid("9872A8A5-3216-4F1A-B10B-B8103C002002")]
    [IntrinsicType("hybridizer::forgedfunc")]
    public class ForgedFunc : ICustomMarshalled
    {
        [HybridizerIgnore]
        public Delegate Del;

        private IntPtr _funcptr;
        public IntPtr funcptr { get { return _funcptr; } set { _funcptr = value; } }
        
        public TRes Invoke<TRes>()
        {
            return (TRes)Del.DynamicInvoke();
        }
        
        public TRes Invoke<T1, TRes>(T1 i1)
        {
            return (TRes)Del.DynamicInvoke(i1);
        }
        
        public TRes Invoke<T1, T2, TRes>(T1 i1, T2 i2)
        {
            return (TRes)Del.DynamicInvoke(i1, i2);
        }
        
        public TRes Invoke<T1, T2, T3, TRes>(T1 i1, T2 i2, T3 i3)
        {
            return (TRes)Del.DynamicInvoke(i1, i2, i3);
        }
        
        public TRes Invoke<T1, T2, T3, T4, TRes>(T1 i1, T2 i2, T3 i3, T4 i4)
        {
            return (TRes)Del.DynamicInvoke(i1, i2, i3, i4);
        }

        public TRes Invoke<T1, T2, T3, T4, T5, TRes>(T1 i1, T2 i2, T3 i3, T4 i4, T5 i5)
        {
            return (TRes)Del.DynamicInvoke(i1, i2, i3, i4, i5);
        }

        public TRes Invoke<T1, T2, T3, T4, T5, T6, TRes>(T1 i1, T2 i2, T3 i3, T4 i4, T5 i5, T6 i6)
        {
            return (TRes)Del.DynamicInvoke(i1, i2, i3, i4, i5, i6);
        }

        public void MarshalTo(BinaryWriter bw, HybridizerFlavor flavor)
        {
            bw.Write(funcptr.ToInt64());
        }

        public void UnmarshalFrom(BinaryReader br, HybridizerFlavor flavor)
        {
            // not implemented as it's basically useless
            if (br != null)
            {
            }
        }
    }

    [ICustomMarshalledSize(8)]
    [IntrinsicType("hybridizer::forgedaction")]
    [Guid("D1EF87D4-3E26-48DB-8C66-6E1A2DCF4F1A")]
    public class ForgedAction : ICustomMarshalled
    {
        [HybridizerIgnore]
        public Delegate Del;

        private IntPtr _funcptr;
        public IntPtr funcptr { get { return _funcptr; } set { _funcptr = value; } }
        
        public void Invoke()
        {
            Del.DynamicInvoke();
        }
        
        public void Invoke<T1>(T1 i1)
        {
            Del.DynamicInvoke(i1);
        }

        public void Invoke<T1, T2>(T1 i1, T2 i2)
        {
            Del.DynamicInvoke(i1, i2);
        }
        
        public void Invoke<T1, T2, T3>(T1 i1, T2 i2, T3 i3)
        {
            Del.DynamicInvoke(i1, i2, i3);
        }
        
        public void Invoke<T1, T2, T3, T4, TRes>(T1 i1, T2 i2, T3 i3, T4 i4)
        {
            Del.DynamicInvoke(i1, i2, i3, i4);
        }

        public void Invoke<T1, T2, T3, T4, T5, TRes>(T1 i1, T2 i2, T3 i3, T4 i4, T5 i5)
        {
            Del.DynamicInvoke(i1, i2, i3, i4, i5);
        }

        public void Invoke<T1, T2, T3, T4, T5, T6, TRes>(T1 i1, T2 i2, T3 i3, T4 i4, T5 i5, T6 i6)
        {
            Del.DynamicInvoke(i1, i2, i3, i4, i5, i6);
        }

        public void MarshalTo(BinaryWriter bw, HybridizerFlavor flavor)
        {
            bw.Write(funcptr.ToInt64());
        }

        public void UnmarshalFrom(BinaryReader br, HybridizerFlavor flavor)
        {
            // not implemented as it's basically useless
            if (br != null)
            {
            }
        }
    }

#pragma warning restore CS1591 // Missing XML comment for publicly visible type or member
}

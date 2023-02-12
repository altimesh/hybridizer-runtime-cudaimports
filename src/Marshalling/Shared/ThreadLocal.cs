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
    class ThreadLocal<T>
    {
        [ThreadStatic]
        static Dictionary<Object, T> threadLocalValues = new Dictionary<Object, T>();

        private T defaultValue;
        private Func<T> defaultValueFactory;

        public ThreadLocal(Func<T> func)
        {
            defaultValueFactory = func;
        }

        public T Value
        {
            get
            {
                if (defaultValue == null)
                {
                    defaultValue = defaultValueFactory.Invoke();
                }
                
                T value = defaultValue;
                if (!threadLocalValues.TryGetValue(this, out value))
                {
                    threadLocalValues[this] = value;
                }
                return value;
            }
            set { threadLocalValues[this] = value; }
        }
    }
}
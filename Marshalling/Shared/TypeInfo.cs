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
    /// <summary>
    /// Type information, used as a caching structure in the marshaller
    /// </summary>
    internal struct TypeInfo
    {
        internal Type type;
        internal long typeId;
        internal FieldTools.FieldDeclaration[] fields;
        internal long size;
        internal IHybCustomMarshaler _customMarshaler;
        internal IHybCustomMarshaler CustomMarshaler {
            get { return _customMarshaler; }
            set { _customMarshaler = value; } 
        }

        internal TypeInfo(Type type, long typeId, FieldTools.FieldDeclaration[] fields, long size)
        {
            this.type = type;
            this.fields = fields;
            this.typeId = typeId;
            this.size = size;
            this._customMarshaler = null;
        }

        public override string ToString()
        {
            var sb = new StringBuilder();
            sb.AppendFormat("TypeId: {0}, Size: {1}", typeId, size);
            sb.AppendLine();
            foreach (var fd in fields)
            {
                sb.AppendFormat("\t{0}", fd);
                sb.AppendLine();
            }
            return sb.ToString();
        }
    }
}
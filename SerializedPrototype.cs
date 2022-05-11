using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Xml.Serialization;

namespace Hybridizer.Runtime.CUDAImports
{
    /// <summary>
    /// Serializing a function prototype
    /// </summary>
    [Serializable]
    public class SerializedPrototype
    {
        /// <summary>
        /// Internal representation for function name
        /// </summary>
        [Serializable]
        public class Name
        {
            /// <summary>
            /// The name in dotnet
            /// </summary>
            [XmlAttribute]
            public string DotNetName { get; set; }
            /// <summary>
            /// The c++/CUDA name
            /// </summary>
            [XmlAttribute]
            public string CxxName { get; set; }
        }

        /// <summary>
        /// embedding type
        /// </summary>
        public Name DeclaringType { get; set; }
        /// <summary>
        /// return type
        /// </summary>
        public Name ReturnType { get; set; }
        /// <summary>
        /// symbol
        /// </summary>
        public Name Symbol { get; set; }
        /// <summary>
        /// list of parameters names
        /// </summary>
        public List<Name> Parameters { get; set; }
    }
}

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Xml.Serialization;

namespace Hybridizer.Runtime.CUDAImports
{
    [Serializable]
    public class SerializedPrototype
    {
        [Serializable]
        public class Name
        {
            [XmlAttribute]
            public string DotNetName { get; set; }
            [XmlAttribute]
            public string CxxName { get; set; }
        }

        public Name DeclaringType { get; set; }
        public Name ReturnType { get; set; }
        public Name Symbol { get; set; }
        public List<Name> Parameters { get; set; }
    }
}

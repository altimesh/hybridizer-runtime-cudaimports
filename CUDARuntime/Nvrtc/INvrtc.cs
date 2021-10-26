/* (c) ALTIMESH 2019 -- all rights reserved */
using Altimesh.Hybridizer.Runtime;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;

namespace Hybridizer.Runtime.CUDAImports
{
    public interface INvrtc
    {
        nvrtcResult DestroyProgram(ref nvrtcProgram prog);
        nvrtcResult CreateProgram(out nvrtcProgram prog, string cudaSource, string cudaSourceName,
                            string[] headers, string[] headerNames);
        nvrtcResult CompileProgram(nvrtcProgram prog, string[] options);
        nvrtcResult GetProgramLog(nvrtcProgram prog, out string log);
        nvrtcResult GetPTX(nvrtcProgram prog, out string log);
        nvrtcResult Version(out int major, out int minor);
    }
}

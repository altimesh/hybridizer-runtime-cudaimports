using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using NUnit.Framework;

namespace Hybridizer.Runtime.CUDAImports.Tests
{
    class ICudaTests
    {
        [Test]
        public void TestGetErrorString()
        {
            // Just make sure that we can get the appropriate error messages
            for (int i = 0; i < 150; i++)
            {
                Console.WriteLine(cuda.GetErrorString((cudaError_t)i));
            }

            // Check a few well-known error codes
            Assert.AreEqual("no error", cuda.GetErrorString((cudaError_t)0));
            Assert.AreEqual("driver shutting down", cuda.GetErrorString((cudaError_t)4));
            Assert.AreEqual("memory size or pointer value too large to fit in 32 bit", cuda.GetErrorString((cudaError_t)32));
        }
    }
}

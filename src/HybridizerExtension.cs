/* (c) ALTIMESH 2018 -- all rights reserved */
using System;

namespace Hybridizer.Runtime.CUDAImports
{
#pragma warning disable 1591
    public static class HybridizerExtension
    {
        public static class CUDA
        {
            public static class BlockDim
            {
                public static int x = 256;
                public static int y = 1;
                public static int z = 1;
            }

            public static class GridDim
            {
                public static int x = 112;
                public static int y = 1;
            }

            public static int SharedSize = 8160;

            public static int Error
            {
                set
                {
                    if (value != 0) throw new ApplicationException(string.Format("CUDA ERROR : {0}", value));
                }
            }
        }

        public static class KEPLER
        {
            public static class BlockDim
            {
                public static int x = 128;
                public static int y = 1;
                public static int z = 1;
            }

            public static class GridDim
            {
                public static int x = 112 ;
                public static int y = 1 ;
            }

            public static int SharedSize = 8160;

            public static int Error
            {
                set
                {
                    if (value != 0) throw new ApplicationException(string.Format("CUDA ERROR : {0}", value));
                }
            }
        }
    }
#pragma warning restore 1591
}

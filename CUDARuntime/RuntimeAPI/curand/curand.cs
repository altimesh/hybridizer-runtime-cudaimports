/* (c) ALTIMESH 2018 -- all rights reserved */
using System;
using System.Runtime.InteropServices;

using size_t = System.UIntPtr ;

namespace Hybridizer.Runtime.CUDAImports
{
	 /// <summary>
    /// curand mapping
    /// Full documentation <see href="https://docs.nvidia.com/cuda/curand/index.html">here</see>
    /// </summary>
    #pragma warning disable 1591
    public partial class curand
    {
        public enum VERSION
        {
            CUDA_101, CUDA_100, CUDA_92, CUDA_91, CUDA_90, CUDA_80, CUDA_75, CUDA_70, CUDA_65, CUDA_60, CUDA_55, CUDA_50
        }
        
        public static ICurand instance { get; set; }

        /// <summary>
        /// select curand version -- this is now done automatically from app settings
        /// </summary>
        /// <param name="v"></param>
        public static void switchToVersion(VERSION v)
        {
            bool is64 = IntPtr.Size == 8;
            if (is64)
            {
                switch (v)
                {
                    case VERSION.CUDA_101:
                        instance = new ICurand64_101();
                        break;
                    case VERSION.CUDA_100:
                        instance = new ICurand64_100();
                        break;
					case VERSION.CUDA_92:
						instance = new ICurand64_92();
						break;
					case VERSION.CUDA_91:
						instance = new ICurand64_91();
						break;
					case VERSION.CUDA_90:
						instance = new ICurand64_90();
						break;
					case VERSION.CUDA_80:
						instance = new ICurand64_80();
						break;
					case VERSION.CUDA_75:
                        instance = new ICurand64_75();
                        break;
                    case VERSION.CUDA_70:
                        instance = new ICurand64_70();
                        break;
                    case VERSION.CUDA_65:
                        instance = new ICurand64_65();
                        break;
                    case VERSION.CUDA_60:
                        instance = new ICurand64_60();
                        break;
                    case VERSION.CUDA_55:
                        instance = new ICurand64_55();
                        break;
                    default:
                        throw new NotSupportedException("not supported version of curand");
                }
            }
            else
            {
                switch (v)
                {
                    case VERSION.CUDA_101:
                    case VERSION.CUDA_100:
					case VERSION.CUDA_92:
					case VERSION.CUDA_91:
					case VERSION.CUDA_90:
						throw new NotSupportedException("curand 32 bits is not supported after cuda 8.0");
					case VERSION.CUDA_80:
                        instance = new ICurand32_80();
                        break;
                    case VERSION.CUDA_75:
                        instance = new ICurand32_75();
                        break;
                    case VERSION.CUDA_70:
                        instance = new ICurand32_70();
                        break;
                    case VERSION.CUDA_65:
                        instance = new ICurand32_65();
                        break;
                    case VERSION.CUDA_60:
                        instance = new ICurand32_60();
                        break;
                    case VERSION.CUDA_55:
                        instance = new ICurand32_55();
                        break;
                    default:
                        throw new NotSupportedException("not supported version of curand");
                }
                
            }
        }

        /// <summary>
        /// Gets CUDA version from app.config
        /// </summary>
        /// <returns></returns>
        public static string GetCudaVersion()
        {
            // If not, get the version configured in app.config
            string cudaVersion = cuda.GetCudaVersion();

            // Otherwise default to latest version
            if (cudaVersion == null) cudaVersion = "80";
            cudaVersion = cudaVersion.Replace(".", ""); // Remove all dots ("7.5" will be understood "75")

            return cudaVersion;
        }

        static curand()
        {
            bool ok = false;
            string cudaVersion = GetCudaVersion();
            try
            {
                switch (cudaVersion)
                {
                    case "50":
                        switchToVersion(VERSION.CUDA_50);
                        ok = true;
                        break;
                    case "55":
                        switchToVersion(VERSION.CUDA_55);
                        ok = true;
                        break;
                    case "60":
                        switchToVersion(VERSION.CUDA_60);
                        ok = true;
                        break;
                    case "65":
                        switchToVersion(VERSION.CUDA_65);
                        ok = true;
                        break;
                    case "70":
                        switchToVersion(VERSION.CUDA_70);
                        ok = true;
                        break;
                    case "75":
                        switchToVersion(VERSION.CUDA_75);
                        ok = true;
                        break;
                    case "80":
                        switchToVersion(VERSION.CUDA_80);
                        ok = true;
                        break;
                    case "90":
                        switchToVersion(VERSION.CUDA_90);
                        ok = true;
                        break;
                    case "91":
                        switchToVersion(VERSION.CUDA_91);
                        ok = true;
                        break;
                    case "92":
                        switchToVersion(VERSION.CUDA_92);
                        ok = true;
                        break;
                    case "100":
                        switchToVersion(VERSION.CUDA_100);
                        ok = true;
                        break;
                    case "101":
                        switchToVersion(VERSION.CUDA_101);
                        ok = true;
                        break;
                    default:
                        Console.Error.WriteLine("invalid cuda version provided");
                        break;
                }
            }
            catch (Exception) {}
            if (!ok)
            {
                Console.Error.WriteLine("No curand dll found");
                instance = new ICurand64_55();
            }
        }

        #region helper functions
        public static curandStatus_t curandCreateGenerator(out curandGenerator_t generator, curandRngType_t type)
        {
            return instance.curandCreateGenerator(out generator, type);
        }

        public static curandStatus_t curandGenerate(curandGenerator_t generator, IntPtr outputPtr, size_t num)
        {
            return instance.curandGenerate(generator, outputPtr, num);
        }
        public static curandStatus_t curandGenerateUniform(curandGenerator_t generator, IntPtr outputPtr, size_t num)
        {
            return instance.curandGenerateUniform(generator, outputPtr, num);
        }

        public static curandStatus_t curandGenerateNormal(curandGenerator_t generator, IntPtr outputPtr, size_t n, float mean, float stddev)
        {
            return instance.curandGenerateNormal(generator, outputPtr, n, mean, stddev);
        }

        public static curandStatus_t curandGenerateLogNormal(curandGenerator_t generator, IntPtr outputPtr, size_t n, float mean, float stddev)
        {
            return instance.curandGenerateLogNormal(generator, outputPtr, n, mean, stddev);
        }

        public static curandStatus_t curandGeneratePoisson(curandGenerator_t generator, IntPtr outputPtr, size_t n, double lambda)
        {
            return instance.curandGeneratePoisson(generator, outputPtr, n, lambda);
        }

        public static curandStatus_t curandGenerateUniformDouble(curandGenerator_t generator, IntPtr outputPtr, size_t num)
        {
            return instance.curandGenerateUniformDouble(generator, outputPtr, num);
        }
        public static curandStatus_t curandGenerateNormalDouble(curandGenerator_t generator, IntPtr outputPtr, size_t n, double mean, double stddev)
        {
            return instance.curandGenerateNormalDouble(generator, outputPtr, n, mean, stddev);
        }

        public static curandStatus_t curandGenerateLogNormalDouble(curandGenerator_t generator, IntPtr outputPtr, size_t n, double mean, double stddev)
        {
            return instance.curandGenerateLogNormalDouble(generator, outputPtr, n, mean, stddev);
        }

        public static curandStatus_t curandSetPseudoRandomGeneratorSeed(curandGenerator_t prngGPU, UInt64 seed)
        {
            return instance.curandSetPseudoRandomGeneratorSeed(prngGPU, seed);
        }

        public static curandStatus_t curandSetGeneratorOrdering(curandGenerator_t prngGPU, curandOrdering_t type)
        {
            return instance.curandSetGeneratorOrdering(prngGPU, type);
        }

        public static curandStatus_t curandSetGeneratorOffset(curandGenerator_t generator, UInt64 type)
        {
            return instance.curandSetGeneratorOffset(generator, type);
        }
        #endregion
    }
#pragma warning restore 1591
}

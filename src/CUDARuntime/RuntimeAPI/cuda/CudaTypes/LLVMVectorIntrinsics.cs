using System.Runtime.InteropServices;

namespace Hybridizer.Runtime.CUDAImports
{
    /// <summary>
    /// 
    /// </summary>
    public class LLVMVectorIntrinsics
    {
        /// <summary>
        /// The ‘insertelement‘ instruction inserts a scalar element into a vector at a specified index.
        /// Documentation <see href="https://llvm.org/docs/LangRef.html#insertelement-instruction">here</see>
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "hybridizer::insertElement<int8>")]
        public unsafe static int8 InsertElement(int8 vector, int valueToInsert, int index)
        {
            int8 result = new int8(vector);
            int* ptr = (int*)&result;
            ptr[index] = valueToInsert;
            return result;
        }

        /// <summary>
        /// The ‘insertelement‘ instruction inserts a scalar element into a vector at a specified index.
        /// Documentation <see href="https://llvm.org/docs/LangRef.html#insertelement-instruction">here</see>
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "hybridizer::insertElement<float8>")]
        public unsafe static float8 InsertElement(float8 vector, float valueToInsert, int index)
        {
            float8 result = new float8(vector);
            float* ptr = (float*)&result;
            ptr[index] = valueToInsert;
            return result;
        }

        /// <summary>
        /// The ‘insertelement‘ instruction inserts a scalar element into a vector at a specified index.
        /// Documentation <see href="https://llvm.org/docs/LangRef.html#insertelement-instruction">here</see>
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "hybridizer::insertElement<float4>")]
        public unsafe static float4 InsertElement(float4 vector, float valueToInsert, int index)
        {
            float4 result = new float4(vector);
            float* ptr = (float*)&result;
            ptr[index] = valueToInsert;
            return result;
        }

        /// <summary>
        /// The ‘insertelement‘ instruction inserts a scalar element into a vector at a specified index.
        /// Documentation <see href="https://llvm.org/docs/LangRef.html#insertelement-instruction">here</see>
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "hybridizer::insertElement<double2>")]
        public unsafe static double2 InsertElement(double2 vector, double valueToInsert, int index)
        {
            double2 result = new double2(vector);
            double* ptr = (double*)&result;
            ptr[index] = valueToInsert;
            return result;
        }

        /// <summary>
        /// The ‘insertelement‘ instruction inserts a scalar element into a vector at a specified index.
        /// Documentation <see href="https://llvm.org/docs/LangRef.html#insertelement-instruction">here</see>
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "hybridizer::insertElement<long2>")]
        public unsafe static long2 InsertElement(long2 vector, long valueToInsert, int index)
        {
            long2 result = new long2(vector);
            long* ptr = (long*)&result;
            ptr[index] = valueToInsert;
            return result;
        }

        /// <summary>
        /// The ‘extractelement‘ instruction extracts a single scalar element from a vector at a specified index.
        /// Documentation <see href="https://llvm.org/docs/LangRef.html#extractelement-instruction">here</see>
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "hybridizer::extractElement<int8, int>")]
        public unsafe static int ExtractElement(int8 vector, int index)
        {
            int* ptr = (int*)&vector;
            return ptr[index];
        }

        /// <summary>
        /// The ‘extractelement‘ instruction extracts a single scalar element from a vector at a specified index.
        /// Documentation <see href="https://llvm.org/docs/LangRef.html#extractelement-instruction">here</see>
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "hybridizer::extractElement<float8, float>")]
        public unsafe static float ExtractElement(float8 vector, int index)
        {
            float* ptr = (float*)&vector;
            return ptr[index];
        }

        /// <summary>
        /// The ‘extractelement‘ instruction extracts a single scalar element from a vector at a specified index.
        /// Documentation <see href="https://llvm.org/docs/LangRef.html#extractelement-instruction">here</see>
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "hybridizer::extractElement<float4, float>")]
        public unsafe static float ExtractElement(float4 vector, int index)
        {
            float* ptr = (float*)&vector;
            return ptr[index];
        }

        /// <summary>
        /// The ‘extractelement‘ instruction extracts a single scalar element from a vector at a specified index.
        /// Documentation <see href="https://llvm.org/docs/LangRef.html#extractelement-instruction">here</see>
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "hybridizer::extractElement<double2, double>")]
        public unsafe static double ExtractElement(double2 vector, int index)
        {
            double* ptr = (double*)&vector;
            return ptr[index];
        }

        /// <summary>
        /// The ‘extractelement‘ instruction extracts a single scalar element from a vector at a specified index.
        /// Documentation <see href="https://llvm.org/docs/LangRef.html#extractelement-instruction">here</see>
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "hybridizer::extractElement<long2, float>")]
        public unsafe static long ExtractElement(long2 vector, int index)
        {
            long* ptr = (long*)&vector;
            return ptr[index];
        }

        /// <summary>
        /// The ‘shufflevector‘ instruction constructs a permutation of elements from two input vectors, returning a vector with the same element type as the input and length that is the same as the shuffle mask.
        /// Documentation <see href="https://llvm.org/docs/LangRef.html#shufflevector-instruction">here</see>
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "hybridizer::shuffleVector<int8, int8, int8>")]
        public unsafe static int8 ShuffleVector(int8 left, int8 right, int8 mask)
        {
            int8 res;
            int* resptr = (int*)&res;
            int* leftptr = (int*)&left;
            int* rightptr = (int*)&right;
            int* maskptr = (int*)&mask;
            for (int i = 0; i < 8; ++i)
            {
                int index = maskptr[i];
                if (index >= 0 && index < 8)
                {
                    resptr[i] = leftptr[index];
                }
                else if (index >= 0)
                {
                    resptr[i] = leftptr[index - 8];
                }
            }

            return res;
        }

        /// <summary>
        /// The ‘shufflevector‘ instruction constructs a permutation of elements from two input vectors, returning a vector with the same element type as the input and length that is the same as the shuffle mask.
        /// Documentation <see href="https://llvm.org/docs/LangRef.html#shufflevector-instruction">here</see>
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "hybridizer::shuffleVector<float8, float8, int8>")]
        public unsafe static float8 ShuffleVector(float8 left, float8 right, int8 mask)
        {
            float8 res;
            float* resptr = (float*)&res;
            float* leftptr = (float*)&left;
            float* rightptr = (float*)&right;
            int* maskptr = (int*)&mask;
            for (int i = 0; i < 8; ++i)
            {
                int index = maskptr[i];
                if (index >= 0 && index < 8)
                {
                    resptr[i] = leftptr[index];
                }
                else if (index >= 0)
                {
                    resptr[i] = leftptr[index - 8];
                }
            }

            return res;
        }

        /// <summary>
        /// The ‘shufflevector‘ instruction constructs a permutation of elements from two input vectors, returning a vector with the same element type as the input and length that is the same as the shuffle mask.
        /// Documentation <see href="https://llvm.org/docs/LangRef.html#shufflevector-instruction">here</see>
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "hybridizer::shuffleVector<long2, long2, int2>")]
        public unsafe static long2 ShuffleVector(long2 left, long2 right, int2 mask)
        {
            long2 res;
            long* resptr = (long*)&res;
            long* leftptr = (long*)&left;
            long* rightptr = (long*)&right;
            int* maskptr = (int*)&mask;
            for (int i = 0; i < 2; ++i)
            {
                int index = maskptr[i];
                if (index >= 0 && index < 2)
                {
                    resptr[i] = leftptr[index];
                }
                else if (index >= 0)
                {
                    resptr[i] = leftptr[index - 2];
                }
            }

            return res;
        }

        /// <summary>
        /// The ‘shufflevector‘ instruction constructs a permutation of elements from two input vectors, returning a vector with the same element type as the input and length that is the same as the shuffle mask.
        /// Documentation <see href="https://llvm.org/docs/LangRef.html#shufflevector-instruction">here</see>
        /// </summary>
        [IntrinsicFunction(IsNaked = true, Name = "hybridizer::shuffleVector<double2, double2, int2>")]
        public unsafe static double2 ShuffleVector(double2 left, double2 right, int2 mask)
        {
            double2 res;
            double* resptr = (double*)&res;
            double* leftptr = (double*)&left;
            double* rightptr = (double*)&right;
            int* maskptr = (int*)&mask;
            for (int i = 0; i < 2; ++i)
            {
                int index = maskptr[i];
                if (index >= 0 && index < 2)
                {
                    resptr[i] = leftptr[index];
                }
                else if (index >= 0)
                {
                    resptr[i] = leftptr[index - 2];
                }
            }

            return res;
        }
    }
}
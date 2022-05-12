using System.Runtime.InteropServices;

namespace Hybridizer.Runtime.CUDAImports
{
    /// <summary>
    /// CUDA texture descriptor
    /// </summary>
    [IntrinsicType("cudaTextureDesc")]
    [StructLayout(LayoutKind.Explicit, Size = 64)]
    public unsafe struct cudaTextureDesc
    {
        /// <summary>
        /// Texture address mode for up to 3 dimensions
        /// </summary>
        [FieldOffset(0)]
        public fixed int addressMode[3];
        /// <summary>
        /// Texture filter mode
        /// </summary>
        [FieldOffset(12)]
        public cudaTextureFilterMode filterMode;
        /// <summary>
        /// Texture read mode
        /// </summary>
        [FieldOffset(16)]
        public cudaTextureReadMode readMode;
        /// <summary>
        /// Perform sRGB->linear conversion during texture read
        /// </summary>
        [FieldOffset(20)]
        public int sRGB;
        /// <summary>
        /// Texture Border Color
        /// </summary>
        [FieldOffset(24)]
        public fixed float borderColor[4];
        /// <summary>
        /// Indicates whether texture reads are normalized or not
        /// </summary>
        [FieldOffset(40)]
        public int normalizedCoords;
        /// <summary>
        /// Limit to the anisotropy ratio
        /// </summary>
        [FieldOffset(44)]
        public uint maxAnisotropy;
        /// <summary>
        /// Mipmap filter mode
        /// </summary>
        [FieldOffset(48)]
        public cudaTextureFilterMode mipmapFilterMode;
        /// <summary>
        /// Offset applied to the supplied mipmap level
        /// </summary>
        [FieldOffset(52)]
        public float mipmapLevelBias;
        /// <summary>
        /// Lower end of the mipmap level range to clamp access to
        /// </summary>
        [FieldOffset(56)]
        public float minMipmapLevelClamp;
        /// <summary>
        /// Upper end of the mipmap level range to clamp access to
        /// </summary>
        [FieldOffset(60)]
        public float maxMipmapLevelClamp;
    }
}
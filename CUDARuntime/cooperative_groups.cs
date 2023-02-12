/* (c) ALTIMESH 2019 -- all rights reserved */
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Hybridizer.Runtime.CUDAImports
{
    /// <summary>
    /// global functions from cooperative_groups.h header
    /// </summary>
    [IntrinsicInclude("cooperative_groups.h", Flavor = (int) HybridizerFlavor.CUDA)]
    public struct cooperative_groups
    {
        /// <summary>
        /// Constructs a generic thread_group containing only the calling thread
        /// </summary>
        [IntrinsicFunction("cooperative_groups::this_thread")]
        public static thread_group this_thread() { return new thread_group(); }
        /// <summary>
        /// Constructs a thread_block group
        /// </summary>
        [IntrinsicFunction("cooperative_groups::this_thread_block")]
        public static thread_block this_thread_block() { return new thread_block(); }
        /// <summary>
        /// Constructs a grid_group
        /// </summary>
        [IntrinsicFunction("cooperative_groups::this_grid")]
        public static grid_group this_grid() { return new grid_group(); }
        /// <summary>
        /// 
        /// </summary>
        [IntrinsicFunction("cooperative_groups::coalesced_threads")]
        public static coalesced_group coalesced_threads() { return new coalesced_group(); }

        /// <summary>
        /// syncs a thread block
        /// </summary>
        [IntrinsicFunction("cooperative_groups::sync")]
        public static void sync(thread_block tb) { }


        /// <summary>
        /// syncs a thread group
        /// </summary>
        [IntrinsicFunction("cooperative_groups::sync")]
        public static void sync(thread_group tb) { }

        /// <summary>
        /// syncs current grid
        /// requires cuLaunchCooperativeKernel
        /// </summary>
        /// <param name="tb"></param>
        [IntrinsicFunction("cooperative_groups::sync")]
        public static void sync(grid_group tb) { }

        /// <summary>
        /// syncs a tiled partition
        /// </summary>
        /// <param name="tb"></param>
        [IntrinsicFunction("cooperative_groups::sync")]
        public static void sync(thread_block_tile_1 tb) { }

        /// <summary>
        /// syncs a tiled partition
        /// </summary>
        /// <param name="tb"></param>
        [IntrinsicFunction("cooperative_groups::sync")]
        public static void sync(thread_block_tile_2 tb) { }

        /// <summary>
        /// syncs a tiled partition
        /// </summary>
        /// <param name="tb"></param>
        [IntrinsicFunction("cooperative_groups::sync")]
        public static void sync(thread_block_tile_4 tb) { }

        /// <summary>
        /// syncs a tiled partition
        /// </summary>
        /// <param name="tb"></param>
        [IntrinsicFunction("cooperative_groups::sync")]
        public static void sync(thread_block_tile_8 tb) { }

        /// <summary>
        /// syncs a tiled partition
        /// </summary>
        /// <param name="tb"></param>
        [IntrinsicFunction("cooperative_groups::sync")]
        public static void sync(thread_block_tile_16 tb) { }

        /// <summary>
        /// syncs a tiled partition
        /// </summary>
        /// <param name="tb"></param>
        [IntrinsicFunction("cooperative_groups::sync")]
        public static void sync(thread_block_tile_32 tb) { }


        /// <summary>
        /// The tiled_partition(parent, tilesz) method is a collective operation that
        /// partitions the parent group into a one-dimensional, row-major, tiling of subgroups.
        /// 
        /// A total of ((size(parent)+tilesz-1)/tilesz) subgroups will
        /// be created where threads having identical k = (thread_rank(parent)/tilesz)
        /// will be members of the same subgroup.
        /// 
        /// The implementation may cause the calling thread to wait until all the members
        /// of the parent group have invoked the operation before resuming execution.
        /// 
        /// Functionality is limited to power-of-two sized subgorup instances of at most
        /// 32 threads. Only thread_block, thread_block_tile&lt;&gt;, and their subgroups can be
        /// tiled_partition() in _CG_VERSION 1000.
        /// </summary>
        [IntrinsicFunction("cooperative_groups::tiled_partition")]
        public static thread_group tiled_partition(thread_group parent, uint tilesz)
        {
            return new thread_group();
        }

        /// <summary>
        /// Thread block type overload: returns a basic thread_group for now (may be specialized later)
        /// </summary>
        [IntrinsicFunction("cooperative_groups::tiled_partition")]
        public static thread_group tiled_partition(thread_block parent, uint tilesz)
        {
            return new thread_group();
        }

        /// <summary>
        /// Coalesced group type overload: retains its ability to stay coalesced
        /// </summary>
        [IntrinsicFunction("cooperative_groups::tiled_partition")]
        public static coalesced_group tiled_partition(coalesced_group parent, uint tilesz)
        {
            return new coalesced_group();
        }

        /// <summary>
        ///  The tiled_partition&lt;tilesz&gt;(parent) method is a collective operation that
        ///  partitions the parent group into a one-dimensional, row-major, tiling of subgroups.
        /// 
        ///  A total of ((size(parent)/tilesz) subgroups will be created,
        ///  therefore the parent group size must be evenly divisible by the tilesz.
        ///  The allow parent groups are thread_block or thread_block_tile&lt;size&gt;.
        /// 
        ///  The implementation may cause the calling thread to wait until all the members
        ///  of the parent group have invoked the operation before resuming execution.
        /// 
        ///  Functionality is limited to native hardware sizes, 1/2/4/8/16/32.
        ///  The size(parent) must be greater than the template Size parameter
        ///  otherwise the results are undefined.
        /// </summary>
        [IntrinsicFunction("cooperative_groups::tiled_partition<32>")]
        public static thread_block_tile_32 tile_partition_32(thread_block group) { return new thread_block_tile_32(); }

        /// <summary>
        ///  The tiled_partition&lt;tilesz&gt;(parent) method is a collective operation that
        ///  partitions the parent group into a one-dimensional, row-major, tiling of subgroups.
        /// 
        ///  A total of ((size(parent)/tilesz) subgroups will be created,
        ///  therefore the parent group size must be evenly divisible by the tilesz.
        ///  The allow parent groups are thread_block or thread_block_tile&lt;size&gt;.
        /// 
        ///  The implementation may cause the calling thread to wait until all the members
        ///  of the parent group have invoked the operation before resuming execution.
        /// 
        ///  Functionality is limited to native hardware sizes, 1/2/4/8/16/32.
        ///  The size(parent) must be greater than the template Size parameter
        ///  otherwise the results are undefined.
        /// </summary>
        [IntrinsicFunction("cooperative_groups::tiled_partition<16>")]
        public static thread_block_tile_16 tile_partition_16(thread_block group) { return new thread_block_tile_16(); }

        /// <summary>
        ///  The tiled_partition&lt;tilesz&gt;(parent) method is a collective operation that
        ///  partitions the parent group into a one-dimensional, row-major, tiling of subgroups.
        /// 
        ///  A total of ((size(parent)/tilesz) subgroups will be created,
        ///  therefore the parent group size must be evenly divisible by the tilesz.
        ///  The allow parent groups are thread_block or thread_block_tile&lt;size&gt;.
        /// 
        ///  The implementation may cause the calling thread to wait until all the members
        ///  of the parent group have invoked the operation before resuming execution.
        /// 
        ///  Functionality is limited to native hardware sizes, 1/2/4/8/16/32.
        ///  The size(parent) must be greater than the template Size parameter
        ///  otherwise the results are undefined.
        /// </summary>
        [IntrinsicFunction("cooperative_groups::tiled_partition<8>")]
        public static thread_block_tile_8 tile_partition_8(thread_block group) { return new thread_block_tile_8(); }

        /// <summary>
        ///  The tiled_partition&lt;tilesz&gt;(parent) method is a collective operation that
        ///  partitions the parent group into a one-dimensional, row-major, tiling of subgroups.
        /// 
        ///  A total of ((size(parent)/tilesz) subgroups will be created,
        ///  therefore the parent group size must be evenly divisible by the tilesz.
        ///  The allow parent groups are thread_block or thread_block_tile&lt;size&gt;.
        /// 
        ///  The implementation may cause the calling thread to wait until all the members
        ///  of the parent group have invoked the operation before resuming execution.
        /// 
        ///  Functionality is limited to native hardware sizes, 1/2/4/8/16/32.
        ///  The size(parent) must be greater than the template Size parameter
        ///  otherwise the results are undefined.
        /// </summary>
        [IntrinsicFunction("cooperative_groups::tiled_partition<4>")]
        public static thread_block_tile_4 tile_partition_4(thread_block group) { return new thread_block_tile_4(); }

        /// <summary>
        ///  The tiled_partition&lt;tilesz&gt;(parent) method is a collective operation that
        ///  partitions the parent group into a one-dimensional, row-major, tiling of subgroups.
        /// 
        ///  A total of ((size(parent)/tilesz) subgroups will be created,
        ///  therefore the parent group size must be evenly divisible by the tilesz.
        ///  The allow parent groups are thread_block or thread_block_tile&lt;size&gt;.
        /// 
        ///  The implementation may cause the calling thread to wait until all the members
        ///  of the parent group have invoked the operation before resuming execution.
        /// 
        ///  Functionality is limited to native hardware sizes, 1/2/4/8/16/32.
        ///  The size(parent) must be greater than the template Size parameter
        ///  otherwise the results are undefined.
        /// </summary>
        [IntrinsicFunction("cooperative_groups::tiled_partition<2>")]
        public static thread_block_tile_2 tile_partition_2(thread_block group) { return new thread_block_tile_2(); }

        /// <summary>
        ///  The tiled_partition&lt;tilesz&gt;(parent) method is a collective operation that
        ///  partitions the parent group into a one-dimensional, row-major, tiling of subgroups.
        /// 
        ///  A total of ((size(parent)/tilesz) subgroups will be created,
        ///  therefore the parent group size must be evenly divisible by the tilesz.
        ///  The allow parent groups are thread_block or thread_block_tile&lt;size&gt;.
        /// 
        ///  The implementation may cause the calling thread to wait until all the members
        ///  of the parent group have invoked the operation before resuming execution.
        /// 
        ///  Functionality is limited to native hardware sizes, 1/2/4/8/16/32.
        ///  The size(parent) must be greater than the template Size parameter
        ///  otherwise the results are undefined.
        /// </summary>
        [IntrinsicFunction("cooperative_groups::tiled_partition<1>")]
        public static thread_block_tile_1 tile_partition_1(thread_block group) { return new thread_block_tile_1(); }

        // TODO : tile_partition of thread_block_tile<S>
    }

    /// <summary>
    /// A handle to a group of threads. The handle is only accessible to members of the group it represents.
    /// </summary>
    [IntrinsicType("cooperative_groups::thread_group")]
    [SingleStaticAssignment]
    public struct thread_group
    {
        /// <summary>
        /// Thread index within the group
        /// </summary>
        public uint thread_rank() { return 0; }
        /// <summary>
        /// synchronize threads within the group
        /// </summary>
        public void sync() { }
        /// <summary>
        /// Get the size (total number of threads) of a group
        /// </summary>
        public uint size() { return 1; }
    }

    /// <summary>
    /// Every GPU kernel is executed by a grid of thread blocks, and threads within
    /// each block are guaranteed to reside on the same streaming multiprocessor.
    /// A thread_block represents a thread block whose dimensions are not known until runtime.
    /// </summary>
    [IntrinsicType("cooperative_groups::thread_block")]
    [SingleStaticAssignment]
    public struct thread_block
    {
        /// <summary>
        /// Thread index within the group
        /// </summary>
        public uint thread_rank() { return 0; }
        /// <summary>
        /// synchronize threads within the group
        /// </summary>
        public void sync() { }
        /// <summary>
        /// Get the size (total number of threads) of a group
        /// </summary>
        public uint size() { return 1; }
        /// <summary>
        /// equivalent of blockIdx
        /// </summary>
        public dim3 group_index() { return new dim3(0, 0, 0); }
        /// <summary>
        /// equivalent of threadIdx
        /// </summary>
        public dim3 thread_index() { return new dim3(0, 0, 0); }
        /// <summary>
        /// equivalent of blockDim
        /// </summary>
        public dim3 group_dim() { return new dim3(1, 0, 0); }

        /// <summary>
        /// creates a thread_group
        /// </summary>
        /// <param name="tb"></param>
        [IntrinsicFunction("")]
        public static implicit operator thread_group(thread_block tb) { return new thread_group(); }
    }

    /// <summary>
    /// A group representing the current set of converged threads in a warp.
    /// The size of the group is not guaranteed and it may return a group of
    /// only one thread (itself).
    /// This group exposes warp-synchronous builtins.
    /// </summary>
    [IntrinsicType("cooperative_groups::coalesced_group")]
    [SingleStaticAssignment]
    public struct coalesced_group
    {
        /// <summary>
        /// Thread index within the group
        /// </summary>
        public uint thread_rank() { return 0; }
        /// <summary>
        /// synchronize threads within the group
        /// </summary>
        public void sync() { }
        /// <summary>
        /// Get the size (total number of threads) of a group
        /// </summary>
        public uint size() { return 1; }

        /// <summary>
        /// Direct copy from indexed lane
        /// </summary>
        public int    shfl(int    v, uint rank) { return v; }
        /// <summary>
        /// Direct copy from indexed lane
        /// </summary>
        public uint   shfl(uint   v, uint rank) { return v; }
        /// <summary>
        /// Direct copy from indexed lane
        /// </summary>
        public long   shfl(long   v, uint rank) { return v; }
        /// <summary>
        /// Direct copy from indexed lane
        /// </summary>
        public ulong  shfl(ulong  v, uint rank) { return v; }
        /// <summary>
        /// Direct copy from indexed lane
        /// </summary>
        public float  shfl(float  v, uint rank) { return v; }
        /// <summary>
        /// Direct copy from indexed lane
        /// </summary>
        public double shfl(double v, uint rank) { return v; }

        /// <summary>
        /// Copy from a lane with lower ID relative to caller
        /// </summary>
        public int    shfl_up(int    v, int delta) { return v; }
        /// <summary>
        /// Copy from a lane with lower ID relative to caller
        /// </summary>
        public uint   shfl_up(uint   v, int delta) { return v; }
        /// <summary>
        /// Copy from a lane with lower ID relative to caller
        /// </summary>
        public long   shfl_up(long   v, int delta) { return v; }
        /// <summary>
        /// Copy from a lane with lower ID relative to caller
        /// </summary>
        public ulong  shfl_up(ulong  v, int delta) { return v; }
        /// <summary>
        /// Copy from a lane with lower ID relative to caller
        /// </summary>
        public float  shfl_up(float  v, int delta) { return v; }
        /// <summary>
        /// Copy from a lane with lower ID relative to caller
        /// </summary>
        public double shfl_up(double v, int delta) { return v; }

        /// <summary>
        /// Copy from a lane with higher ID relative to caller
        /// </summary>
        public int    shfl_down(int    v, int delta) { return v; }
        /// <summary>
        /// Copy from a lane with higher ID relative to caller
        /// </summary>
        public uint   shfl_down(uint   v, int delta) { return v; }
        /// <summary>
        /// Copy from a lane with higher ID relative to caller
        /// </summary>
        public long   shfl_down(long   v, int delta) { return v; }
        /// <summary>
        /// Copy from a lane with higher ID relative to caller
        /// </summary>
        public ulong  shfl_down(ulong  v, int delta) { return v; }
        /// <summary>
        /// Copy from a lane with higher ID relative to caller
        /// </summary>
        public float  shfl_down(float  v, int delta) { return v; }
        /// <summary>
        /// Copy from a lane with higher ID relative to caller
        /// </summary>
        public double shfl_down(double v, int delta) { return v; }

        /// <summary>
        /// Evaluate predicate for all non-exited threads and return non-zero if and only if predicate evaluates to non-zero for any of them. 
        /// </summary>
        public int any(int predicate) { return predicate; }
        /// <summary>
        /// Evaluate predicate for all non-exited threads and return non-zero if and only if predicate evaluates to non-zero for any of them. 
        /// </summary>
        public int all(int predicate) { return predicate; }
        /// <summary>
        /// Evaluate predicate for all non-exited threads and return an integer whose Nth bit is set if and only if predicate evaluates to non-zero for the Nth thread of the warp and the Nth thread is active. 
        /// </summary>
        public int ballot(int predicate) { return predicate; }

        // TODO: match_all, match_any
        /// <summary>
        /// creates a thread_group
        /// </summary>
        /// <param name="tb"></param>
        [IntrinsicFunction("")]
        public static implicit operator thread_group(coalesced_group tb) { return new thread_group(); }
    }

    /// <summary>
    /// dotnet representation of cooperative_groups::grid_group
    /// </summary>
    [IntrinsicType("cooperative_groups::grid_group")]
    [SingleStaticAssignment]
    public struct grid_group
    {
        /// <summary>
        /// Thread index within the group
        /// </summary>
        public uint thread_rank() { return 0; }
        /// <summary>
        /// synchronize threads within the group
        /// </summary>
        public void sync() { }
        /// <summary>
        /// Get the size (total number of threads) of a group
        /// </summary>
        public uint size() { return 1; }
        /// <summary>
        /// is valid ?
        /// </summary>
        /// <returns></returns>
        public bool is_valid() { return true; }
        /// <summary>
        /// returns group dimension
        /// </summary>
        /// <returns></returns>
        public dim3 group_dim() { return new dim3(1, 0, 0); }

        /// <summary>
        /// creates a thread_group
        /// </summary>
        /// <param name="tb"></param>
        [IntrinsicFunction("")]
        public static implicit operator thread_group(grid_group tb) { return new thread_group(); }
    }

    // TODO: multi_grid_group
    /// <summary>
    /// a thread block tile of 32 threads
    /// </summary>
    [IntrinsicType("cooperative_groups::thread_block_tile<32>")]
    [SingleStaticAssignment]
    public struct thread_block_tile_32
    {
        /// <summary>
        /// creates a thread group
        /// </summary>
        /// <param name="tb"></param>
        [IntrinsicFunction("")]
        public static implicit operator thread_group(thread_block_tile_32 tb) { return new thread_group(); }
        /// <summary>
        /// Thread index within the group
        /// </summary>
        public uint thread_rank() { return 0; }
        /// <summary>
        /// synchronize threads within the group
        /// </summary>
        public void sync() { }
        /// <summary>
        /// Get the size (total number of threads) of a group
        /// </summary>
        public uint size() { return 1; }

        /// <summary>
        /// Direct copy from indexed lane
        /// </summary>
        public int shfl(int v, uint rank) { return v; }
        /// <summary>
        /// Direct copy from indexed lane
        /// </summary>
        public uint shfl(uint v, uint rank) { return v; }
        /// <summary>
        /// Direct copy from indexed lane
        /// </summary>
        public long shfl(long v, uint rank) { return v; }
        /// <summary>
        /// Direct copy from indexed lane
        /// </summary>
        public ulong shfl(ulong v, uint rank) { return v; }
        /// <summary>
        /// Direct copy from indexed lane
        /// </summary>
        public float shfl(float v, uint rank) { return v; }
        /// <summary>
        /// Direct copy from indexed lane
        /// </summary>
        public double shfl(double v, uint rank) { return v; }

        /// <summary>
        /// Copy from a lane with lower ID relative to caller
        /// </summary>
        public int shfl_up(int v, int delta) { return v; }
        /// <summary>
        /// Copy from a lane with lower ID relative to caller
        /// </summary>
        public uint shfl_up(uint v, int delta) { return v; }
        /// <summary>
        /// Copy from a lane with lower ID relative to caller
        /// </summary>
        public long shfl_up(long v, int delta) { return v; }
        /// <summary>
        /// Copy from a lane with lower ID relative to caller
        /// </summary>
        public ulong shfl_up(ulong v, int delta) { return v; }
        /// <summary>
        /// Copy from a lane with lower ID relative to caller
        /// </summary>
        public float shfl_up(float v, int delta) { return v; }
        /// <summary>
        /// Copy from a lane with lower ID relative to caller
        /// </summary>
        public double shfl_up(double v, int delta) { return v; }

        /// <summary>
        /// Copy from a lane with higher ID relative to caller
        /// </summary>
        public int shfl_down(int v, int delta) { return v; }
        /// <summary>
        /// Copy from a lane with higher ID relative to caller
        /// </summary>
        public uint shfl_down(uint v, int delta) { return v; }
        /// <summary>
        /// Copy from a lane with higher ID relative to caller
        /// </summary>
        public long shfl_down(long v, int delta) { return v; }
        /// <summary>
        /// Copy from a lane with higher ID relative to caller
        /// </summary>
        public ulong shfl_down(ulong v, int delta) { return v; }
        /// <summary>
        /// Copy from a lane with higher ID relative to caller
        /// </summary>
        public float shfl_down(float v, int delta) { return v; }
        /// <summary>
        /// Copy from a lane with higher ID relative to caller
        /// </summary>
        public double shfl_down(double v, int delta) { return v; }

        /// <summary>
        /// Evaluate predicate for all non-exited threads and return non-zero if and only if predicate evaluates to non-zero for any of them. 
        /// </summary>
        public int any(int predicate) { return predicate; }
        /// <summary>
        /// Evaluate predicate for all non-exited threads and return non-zero if and only if predicate evaluates to non-zero for any of them. 
        /// </summary>
        public int all(int predicate) { return predicate; }
        /// <summary>
        /// Evaluate predicate for all non-exited threads and return an integer whose Nth bit is set if and only if predicate evaluates to non-zero for the Nth thread of the warp and the Nth thread is active. 
        /// </summary>
        public int ballot(int predicate) { return predicate; }

        // TODO: match_all, match_any
        /// <summary>
        /// Copy from a lane based on bitwise XOR of own lane ID
        /// </summary>
        public int    shfl_xor(int    v, uint laneMask) { return v; }
        /// <summary>
        /// Copy from a lane based on bitwise XOR of own lane ID
        /// </summary>
        public uint   shfl_xor(uint   v, uint laneMask) { return v; }
        /// <summary>
        /// Copy from a lane based on bitwise XOR of own lane ID
        /// </summary>
        public long   shfl_xor(long   v, uint laneMask) { return v; }
        /// <summary>
        ///Copy from a lane based on bitwise XOR of own lane ID
        /// </summary>
        public ulong  shfl_xor(ulong  v, uint laneMask) { return v; }
        /// <summary>
        /// Copy from a lane based on bitwise XOR of own lane ID
        /// </summary>
        public float  shfl_xor(float  v, uint laneMask) { return v; }
        /// <summary>
        /// Copy from a lane based on bitwise XOR of own lane ID
        /// </summary>
        public double shfl_xor(double v, uint laneMask) { return v; }
    }
    
    /// <summary>
    /// a thread block tile of 16 threads
    /// </summary>
    [IntrinsicType("cooperative_groups::thread_block_tile<16>")]
    [SingleStaticAssignment]
    public struct thread_block_tile_16
    {
        /// <summary>
        /// converts to a thread_group
        /// </summary>
        /// <param name="tb"></param>
        [IntrinsicFunction("")]
        public static implicit operator thread_group(thread_block_tile_16 tb) { return new thread_group(); }
        /// <summary>
        /// Thread index within the group
        /// </summary>
        public uint thread_rank() { return 0; }
        /// <summary>
        /// synchronize threads within the group
        /// </summary>
        public void sync() { }
        /// <summary>
        /// Get the size (total number of threads) of a group
        /// </summary>
        public uint size() { return 1; }

        /// <summary>
        /// Direct copy from indexed lane
        /// </summary>
        public int shfl(int v, uint rank) { return v; }
        /// <summary>
        /// Direct copy from indexed lane
        /// </summary>
        public uint shfl(uint v, uint rank) { return v; }
        /// <summary>
        /// Direct copy from indexed lane
        /// </summary>
        public long shfl(long v, uint rank) { return v; }
        /// <summary>
        /// Direct copy from indexed lane
        /// </summary>
        public ulong shfl(ulong v, uint rank) { return v; }
        /// <summary>
        /// Direct copy from indexed lane
        /// </summary>
        public float shfl(float v, uint rank) { return v; }
        /// <summary>
        /// Direct copy from indexed lane
        /// </summary>
        public double shfl(double v, uint rank) { return v; }

        /// <summary>
        /// Copy from a lane with lower ID relative to caller
        /// </summary>
        public int shfl_up(int v, int delta) { return v; }
        /// <summary>
        /// Copy from a lane with lower ID relative to caller
        /// </summary>
        public uint shfl_up(uint v, int delta) { return v; }
        /// <summary>
        /// Copy from a lane with lower ID relative to caller
        /// </summary>
        public long shfl_up(long v, int delta) { return v; }
        /// <summary>
        /// Copy from a lane with lower ID relative to caller
        /// </summary>
        public ulong shfl_up(ulong v, int delta) { return v; }
        /// <summary>
        /// Copy from a lane with lower ID relative to caller
        /// </summary>
        public float shfl_up(float v, int delta) { return v; }
        /// <summary>
        /// Copy from a lane with lower ID relative to caller
        /// </summary>
        public double shfl_up(double v, int delta) { return v; }

        /// <summary>
        /// Copy from a lane with higher ID relative to caller
        /// </summary>
        public int shfl_down(int v, int delta) { return v; }
        /// <summary>
        /// Copy from a lane with higher ID relative to caller
        /// </summary>
        public uint shfl_down(uint v, int delta) { return v; }
        /// <summary>
        /// Copy from a lane with higher ID relative to caller
        /// </summary>
        public long shfl_down(long v, int delta) { return v; }
        /// <summary>
        /// Copy from a lane with higher ID relative to caller
        /// </summary>
        public ulong shfl_down(ulong v, int delta) { return v; }
        /// <summary>
        /// Copy from a lane with higher ID relative to caller
        /// </summary>
        public float shfl_down(float v, int delta) { return v; }
        /// <summary>
        /// Copy from a lane with higher ID relative to caller
        /// </summary>
        public double shfl_down(double v, int delta) { return v; }

        /// <summary>
        /// Evaluate predicate for all non-exited threads and return non-zero if and only if predicate evaluates to non-zero for any of them. 
        /// </summary>
        public int any(int predicate) { return predicate; }
        /// <summary>
        /// Evaluate predicate for all non-exited threads and return non-zero if and only if predicate evaluates to non-zero for any of them. 
        /// </summary>
        public int all(int predicate) { return predicate; }
        /// <summary>
        /// Evaluate predicate for all non-exited threads and return an integer whose Nth bit is set if and only if predicate evaluates to non-zero for the Nth thread of the warp and the Nth thread is active. 
        /// </summary>
        public int ballot(int predicate) { return predicate; }

        // TODO: match_all, match_any
        /// <summary>
        /// Copy from a lane based on bitwise XOR of own lane ID
        /// </summary>
        public int shfl_xor(int v, uint laneMask) { return v; }
        /// <summary>
        /// Copy from a lane based on bitwise XOR of own lane ID
        /// </summary>
        public uint shfl_xor(uint v, uint laneMask) { return v; }
        /// <summary>
        /// Copy from a lane based on bitwise XOR of own lane ID
        /// </summary>
        public long shfl_xor(long v, uint laneMask) { return v; }
        /// <summary>
        ///Copy from a lane based on bitwise XOR of own lane ID
        /// </summary>
        public ulong shfl_xor(ulong v, uint laneMask) { return v; }
        /// <summary>
        /// Copy from a lane based on bitwise XOR of own lane ID
        /// </summary>
        public float shfl_xor(float v, uint laneMask) { return v; }
        /// <summary>
        /// Copy from a lane based on bitwise XOR of own lane ID
        /// </summary>
        public double shfl_xor(double v, uint laneMask) { return v; }
    }

    /// <summary>
    /// a thread block tile of 8 threads
    /// </summary>
    [IntrinsicType("cooperative_groups::thread_block_tile<8>")]
    [SingleStaticAssignment]
    public struct thread_block_tile_8
    {
        /// <summary>
        /// converts to a thread_group
        /// </summary>
        /// <param name="tb"></param>
        [IntrinsicFunction("")]
        public static implicit operator thread_group(thread_block_tile_8 tb) { return new thread_group(); }
        /// <summary>
        /// Thread index within the group
        /// </summary>
        public uint thread_rank() { return 0; }
        /// <summary>
        /// synchronize threads within the group
        /// </summary>
        public void sync() { }
        /// <summary>
        /// Get the size (total number of threads) of a group
        /// </summary>
        public uint size() { return 1; }

        /// <summary>
        /// Direct copy from indexed lane
        /// </summary>
        public int shfl(int v, uint rank) { return v; }
        /// <summary>
        /// Direct copy from indexed lane
        /// </summary>
        public uint shfl(uint v, uint rank) { return v; }
        /// <summary>
        /// Direct copy from indexed lane
        /// </summary>
        public long shfl(long v, uint rank) { return v; }
        /// <summary>
        /// Direct copy from indexed lane
        /// </summary>
        public ulong shfl(ulong v, uint rank) { return v; }
        /// <summary>
        /// Direct copy from indexed lane
        /// </summary>
        public float shfl(float v, uint rank) { return v; }
        /// <summary>
        /// Direct copy from indexed lane
        /// </summary>
        public double shfl(double v, uint rank) { return v; }

        /// <summary>
        /// Copy from a lane with lower ID relative to caller
        /// </summary>
        public int shfl_up(int v, int delta) { return v; }
        /// <summary>
        /// Copy from a lane with lower ID relative to caller
        /// </summary>
        public uint shfl_up(uint v, int delta) { return v; }
        /// <summary>
        /// Copy from a lane with lower ID relative to caller
        /// </summary>
        public long shfl_up(long v, int delta) { return v; }
        /// <summary>
        /// Copy from a lane with lower ID relative to caller
        /// </summary>
        public ulong shfl_up(ulong v, int delta) { return v; }
        /// <summary>
        /// Copy from a lane with lower ID relative to caller
        /// </summary>
        public float shfl_up(float v, int delta) { return v; }
        /// <summary>
        /// Copy from a lane with lower ID relative to caller
        /// </summary>
        public double shfl_up(double v, int delta) { return v; }

        /// <summary>
        /// Copy from a lane with higher ID relative to caller
        /// </summary>
        public int shfl_down(int v, int delta) { return v; }
        /// <summary>
        /// Copy from a lane with higher ID relative to caller
        /// </summary>
        public uint shfl_down(uint v, int delta) { return v; }
        /// <summary>
        /// Copy from a lane with higher ID relative to caller
        /// </summary>
        public long shfl_down(long v, int delta) { return v; }
        /// <summary>
        /// Copy from a lane with higher ID relative to caller
        /// </summary>
        public ulong shfl_down(ulong v, int delta) { return v; }
        /// <summary>
        /// Copy from a lane with higher ID relative to caller
        /// </summary>
        public float shfl_down(float v, int delta) { return v; }
        /// <summary>
        /// Copy from a lane with higher ID relative to caller
        /// </summary>
        public double shfl_down(double v, int delta) { return v; }

        /// <summary>
        /// Evaluate predicate for all non-exited threads and return non-zero if and only if predicate evaluates to non-zero for any of them. 
        /// </summary>
        public int any(int predicate) { return predicate; }
        /// <summary>
        /// Evaluate predicate for all non-exited threads and return non-zero if and only if predicate evaluates to non-zero for any of them. 
        /// </summary>
        public int all(int predicate) { return predicate; }
        /// <summary>
        /// Evaluate predicate for all non-exited threads and return an integer whose Nth bit is set if and only if predicate evaluates to non-zero for the Nth thread of the warp and the Nth thread is active. 
        /// </summary>
        public int ballot(int predicate) { return predicate; }

        // TODO: match_all, match_any
        /// <summary>
        /// Copy from a lane based on bitwise XOR of own lane ID
        /// </summary>
        public int shfl_xor(int v, uint laneMask) { return v; }
        /// <summary>
        /// Copy from a lane based on bitwise XOR of own lane ID
        /// </summary>
        public uint shfl_xor(uint v, uint laneMask) { return v; }
        /// <summary>
        /// Copy from a lane based on bitwise XOR of own lane ID
        /// </summary>
        public long shfl_xor(long v, uint laneMask) { return v; }
        /// <summary>
        ///Copy from a lane based on bitwise XOR of own lane ID
        /// </summary>
        public ulong shfl_xor(ulong v, uint laneMask) { return v; }
        /// <summary>
        /// Copy from a lane based on bitwise XOR of own lane ID
        /// </summary>
        public float shfl_xor(float v, uint laneMask) { return v; }
        /// <summary>
        /// Copy from a lane based on bitwise XOR of own lane ID
        /// </summary>
        public double shfl_xor(double v, uint laneMask) { return v; }
    }

    /// <summary>
    /// a thread block tile of 4 threads
    /// </summary>
    [IntrinsicType("cooperative_groups::thread_block_tile<4>")]
    [SingleStaticAssignment]
    public struct thread_block_tile_4
    {
        /// <summary>
        /// converts to a thread_group
        /// </summary>
        /// <param name="tb"></param>
        [IntrinsicFunction("")]
        public static implicit operator thread_group(thread_block_tile_4 tb) { return new thread_group(); }
        /// <summary>
        /// Thread index within the group
        /// </summary>
        public uint thread_rank() { return 0; }
        /// <summary>
        /// synchronize threads within the group
        /// </summary>
        public void sync() { }
        /// <summary>
        /// Get the size (total number of threads) of a group
        /// </summary>
        public uint size() { return 1; }

        /// <summary>
        /// Direct copy from indexed lane
        /// </summary>
        public int shfl(int v, uint rank) { return v; }
        /// <summary>
        /// Direct copy from indexed lane
        /// </summary>
        public uint shfl(uint v, uint rank) { return v; }
        /// <summary>
        /// Direct copy from indexed lane
        /// </summary>
        public long shfl(long v, uint rank) { return v; }
        /// <summary>
        /// Direct copy from indexed lane
        /// </summary>
        public ulong shfl(ulong v, uint rank) { return v; }
        /// <summary>
        /// Direct copy from indexed lane
        /// </summary>
        public float shfl(float v, uint rank) { return v; }
        /// <summary>
        /// Direct copy from indexed lane
        /// </summary>
        public double shfl(double v, uint rank) { return v; }

        /// <summary>
        /// Copy from a lane with lower ID relative to caller
        /// </summary>
        public int shfl_up(int v, int delta) { return v; }
        /// <summary>
        /// Copy from a lane with lower ID relative to caller
        /// </summary>
        public uint shfl_up(uint v, int delta) { return v; }
        /// <summary>
        /// Copy from a lane with lower ID relative to caller
        /// </summary>
        public long shfl_up(long v, int delta) { return v; }
        /// <summary>
        /// Copy from a lane with lower ID relative to caller
        /// </summary>
        public ulong shfl_up(ulong v, int delta) { return v; }
        /// <summary>
        /// Copy from a lane with lower ID relative to caller
        /// </summary>
        public float shfl_up(float v, int delta) { return v; }
        /// <summary>
        /// Copy from a lane with lower ID relative to caller
        /// </summary>
        public double shfl_up(double v, int delta) { return v; }

        /// <summary>
        /// Copy from a lane with higher ID relative to caller
        /// </summary>
        public int shfl_down(int v, int delta) { return v; }
        /// <summary>
        /// Copy from a lane with higher ID relative to caller
        /// </summary>
        public uint shfl_down(uint v, int delta) { return v; }
        /// <summary>
        /// Copy from a lane with higher ID relative to caller
        /// </summary>
        public long shfl_down(long v, int delta) { return v; }
        /// <summary>
        /// Copy from a lane with higher ID relative to caller
        /// </summary>
        public ulong shfl_down(ulong v, int delta) { return v; }
        /// <summary>
        /// Copy from a lane with higher ID relative to caller
        /// </summary>
        public float shfl_down(float v, int delta) { return v; }
        /// <summary>
        /// Copy from a lane with higher ID relative to caller
        /// </summary>
        public double shfl_down(double v, int delta) { return v; }

        /// <summary>
        /// Evaluate predicate for all non-exited threads and return non-zero if and only if predicate evaluates to non-zero for any of them. 
        /// </summary>
        public int any(int predicate) { return predicate; }
        /// <summary>
        /// Evaluate predicate for all non-exited threads and return non-zero if and only if predicate evaluates to non-zero for any of them. 
        /// </summary>
        public int all(int predicate) { return predicate; }
        /// <summary>
        /// Evaluate predicate for all non-exited threads and return an integer whose Nth bit is set if and only if predicate evaluates to non-zero for the Nth thread of the warp and the Nth thread is active. 
        /// </summary>
        public int ballot(int predicate) { return predicate; }

        // TODO: match_all, match_any
        /// <summary>
        /// Copy from a lane based on bitwise XOR of own lane ID
        /// </summary>
        public int shfl_xor(int v, uint laneMask) { return v; }
        /// <summary>
        /// Copy from a lane based on bitwise XOR of own lane ID
        /// </summary>
        public uint shfl_xor(uint v, uint laneMask) { return v; }
        /// <summary>
        /// Copy from a lane based on bitwise XOR of own lane ID
        /// </summary>
        public long shfl_xor(long v, uint laneMask) { return v; }
        /// <summary>
        ///Copy from a lane based on bitwise XOR of own lane ID
        /// </summary>
        public ulong shfl_xor(ulong v, uint laneMask) { return v; }
        /// <summary>
        /// Copy from a lane based on bitwise XOR of own lane ID
        /// </summary>
        public float shfl_xor(float v, uint laneMask) { return v; }
        /// <summary>
        /// Copy from a lane based on bitwise XOR of own lane ID
        /// </summary>
        public double shfl_xor(double v, uint laneMask) { return v; }
    }

    /// <summary>
    /// a thread block tile of 2 threads
    /// </summary>
    [IntrinsicType("cooperative_groups::thread_block_tile<2>")]
    [SingleStaticAssignment]
    public struct thread_block_tile_2
    {
        /// <summary>
        /// converts to a thread_group
        /// </summary>
        /// <param name="tb"></param>
        [IntrinsicFunction("")]
        public static implicit operator thread_group(thread_block_tile_2 tb) { return new thread_group(); }
        /// <summary>
        /// Thread index within the group
        /// </summary>
        public uint thread_rank() { return 0; }
        /// <summary>
        /// synchronize threads within the group
        /// </summary>
        public void sync() { }
        /// <summary>
        /// Get the size (total number of threads) of a group
        /// </summary>
        public uint size() { return 1; }

        /// <summary>
        /// Direct copy from indexed lane
        /// </summary>
        public int shfl(int v, uint rank) { return v; }
        /// <summary>
        /// Direct copy from indexed lane
        /// </summary>
        public uint shfl(uint v, uint rank) { return v; }
        /// <summary>
        /// Direct copy from indexed lane
        /// </summary>
        public long shfl(long v, uint rank) { return v; }
        /// <summary>
        /// Direct copy from indexed lane
        /// </summary>
        public ulong shfl(ulong v, uint rank) { return v; }
        /// <summary>
        /// Direct copy from indexed lane
        /// </summary>
        public float shfl(float v, uint rank) { return v; }
        /// <summary>
        /// Direct copy from indexed lane
        /// </summary>
        public double shfl(double v, uint rank) { return v; }

        /// <summary>
        /// Copy from a lane with lower ID relative to caller
        /// </summary>
        public int shfl_up(int v, int delta) { return v; }
        /// <summary>
        /// Copy from a lane with lower ID relative to caller
        /// </summary>
        public uint shfl_up(uint v, int delta) { return v; }
        /// <summary>
        /// Copy from a lane with lower ID relative to caller
        /// </summary>
        public long shfl_up(long v, int delta) { return v; }
        /// <summary>
        /// Copy from a lane with lower ID relative to caller
        /// </summary>
        public ulong shfl_up(ulong v, int delta) { return v; }
        /// <summary>
        /// Copy from a lane with lower ID relative to caller
        /// </summary>
        public float shfl_up(float v, int delta) { return v; }
        /// <summary>
        /// Copy from a lane with lower ID relative to caller
        /// </summary>
        public double shfl_up(double v, int delta) { return v; }

        /// <summary>
        /// Copy from a lane with higher ID relative to caller
        /// </summary>
        public int shfl_down(int v, int delta) { return v; }
        /// <summary>
        /// Copy from a lane with higher ID relative to caller
        /// </summary>
        public uint shfl_down(uint v, int delta) { return v; }
        /// <summary>
        /// Copy from a lane with higher ID relative to caller
        /// </summary>
        public long shfl_down(long v, int delta) { return v; }
        /// <summary>
        /// Copy from a lane with higher ID relative to caller
        /// </summary>
        public ulong shfl_down(ulong v, int delta) { return v; }
        /// <summary>
        /// Copy from a lane with higher ID relative to caller
        /// </summary>
        public float shfl_down(float v, int delta) { return v; }
        /// <summary>
        /// Copy from a lane with higher ID relative to caller
        /// </summary>
        public double shfl_down(double v, int delta) { return v; }

        /// <summary>
        /// Evaluate predicate for all non-exited threads and return non-zero if and only if predicate evaluates to non-zero for any of them. 
        /// </summary>
        public int any(int predicate) { return predicate; }
        /// <summary>
        /// Evaluate predicate for all non-exited threads and return non-zero if and only if predicate evaluates to non-zero for any of them. 
        /// </summary>
        public int all(int predicate) { return predicate; }
        /// <summary>
        /// Evaluate predicate for all non-exited threads and return an integer whose Nth bit is set if and only if predicate evaluates to non-zero for the Nth thread of the warp and the Nth thread is active. 
        /// </summary>
        public int ballot(int predicate) { return predicate; }

        // TODO: match_all, match_any
        /// <summary>
        /// Copy from a lane based on bitwise XOR of own lane ID
        /// </summary>
        public int shfl_xor(int v, uint laneMask) { return v; }
        /// <summary>
        /// Copy from a lane based on bitwise XOR of own lane ID
        /// </summary>
        public uint shfl_xor(uint v, uint laneMask) { return v; }
        /// <summary>
        /// Copy from a lane based on bitwise XOR of own lane ID
        /// </summary>
        public long shfl_xor(long v, uint laneMask) { return v; }
        /// <summary>
        ///Copy from a lane based on bitwise XOR of own lane ID
        /// </summary>
        public ulong shfl_xor(ulong v, uint laneMask) { return v; }
        /// <summary>
        /// Copy from a lane based on bitwise XOR of own lane ID
        /// </summary>
        public float shfl_xor(float v, uint laneMask) { return v; }
        /// <summary>
        /// Copy from a lane based on bitwise XOR of own lane ID
        /// </summary>
        public double shfl_xor(double v, uint laneMask) { return v; }
    }

    /// <summary>
    /// a thread block tile of 1 thread
    /// </summary>
    [IntrinsicType("cooperative_groups::thread_block_tile<1>")]
    [SingleStaticAssignment]
    public struct thread_block_tile_1
    {
        /// <summary>
        /// converts to a thread_group
        /// </summary>
        /// <param name="tb"></param>
        [IntrinsicFunction("")]
        public static implicit operator thread_group(thread_block_tile_1 tb) { return new thread_group(); }
        /// <summary>
        /// Thread index within the group
        /// </summary>
        public uint thread_rank() { return 0; }
        /// <summary>
        /// synchronize threads within the group
        /// </summary>
        public void sync() { }
        /// <summary>
        /// Get the size (total number of threads) of a group
        /// </summary>
        public uint size() { return 1; }

        /// <summary>
        /// Direct copy from indexed lane
        /// </summary>
        public int shfl(int v, uint rank) { return v; }
        /// <summary>
        /// Direct copy from indexed lane
        /// </summary>
        public uint shfl(uint v, uint rank) { return v; }
        /// <summary>
        /// Direct copy from indexed lane
        /// </summary>
        public long shfl(long v, uint rank) { return v; }
        /// <summary>
        /// Direct copy from indexed lane
        /// </summary>
        public ulong shfl(ulong v, uint rank) { return v; }
        /// <summary>
        /// Direct copy from indexed lane
        /// </summary>
        public float shfl(float v, uint rank) { return v; }
        /// <summary>
        /// Direct copy from indexed lane
        /// </summary>
        public double shfl(double v, uint rank) { return v; }

        /// <summary>
        /// Copy from a lane with lower ID relative to caller
        /// </summary>
        public int shfl_up(int v, int delta) { return v; }
        /// <summary>
        /// Copy from a lane with lower ID relative to caller
        /// </summary>
        public uint shfl_up(uint v, int delta) { return v; }
        /// <summary>
        /// Copy from a lane with lower ID relative to caller
        /// </summary>
        public long shfl_up(long v, int delta) { return v; }
        /// <summary>
        /// Copy from a lane with lower ID relative to caller
        /// </summary>
        public ulong shfl_up(ulong v, int delta) { return v; }
        /// <summary>
        /// Copy from a lane with lower ID relative to caller
        /// </summary>
        public float shfl_up(float v, int delta) { return v; }
        /// <summary>
        /// Copy from a lane with lower ID relative to caller
        /// </summary>
        public double shfl_up(double v, int delta) { return v; }

        /// <summary>
        /// Copy from a lane with higher ID relative to caller
        /// </summary>
        public int shfl_down(int v, int delta) { return v; }
        /// <summary>
        /// Copy from a lane with higher ID relative to caller
        /// </summary>
        public uint shfl_down(uint v, int delta) { return v; }
        /// <summary>
        /// Copy from a lane with higher ID relative to caller
        /// </summary>
        public long shfl_down(long v, int delta) { return v; }
        /// <summary>
        /// Copy from a lane with higher ID relative to caller
        /// </summary>
        public ulong shfl_down(ulong v, int delta) { return v; }
        /// <summary>
        /// Copy from a lane with higher ID relative to caller
        /// </summary>
        public float shfl_down(float v, int delta) { return v; }
        /// <summary>
        /// Copy from a lane with higher ID relative to caller
        /// </summary>
        public double shfl_down(double v, int delta) { return v; }

        /// <summary>
        /// Evaluate predicate for all non-exited threads and return non-zero if and only if predicate evaluates to non-zero for any of them. 
        /// </summary>
        public int any(int predicate) { return predicate; }
        /// <summary>
        /// Evaluate predicate for all non-exited threads and return non-zero if and only if predicate evaluates to non-zero for any of them. 
        /// </summary>
        public int all(int predicate) { return predicate; }
        /// <summary>
        /// Evaluate predicate for all non-exited threads and return an integer whose Nth bit is set if and only if predicate evaluates to non-zero for the Nth thread of the warp and the Nth thread is active. 
        /// </summary>
        public int ballot(int predicate) { return predicate; }

        // TODO: match_all, match_any
        /// <summary>
        /// Copy from a lane based on bitwise XOR of own lane ID
        /// </summary>
        public int shfl_xor(int v, uint laneMask) { return v; }
        /// <summary>
        /// Copy from a lane based on bitwise XOR of own lane ID
        /// </summary>
        public uint shfl_xor(uint v, uint laneMask) { return v; }
        /// <summary>
        /// Copy from a lane based on bitwise XOR of own lane ID
        /// </summary>
        public long shfl_xor(long v, uint laneMask) { return v; }
        /// <summary>
        ///Copy from a lane based on bitwise XOR of own lane ID
        /// </summary>
        public ulong shfl_xor(ulong v, uint laneMask) { return v; }
        /// <summary>
        /// Copy from a lane based on bitwise XOR of own lane ID
        /// </summary>
        public float shfl_xor(float v, uint laneMask) { return v; }
        /// <summary>
        /// Copy from a lane based on bitwise XOR of own lane ID
        /// </summary>
        public double shfl_xor(double v, uint laneMask) { return v; }
    }
}
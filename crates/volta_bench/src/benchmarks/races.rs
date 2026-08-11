//! FaialAA data-race benchmarks (Table 7): pre-fix (racy) and post-fix
//! versions of OpenMM and Megatron-LM kernels.
//!
//! Scalar dimensions are baked into these PTX files (they were compiled
//! from specialized sources); array extents here are sized generously and
//! only serve bounds checking. The pre-fix PTX line counts match paper
//! Table 7 exactly (94/132/1190/924/546), so the corpus is the paper's
//! artifact. The three OpenMM kernels run as a single work group
//! (grid (1,1,1); none of their PTX reads `%nctaid`).

use volta_analysis::eval::{AnalysisConfig, ParamValue};

use crate::config::{
    BenchmarkCategory, BenchmarkDef, KernelRun, f32_inout, f32_input, f32_output, u32_index,
    u32_inout, u32_input, u32_output,
};

fn pair(name: &str, file_base: &str, kernel: &str, config: AnalysisConfig) -> [BenchmarkDef; 2] {
    pair_with(name, file_base, kernel, config.clone(), config)
}

fn pair_with(
    name: &str,
    file_base: &str,
    kernel: &str,
    pre_config: AnalysisConfig,
    post_config: AnalysisConfig,
) -> [BenchmarkDef; 2] {
    let pre = KernelRun::new(
        &format!("08_races/{}-pre_racy.ptx", file_base),
        kernel,
        pre_config,
    );
    let post = KernelRun::new(
        &format!("08_races/{}-post_fixed.ptx", file_base),
        kernel,
        post_config,
    );
    [
        BenchmarkDef::race_check(
            format!("{} (pre-fix)", name),
            BenchmarkCategory::DataRace,
            pre,
            true,
        ),
        BenchmarkDef::race_check(
            format!("{} (post-fix)", name),
            BenchmarkCategory::DataRace,
            post,
            false,
        ),
    ]
}

/// `computeBucketPositions(unsigned int* bucketOffset)`: an exclusive-scan
/// over bucket counts; the pre-fix version misses a barrier in the scan.
///
/// 64 threads is deliberate: `numBuckets = 1024` is baked into the .cu, and
/// the pre-fix race is *cross-pass* (pass i+1's `posBuffer[tid]` store vs
/// pass i's post-barrier `posBuffer[blockDim.x-1]` reads), so the outer
/// loop must run more than once, i.e. blockDim.x < 1024 (here: 16 passes).
/// The GPUVerify header's `--blockDim=1024` would run a single pass and
/// mask the race. `bucketOffset` is read and written back over [0..1024)
/// (inout); 1 << 20 is generous bounds slack. Dynamic shared is real
/// (`.extern .shared posBuffer[]`) but touches only blockDim.x u32s =
/// 256 bytes; 64 KiB is a generous uniform figure for this category.
fn bucket_positions() -> AnalysisConfig {
    let mut config = AnalysisConfig::new((64, 1, 1));
    config.arrays = vec![u32_inout("bucketOffset", 0x1_0000_0000, 1 << 20)];
    config.params = vec![ParamValue::ArrayPtr("bucketOffset".to_string())];
    config.dynamic_shared_bytes = 64 * 1024;
    config
}

/// `computeRange(const int* data, int* range)` min/max reduction.
///
/// 64 threads: the .cu bakes `length = 1024`, and its GPUVerify header says
/// `--blockDim=1024`, but the pre-fix missing barrier (between the
/// `rangeBuffer[0]` reads and the `rangeBuffer[tid] = maximum` stores)
/// races at any blockDim >= 2; 64 keeps the run cheap. The .cu's
/// `getValue(v)` macro discards its argument in favor of the `SORT_KEY`
/// module global, and nvcc elides the dead `data[index]` load entirely
/// (param_0 is never `ld.param`ed), so the `data` array is an unread
/// placeholder for the pointer param. Of the three `.global` keys the PTX
/// loads, only `MAX_KEY` gets an (arbitrary) concrete value; `MIN_KEY` and
/// `SORT_KEY` stay unset - they feed pure data values (never an address
/// or branch), which a race check tolerates. `range` gets [0..2) written
/// by thread 0 (output); extents are generous slack. Dynamic shared is
/// real but rangeBuffer touches blockDim.x u32s = 256 bytes << 64 KiB.
fn compute_range() -> AnalysisConfig {
    let mut config = AnalysisConfig::new((64, 1, 1));
    config.arrays = vec![
        u32_input("data", 0x1_0000_0000, 1 << 20),
        u32_output("range", 0x2_0000_0000, 1 << 20),
    ];
    config.params = vec![
        ParamValue::ArrayPtr("data".to_string()),
        ParamValue::ArrayPtr("range".to_string()),
    ];
    config.global_values = vec![("MAX_KEY".to_string(), 32)];
    config.dynamic_shared_bytes = 64 * 1024;
    config
}

/// `computeRMSDPart1(posq, referencePos, particles, buffer)` with the
/// particle count (`numParticles = 1024`) baked into the PTX. `particles`
/// is an index array (`posq[particles[i]]`), so it holds concrete identity
/// indices - Volta needs concrete addresses, and this is exactly the
/// `M[x]` pattern the paper's FaialAA comparison discusses.
///
/// 64 threads: the GPUVerify header says `--blockDim=1024`, but the
/// pre-fix race (`temp[thread] = value` racing the previous
/// `reduceValue`'s post-barrier `temp[0]` reads) fires at any
/// blockDim >= 2; 64 keeps the run cheap. Extents are generous slack:
/// posq/refpos are float4 arrays read at identity indices [0..1024), i.e.
/// f32 elements [0..4096+3); `buffer` is written only ([0..13), by thread
/// 0; inout is conservative - OpenMM treats it as an accumulation
/// buffer). Dynamic shared is real; `temp` touches blockDim.x f32s =
/// 256 bytes << 64 KiB.
fn reduce_value() -> AnalysisConfig {
    let mut config = AnalysisConfig::new((64, 1, 1));
    config.arrays = vec![
        f32_input("posq", 0x1_0000_0000, 1 << 20),
        f32_input("refpos", 0x2_0000_0000, 1 << 20),
        u32_index("particles", 0x3_0000_0000, 1 << 20),
        f32_inout("buffer", 0x4_0000_0000, 1 << 20),
    ];
    config.params = vec![
        ParamValue::ArrayPtr("posq".to_string()),
        ParamValue::ArrayPtr("refpos".to_string()),
        ParamValue::ArrayPtr("particles".to_string()),
        ParamValue::ArrayPtr("buffer".to_string()),
    ];
    config.dynamic_shared_bytes = 64 * 1024;
    config
}

/// `cuApplyLayerNorm(out, mean, invvar, vals, n1, n2, eps, gamma, beta,
/// has_gamma, has_beta)`.
///
/// This Megatron fork is a "small flat test" specialization: the .cu bakes
/// `N1_ROWS = 1`, `N2_COLS = 1`, and `BLK_X = 8`/`BLK_Y = 1` into the
/// index math, and the PTX loads only params 0-3 and 6 (the four pointers
/// and eps) - n1, n2, gamma, beta, has_gamma, and has_beta are declared
/// but never read, so their config values are placeholders and the array
/// extents (sized for the nominal N1 = 2, N2 = 1024 problem) are generous
/// slack. CTA (0,0) touches only vals[0] (read) and out[0]/mean[0]/
/// invvar[0] (written by thread (0,0)).
///
/// One warp per block (y = 1), for both versions - and exactly 32
/// threads, because the Welford reduction uses full-mask
/// `__shfl_sync(0xffffffff, ..)`, which needs all 32 lanes resident.
/// Both compilations of this fork dereference an unassigned
/// `__shared__ U* buf` on the `blockDim.y > 1` inter-warp reduction path,
/// which nvcc compiles to `trap`; with y > 1 the first thread of warp 2
/// traps before any race is reachable. One warp is also all the pre-fix
/// race needs: thread (0,0) writes the `smu`/`sinv` shared broadcast
/// values that every thread then reads with no barrier in between (the
/// `__syncthreads()` the fix commit adds).
///
/// grid.y = 2 reflects the nominal one-CTA-per-row launch; `%nctaid.y` is
/// read only as the row-loop stride, and with N1_ROWS = 1 baked any
/// grid.y >= 1 gives CTA (0,0) exactly row 0. No dynamic shared: neither
/// PTX has `.extern .shared` (`smu`/`sinv` are demoted static `.shared`
/// f32s and `buf` a static shared pointer), so the interpreter would
/// ignore any value here.
fn layer_norm() -> AnalysisConfig {
    const N1: i64 = 2;
    const N2: i64 = 1024;
    let mut config = AnalysisConfig::new((32, 1, 1));
    config.grid_dim = (1, 2, 1);
    config.arrays = vec![
        f32_output("out", 0x1_0000_0000, (N1 * N2) as u64),
        f32_output("mean", 0x2_0000_0000, N1 as u64),
        f32_output("invvar", 0x3_0000_0000, N1 as u64),
        f32_input("vals", 0x4_0000_0000, (N1 * N2) as u64),
        f32_input("gamma", 0x5_0000_0000, N2 as u64),
        f32_input("beta", 0x6_0000_0000, N2 as u64),
    ];
    config.params = vec![
        ParamValue::ArrayPtr("out".to_string()),
        ParamValue::ArrayPtr("mean".to_string()),
        ParamValue::ArrayPtr("invvar".to_string()),
        ParamValue::ArrayPtr("vals".to_string()),
        ParamValue::Int(N1),
        ParamValue::Int(N2),
        ParamValue::Float(1e-5),
        ParamValue::ArrayPtr("gamma".to_string()),
        ParamValue::ArrayPtr("beta".to_string()),
        ParamValue::Int(1),
        ParamValue::Int(1),
    ];
    config
}

/// `cuComputeGradInput(dout, input, mean, invvar, eps, gamma, grad_input)` -
/// the .cu comments n1/n2/has_gamma out of the signature (7 params) and
/// bakes `n1 = n2 = 8`, `has_gamma = 1` ("small param for flat test"; eps
/// is also never `ld.param`ed, so `1e-5` is a placeholder).
///
/// grid.y = 1 << 20 is load-bearing: the PTX reads `%nctaid.y` as the
/// grid-strided row-loop stride, so any grid.y > n1 = 8 makes CTA (0,0)
/// process only row 0; huge documents the intent. The per-version
/// `block_y` is the point of the pair: the kernel's shared reduction
/// indexes `buf[2*threadIdx.x]`, ignoring threadIdx.y, so pre-fix with
/// y = 4 the four y-threads of each x collide on one slot (the reported
/// race); the fixed kernel keeps the flat-1D layout and is only y-safe at
/// y = 1, the layout the fix commit assumes. Extents are generous slack:
/// row 0 means dout/input/gamma are read on [0..8), mean/invvar on
/// [0..1), and grad_input written on [0..8). Dynamic shared is real
/// (`.extern .shared buf[]`) but touches 2*blockDim.x f32s = 256 bytes
/// << 64 KiB.
fn grad_input(block_y: u32) -> AnalysisConfig {
    let mut config = AnalysisConfig::new((32, block_y, 1));
    config.grid_dim = (1, 1 << 20, 1);
    config.arrays = vec![
        f32_input("dout", 0x1_0000_0000, 1 << 22),
        f32_input("input", 0x2_0000_0000, 1 << 22),
        f32_input("mean", 0x3_0000_0000, 1 << 20),
        f32_input("invvar", 0x4_0000_0000, 1 << 20),
        f32_input("gamma", 0x5_0000_0000, 1 << 12),
        f32_output("grad_input", 0x6_0000_0000, 1 << 22),
    ];
    config.params = vec![
        ParamValue::ArrayPtr("dout".to_string()),
        ParamValue::ArrayPtr("input".to_string()),
        ParamValue::ArrayPtr("mean".to_string()),
        ParamValue::ArrayPtr("invvar".to_string()),
        ParamValue::Float(1e-5),
        ParamValue::ArrayPtr("gamma".to_string()),
        ParamValue::ArrayPtr("grad_input".to_string()),
    ];
    config.dynamic_shared_bytes = 64 * 1024;
    config
}

pub fn benchmarks() -> Vec<BenchmarkDef> {
    let mut benches = Vec::new();
    benches.extend(pair(
        "BucketPositions",
        "BucketPositions",
        "_Z22computeBucketPositionsPj",
        bucket_positions(),
    ));
    benches.extend(pair(
        "ComputeRange",
        "ComputeRange",
        "_Z12computeRangePKiPi",
        compute_range(),
    ));
    benches.extend(pair(
        "ReduceValue",
        "ReduceValue",
        "computeRMSDPart1",
        reduce_value(),
    ));
    benches.extend(pair(
        "LayerNorm",
        "LayerNorm",
        "_Z16cuApplyLayerNormIfffEvPT1_PT0_S3_PKT_iiS2_PKS0_S8_ii",
        layer_norm(),
    ));
    benches.extend(pair_with(
        "GradInput",
        "GradInput",
        "_Z18cuComputeGradInputIfffEvPKT1_PKT_PKT0_S8_S6_S2_PS3_",
        grad_input(4),
        grad_input(1),
    ));
    benches
}

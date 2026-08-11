//! TileLang compiler-generated GEMMs (Table 6): reference (`gemm_basic`) vs
//! tensor-core (`gemm_tc`) pairs at three CTA tile sizes. Operands are f16
//! (A, B, and C), matrices are the full ML problem (4096-column row
//! strides); each pair's CTA (0,0) computes the same tile.

use volta_analysis::eval::{AnalysisConfig, ArrayDef, ArrayKind, ParamValue};

use crate::config::{BenchmarkCategory, BenchmarkDef, KernelRun, f16_input};

/// Full matrix extent (row stride observed in the PTX: 8192 bytes = 4096 f16)
const N: u64 = 4096;

const A_BASE: u64 = 0x1_0000_0000;
const B_BASE: u64 = 0x2_0000_0000;
const C_BASE: u64 = 0x3_0000_0000;

/// `main_kernel(A, B, C)` with 128 threads and dynamic shared memory.
///
/// There are no .cu sources for this category (TileLang emits the CUDA and
/// the launch), so every number is read off the PTX: 128 threads from the
/// `.maxntid 128, 1, 1` directive in all six files (paper Table 6 lists
/// (128, 128) per pair); the 4096-element row stride from the constant
/// address math (e.g. the +8192-byte row step in the C-store epilogues).
/// The grid is the 4096^2 launch for the 32x32 CTA tile, (4096/32)^2; the
/// larger tiles would use (128,64)/(64,64), but no PTX reads `%nctaid`
/// and only CTA (0,0) is analyzed, so the shared value is inert.
fn config() -> AnalysisConfig {
    let mut config = AnalysisConfig::new((128, 1, 1));
    config.grid_dim = (128, 128, 1);
    config.arrays = vec![
        f16_input("A", A_BASE, N * N),
        f16_input("B", B_BASE, N * N),
        ArrayDef {
            name: "C".to_string(),
            base: C_BASE,
            elem_width: 2,
            len: N * N,
            kind: ArrayKind::Output,
        },
    ];
    config.params = vec![
        ParamValue::ArrayPtr("A".to_string()),
        ParamValue::ArrayPtr("B".to_string()),
        ParamValue::ArrayPtr("C".to_string()),
    ];
    // TileLang sizes the dynamic shared allocation in its (absent) host
    // launcher, so the exact figure is not recoverable from the corpus.
    // Measured footprints (smallest window each kernel completes under):
    // 4 KiB for 32x32x32, 6 KiB for 64x32x32, 8 KiB for 64x64x32 - the
    // f16 A (M x K) + B (K x N) staging tiles, identical for -ref and
    // -opt. 160 KiB is a deliberately generous ceiling (just under
    // sm_80's 163 KiB opt-in maximum; the PTX targets sm_80); slack only
    // costs bounds-check precision inside the window.
    config.dynamic_shared_bytes = 160 * 1024;
    config
}

pub fn benchmarks() -> Vec<BenchmarkDef> {
    ["32x32x32", "64x32x32", "64x64x32"]
        .into_iter()
        .map(|size| {
            BenchmarkDef::equivalence(
                format!("(TL-{size}-ref, TL-{size}-opt)"),
                BenchmarkCategory::CompilerGenerated,
                KernelRun::new(
                    &format!("07_tilelang/{size}-ref.ptx"),
                    "main_kernel",
                    config(),
                ),
                KernelRun::new(
                    &format!("07_tilelang/{size}-opt.ptx"),
                    "main_kernel",
                    config(),
                ),
            )
        })
        .collect()
}

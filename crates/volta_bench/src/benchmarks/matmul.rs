//! Boehm SGEMM tutorial (Table 2): MatMul-1..7, all checked against
//! MatMul-1. All kernels compute a 64x64 tile of C = alpha*A*B + beta*C
//! with M = N = K = 4096; they differ in tiling/vectorization, which shows
//! up as different thread counts.

use volta_analysis::eval::{AnalysisConfig, ParamValue};

use crate::config::{BenchmarkCategory, BenchmarkDef, KernelRun, f32_inout, f32_input};

const N: u64 = 4096;
const A_BASE: u64 = 0x1_0000_0000;
const B_BASE: u64 = 0x2_0000_0000;
const C_BASE: u64 = 0x3_0000_0000;

/// `sgemm(A, B, C, alpha, beta)`; alpha/beta stay symbolic.
///
/// N = 4096 is the `M`/`N`/`K` macro set in every MatMul-*.cu; A/B are
/// read-only and C is read (`beta * C[..]`) then written, hence inout.
/// The grid is the tutorial's `CEIL_DIV(4096, 64)`-per-axis launch for the
/// 64x64 CTA tile; none of the seven PTX files reads `%nctaid`, so it is
/// documentation only (CTA (0,0) computes C's top-left 64x64 tile).
fn config(threads: u32) -> AnalysisConfig {
    let mut config = AnalysisConfig::new((threads, 1, 1));
    config.grid_dim = (64, 64, 1);
    config.arrays = vec![
        f32_input("A", A_BASE, N * N),
        f32_input("B", B_BASE, N * N),
        f32_inout("C", C_BASE, N * N),
    ];
    config.params = vec![
        ParamValue::ArrayPtr("A".to_string()),
        ParamValue::ArrayPtr("B".to_string()),
        ParamValue::ArrayPtr("C".to_string()),
        ParamValue::SymFloat("alpha".to_string()),
        ParamValue::SymFloat("beta".to_string()),
    ];
    config
}

fn kernel(file: &str, name: &str, threads: u32) -> KernelRun {
    KernelRun::new(&format!("02_matmul/{}", file), name, config(threads))
}

fn matmul1() -> KernelRun {
    // 512 threads = BM*BN/TM = 64*64/8 results per thread (MatMul-1.cu).
    kernel("MatMul-1.ptx", "_Z5sgemmPKfS0_Pfff", 512)
}

pub fn benchmarks() -> Vec<BenchmarkDef> {
    // Thread counts follow each .cu's per-thread tiling over the 64x64 CTA
    // tile: MatMul-2 = 64*64/(TM*TN = 4*2) = 512; MatMul-3/4/5 =
    // 64*64/(8*4) = 128; MatMul-6/7 declare `NUM_THREADS = 256` (warp
    // tiling). Cross-check: the paper's Table 2 block-sync counts are
    // threads x 1024 bar.syncs (2 per K-tile, K/BK = 512 tiles):
    // 524k/524k/131k/131k/131k/262k/262k.
    let defs = [
        ("MatMul-2.ptx", "_Z5sgemmPKfS0_Pfff", 512u32),
        ("MatMul-3.ptx", "_Z5sgemmPfS_S_ff", 128),
        ("MatMul-4.ptx", "_Z5sgemmPfS_S_ff", 128),
        ("MatMul-5.ptx", "_Z5sgemmPfS_S_ff", 128),
        ("MatMul-6.ptx", "_Z5sgemmPKfS0_Pfff", 256),
        ("MatMul-7.ptx", "_Z5sgemmPKfS0_Pfff", 256),
    ];

    let mut benches = vec![BenchmarkDef::equivalence(
        "(MatMul-1, MatMul-1)",
        BenchmarkCategory::MatMul,
        matmul1(),
        matmul1(),
    )];
    for (i, (file, name, threads)) in defs.into_iter().enumerate() {
        benches.push(BenchmarkDef::equivalence(
            format!("(MatMul-1, MatMul-{})", i + 2),
            BenchmarkCategory::MatMul,
            matmul1(),
            kernel(file, name, threads),
        ));
    }
    benches
}

//! Side-by-side comparison of Volta's own decision procedure against Z3 on
//! the same equivalence benchmarks: for each backend, the per-element
//! outcomes and the time spent deciding them. The interesting axis is what
//! each backend can decide (Z3 also reports "unknown", "timeout", and
//! "unsupported"), not just how fast it gets there.
//!
//! Reproduces the paper's section 6.5 / Table 8 methodology: benchmarks
//! whose VCs contain no exponentials are decided by Z3 in milliseconds; the
//! attention benchmarks (exponentials via softmax) come back `unknown`
//! under the default exp encoding, and running them again with the
//! addition-law axiom `forall x y. e^x e^y = e^(x+y)` instead drives Z3
//! past the time budget (`timeout`). Exp-containing benchmarks therefore
//! get a second, `+exp-axiom` result set; the paper used a 10-minute
//! budget per query (`--z3-timeout 600`) and one element per output tensor
//! (`--sample 1`).

use std::path::Path;
use std::time::{Duration, Instant};

use volta_analysis::driver::{EquivCheckOptions, EquivOutcome, check_output_equivalence_with};
use volta_analysis::eval::AnalysisOutput;
use volta_analysis::symbolic::ExprNode;
use volta_z3::ExpMode;

use crate::config::{BenchmarkCategory, BenchmarkDef};
use crate::runner::run_kernel;

/// One benchmark's result under both backends.
#[derive(Debug, Clone)]
pub struct Z3CompareRow {
    pub name: String,
    pub category: BenchmarkCategory,
    /// Symbolic-execution time (both kernels) - identical setup cost for
    /// both backends, included for context, not part of the comparison.
    pub exec_secs: f64,
    pub decision_secs: f64,
    pub decision_status: String,
    /// Z3 solve time under the default exp encoding, summed across all
    /// checked elements.
    pub z3_secs: f64,
    /// Per-outcome Z3 element counts under the default exp encoding.
    pub z3: volta_z3::Z3Counts,
    /// The same pair rerun under `ExpMode::AdditionAxiom` - only for
    /// benchmarks whose VCs actually contain exponentials (the axiom is
    /// vacuous otherwise). `(solve_secs, counts)`.
    pub z3_axiom: Option<(f64, volta_z3::Z3Counts)>,
    /// Set when the row couldn't be produced at all (bad kernel, not an
    /// equivalence benchmark, decision-procedure error, ...); the other
    /// fields are left at their defaults in that case.
    pub error: Option<String>,
}

fn empty_row(def: &BenchmarkDef, error: String) -> Z3CompareRow {
    Z3CompareRow {
        name: def.name.clone(),
        category: def.category,
        exec_secs: 0.0,
        decision_secs: 0.0,
        decision_status: "N/A".to_string(),
        z3_secs: 0.0,
        z3: volta_z3::Z3Counts::default(),
        z3_axiom: None,
        error: Some(error),
    }
}

/// Does any expression reachable from the written outputs contain an
/// `Exp` node? Decides whether the `+exp-axiom` rerun is meaningful.
/// Iterative DFS (attention accumulator chains are deep) over output
/// roots only - dead arena nodes don't count.
fn vc_uses_exp(output: &AnalysisOutput) -> bool {
    let mut visited = vec![false; output.arena.node_count()];
    let mut stack: Vec<_> = output
        .outputs
        .iter()
        .flat_map(|(_, elems)| elems.iter().map(|&(_, root)| root))
        .collect();
    while let Some(id) = stack.pop() {
        let index = id_collections::Id::to_index(id) as usize;
        if std::mem::replace(&mut visited[index], true) {
            continue;
        }
        let node = output.arena.node(id);
        if matches!(node, ExprNode::Exp(_)) {
            return true;
        }
        node.for_each_child(|child| stack.push(child));
    }
    false
}

/// Run one equivalence benchmark through both backends (and, when its VCs
/// contain exponentials, through Z3 a second time with the addition-law
/// axiom). Never panics or aborts a batch: failures (missing optimized
/// kernel, analysis error, decision-procedure error) become
/// `Z3CompareRow::error`, so a caller looping over many benchmarks can
/// keep going.
pub fn compare_one(
    kernels_dir: &Path,
    def: &BenchmarkDef,
    sample: u64,
    verify_numeric: bool,
    recycle_terms: usize,
    z3_timeout: Option<Duration>,
) -> Z3CompareRow {
    let Some(optimized_run) = &def.optimized else {
        return empty_row(
            def,
            "no optimized kernel (not an equivalence benchmark)".to_string(),
        );
    };

    let exec0 = Instant::now();
    let reference = match run_kernel(kernels_dir, &def.reference) {
        Ok(o) => o,
        Err(e) => return empty_row(def, format!("reference kernel: {:#}", e)),
    };
    let optimized = match run_kernel(kernels_dir, optimized_run) {
        Ok(o) => o,
        Err(e) => return empty_row(def, format!("optimized kernel: {:#}", e)),
    };
    let exec_secs = exec0.elapsed().as_secs_f64();
    // Both backends check along the reference config's declared outputs.
    let arrays = def.reference.config.output_array_names();

    let options = EquivCheckOptions {
        sample,
        verify_numeric,
        recycle_terms,
    };
    let d0 = Instant::now();
    let decision_status =
        match check_output_equivalence_with(&reference, &optimized, &arrays, &options) {
            Ok(report) => match report.outcome {
                EquivOutcome::Equivalent => "EQUIV".to_string(),
                EquivOutcome::NotEquivalent { mismatches } => format!("DIFF({})", mismatches.len()),
            },
            Err(e) => return empty_row(def, format!("decision procedure: {}", e)),
        };
    let decision_secs = d0.elapsed().as_secs_f64();

    let run_z3 = |mode: ExpMode| -> Result<(f64, volta_z3::Z3Counts), String> {
        volta_z3::check_output_equivalence(
            &reference, &optimized, &arrays, sample, z3_timeout, mode,
        )
        .map(|report| (report.total_solve_secs(), report.counts()))
        .map_err(|e| format!("z3: {}", e))
    };

    let (z3_secs, z3, error) = match run_z3(ExpMode::PowerBounded) {
        Ok((secs, counts)) => (secs, counts, None),
        Err(e) => (0.0, volta_z3::Z3Counts::default(), Some(e)),
    };

    let z3_axiom = if error.is_none() && (vc_uses_exp(&reference) || vc_uses_exp(&optimized)) {
        run_z3(ExpMode::AdditionAxiom).ok()
    } else {
        None
    };

    Z3CompareRow {
        name: def.name.clone(),
        category: def.category,
        exec_secs,
        decision_secs,
        decision_status,
        z3_secs,
        z3,
        z3_axiom,
        error,
    }
}

pub fn export_json(rows: &[Z3CompareRow], path: &Path) -> anyhow::Result<()> {
    let entries: Vec<_> = rows
        .iter()
        .map(|r| {
            serde_json::json!({
                "name": r.name,
                "category": r.category.name(),
                "exec_secs": r.exec_secs,
                "decision_secs": r.decision_secs,
                "decision_status": r.decision_status,
                "z3_secs": r.z3_secs,
                "z3_equivalent": r.z3.equivalent,
                "z3_not_equivalent": r.z3.not_equivalent,
                "z3_unknown": r.z3.unknown,
                "z3_timeout": r.z3.timeout,
                "z3_unsupported": r.z3.unsupported,
                "z3_error": r.z3.error,
                "z3_axiom": r.z3_axiom.map(|(secs, counts)| {
                    serde_json::json!({
                        "z3_secs": secs,
                        "z3_equivalent": counts.equivalent,
                        "z3_not_equivalent": counts.not_equivalent,
                        "z3_unknown": counts.unknown,
                        "z3_timeout": counts.timeout,
                        "z3_unsupported": counts.unsupported,
                        "z3_error": counts.error,
                    })
                }),
                "error": r.error,
            })
        })
        .collect();
    std::fs::write(path, serde_json::to_string_pretty(&entries)?)?;
    Ok(())
}

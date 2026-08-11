//! Side-by-side comparison of Volta's own decision procedure against Z3 on
//! the same equivalence benchmarks: for each backend, the per-element
//! outcomes and the time spent deciding them. The interesting axis is what
//! each backend can decide (Z3 also reports "unknown", "timeout", and
//! "unsupported"), not just how fast it gets there.
//!
//! Both time columns measure the deciding work only, so they are
//! comparable: the decision column is the summed canon equivalence
//! checks (`EquivSession::check` - VC pairing and the optional
//! `--verify-numeric` oracle excluded); the Z3 column is the in-worker
//! libz3 solve time (query translation and the worker's fixed
//! scaffolding excluded - process spawn/exec plus z3 context/frontend
//! setup is ~10.5ms, several times a whole polynomial-fragment solve).
//! Timeout elements count their full budget (the paper's convention for
//! timeout rows).
//!
//! Both solve phases honor `--iterations`: each iteration re-solves the
//! same sampled elements (fresh decision session per iteration; fresh z3
//! worker per query anyway), and the tables report the median. One Z3
//! carve-out: an element whose iteration-1 outcome is timeout,
//! unsupported, or error is *not* re-solved in later iterations -
//! re-solving a timeout would multiply its full budget into every
//! iteration, and unsupported/error elements never reach the solver -
//! its iteration-1 solve time is charged to every iteration's total
//! instead. Z3 verdict counts always come from iteration 1.
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

use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use volta_analysis::driver::{
    EquivCheckOptions, EquivOutcome, check_output_equivalence_with, paired_elements,
};
use volta_analysis::eval::AnalysisOutput;
use volta_analysis::symbolic::{ExprId, ExprNode};
use volta_z3::{ElementOutcome, ExpMode};

use crate::config::{BenchmarkCategory, BenchmarkDef};
use crate::results::median;
use crate::runner::{persist_vcs, run_kernel};

/// Options for [`compare_one`], mirroring the runner's `RunnerConfig`
/// fields that apply to a two-backend comparison.
#[derive(Debug, Clone)]
pub struct Z3CompareOptions {
    /// Check at most this many output elements per array (0 = all).
    pub sample: u64,
    /// Confirm every decision-procedure verdict with the f64 oracle.
    pub verify_numeric: bool,
    /// Recycle the VC intern tables past this many terms (0 = never).
    pub recycle_terms: usize,
    /// Solve-phase iterations for both backends (see the module docs for
    /// the Z3 carve-out).
    pub iterations: NonZeroUsize,
    /// Per-query Z3 time budget (`None` = no limit).
    pub z3_timeout: Option<Duration>,
    /// Write each benchmark's VC dump under this directory (`None` = skip).
    pub vcs_dir: Option<PathBuf>,
}

/// The `+exp-axiom` rerun's results: per-iteration solve totals and the
/// iteration-1 verdict counts.
#[derive(Debug, Clone)]
pub struct Z3AxiomRerun {
    pub iters_secs: Vec<f64>,
    pub counts: volta_z3::Z3Counts,
}

/// One benchmark's result under both backends.
#[derive(Debug, Clone)]
pub struct Z3CompareRow {
    pub name: String,
    pub category: BenchmarkCategory,
    /// VC-generation time: both kernels' symbolic executions plus
    /// footprint pairing - identical setup cost for both backends,
    /// included for context, not part of the comparison. Dump-file
    /// writing is excluded.
    pub vc_gen_secs: f64,
    /// Decision-procedure time per solve iteration: each entry is one
    /// iteration's summed canon equivalence checks only (excludes VC
    /// pairing and the optional numeric oracle - see
    /// `EquivCheckReport::check_iters`).
    pub decision_iters_secs: Vec<f64>,
    pub decision_status: String,
    /// Z3 solver time per iteration under the default exp encoding, each
    /// entry summed across all checked elements: in-worker libz3 solve
    /// time, excluding worker spawn/exec and translation; timeout
    /// elements count their budget (see `volta_z3::Z3CheckResult`) and,
    /// like unsupported/error elements, are solved in iteration 1 only
    /// (the carve-out; see the module docs).
    pub z3_iters_secs: Vec<f64>,
    /// Per-outcome Z3 element counts under the default exp encoding
    /// (iteration 1's verdicts).
    pub z3: volta_z3::Z3Counts,
    /// The same pair rerun under `ExpMode::AdditionAxiom` - only for
    /// benchmarks whose VCs actually contain exponentials (the axiom is
    /// vacuous otherwise).
    pub z3_axiom: Option<Z3AxiomRerun>,
    /// Where this benchmark's VC dump was written (same path as a default
    /// `all`/`category` run would use - one write covers both).
    pub dump_path: Option<PathBuf>,
    /// Set when the row couldn't be produced at all (bad kernel, not an
    /// equivalence benchmark, decision-procedure error, ...); the other
    /// fields are left at their defaults in that case.
    pub error: Option<String>,
}

impl Z3CompareRow {
    /// Median decision-procedure solve time across iterations.
    pub fn decision_median_secs(&self) -> f64 {
        median(&self.decision_iters_secs).unwrap_or(0.0)
    }

    /// Median Z3 solve time across iterations (default exp encoding).
    pub fn z3_median_secs(&self) -> f64 {
        median(&self.z3_iters_secs).unwrap_or(0.0)
    }
}

fn empty_row(def: &BenchmarkDef, error: String) -> Z3CompareRow {
    Z3CompareRow {
        name: def.name.clone(),
        category: def.category,
        vc_gen_secs: 0.0,
        decision_iters_secs: Vec::new(),
        decision_status: "N/A".to_string(),
        z3_iters_secs: Vec::new(),
        z3: volta_z3::Z3Counts::default(),
        z3_axiom: None,
        dump_path: None,
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

/// The carve-out predicate: outcomes that are solved in iteration 1 only.
/// Re-solving a timeout multiplies its full budget into every iteration;
/// unsupported/error elements never reached the solver at all.
fn carved_out(outcome: &ElementOutcome) -> bool {
    matches!(
        outcome,
        ElementOutcome::Timeout | ElementOutcome::Unsupported(_) | ElementOutcome::Error(_)
    )
}

/// One element's Z3 solve time for a re-solve iteration. Verdicts of
/// later iterations are not compared (z3's unknown/timeout boundary is
/// budget-dependent, unlike the deterministic decision procedure);
/// elements that fail to reach the solver contribute zero, matching
/// iteration 1's accounting for unsupported/error elements.
fn z3_resolve_time(
    reference: &AnalysisOutput,
    r: ExprId,
    optimized: &AnalysisOutput,
    o: ExprId,
    timeout: Option<Duration>,
    mode: ExpMode,
) -> Duration {
    match volta_z3::check_equivalent(&reference.arena, r, &optimized.arena, o, timeout, mode) {
        Ok(res) => res.solve,
        Err(_) => Duration::ZERO,
    }
}

/// Run the Z3 backend over the same sampled elements
/// `volta_z3::check_output_equivalence` would check, `iterations` times,
/// applying the carve-out on re-solves. Returns per-iteration solve
/// totals plus iteration 1's verdict counts.
fn z3_iterations(
    reference: &AnalysisOutput,
    optimized: &AnalysisOutput,
    arrays: &[String],
    sampled: &[(String, u64, ExprId, ExprId)],
    options: &Z3CompareOptions,
    mode: ExpMode,
) -> Result<(Vec<f64>, volta_z3::Z3Counts), String> {
    // Iteration 1: the ordinary full run - per-element verdicts and times.
    let report = volta_z3::check_output_equivalence(
        reference,
        optimized,
        arrays,
        options.sample,
        options.z3_timeout,
        mode,
    )
    .map_err(|e| format!("z3: {}", e))?;
    let counts = report.counts();
    // `check_output_equivalence` samples each array's prefix, exactly like
    // the flattened list we re-solve from.
    debug_assert_eq!(report.elements.len(), sampled.len());

    // Iteration-1 time of the carved-out elements, charged to every
    // iteration's total (timeouts report their budget - see
    // `volta_z3::Z3CheckResult`).
    let carved_secs: f64 = report
        .elements
        .iter()
        .filter(|e| carved_out(&e.outcome))
        .map(|e| e.solve.as_secs_f64())
        .sum();

    let mut iters_secs = vec![report.total_solve_secs()];
    for _ in 1..options.iterations.get() {
        let mut total = carved_secs;
        for (element, &(_, _, r, o)) in report.elements.iter().zip(sampled) {
            if !carved_out(&element.outcome) {
                total += z3_resolve_time(reference, r, optimized, o, options.z3_timeout, mode)
                    .as_secs_f64();
            }
        }
        iters_secs.push(total);
    }
    Ok((iters_secs, counts))
}

/// Run one equivalence benchmark through both backends (and, when its VCs
/// contain exponentials, through Z3 a second time with the addition-law
/// axiom). Also persists the benchmark's VC dump exactly as a default
/// `all`/`category` run would. Never panics or aborts a batch: failures
/// (missing optimized kernel, analysis error, decision-procedure error)
/// become `Z3CompareRow::error`, so a caller looping over many benchmarks
/// can keep going.
pub fn compare_one(
    kernels_dir: &Path,
    def: &BenchmarkDef,
    options: &Z3CompareOptions,
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

    // Persist the VCs (excluded from VC-generation timing).
    let persisted = persist_vcs(options.vcs_dir.as_deref(), &def.name, reference, optimized);
    let (reference, optimized) = (persisted.reference, persisted.optimized);
    let dump_path = persisted.path;

    // Pair the footprints once (the tail of VC generation) and flatten the
    // sampled elements for the Z3 re-solve iterations.
    let pair0 = Instant::now();
    let paired = match paired_elements(&reference, &optimized, &arrays) {
        Ok(p) => p,
        Err(e) => return empty_row(def, format!("pairing footprints: {}", e)),
    };
    let vc_gen_secs = exec_secs + pair0.elapsed().as_secs_f64();
    let mut sampled: Vec<(String, u64, ExprId, ExprId)> = Vec::new();
    for (name, common) in &paired {
        let limit = match options.sample {
            0 => common.len(),
            n => common.len().min(n as usize),
        };
        for &(index, r, o) in common.iter().take(limit) {
            sampled.push((name.clone(), index, r, o));
        }
    }

    let equiv_options = EquivCheckOptions {
        sample: options.sample,
        verify_numeric: options.verify_numeric,
        recycle_terms: options.recycle_terms,
        iterations: options.iterations,
    };
    let (decision_status, decision_iters_secs) =
        match check_output_equivalence_with(&reference, &optimized, &arrays, &equiv_options) {
            Ok(report) => {
                let status = match report.outcome {
                    EquivOutcome::Equivalent => "EQUIV".to_string(),
                    EquivOutcome::NotEquivalent { mismatches } => {
                        format!("DIFF({})", mismatches.len())
                    }
                };
                // The canon checks themselves, not the wall clock around
                // the whole call - see the module docs on comparability.
                let iters: Vec<f64> = report.check_iters.iter().map(|d| d.as_secs_f64()).collect();
                (status, iters)
            }
            Err(e) => return empty_row(def, format!("decision procedure: {}", e)),
        };

    let run_z3 =
        |mode: ExpMode| z3_iterations(&reference, &optimized, &arrays, &sampled, options, mode);

    let (z3_iters_secs, z3, error) = match run_z3(ExpMode::PowerBounded) {
        Ok((iters, counts)) => (iters, counts, None),
        Err(e) => (Vec::new(), volta_z3::Z3Counts::default(), Some(e)),
    };

    let z3_axiom = if error.is_none() && (vc_uses_exp(&reference) || vc_uses_exp(&optimized)) {
        run_z3(ExpMode::AdditionAxiom)
            .ok()
            .map(|(iters_secs, counts)| Z3AxiomRerun { iters_secs, counts })
    } else {
        None
    };

    Z3CompareRow {
        name: def.name.clone(),
        category: def.category,
        vc_gen_secs,
        decision_iters_secs,
        decision_status,
        z3_iters_secs,
        z3,
        z3_axiom,
        dump_path,
        error,
    }
}

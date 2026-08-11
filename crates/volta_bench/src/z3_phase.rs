//! The optional Z3 phase of the benchmark pipeline (`--z3`): after the
//! decision procedure has solved a benchmark's verification conditions,
//! solve the exact same sampled elements with Z3 for a side-by-side
//! capability/timing comparison. The interesting axis is what each
//! backend can decide (Z3 also reports "unknown", "timeout", and
//! "unsupported"), not just how fast it gets there.
//!
//! The two backends' solve times are comparable because both measure the
//! deciding work only: the decision phase is the summed canon equivalence
//! checks (`EquivSession::check` - VC pairing and the optional
//! `--verify-numeric` oracle excluded); the Z3 phase is the in-worker
//! libz3 solve time (query translation and the worker's fixed
//! scaffolding excluded - process spawn/exec plus z3 context/frontend
//! setup is ~10.5ms, several times a whole polynomial-fragment solve).
//! Timeout elements count their full budget (the paper's convention for
//! timeout rows).
//!
//! Like the other timed phases, the Z3 phase honors `--iterations`: each
//! iteration re-solves the same sampled elements (fresh z3 worker per
//! query anyway), with one carve-out: an element whose iteration-1
//! outcome is timeout, unsupported, or error is *not* re-solved in later
//! iterations - re-solving a timeout would multiply its full budget into
//! every iteration, and unsupported/error elements never reach the
//! solver - its iteration-1 solve time is charged to every iteration's
//! total instead. Z3 verdict counts and the per-element results always
//! come from iteration 1.
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
use std::time::Duration;

use volta_analysis::eval::AnalysisOutput;
use volta_analysis::symbolic::{ExprId, ExprNode};
use volta_z3::{ElementOutcome, ElementResult, ExpMode, Z3Counts};

use crate::results::median;

/// Configuration for the Z3 phase, carried inside `RunnerConfig` (`Some`
/// exactly when `--z3` was passed).
#[derive(Debug, Clone)]
pub struct Z3Options {
    /// Per-query Z3 time budget (`None` = no limit).
    pub timeout: Option<Duration>,
}

/// One Z3 run over a benchmark's sampled elements under a single exp
/// encoding: per-iteration solve totals, iteration 1's verdict counts,
/// and iteration 1's per-element results.
#[derive(Debug, Clone)]
pub struct Z3ModeRun {
    /// Z3 solver time per iteration, each entry summed across all checked
    /// elements: in-worker libz3 solve time, excluding worker spawn/exec
    /// and translation; timeout elements count their budget (see
    /// `volta_z3::Z3CheckResult`) and, like unsupported/error elements,
    /// are solved in iteration 1 only (the carve-out; see the module
    /// docs), their iteration-1 time charged to every entry here.
    pub iters_secs: Vec<f64>,
    /// Per-outcome element counts (iteration 1's verdicts).
    pub counts: Z3Counts,
    /// Iteration 1's per-element outcomes and solve times, in
    /// `driver::sampled_elements` order - positionally aligned with the
    /// decision procedure's per-element results.
    pub elements: Vec<ElementResult>,
}

impl Z3ModeRun {
    /// Median solve time across iterations (the table's Z3 column).
    pub fn median_secs(&self) -> f64 {
        median(&self.iters_secs).unwrap_or(0.0)
    }
}

/// The full Z3 phase for one benchmark: the default exp encoding, plus
/// the `+exp-axiom` rerun for benchmarks whose VCs contain exponentials
/// (the axiom is vacuous otherwise).
#[derive(Debug, Clone)]
pub struct Z3Phase {
    /// The default `ExpMode::PowerBounded` encoding.
    pub plain: Z3ModeRun,
    /// The same elements rerun under `ExpMode::AdditionAxiom`; `None`
    /// when the VCs contain no `Exp` node.
    pub axiom: Option<Z3ModeRun>,
}

/// Whether the Z3 phase ran to completion. A `Failed` phase does not
/// invalidate the benchmark's decision verdict (that already happened),
/// but it fails the benchmark's pass/fail: the user asked for a Z3
/// comparison and did not get one.
#[derive(Debug, Clone)]
pub enum Z3PhaseOutcome {
    Ran(Z3Phase),
    Failed(String),
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
/// unsupported/error elements never reached the solver at all. An
/// exhaustive match (mirroring `Z3EquivReport::counts`) so a future
/// `ElementOutcome` variant is a compile error here, not a silent
/// re-solve decision.
fn carved_out(outcome: &ElementOutcome) -> bool {
    match outcome {
        ElementOutcome::Timeout | ElementOutcome::Unsupported(_) | ElementOutcome::Error(_) => true,
        ElementOutcome::Equivalent | ElementOutcome::NotEquivalent | ElementOutcome::Unknown => {
            false
        }
    }
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

/// Run Z3 over the same sampled elements
/// `volta_z3::check_output_equivalence` would check, `iterations` times,
/// applying the carve-out on re-solves. `sampled` is the caller's
/// `driver::sampled_elements` list (borrowed array names).
fn z3_iterations(
    reference: &AnalysisOutput,
    optimized: &AnalysisOutput,
    arrays: &[String],
    sampled: &[(&str, u64, ExprId, ExprId)],
    sample: u64,
    iterations: NonZeroUsize,
    timeout: Option<Duration>,
    mode: ExpMode,
) -> Result<Z3ModeRun, String> {
    // Iteration 1: the ordinary full run - per-element verdicts and times.
    let report =
        volta_z3::check_output_equivalence(reference, optimized, arrays, sample, timeout, mode)
            .map_err(|e| format!("z3: {}", e))?;
    let counts = report.counts();
    // The re-solve loop below pairs `report.elements` with `sampled`
    // positionally. Both lists derive from `driver::sampled_elements`
    // over the same footprints, so they must agree element for element -
    // verified for real (not just in debug builds): a silent mispairing
    // would re-solve the wrong expressions.
    if report.elements.len() != sampled.len()
        || report
            .elements
            .iter()
            .zip(sampled)
            .any(|(e, &(name, index, _, _))| e.array != name || e.index != index)
    {
        return Err(format!(
            "z3 element list ({} elements) does not match the sampled VC list ({}): \
             volta_z3::check_output_equivalence and driver::sampled_elements must \
             sample identically",
            report.elements.len(),
            sampled.len(),
        ));
    }

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
    for _ in 1..iterations.get() {
        let mut total = carved_secs;
        for (element, &(_, _, r, o)) in report.elements.iter().zip(sampled) {
            if !carved_out(&element.outcome) {
                total += z3_resolve_time(reference, r, optimized, o, timeout, mode).as_secs_f64();
            }
        }
        iters_secs.push(total);
    }
    Ok(Z3ModeRun {
        iters_secs,
        counts,
        elements: report.elements,
    })
}

/// Run one benchmark's Z3 phase over the last generation's outputs: the
/// default exp encoding, plus the `+exp-axiom` rerun when the VCs
/// contain exponentials. `sampled` must be `driver::sampled_elements`
/// over the same footprints (the runner derives it from the generation
/// phase's pairing). Never panics: any failure becomes
/// [`Z3PhaseOutcome::Failed`], so a batch run keeps going.
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_z3_phase(
    reference: &AnalysisOutput,
    optimized: &AnalysisOutput,
    arrays: &[String],
    sampled: &[(&str, u64, ExprId, ExprId)],
    sample: u64,
    iterations: NonZeroUsize,
    options: &Z3Options,
) -> Z3PhaseOutcome {
    let run = |mode: ExpMode| {
        z3_iterations(
            reference,
            optimized,
            arrays,
            sampled,
            sample,
            iterations,
            options.timeout,
            mode,
        )
    };
    let plain = match run(ExpMode::PowerBounded) {
        Ok(run) => run,
        Err(e) => return Z3PhaseOutcome::Failed(e),
    };
    let axiom = if vc_uses_exp(reference) || vc_uses_exp(optimized) {
        match run(ExpMode::AdditionAxiom) {
            Ok(run) => Some(run),
            Err(e) => return Z3PhaseOutcome::Failed(format!("+exp-axiom rerun: {}", e)),
        }
    } else {
        None
    };
    Z3PhaseOutcome::Ran(Z3Phase { plain, axiom })
}

//! The one benchmark pipeline: generate VCs (`--iterations` timed runs),
//! persist the last generation's dump, solve with the decision procedure
//! (`--iterations` timed runs), optionally solve with Z3 (`--z3`, same
//! iteration scheme), and record everything.
//!
//! Phase timing:
//!
//! - **VC generation** re-runs everything it takes to produce the
//!   verification conditions from the parsed modules - lowering, both
//!   symbolic executions, and footprint pairing - once per iteration.
//!   Each kernel file is read and parsed once per benchmark, *outside*
//!   the timed loop: file I/O and parsing are not VC generation. The
//!   last iteration's outputs feed the dump file and both solve phases
//!   (earlier ones are dropped before the next starts, so peak memory is
//!   one generation); every later iteration is fingerprint-checked
//!   against iteration 1 (same outcome kind, same per-array footprints,
//!   and same expression identities: arena node count plus per-element
//!   `ExprId`s), so a nondeterministic interpreter regression fails
//!   loudly instead of silently timing different work.
//! - **Decision solve** and the optional **Z3 solve** re-solve the same
//!   sampled elements per iteration (see
//!   `EquivCheckOptions::iterations` and `crate::z3_phase`).
//!
//! Race-check benchmarks have only the generation phase (their whole
//! analysis is the symbolic execution); both solve phases and the dump
//! are skipped, and the Z3 section stays empty even under `--z3`.

use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result, anyhow};
use volta_analysis::driver::{
    AnalysisError, ElementCheckTime, EquivCheckOptions, EquivOutcome, VcDump, VcSnapshot,
    analyze_kernel, check_output_equivalence_with, paired_elements, sampled_elements,
    vc_dump::write_vc_dump,
};
use volta_analysis::eval::{AnalysisOutput, EvalError};
use volta_analysis::symbolic::ExprId;
use volta_frontend::ascii::AsAscii;
use volta_frontend::ast::Module;
use volta_frontend::parse::Parser;

use crate::config::{BenchmarkCategory, BenchmarkDef, ExpectedOutcome, KernelRun};
use crate::results::{cv, median, vc_dump_path};
use crate::z3_phase::{Z3Options, Z3PhaseOutcome, run_z3_phase};

/// A phase's per-iteration coefficient of variation above this prints a
/// noisy-timing warning (see [`warn_noisy_phases`]).
pub const NOISY_CV_THRESHOLD: f64 = 0.10;

/// Statistics collected from a benchmark run
#[derive(Debug, Clone, Default)]
pub struct BenchmarkStats {
    /// VC-generation wall time per iteration, seconds: each entry is one
    /// full generation from the parsed modules - lowering plus symbolic
    /// execution for both kernels (just the reference for race-check
    /// benchmarks) plus footprint pairing (nothing to pair for race-check
    /// and rejected benchmarks). File reading and parsing happen once per
    /// benchmark, outside the timed loop; writing the VC dump file is
    /// excluded too (tracked in `dump_write_secs`). Empty only for
    /// infrastructure failures.
    pub vc_gen_iters_secs: Vec<f64>,
    /// Time writing the `.vcdump` file (once, from the last generation);
    /// `None` when no dump was written.
    pub dump_write_secs: Option<f64>,
    /// Decision-procedure solve time per iteration, seconds: each entry
    /// is one solve iteration's summed canon equivalence checks only
    /// (`EquivCheckReport::check_iters`) - excludes VC pairing and the
    /// optional `--verify-numeric` oracle, so the solve columns report
    /// the same quantity whether or not verification aids are switched
    /// on. Empty for race-check benchmarks and failures.
    pub solve_iters_secs: Vec<f64>,
    /// Time in the `--verify-numeric` f64-oracle confirmations (they run
    /// on solve iteration 1 only); `Some` exactly when the flag was on.
    /// Excluded from `solve_iters_secs` - see
    /// `EquivCheckReport::verify_time`.
    pub verify_numeric_secs: Option<f64>,
    /// Iteration 1's per-element decision-procedure check durations, in
    /// `driver::sampled_elements` order (summing to
    /// `solve_iters_secs[0]`); empty when no solve ran.
    pub decision_elements: Vec<ElementCheckTime>,
    /// bar.sync executions across all threads (optimized kernel if present)
    pub block_syncs: u64,
    /// Warp-level sync executions across all threads
    pub warp_syncs: u64,
    /// Instructions executed (both kernels)
    pub instructions: u64,
    /// Output elements compared
    pub elements_checked: u64,
    /// Output elements in the footprint (>= elements_checked when sampling)
    pub elements_total: u64,
    /// Reference kernel's instructions executed, broken down by kind.
    pub reference_op_counts: std::collections::BTreeMap<&'static str, u64>,
    /// Optimized kernel's instructions executed, broken down by kind.
    /// Empty for race-only benchmarks (no optimized kernel).
    pub optimized_op_counts: std::collections::BTreeMap<&'static str, u64>,
}

impl BenchmarkStats {
    /// Median VC-generation time (the table's "Gen (s)" column); 0 when
    /// nothing ran.
    pub fn vc_gen_median_secs(&self) -> f64 {
        median(&self.vc_gen_iters_secs).unwrap_or(0.0)
    }

    /// Median decision-solve time (the table's "Solve (s)" column); 0
    /// when no solve ran.
    pub fn solve_median_secs(&self) -> f64 {
        median(&self.solve_iters_secs).unwrap_or(0.0)
    }
}

/// Actual outcome of running a benchmark
#[derive(Debug, Clone)]
pub enum ActualOutcome {
    Equivalent,
    NotEquivalent {
        mismatches: usize,
        first: String,
    },
    /// The analysis rejected the kernel (data race, deadlock, or another
    /// soundness error); `is_race` distinguishes true races.
    Rejected {
        description: String,
        is_race: bool,
    },
    RaceFree,
    Error {
        message: String,
    },
}

impl ActualOutcome {
    pub fn matches(&self, expected: ExpectedOutcome) -> bool {
        match (self, expected) {
            (Self::Equivalent, ExpectedOutcome::Equivalent) => true,
            (Self::RaceFree, ExpectedOutcome::RaceFree) => true,
            (Self::Rejected { is_race, .. }, ExpectedOutcome::DataRace) => *is_race,
            _ => false,
        }
    }

    pub fn status(&self) -> &'static str {
        match self {
            Self::Equivalent => "EQUIV",
            Self::NotEquivalent { .. } => "DIFF",
            Self::Rejected { is_race: true, .. } => "RACE",
            Self::Rejected { is_race: false, .. } => "REJECT",
            Self::RaceFree => "OK",
            Self::Error { .. } => "ERR",
        }
    }
}

/// Result of running a single benchmark
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    pub name: String,
    pub category: BenchmarkCategory,
    pub elapsed_secs: f64,
    pub outcome: ActualOutcome,
    pub stats: BenchmarkStats,
    /// The outcome matched the benchmark's expectation (the Z3 phase
    /// plays no part in this).
    pub outcome_matched: bool,
    /// `outcome_matched` and, when `--z3` was on, the Z3 phase ran to
    /// completion. Z3 *verdicts* (unknown/timeout/...) never affect this
    /// - they are the comparison's data, not failures.
    pub passed: bool,
    /// Where this benchmark's VC dump was written (equivalence benchmarks
    /// under a configured `vcs_dir` only).
    pub dump_path: Option<PathBuf>,
    /// The Z3 phase's results: `None` when `--z3` was off or the
    /// benchmark has no solve phase (race checks, rejections, failures).
    pub z3: Option<Z3PhaseOutcome>,
}

/// Benchmark runner configuration
#[derive(Debug, Clone)]
pub struct RunnerConfig {
    /// Base directory for kernel files
    pub kernels_dir: PathBuf,
    pub verbose: bool,
    /// Check at most this many output elements per array (0 = all).
    pub sample: u64,
    /// Confirm every verdict with the f64 numeric oracle.
    pub verify_numeric: bool,
    /// Recycle the VC intern tables past this many terms (0 = never).
    pub recycle_terms: usize,
    /// How many times each timed phase runs (VC generation, decision
    /// solve, and the Z3 solve when enabled); tables report medians.
    pub iterations: NonZeroUsize,
    /// Write each equivalence benchmark's VC dump under this directory
    /// (`None` = don't persist VCs).
    pub vcs_dir: Option<PathBuf>,
    /// Run the Z3 solve phase (`--z3`); `None` = decision procedure only.
    pub z3: Option<Z3Options>,
}

impl Default for RunnerConfig {
    fn default() -> Self {
        Self {
            kernels_dir: PathBuf::from(crate::KERNELS_DIR),
            verbose: false,
            sample: 0,
            verify_numeric: false,
            recycle_terms: volta_analysis::equiv::DEFAULT_RECYCLE_TERMS,
            iterations: NonZeroUsize::MIN,
            vcs_dir: None,
            z3: None,
        }
    }
}

/// An infrastructure failure inside [`BenchmarkRunner::run_inner`],
/// carrying any VC dump written before the failure: the file exists on
/// disk, and the benchmark's record should say so even when the run
/// errors after the dump (e.g. a failed equivalence check).
struct RunFailure {
    error: anyhow::Error,
    dump_path: Option<PathBuf>,
}

impl From<anyhow::Error> for RunFailure {
    /// For `?` on failures that occur before any dump is written: no
    /// path to preserve. Post-dump failure sites attach the path
    /// explicitly instead of using this.
    fn from(error: anyhow::Error) -> Self {
        Self {
            error,
            dump_path: None,
        }
    }
}

/// Everything a successful (in the infrastructure sense) run produces.
struct RunOutput {
    outcome: ActualOutcome,
    stats: BenchmarkStats,
    dump_path: Option<PathBuf>,
    z3: Option<Z3PhaseOutcome>,
}

/// One VC-generation iteration's product.
enum Generated {
    /// The analysis rejected a kernel (data race, deadlock, another
    /// soundness error) - the expected outcome for racy benchmarks.
    Rejected { outcome: ActualOutcome },
    /// A race-check benchmark ran to completion: race-free.
    RaceFree { reference: AnalysisOutput },
    /// An equivalence benchmark's full VCs: both outputs plus the paired
    /// footprints along the reference config's declared output arrays.
    Equivalence {
        reference: AnalysisOutput,
        optimized: AnalysisOutput,
        paired: Vec<(String, Vec<(u64, ExprId, ExprId)>)>,
    },
}

/// One benchmark's kernel file(s), read and parsed once per run (see
/// [`BenchmarkRunner::load_benchmark`]): every generation iteration
/// re-analyzes the same parsed modules. Each module is paired with its
/// `KernelRun` so [`generate`] cannot mix a module up with the wrong
/// launch config.
struct LoadedBenchmark<'d> {
    reference: (&'d KernelRun, Module),
    optimized: Option<(&'d KernelRun, Module)>,
}

/// One VC-generation iteration over the already-parsed modules: lower
/// and run the kernel(s), then pair the footprints. `Err` is an
/// infrastructure failure (lowering, footprint pairing); an analysis
/// *rejection* (race, deadlock, ...) is a `Generated::Rejected` outcome,
/// expected for racy benchmarks.
fn generate(loaded: &LoadedBenchmark) -> Result<Generated> {
    let (reference_run, reference_module) = &loaded.reference;
    let reference = match analyze(reference_module, reference_run)? {
        Ok(output) => output,
        Err(e) => {
            return Ok(Generated::Rejected {
                outcome: rejected_outcome(e),
            });
        }
    };
    let Some((optimized_run, optimized_module)) = &loaded.optimized else {
        // Race-check benchmark: reaching the end means no race.
        return Ok(Generated::RaceFree { reference });
    };
    let optimized = match analyze(optimized_module, optimized_run)? {
        Ok(output) => output,
        Err(e) => {
            return Ok(Generated::Rejected {
                outcome: rejected_outcome(e),
            });
        }
    };
    // Pair along the reference config's declared output arrays - the
    // tail of VC generation, shared by both solve backends.
    let arrays = reference_run.config.output_array_names();
    let paired = paired_elements(&reference, &optimized, &arrays)
        .map_err(|e| anyhow!("pairing footprints: {}", e))?;
    Ok(Generated::Equivalence {
        reference,
        optimized,
        paired,
    })
}

/// Lower and run one kernel from its parsed module, splitting the two
/// failure modes the runner cares about: the outer error is an
/// infrastructure failure (lowering); the inner error is an analysis
/// rejection (race, deadlock, structured-CTA violation, ...), which for
/// a race-check benchmark is itself the expected outcome.
fn analyze(module: &Module, run: &KernelRun) -> Result<Result<AnalysisOutput, EvalError>> {
    match analyze_kernel(module, Some(&run.kernel), run.config.clone()) {
        Ok(output) => Ok(Ok(output)),
        Err(AnalysisError::Eval(e)) => Ok(Err(e)),
        Err(e) => Err(anyhow!("{}: {}", run.path, e)),
    }
}

/// The cheap fingerprint of one generation, kept across iterations
/// (without the arenas) to check that every iteration generated the same
/// thing - footprints *and* expression identities.
///
/// For rejections only the verdict-bearing part is compared - the status
/// (RACE vs REJECT), not the diagnostic text: diagnostic text may embed
/// schedule-dependent details, so verdict kinds are the contract; a
/// rejection *kind* flip would change the benchmark verdict and must
/// fail loudly.
#[derive(PartialEq)]
enum GenShape {
    Rejected {
        status: &'static str,
    },
    RaceFree {
        reference: KernelFingerprint,
    },
    Equivalence {
        reference: KernelFingerprint,
        optimized: KernelFingerprint,
    },
}

/// One kernel's generation fingerprint: the arena's node count plus the
/// full per-array `(index, ExprId)` output lists. Each generation builds
/// a fresh arena deterministically, so identical construction order is
/// equivalent to identical ids - `ExprId` equality across independent
/// arenas is a strong expression-identity check that costs nothing (no
/// rendering, no arena retained).
#[derive(PartialEq)]
struct KernelFingerprint {
    node_count: usize,
    outputs: Vec<(String, Vec<(u64, ExprId)>)>,
}

fn kernel_fingerprint(output: &AnalysisOutput) -> KernelFingerprint {
    KernelFingerprint {
        node_count: output.arena.node_count(),
        outputs: output.outputs.clone(),
    }
}

impl Generated {
    fn shape(&self) -> GenShape {
        match self {
            Self::Rejected { outcome } => GenShape::Rejected {
                status: outcome.status(),
            },
            Self::RaceFree { reference } => GenShape::RaceFree {
                reference: kernel_fingerprint(reference),
            },
            Self::Equivalence {
                reference,
                optimized,
                ..
            } => GenShape::Equivalence {
                reference: kernel_fingerprint(reference),
                optimized: kernel_fingerprint(optimized),
            },
        }
    }
}

impl GenShape {
    fn kind(&self) -> &'static str {
        match self {
            Self::Rejected { .. } => "a rejection",
            Self::RaceFree { .. } => "a race-free completion",
            Self::Equivalence { .. } => "equivalence footprints",
        }
    }
}

/// First difference between one kernel's fingerprints across two
/// generation iterations, as a message fragment; `None` when identical.
fn fingerprint_mismatch(
    kernel: &str,
    a: &KernelFingerprint,
    b: &KernelFingerprint,
) -> Option<String> {
    if a == b {
        return None;
    }
    if a.node_count != b.node_count {
        return Some(format!(
            "{} kernel: built {} vs {} arena nodes",
            kernel, a.node_count, b.node_count
        ));
    }
    for ((an, ae), (bn, be)) in a.outputs.iter().zip(&b.outputs) {
        if an != bn {
            return Some(format!(
                "{} kernel: output array '{}' vs '{}'",
                kernel, an, bn
            ));
        }
        if ae.len() != be.len() {
            return Some(format!(
                "{} kernel: array '{}' wrote {} vs {} elements",
                kernel,
                an,
                ae.len(),
                be.len()
            ));
        }
        for (&(ai, a_expr), &(bi, b_expr)) in ae.iter().zip(be) {
            if ai != bi {
                return Some(format!(
                    "{} kernel: array '{}' wrote element {} vs {}",
                    kernel, an, ai, bi
                ));
            }
            if a_expr != b_expr {
                return Some(format!(
                    "{} kernel: array '{}' element {} built expression {:?} vs {:?}",
                    kernel, an, ai, a_expr, b_expr
                ));
            }
        }
    }
    Some(format!(
        "{} kernel: {} vs {} output arrays",
        kernel,
        a.outputs.len(),
        b.outputs.len()
    ))
}

/// How a later generation iteration's fingerprint disagrees with
/// iteration 1's; `None` when they agree. The interpreter is
/// deterministic, so any disagreement is a bug to fail loudly on, not to
/// time quietly.
fn gen_shape_mismatch(first: &GenShape, later: &GenShape) -> Option<String> {
    match (first, later) {
        (GenShape::Rejected { status: a }, GenShape::Rejected { status: b }) => {
            (a != b).then(|| format!("iteration 1 rejected as {}, this one as {}", a, b))
        }
        (GenShape::RaceFree { reference: a }, GenShape::RaceFree { reference: b }) => {
            fingerprint_mismatch("the", a, b)
        }
        (
            GenShape::Equivalence {
                reference: r1,
                optimized: o1,
            },
            GenShape::Equivalence {
                reference: r2,
                optimized: o2,
            },
        ) => fingerprint_mismatch("reference", r1, r2)
            .or_else(|| fingerprint_mismatch("optimized", o1, o2)),
        _ => Some(format!(
            "iteration 1 produced {}, this one {}",
            first.kind(),
            later.kind()
        )),
    }
}

pub struct BenchmarkRunner {
    config: RunnerConfig,
}

impl BenchmarkRunner {
    pub fn new(config: RunnerConfig) -> Self {
        Self { config }
    }

    pub fn run(&self, def: &BenchmarkDef) -> BenchmarkResult {
        if self.config.vcs_dir.is_some() && def.optimized.is_none() {
            // stderr, like the runner's other progress chatter: it must
            // not panic mid-run when stdout is a closed pipe (`| head`) -
            // the results files still have to be written afterwards.
            eprintln!(
                "note: {}: race-check benchmark (no optimized kernel) - no VC dump",
                def.name
            );
        }
        let start = Instant::now();
        let output = match self.run_inner(def) {
            Ok(output) => output,
            Err(failure) => RunOutput {
                outcome: ActualOutcome::Error {
                    message: format!("{:#}", failure.error),
                },
                stats: BenchmarkStats::default(),
                // A failure after the dump was written (e.g. the
                // equivalence check erroring) still leaves the dump on
                // disk; the record keeps pointing at it.
                dump_path: failure.dump_path,
                z3: None,
            },
        };
        let elapsed_secs = start.elapsed().as_secs_f64();
        let outcome_matched = output.outcome.matches(def.expected);
        let passed = outcome_matched && !matches!(output.z3, Some(Z3PhaseOutcome::Failed(_)));
        let result = BenchmarkResult {
            name: def.name.clone(),
            category: def.category,
            elapsed_secs,
            outcome: output.outcome,
            stats: output.stats,
            outcome_matched,
            passed,
            dump_path: output.dump_path,
            z3: output.z3,
        };
        warn_noisy_phases(&result);
        result
    }

    pub fn run_all(&self, defs: &[BenchmarkDef]) -> Vec<BenchmarkResult> {
        defs.iter()
            .map(|def| {
                if self.config.verbose {
                    eprintln!("running {} ...", def.name);
                }
                let result = self.run(def);
                if self.config.verbose {
                    eprintln!(
                        "  -> {} in {:.1}s",
                        result.outcome.status(),
                        result.elapsed_secs
                    );
                }
                result
            })
            .collect()
    }

    /// Read and parse the benchmark's kernel file(s) - once per run,
    /// before the timed generation loop: file I/O and parsing are not
    /// part of VC generation (lowering is; it happens inside
    /// `analyze_kernel`, per iteration).
    fn load_benchmark<'d>(&self, def: &'d BenchmarkDef) -> Result<LoadedBenchmark<'d>> {
        let load = |run: &'d KernelRun| {
            load_module(&self.config.kernels_dir.join(&run.path)).map(|module| (run, module))
        };
        Ok(LoadedBenchmark {
            reference: load(&def.reference)?,
            optimized: def.optimized.as_ref().map(load).transpose()?,
        })
    }

    fn run_inner(&self, def: &BenchmarkDef) -> Result<RunOutput, RunFailure> {
        let mut stats = BenchmarkStats::default();

        // Kernel files are read and parsed once, outside the timed
        // generation loop below - file I/O and parsing are not part of
        // the VC-generation phase.
        let loaded = self.load_benchmark(def)?;

        // --- VC generation: `iterations` timed runs. Only the last
        // one's outputs are kept (dropping the previous before the next
        // starts, so peak memory is a single generation); every later
        // iteration's fingerprint (outcome kind, footprints, expression
        // identities) must match iteration 1's.
        let mut first_shape: Option<GenShape> = None;
        let mut last: Option<Generated> = None;
        for iteration in 1..=self.config.iterations.get() {
            drop(last.take());
            let gen_start = Instant::now();
            let generated = generate(&loaded)?;
            stats
                .vc_gen_iters_secs
                .push(gen_start.elapsed().as_secs_f64());
            let shape = generated.shape();
            match &first_shape {
                None => first_shape = Some(shape),
                Some(first) => {
                    if let Some(diff) = gen_shape_mismatch(first, &shape) {
                        return Err(anyhow!(
                            "VC generation is nondeterministic: iteration {} disagrees \
                             with iteration 1: {}",
                            iteration,
                            diff
                        )
                        .into());
                    }
                }
            }
            last = Some(generated);
        }

        let (reference, optimized, paired) = match last.expect("iterations >= 1") {
            Generated::Rejected { outcome } => {
                return Ok(RunOutput {
                    outcome,
                    stats,
                    dump_path: None,
                    z3: None,
                });
            }
            Generated::RaceFree { reference } => {
                stats.instructions = reference.stats.instructions;
                stats.block_syncs = reference.stats.block_syncs;
                stats.warp_syncs = reference.stats.warp_syncs;
                stats.reference_op_counts = reference.op_counts.clone();
                return Ok(RunOutput {
                    outcome: ActualOutcome::RaceFree,
                    stats,
                    dump_path: None,
                    z3: None,
                });
            }
            Generated::Equivalence {
                reference,
                optimized,
                paired,
            } => (reference, optimized, paired),
        };

        // Execution counters from the last generation (every iteration
        // executes identically; the fingerprint check above guards the
        // footprint-and-expression part of that). The paper's tables
        // report the optimized kernel's sync counts.
        stats.instructions = reference.stats.instructions + optimized.stats.instructions;
        stats.block_syncs = optimized.stats.block_syncs;
        stats.warp_syncs = optimized.stats.warp_syncs;
        stats.reference_op_counts = reference.op_counts.clone();
        stats.optimized_op_counts = optimized.op_counts.clone();

        // Persist the last generation's verification conditions (the
        // write itself is timed into `dump_write_secs`, not the
        // generation iterations).
        let persisted = persist_vcs(
            self.config.vcs_dir.as_deref(),
            &def.name,
            reference,
            optimized,
        );
        stats.dump_write_secs = persisted.write_secs;
        let (reference, optimized) = (persisted.reference, persisted.optimized);

        // --- Decision solve: `iterations` runs over the same sampled
        // elements. A failure past this point happens *after* the dump
        // was written, so it carries the dump path.
        let arrays = def.reference.config.output_array_names();
        let outcome = self
            .check_equivalence(&reference, &optimized, &arrays, &mut stats)
            .map_err(|error| RunFailure {
                error,
                dump_path: persisted.path.clone(),
            })?;

        // --- Z3 solve (optional): the exact same sampled elements.
        let z3 = self.config.z3.as_ref().map(|options| {
            let sampled = sampled_elements(&paired, self.config.sample);
            run_z3_phase(
                &reference,
                &optimized,
                &arrays,
                &sampled,
                self.config.sample,
                self.config.iterations,
                options,
            )
        });

        Ok(RunOutput {
            outcome,
            stats,
            dump_path: persisted.path,
            z3,
        })
    }

    /// Compare the two outputs element for element along the named
    /// arrays. The actual element loop lives in `volta_analysis::driver`.
    fn check_equivalence(
        &self,
        reference: &AnalysisOutput,
        optimized: &AnalysisOutput,
        arrays: &[String],
        stats: &mut BenchmarkStats,
    ) -> Result<ActualOutcome> {
        let options = EquivCheckOptions {
            sample: self.config.sample,
            verify_numeric: self.config.verify_numeric,
            recycle_terms: self.config.recycle_terms,
            iterations: self.config.iterations,
        };
        let report = check_output_equivalence_with(reference, optimized, arrays, &options)
            .context("checking output equivalence")?;
        stats.elements_checked = report.elements_checked;
        stats.elements_total = report.elements_total;
        stats.solve_iters_secs = report.check_iters.iter().map(|d| d.as_secs_f64()).collect();
        stats.verify_numeric_secs = report.verify_time.map(|d| d.as_secs_f64());
        stats.decision_elements = report.element_checks;
        Ok(match report.outcome {
            EquivOutcome::Equivalent => ActualOutcome::Equivalent,
            EquivOutcome::NotEquivalent { mismatches } => {
                let first = mismatches
                    .first()
                    .map(|m| format!("{}[{}]", m.array, m.index))
                    .unwrap_or_default();
                ActualOutcome::NotEquivalent {
                    mismatches: mismatches.len(),
                    first,
                }
            }
        })
    }
}

/// Print a stderr warning for every timed phase whose per-iteration
/// coefficient of variation exceeds [`NOISY_CV_THRESHOLD`]: the median
/// is still the headline number, but the reader should know it came from
/// noisy samples.
fn warn_noisy_phases(result: &BenchmarkResult) {
    let mut phases: Vec<(&str, &[f64])> = vec![
        ("VC generation", &result.stats.vc_gen_iters_secs),
        ("decision solve", &result.stats.solve_iters_secs),
    ];
    if let Some(Z3PhaseOutcome::Ran(phase)) = &result.z3 {
        phases.push(("z3 solve", &phase.plain.iters_secs));
        if let Some(axiom) = &phase.axiom {
            phases.push(("z3 +exp-axiom solve", &axiom.iters_secs));
        }
    }
    for (phase, iters) in phases {
        if let Some(cv) = cv(iters)
            && cv > NOISY_CV_THRESHOLD
        {
            eprintln!(
                "warning: {}: {} timing noisy (CV {:.2} > {:.2}); \
                 consider more iterations or a quieter machine",
                result.name, phase, cv, NOISY_CV_THRESHOLD
            );
        }
    }
}

/// The result of [`persist_vcs`]: the analysis outputs (moved through the
/// dump, never cloned - the arenas can be GiB-scale) plus what got written.
struct PersistedVcs {
    reference: AnalysisOutput,
    optimized: AnalysisOutput,
    /// The `.vcdump` path, when a dump was written.
    path: Option<PathBuf>,
    /// Time spent writing it, when a dump was written.
    write_secs: Option<f64>,
}

/// Persist one equivalence benchmark's verification conditions to
/// `<vcs_dir>/<sanitized-name>.vcdump` via the shared
/// `volta_analysis::driver::vc_dump` format (the same file `volta compare
/// --dump-vcs` writes and `--from-dump` reads), overwriting any previous
/// run's dump - VCs are deterministic (and the generation phase's
/// fingerprint check enforces the footprint-and-expression-identity part
/// of that per run). A write failure
/// warns and carries on: a full disk should not change a benchmark
/// verdict.
///
/// The outputs are moved into the dump and moved back out
/// (`into_analysis_output`), which clears their `stats`/`op_counts` -
/// callers must record those before calling. Byte-identity across runs
/// rests on one premise: no production code path creates machine symbols
/// (`ExprArena::symbol`, the only id drawn from a process-global
/// counter), so every id in a dump is deterministic; a future
/// machine-symbol caller would void byte-identity across runs but not
/// the dumps' validity - `--from-dump` never depends on the numeric id
/// values.
fn persist_vcs(
    vcs_dir: Option<&Path>,
    benchmark_name: &str,
    reference: AnalysisOutput,
    optimized: AnalysisOutput,
) -> PersistedVcs {
    let Some(vcs_dir) = vcs_dir else {
        return PersistedVcs {
            reference,
            optimized,
            path: None,
            write_secs: None,
        };
    };
    let path = vc_dump_path(vcs_dir, benchmark_name);
    let dump = VcDump {
        reference: VcSnapshot::from_output(reference),
        optimized: VcSnapshot::from_output(optimized),
    };
    // The directory is a one-time setup cost, not part of any dump's
    // write time - create it before starting the write timer.
    let created = std::fs::create_dir_all(vcs_dir);
    let write0 = Instant::now();
    let written = created.and_then(|_| write_vc_dump(&path, &dump));
    let (path, write_secs) = match written {
        Ok(()) => (Some(path), Some(write0.elapsed().as_secs_f64())),
        Err(e) => {
            eprintln!("warning: could not write VC dump {}: {}", path.display(), e);
            (None, None)
        }
    };
    PersistedVcs {
        reference: dump.reference.into_analysis_output(),
        optimized: dump.optimized.into_analysis_output(),
        path,
        write_secs,
    }
}

fn rejected_outcome(e: EvalError) -> ActualOutcome {
    let is_race = matches!(e, EvalError::DataRace { .. });
    ActualOutcome::Rejected {
        description: e.to_string(),
        is_race,
    }
}

#[cfg(test)]
mod tests {
    use id_collections::Id;

    use super::*;

    fn fp(node_count: usize, elems: &[(u64, u32)]) -> KernelFingerprint {
        KernelFingerprint {
            node_count,
            outputs: vec![(
                "out".to_string(),
                elems
                    .iter()
                    .map(|&(i, id)| (i, ExprId::from_index(id)))
                    .collect(),
            )],
        }
    }

    #[test]
    fn identical_fingerprints_agree() {
        let a = fp(7, &[(0, 3), (1, 5)]);
        let b = fp(7, &[(0, 3), (1, 5)]);
        assert_eq!(fingerprint_mismatch("reference", &a, &b), None);
    }

    #[test]
    fn node_count_divergence_is_named() {
        let a = fp(7, &[(0, 3)]);
        let b = fp(8, &[(0, 3)]);
        let msg = fingerprint_mismatch("reference", &a, &b).unwrap();
        assert!(msg.contains("7 vs 8 arena nodes"), "{}", msg);
    }

    #[test]
    fn expression_identity_divergence_is_named() {
        // Same footprint indices, different ExprIds: exactly the case
        // the pre-fingerprint shape check let slip through.
        let a = fp(7, &[(0, 3), (1, 5)]);
        let b = fp(7, &[(0, 3), (1, 6)]);
        let msg = fingerprint_mismatch("optimized", &a, &b).unwrap();
        assert!(
            msg.contains("array 'out' element 1 built expression"),
            "{}",
            msg
        );
    }

    #[test]
    fn footprint_index_divergence_is_named() {
        let a = fp(7, &[(0, 3)]);
        let b = fp(7, &[(2, 3)]);
        let msg = fingerprint_mismatch("the", &a, &b).unwrap();
        assert!(msg.contains("wrote element 0 vs 2"), "{}", msg);
    }

    #[test]
    fn rejection_kind_flip_is_named_but_text_is_not_compared() {
        // Rejections compare by verdict kind only: diagnostic text may
        // embed schedule-dependent details.
        let race = GenShape::Rejected { status: "RACE" };
        let reject = GenShape::Rejected { status: "REJECT" };
        assert_eq!(
            gen_shape_mismatch(&race, &GenShape::Rejected { status: "RACE" }),
            None
        );
        let msg = gen_shape_mismatch(&race, &reject).unwrap();
        assert!(msg.contains("RACE") && msg.contains("REJECT"), "{}", msg);
    }
}

/// Load and parse a PTX module.
pub fn load_module(path: &Path) -> Result<Module> {
    let contents = std::fs::read_to_string(path)
        .with_context(|| format!("failed to read {}", path.display()))?;
    let ascii_src = contents
        .as_bytes()
        .as_ascii_slice()
        .context("file contains non-ASCII characters")?;
    let mut parser = Parser::new(ascii_src);
    parser
        .parse_module()
        .map_err(|e| anyhow!("parse error: {}", e.error.title()))
}

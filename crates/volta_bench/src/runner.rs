//! Benchmark execution: parse, lower, symbolically execute, persist the
//! verification conditions, and (for equivalence benchmarks) check them.

use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result, anyhow};
use volta_analysis::driver::{
    AnalysisError, EquivCheckOptions, EquivOutcome, VcDump, VcSnapshot, analyze_kernel,
    check_output_equivalence_with, vc_dump::write_vc_dump,
};
use volta_analysis::eval::{AnalysisOutput, EvalError, Stats};
use volta_frontend::ascii::AsAscii;
use volta_frontend::ast::Module;
use volta_frontend::parse::Parser;

use crate::config::{BenchmarkCategory, BenchmarkDef, ExpectedOutcome, KernelRun};
use crate::results::{median, vc_dump_path};

/// Statistics collected from a benchmark run
#[derive(Debug, Clone, Default)]
pub struct BenchmarkStats {
    /// Symbolic-execution wall time (both kernels), seconds
    pub exec_secs: f64,
    /// VC-generation wall time, seconds: the two symbolic executions
    /// (`exec_secs`) plus footprint pairing (`paired_elements`). Writing
    /// the VC dump file is excluded (tracked in `dump_write_secs`). For
    /// race-check and rejected benchmarks there is nothing to pair, so
    /// this equals `exec_secs`.
    pub vc_gen_secs: f64,
    /// Time writing the `.vcdump` file; `None` when no dump was written.
    pub dump_write_secs: Option<f64>,
    /// VC-solving time per iteration, seconds: each entry is one solve
    /// iteration's summed canon equivalence checks only
    /// (`EquivCheckReport::check_iters`) - excludes VC pairing and the
    /// optional `--verify-numeric` oracle, so the solve columns report
    /// the same quantity whether or not verification aids are switched
    /// on. Empty for race-check benchmarks and failures.
    pub solve_iters_secs: Vec<f64>,
    /// Median of `solve_iters_secs` (the table's "VC (s)" column); 0 when
    /// no solve ran.
    pub vc_secs: f64,
    /// Time in the `--verify-numeric` f64-oracle confirmations (they run
    /// on solve iteration 1 only); `Some` exactly when the flag was on.
    /// Excluded from `solve_iters_secs` - see
    /// `EquivCheckReport::verify_time`.
    pub verify_numeric_secs: Option<f64>,
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
    pub passed: bool,
    /// Where this benchmark's VC dump was written (equivalence benchmarks
    /// under a configured `vcs_dir` only).
    pub dump_path: Option<PathBuf>,
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
    /// Solve-phase iterations (see `EquivCheckOptions::iterations`): the
    /// per-element check loop runs this many times, each from a fresh
    /// session; the table reports the median.
    pub iterations: NonZeroUsize,
    /// Write each equivalence benchmark's VC dump under this directory
    /// (`None` = don't persist VCs).
    pub vcs_dir: Option<PathBuf>,
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
        let (outcome, stats, dump_path) = match self.run_inner(def) {
            Ok((outcome, stats, dump_path)) => (outcome, stats, dump_path),
            Err(failure) => (
                ActualOutcome::Error {
                    message: format!("{:#}", failure.error),
                },
                BenchmarkStats::default(),
                // A failure after the dump was written (e.g. the
                // equivalence check erroring) still leaves the dump on
                // disk; the record keeps pointing at it.
                failure.dump_path,
            ),
        };
        let elapsed_secs = start.elapsed().as_secs_f64();
        let passed = outcome.matches(def.expected);
        BenchmarkResult {
            name: def.name.clone(),
            category: def.category,
            elapsed_secs,
            outcome,
            stats,
            passed,
            dump_path,
        }
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

    fn run_inner(
        &self,
        def: &BenchmarkDef,
    ) -> Result<(ActualOutcome, BenchmarkStats, Option<PathBuf>), RunFailure> {
        let mut stats = BenchmarkStats::default();

        // Analyze the reference kernel.
        let exec0 = Instant::now();
        let reference = match self.analyze(&def.reference)? {
            Ok(output) => output,
            Err(e) => {
                stats.exec_secs = exec0.elapsed().as_secs_f64();
                stats.vc_gen_secs = stats.exec_secs;
                return Ok((rejected_outcome(e), stats, None));
            }
        };
        record_exec_stats(&mut stats, reference.stats);
        stats.reference_op_counts = reference.op_counts.clone();

        let Some(optimized_run) = &def.optimized else {
            // Race-check benchmark: reaching the end means no race.
            stats.exec_secs = exec0.elapsed().as_secs_f64();
            stats.vc_gen_secs = stats.exec_secs;
            return Ok((ActualOutcome::RaceFree, stats, None));
        };

        // Analyze the optimized kernel.
        let optimized = match self.analyze(optimized_run)? {
            Ok(output) => output,
            Err(e) => {
                stats.exec_secs = exec0.elapsed().as_secs_f64();
                stats.vc_gen_secs = stats.exec_secs;
                return Ok((rejected_outcome(e), stats, None));
            }
        };
        // Report the optimized kernel's sync counts (the paper's tables).
        stats.block_syncs = optimized.stats.block_syncs;
        stats.warp_syncs = optimized.stats.warp_syncs;
        stats.instructions += optimized.stats.instructions;
        stats.optimized_op_counts = optimized.op_counts.clone();
        stats.exec_secs = exec0.elapsed().as_secs_f64();
        stats.vc_gen_secs = stats.exec_secs;

        // Persist the verification conditions (excluded from VC-generation
        // timing; the write itself is timed into `dump_write_secs`).
        let persisted = persist_vcs(
            self.config.vcs_dir.as_deref(),
            &def.name,
            reference,
            optimized,
        );
        stats.dump_write_secs = persisted.write_secs;
        let (reference, optimized) = (persisted.reference, persisted.optimized);

        // Check the verification conditions along the reference config's
        // declared output arrays (`check_equivalence` fills in the
        // solve times and completes `vc_gen_secs` with the pairing time).
        // A failure past this point happens *after* the dump was written,
        // so it carries the dump path.
        let arrays = def.reference.config.output_array_names();
        let outcome = self
            .check_equivalence(&reference, &optimized, &arrays, &mut stats)
            .map_err(|error| RunFailure {
                error,
                dump_path: persisted.path.clone(),
            })?;
        Ok((outcome, stats, persisted.path))
    }

    /// Run one kernel, splitting the two failure modes the runner cares
    /// about: the outer error is an infrastructure failure (I/O, parse,
    /// lowering); the inner error is an analysis rejection (race, deadlock,
    /// structured-CTA violation, ...), which for a race-check benchmark is
    /// itself the expected outcome. Callers that don't need that
    /// distinction (e.g. `z3_compare`) use the flat [`run_kernel`] instead.
    fn analyze(&self, run: &KernelRun) -> Result<Result<AnalysisOutput, EvalError>> {
        let path = self.config.kernels_dir.join(&run.path);
        let module = load_module(&path)?;
        match analyze_kernel(&module, Some(&run.kernel), run.config.clone()) {
            Ok(output) => Ok(Ok(output)),
            Err(AnalysisError::Eval(e)) => Ok(Err(e)),
            Err(e) => Err(anyhow!("{}: {}", run.path, e)),
        }
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
        stats.vc_gen_secs = stats.exec_secs + report.pair_time.as_secs_f64();
        stats.solve_iters_secs = report.check_iters.iter().map(|d| d.as_secs_f64()).collect();
        stats.vc_secs = median(&stats.solve_iters_secs).unwrap_or(0.0);
        stats.verify_numeric_secs = report.verify_time.map(|d| d.as_secs_f64());
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

/// The result of [`persist_vcs`]: the analysis outputs (moved through the
/// dump, never cloned - the arenas can be GiB-scale) plus what got written.
pub(crate) struct PersistedVcs {
    pub reference: AnalysisOutput,
    pub optimized: AnalysisOutput,
    /// The `.vcdump` path, when a dump was written.
    pub path: Option<PathBuf>,
    /// Time spent writing it, when a dump was written.
    pub write_secs: Option<f64>,
}

/// Persist one equivalence benchmark's verification conditions to
/// `<vcs_dir>/<sanitized-name>.vcdump` via the shared
/// `volta_analysis::driver::vc_dump` format (the same file `volta compare
/// --dump-vcs` writes and `--from-dump` reads), overwriting any previous
/// run's dump - VCs are deterministic. A write failure warns and carries
/// on: a full disk should not change a benchmark verdict.
///
/// The outputs are moved into the dump and moved back out
/// (`into_analysis_output`), which clears their `stats`/`op_counts` -
/// callers must record those before calling. Shared by the benchmark
/// runner and `z3_compare`, so both write byte-identical dumps to the
/// same path. Byte-identity rests on one premise: no production code
/// path creates machine symbols (`ExprArena::symbol`, the only id drawn
/// from a process-global counter), so every id in a dump is
/// deterministic; a future machine-symbol caller would void byte-identity
/// across runs but not the dumps' validity - `--from-dump` never depends
/// on the numeric id values.
pub(crate) fn persist_vcs(
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

fn record_exec_stats(stats: &mut BenchmarkStats, s: Stats) {
    stats.instructions += s.instructions;
    stats.block_syncs = s.block_syncs;
    stats.warp_syncs = s.warp_syncs;
}

fn rejected_outcome(e: EvalError) -> ActualOutcome {
    let is_race = matches!(e, EvalError::DataRace { .. });
    ActualOutcome::Rejected {
        description: e.to_string(),
        is_race,
    }
}

/// Load and analyze one kernel, flattening every failure (I/O, parse,
/// lowering, or an analysis rejection such as a data race) into a single
/// `anyhow` error tagged with the kernel's path - the runner's own
/// path-context formatting. Used by callers that treat any failure as an
/// error rather than a benchmark outcome (see `z3_compare`); the runner's
/// `analyze` method keeps the typed race/deadlock split it needs.
pub fn run_kernel(kernels_dir: &Path, run: &KernelRun) -> Result<AnalysisOutput> {
    let path = kernels_dir.join(&run.path);
    let module = load_module(&path)?;
    analyze_kernel(&module, Some(&run.kernel), run.config.clone())
        .map_err(|e| anyhow!("{}: {}", run.path, e))
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

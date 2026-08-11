//! The `solve` subcommand: replay the solve phase(s) of the pipeline
//! from previously generated VC dumps (`volta-bench generate`), with no
//! parsing, lowering, or symbolic execution involved.
//!
//! Per equivalence benchmark: load `<vcs_dir>/<slug>.vcdump` through the
//! shared `volta_analysis::driver::vc_dump` reader (header + id
//! validation as usual), check it against the vcs manifest
//! (`crate::manifest` - the staleness guard), rehydrate both snapshots,
//! and run the *same* solve-phase functions the one-shot pipeline runs:
//! `BenchmarkRunner::check_equivalence` (the decision procedure, under
//! `--backend decision|both`) and `crate::z3_phase::run_z3_phase` (under
//! `--backend z3|both`). Verdicts and pass/fail are judged by the same
//! `assemble_result` tail as the one-shot path.
//!
//! Race-check benchmarks have no VCs and are skipped with a note: their
//! verdicts are produced by generation (`volta-bench generate` runs them
//! fully). A missing, corrupt, or manifest-contradicting dump is a
//! per-benchmark failure naming the `generate` command to run.
//!
//! Dump load time is recorded (`dump_load_secs`) but excluded from the
//! solve spans - loading is transport, not solving.

use std::time::Instant;

use anyhow::{Context, Result, anyhow, bail};
use volta_analysis::driver::{paired_elements, sampled_elements, vc_dump::read_vc_dump};

use crate::config::{BenchmarkCategory, BenchmarkDef};
use crate::manifest::{self, ManifestCheck, VcsManifest};
use crate::results::vc_dump_path;
use crate::runner::{ActualOutcome, BenchmarkResult, BenchmarkRunner, BenchmarkStats, RunOutput};
use crate::z3_phase::run_z3_phase;

/// Which solver backend(s) a `solve` run applies to the dumped VCs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SolveBackend {
    /// The decision procedure only (the default).
    Decision,
    /// Z3 only: no decision verdict is produced
    /// ([`ActualOutcome::Z3Only`]); the Z3 per-element outcomes are the
    /// run's data.
    Z3,
    /// Both, side by side over the same sampled elements - the dumps'
    /// analogue of the one-shot `--z3` comparison.
    Both,
}

impl SolveBackend {
    pub fn runs_decision(self) -> bool {
        matches!(self, Self::Decision | Self::Both)
    }

    pub fn runs_z3(self) -> bool {
        matches!(self, Self::Z3 | Self::Both)
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Decision => "decision",
            Self::Z3 => "z3",
            Self::Both => "both",
        }
    }
}

/// The console/record note for benchmarks `solve` does not run.
pub const RACE_SKIP_NOTE: &str =
    "race-check benchmark: no VC to solve; its verdict comes from `volta-bench generate`";

/// One benchmark's fate under `solve`.
#[derive(Debug)]
pub enum SolveItem {
    /// An equivalence benchmark: solved (or failed loudly trying).
    Solved(BenchmarkResult),
    /// A race-check benchmark: nothing to solve (see [`RACE_SKIP_NOTE`]).
    Skipped {
        name: String,
        category: BenchmarkCategory,
    },
}

impl BenchmarkRunner {
    /// `solve` over a benchmark list: read the vcs manifest once (absent
    /// manifest: warn and skip the staleness guard, so hand-copied dump
    /// directories stay usable; unreadable manifest: hard error), then
    /// [`run_solve`](Self::run_solve) each benchmark with the same
    /// verbose chatter as `run_all`.
    pub fn solve_suite(
        &self,
        defs: &[BenchmarkDef],
        backend: SolveBackend,
    ) -> Result<Vec<SolveItem>> {
        let vcs_dir = self
            .config
            .vcs_dir
            .as_deref()
            .ok_or_else(|| anyhow!("solve requires a VC dump directory"))?;
        let manifest = manifest::read_manifest(vcs_dir)?;
        if manifest.is_none() {
            eprintln!(
                "warning: no {} - skipping the staleness check (dumps not \
                 written by `volta-bench generate`?)",
                manifest::manifest_path(vcs_dir).display()
            );
        }
        Ok(defs
            .iter()
            .map(|def| {
                if self.config.verbose && def.optimized.is_some() {
                    eprintln!("solving {} ...", def.name);
                }
                let item = self.run_solve(def, backend, manifest.as_ref());
                if self.config.verbose
                    && let SolveItem::Solved(result) = &item
                {
                    eprintln!(
                        "  -> {} in {:.1}s",
                        result.outcome.status(),
                        result.elapsed_secs
                    );
                }
                item
            })
            .collect())
    }

    /// Solve one benchmark from its dump (see the module docs). Any
    /// failure - missing/corrupt dump, manifest disagreement, solve
    /// error - becomes a per-benchmark `Error` result rather than
    /// aborting the run.
    pub fn run_solve(
        &self,
        def: &BenchmarkDef,
        backend: SolveBackend,
        manifest: Option<&VcsManifest>,
    ) -> SolveItem {
        if def.optimized.is_none() {
            // stderr like the rest of the progress chatter (must survive
            // a closed stdout pipe).
            eprintln!("note: {}: skipped - {}", def.name, RACE_SKIP_NOTE);
            return SolveItem::Skipped {
                name: def.name.clone(),
                category: def.category,
            };
        }
        let start = Instant::now();
        let output = self
            .solve_inner(def, backend, manifest)
            .unwrap_or_else(|e| RunOutput {
                outcome: ActualOutcome::Error {
                    message: format!("{:#}", e),
                },
                stats: BenchmarkStats::default(),
                dump_path: None,
                z3: None,
            });
        SolveItem::Solved(self.assemble_result(def, start, output))
    }

    fn solve_inner(
        &self,
        def: &BenchmarkDef,
        backend: SolveBackend,
        manifest: Option<&VcsManifest>,
    ) -> Result<RunOutput> {
        let vcs_dir = self
            .config
            .vcs_dir
            .as_deref()
            .ok_or_else(|| anyhow!("solve requires a VC dump directory"))?;
        let path = vc_dump_path(vcs_dir, &def.name);
        if !path.exists() {
            bail!(
                "no VC dump at {}; run `volta-bench generate single \"{}\"` \
                 (or `volta-bench generate all`) first",
                path.display(),
                def.name
            );
        }
        // Load (timed into `dump_load_secs`, never into the solve
        // spans). `read_vc_dump` checks the header and validates every
        // id, so stale-format/corrupt files fail here with its message.
        let load_start = Instant::now();
        let dump = read_vc_dump(&path).with_context(|| {
            format!(
                "loading VC dump {} (if it is stale or corrupt, re-run \
                 `volta-bench generate`)",
                path.display()
            )
        })?;
        let dump_load_secs = load_start.elapsed().as_secs_f64();

        // The staleness guard: the dump must hold what `generate` last
        // recorded for this benchmark.
        if let Some(manifest) = manifest {
            match manifest::check_dump(
                manifest,
                &def.name,
                &dump.reference.outputs,
                &dump.optimized.outputs,
            )? {
                ManifestCheck::Verified => {}
                ManifestCheck::NoEntry => eprintln!(
                    "warning: {}: dump has no manifest entry (hand-copied?); \
                     skipping the staleness check",
                    def.name
                ),
            }
        }

        let reference = dump.reference.into_analysis_output();
        let optimized = dump.optimized.into_analysis_output();
        let arrays = def.reference.config.output_array_names();
        let mut stats = BenchmarkStats {
            dump_load_secs: Some(dump_load_secs),
            ..BenchmarkStats::default()
        };
        let paired = paired_elements(&reference, &optimized, &arrays)
            .map_err(|e| anyhow!("pairing the dumped footprints: {}", e))?;
        stats.elements_total = paired.iter().map(|(_, common)| common.len() as u64).sum();

        // --- Decision solve (backend decision|both): the same phase
        // function as the one-shot pipeline, iterations and all.
        let outcome = if backend.runs_decision() {
            self.check_equivalence(&reference, &optimized, &arrays, &mut stats)?
        } else {
            stats.elements_checked = sampled_elements(&paired, self.config.sample).len() as u64;
            ActualOutcome::Z3Only
        };

        // --- Z3 solve (backend z3|both): the same phase function as the
        // one-shot `--z3` comparison, over the same sampled elements.
        let z3 = if backend.runs_z3() {
            let options = self.config.z3.as_ref().ok_or_else(|| {
                anyhow!(
                    "solve backend '{}' requires Z3 options in the runner config",
                    backend.name()
                )
            })?;
            let sampled = sampled_elements(&paired, self.config.sample);
            Some(run_z3_phase(
                &reference,
                &optimized,
                &arrays,
                &sampled,
                self.config.sample,
                self.config.iterations,
                options,
            ))
        } else {
            None
        };

        Ok(RunOutput {
            outcome,
            stats,
            dump_path: Some(path),
            z3,
        })
    }
}

#[cfg(test)]
mod tests {
    use volta_analysis::driver::{VcDump, VcSnapshot, vc_dump::write_vc_dump};
    use volta_analysis::eval::AnalysisConfig;
    use volta_analysis::symbolic::{ExprArena, ExprId};

    use super::*;
    use crate::config::{BenchmarkDef, KernelRun, f32_output};
    use crate::runner::RunnerConfig;

    /// An equivalence benchmark whose config declares one output array
    /// `out`; `solve` never reads the kernel paths, only the dump.
    fn synthetic_def() -> BenchmarkDef {
        let mut config = AnalysisConfig::new((1, 1, 1));
        config.arrays = vec![f32_output("out", 0x1000, 2)];
        let run = || KernelRun::new("unused.ptx", "k", config.clone());
        BenchmarkDef::equivalence("Synthetic Pair", BenchmarkCategory::Reduction, run(), run())
    }

    /// A snapshot writing `out[0..len)` as reads of input array `src`.
    fn snapshot(src: &str, len: u64) -> VcSnapshot {
        let mut arena = ExprArena::new();
        let sid = arena.intern_string(src);
        let elems: Vec<(u64, ExprId)> =
            (0..len).map(|i| (i, arena.input_element(sid, i))).collect();
        VcSnapshot {
            arena,
            outputs: vec![("out".to_string(), elems)],
        }
    }

    fn runner(vcs_dir: &std::path::Path) -> BenchmarkRunner {
        BenchmarkRunner::new(RunnerConfig {
            vcs_dir: Some(vcs_dir.to_path_buf()),
            ..RunnerConfig::default()
        })
    }

    fn temp_vcs_dir(tag: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("volta_solve_{}_{}", tag, std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// A missing dump is a per-benchmark FAILURE record that names the
    /// generate command to run - not a panic, not a silent skip.
    #[test]
    fn missing_dump_is_a_failure_naming_generate() {
        let dir = temp_vcs_dir("missing");
        let def = synthetic_def();
        let SolveItem::Solved(result) = runner(&dir).run_solve(&def, SolveBackend::Decision, None)
        else {
            panic!("equivalence benchmarks are solved, not skipped");
        };
        assert!(!result.passed);
        assert_eq!(result.outcome.status(), "ERR");
        let ActualOutcome::Error { message } = &result.outcome else {
            panic!("missing dump must be an Error outcome");
        };
        assert!(message.contains("no VC dump"), "{message}");
        assert!(
            message.contains("volta-bench generate single \"Synthetic Pair\""),
            "{message}"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Race-check benchmarks are skipped (their verdicts come from
    /// `generate`), even when a dump directory exists.
    #[test]
    fn race_benchmarks_are_skipped() {
        let dir = temp_vcs_dir("race");
        let def = BenchmarkDef::race_check(
            "Racy Kernel",
            BenchmarkCategory::DataRace,
            KernelRun::new("unused.ptx", "k", AnalysisConfig::new((1, 1, 1))),
            true,
        );
        let item = runner(&dir).run_solve(&def, SolveBackend::Decision, None);
        let SolveItem::Skipped { name, .. } = item else {
            panic!("race-check benchmarks must be skipped by solve");
        };
        assert_eq!(name, "Racy Kernel");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A well-formed dump solves to the expected verdict with the load
    /// time recorded, and a disagreeing manifest turns the same dump into
    /// a hard per-benchmark failure (the stale/mixed-directory guard).
    #[test]
    fn solve_from_dump_checks_manifest_and_solves() {
        let dir = temp_vcs_dir("dump");
        let def = synthetic_def();
        let dump = VcDump {
            reference: snapshot("in", 2),
            optimized: snapshot("in", 2),
        };
        write_vc_dump(&vc_dump_path(&dir, &def.name), &dump).unwrap();

        // Matching manifest: verified and solved (identical expressions
        // on both sides -> EQUIV, which matches the expectation).
        let mut manifest = VcsManifest::new();
        manifest::record_dump(
            &mut manifest,
            &def.name,
            &dump.reference.outputs,
            &dump.optimized.outputs,
        );
        let SolveItem::Solved(result) =
            runner(&dir).run_solve(&def, SolveBackend::Decision, Some(&manifest))
        else {
            panic!("solved, not skipped");
        };
        assert_eq!(result.outcome.status(), "EQUIV");
        assert!(result.passed);
        assert!(result.stats.dump_load_secs.is_some());
        assert_eq!(result.stats.elements_checked, 2);
        assert_eq!(result.stats.elements_total, 2);
        assert!(!result.stats.solve_iters_secs.is_empty());

        // Stale manifest (recorded a different footprint): hard failure
        // pointing at generate.
        let mut stale = VcsManifest::new();
        manifest::record_dump(
            &mut stale,
            &def.name,
            &snapshot("in", 5).outputs,
            &dump.optimized.outputs,
        );
        let SolveItem::Solved(result) =
            runner(&dir).run_solve(&def, SolveBackend::Decision, Some(&stale))
        else {
            panic!("solved, not skipped");
        };
        assert!(!result.passed);
        let ActualOutcome::Error { message } = &result.outcome else {
            panic!("manifest disagreement must be an Error outcome");
        };
        assert!(message.contains("stale or mixed"), "{message}");
        assert!(message.contains("generate"), "{message}");
        let _ = std::fs::remove_dir_all(&dir);
    }
}

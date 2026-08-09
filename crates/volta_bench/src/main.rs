//! Volta benchmark runner CLI.
//!
//! Note: run in release mode; symbolic execution of the larger kernels is
//! ~20x slower unoptimized.
//!
//! ```bash
//! cargo run --release -p volta_bench -- all --sample 16
//! cargo run --release -p volta_bench -- category reduction
//! cargo run --release -p volta_bench -- single "(Red-1, Red-2)"
//! cargo run --release -p volta_bench -- list
//! ```

use std::path::PathBuf;
use std::process::ExitCode;

#[cfg(feature = "logging")]
use clap::ValueEnum;
use clap::{Parser, Subcommand};
use volta_analysis::driver::write_op_counts;
use volta_bench::{
    BenchmarkCategory, BenchmarkRunner, KERNELS_DIR, RunnerConfig, all_benchmarks, export_json,
    print_all_results, print_results_table, print_summary,
};
use volta_common::run_log::RunLog;

/// Log level for controlling `log`-crate output verbosity (mirrors
/// `volta_cli`'s so the two tools take the same `--log-level` values).
#[cfg(feature = "logging")]
#[derive(Debug, Clone, Copy, Default, ValueEnum)]
enum LogLevel {
    /// Only show errors
    Error,
    /// Show warnings and errors
    #[default]
    Warn,
    /// Show info, warnings, and errors
    Info,
    /// Show debug output and above
    Debug,
    /// Show all log output including trace
    Trace,
}

#[cfg(feature = "logging")]
impl From<LogLevel> for log::LevelFilter {
    fn from(level: LogLevel) -> Self {
        match level {
            LogLevel::Error => log::LevelFilter::Error,
            LogLevel::Warn => log::LevelFilter::Warn,
            LogLevel::Info => log::LevelFilter::Info,
            LogLevel::Debug => log::LevelFilter::Debug,
            LogLevel::Trace => log::LevelFilter::Trace,
        }
    }
}

#[derive(Parser)]
#[command(name = "volta-bench")]
#[command(about = "Volta benchmark runner - reproduces the paper evaluation")]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Commands,

    /// Verbose output (prints progress per benchmark)
    #[arg(short, long, global = true)]
    verbose: bool,

    /// Custom kernels directory
    #[arg(long, global = true)]
    kernels_dir: Option<PathBuf>,

    /// Check at most this many output elements per array (0 = all).
    #[arg(long, global = true, default_value_t = 0)]
    sample: u64,

    /// Confirm every equivalence verdict with the f64 numeric oracle
    #[arg(long, global = true)]
    verify_numeric: bool,

    /// Recycle the VC intern tables past this many interned terms. Lower
    /// values bound VC memory at the cost of re-canonicalizing shared
    /// structure (0 = never recycle).
    #[arg(long, global = true, default_value_t = volta_analysis::equiv::DEFAULT_RECYCLE_TERMS)]
    recycle_terms: usize,

    /// Log level for `log`-crate output verbosity
    #[cfg(feature = "logging")]
    #[arg(long, value_enum, default_value = "warn", global = true)]
    log_level: LogLevel,

    /// Directory for per-run log files
    #[arg(long, global = true, default_value = "volta-logs")]
    log_dir: PathBuf,

    /// Don't write a per-run log file
    #[arg(long, global = true)]
    no_log_file: bool,
}

#[derive(Subcommand)]
enum Commands {
    /// Run all benchmarks
    All {
        /// Export results to JSON file
        #[arg(long)]
        json: Option<PathBuf>,
    },
    /// Run benchmarks for one category
    Category {
        /// reduction | matmul | attention | causal | conv | agent | tilelang | race
        category: String,
        #[arg(long)]
        json: Option<PathBuf>,
    },
    /// Run a single benchmark by name
    Single { name: String },
    /// List all benchmarks
    List,
    /// Compare the decision procedure against Z3 (SMT-LIB2 evaluated
    /// in-process via libz3) on the same equivalence benchmarks: timing
    /// and per-element outcome side by side.
    Z3Compare {
        /// "all", a category, or an exact benchmark name
        selector: String,
        #[arg(long)]
        json: Option<PathBuf>,
        /// Soft per-query Z3 timeout in seconds (0 = no limit; expiry
        /// reports `unknown`)
        #[arg(long, default_value_t = 30)]
        z3_timeout: u64,
    },
}

fn main() -> ExitCode {
    // Must precede everything: if this process was spawned as a z3
    // solver worker, this runs the query and exits (see volta_z3::ffi).
    volta_z3::init_worker();

    let cli = Cli::parse();

    let command_name = match &cli.command {
        Commands::All { .. } => "run-all",
        Commands::Category { .. } => "category",
        Commands::Single { .. } => "single",
        Commands::List => "list",
        Commands::Z3Compare { .. } => "z3-compare",
    };
    let mut log = RunLog::open(&cli.log_dir, command_name, cli.no_log_file);

    // env_logger's target borrows the log file (via `tee`) before the
    // command match borrows `log` mutably for `record` - initialize it here.
    #[cfg(feature = "logging")]
    env_logger::Builder::new()
        .filter_level(cli.log_level.into())
        .format_timestamp(None)
        .format_target(false)
        .target(env_logger::Target::Pipe(log.tee(std::io::stderr())))
        .init();

    let runner_config = RunnerConfig {
        kernels_dir: cli
            .kernels_dir
            .unwrap_or_else(|| PathBuf::from(KERNELS_DIR)),
        verbose: cli.verbose,
        sample: cli.sample,
        verify_numeric: cli.verify_numeric,
        recycle_terms: cli.recycle_terms,
    };

    let code = match cli.command {
        Commands::All { json } => {
            let suite = all_benchmarks();
            println!("Running {} benchmarks...", suite.benchmarks.len());
            let runner = BenchmarkRunner::new(runner_config);
            let results = runner.run_all(&suite.benchmarks);
            let mut stdout = std::io::stdout();
            print_all_results(&mut stdout, &results).unwrap();
            if let Some(path) = json {
                export_json(&results, &path).unwrap();
                println!("Results exported to {}", path.display());
            }
            let passed = results.iter().filter(|r| r.passed).count();
            log.record(&format!(
                "run-all: {}/{} benchmarks passed",
                passed,
                results.len()
            ));
            exit_by_pass(passed == results.len())
        }
        Commands::Category { category, json } => {
            let Some(category) = parse_category(&category) else {
                eprintln!("Unknown category: {}", category);
                eprintln!(
                    "Available: reduction, matmul, attention, causal, conv, agent, tilelang, race"
                );
                log.record(&format!("category: unknown category '{}'", category));
                return finish(log, ExitCode::FAILURE);
            };
            let suite = all_benchmarks();
            let filtered: Vec<_> = suite
                .filter_category(category)
                .into_iter()
                .cloned()
                .collect();
            println!(
                "Running {} benchmarks for {}...",
                filtered.len(),
                category.name()
            );
            let runner = BenchmarkRunner::new(runner_config);
            let results = runner.run_all(&filtered);
            let mut stdout = std::io::stdout();
            print_results_table(&mut stdout, &results, category).unwrap();
            print_summary(&mut stdout, &results).unwrap();
            if let Some(path) = json {
                export_json(&results, &path).unwrap();
                println!("Results exported to {}", path.display());
            }
            let passed = results.iter().filter(|r| r.passed).count();
            log.record(&format!(
                "category {}: {}/{} benchmarks passed",
                category.name(),
                passed,
                results.len()
            ));
            exit_by_pass(passed == results.len())
        }
        Commands::Single { name } => {
            let suite = all_benchmarks();
            let Some(def) = suite.benchmarks.iter().find(|b| b.name == name) else {
                eprintln!("Benchmark not found: {}", name);
                eprintln!("Use 'volta-bench list' to see available benchmarks.");
                log.record(&format!("single: benchmark not found '{}'", name));
                return finish(log, ExitCode::FAILURE);
            };
            println!("Running {} ...", name);
            let runner = BenchmarkRunner::new(runner_config);
            let result = runner.run(def);
            println!("Status:  {}", result.outcome.status());
            println!(
                "Detail:  {}",
                volta_bench::reporter::describe(&result.outcome)
            );
            println!("Passed:  {}", if result.passed { "yes" } else { "no" });
            println!("Exec:    {:.2}s", result.stats.exec_secs);
            println!("VC:      {:.2}s", result.stats.vc_secs);
            println!("Instrs:  {}", result.stats.instructions);
            println!(
                "Syncs:   {} block, {} warp",
                result.stats.block_syncs, result.stats.warp_syncs
            );
            println!(
                "Elems:   {} checked of {}",
                result.stats.elements_checked, result.stats.elements_total
            );
            let mut stdout = std::io::stdout().lock();
            write_op_counts(&mut stdout, "reference", &result.stats.reference_op_counts).unwrap();
            write_op_counts(&mut stdout, "optimized", &result.stats.optimized_op_counts).unwrap();
            if !result.passed {
                let mut out = Vec::new();
                print_summary(&mut out, std::slice::from_ref(&result)).unwrap();
                print!("{}", String::from_utf8_lossy(&out));
            }
            log.record(&format!(
                "single {}: {} ({})",
                name,
                result.outcome.status(),
                if result.passed { "pass" } else { "FAIL" }
            ));
            exit_by_pass(result.passed)
        }
        Commands::Z3Compare {
            selector,
            json,
            z3_timeout,
        } => {
            println!("z3 {} (libz3, in-process)", volta_z3::z3_version());
            let suite = all_benchmarks();
            // "all" and category selectors silently skip race-check
            // benchmarks (no optimized kernel to compare against); an exact
            // NAME that has no optimized kernel is a user error and still
            // fails loudly below via `compare_one`'s error row.
            let by_name = suite.benchmarks.iter().find(|b| b.name == selector);
            let defs: Vec<&volta_bench::BenchmarkDef> = if selector.eq_ignore_ascii_case("all") {
                skip_race_check(suite.benchmarks.iter().collect())
            } else if let Some(category) = parse_category(&selector) {
                skip_race_check(suite.filter_category(category))
            } else if let Some(def) = by_name {
                vec![def]
            } else {
                eprintln!(
                    "Unknown selector '{}' (not 'all', a category, or an exact benchmark name)",
                    selector
                );
                log.record(&format!("z3-compare: unknown selector '{}'", selector));
                return finish(log, ExitCode::FAILURE);
            };

            let kernels_dir = runner_config.kernels_dir.clone();
            let z3_timeout = volta_z3::timeout_from_secs(z3_timeout);

            println!(
                "{:<28} {:>8} {:>8} {:>8} {:>9}  {}",
                "Benchmark",
                "Exec(s)",
                "Dec(s)",
                "Z3(s)",
                "Decision",
                "Z3: equiv/diff/unk/to/unsup/err"
            );
            println!("{}", "-".repeat(100));
            let mut rows = Vec::new();
            for def in &defs {
                let row = volta_bench::compare_one(
                    &kernels_dir,
                    def,
                    cli.sample,
                    cli.verify_numeric,
                    cli.recycle_terms,
                    z3_timeout,
                );
                if let Some(err) = &row.error {
                    println!("{:<28} {}", row.name, err);
                } else {
                    println!(
                        "{:<28} {:>8.3} {:>8.3} {:>8.3} {:>9}  {}",
                        row.name,
                        row.exec_secs,
                        row.decision_secs,
                        row.z3_secs,
                        row.decision_status,
                        row.z3.compact(),
                    );
                    if let Some((axiom_secs, axiom_counts)) = &row.z3_axiom {
                        println!(
                            "{:<28} {:>8} {:>8} {:>8.3} {:>9}  {}",
                            "  +exp-axiom",
                            "",
                            "",
                            axiom_secs,
                            "",
                            axiom_counts.compact(),
                        );
                    }
                }
                rows.push(row);
            }
            if let Some(path) = json {
                volta_bench::z3_compare::export_json(&rows, &path).unwrap();
                println!("Results exported to {}", path.display());
            }
            let errors = rows.iter().filter(|r| r.error.is_some()).count();
            log.record(&format!(
                "z3-compare {}: {} row(s), {} error(s)",
                selector,
                rows.len(),
                errors
            ));
            exit_by_pass(errors == 0)
        }
        Commands::List => {
            let suite = all_benchmarks();
            for category in suite.categories() {
                println!("{}:", category.name());
                for b in suite.filter_category(category) {
                    println!("  - {}", b.name);
                }
            }
            println!("Total: {} benchmarks", suite.benchmarks.len());
            log.record("list");
            ExitCode::SUCCESS
        }
    };

    finish(log, code)
}

/// Print the run-log path (if any) and return the exit code - the last
/// thing every command path does, mirroring `volta_cli`.
fn finish(log: RunLog, code: ExitCode) -> ExitCode {
    if let Some(path) = log.path() {
        eprintln!("log: {}", path.display());
    }
    code
}

/// Drop the race-check benchmarks (no optimized kernel), announcing how
/// many were skipped so a `z3-compare all`/category run isn't silently
/// short. Called only for the "all"/category selectors.
fn skip_race_check(defs: Vec<&volta_bench::BenchmarkDef>) -> Vec<&volta_bench::BenchmarkDef> {
    let (kept, skipped): (Vec<_>, Vec<_>) = defs.into_iter().partition(|d| d.optimized.is_some());
    if !skipped.is_empty() {
        println!(
            "skipping {} race-check benchmark(s) (no optimized kernel)",
            skipped.len()
        );
    }
    kept
}

fn exit_by_pass(passed: bool) -> ExitCode {
    if passed {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}

fn parse_category(name: &str) -> Option<BenchmarkCategory> {
    match name.to_lowercase().as_str() {
        "reduction" | "red" => Some(BenchmarkCategory::Reduction),
        "matmul" | "mm" => Some(BenchmarkCategory::MatMul),
        "attention" | "attn" => Some(BenchmarkCategory::Attention),
        "causal" | "causal-attention" | "causal-attn" => Some(BenchmarkCategory::CausalAttention),
        "convolution" | "conv" => Some(BenchmarkCategory::Convolution),
        "agent" | "agent-generated" => Some(BenchmarkCategory::AgentGenerated),
        "compiler" | "compiler-generated" | "tilelang" | "tl" => {
            Some(BenchmarkCategory::CompilerGenerated)
        }
        "datarace" | "race" | "races" => Some(BenchmarkCategory::DataRace),
        _ => None,
    }
}

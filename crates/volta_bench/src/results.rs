//! Persistent run artifacts: the `--out-dir` layout, benchmark-name
//! sanitization for VC-dump filenames, solve-timing summary statistics,
//! and the per-run results JSON document.
//!
//! Layout under `--out-dir` (default `bench-out/`):
//!
//! - `vcs/<sanitized-benchmark-name>.vcdump` - each equivalence
//!   benchmark's verification conditions (both kernels' arenas + output
//!   footprints), written once per run through
//!   `volta_analysis::driver::vc_dump` - the same format as `volta
//!   compare --dump-vcs`, so `volta compare --from-dump` replays them.
//!   Overwritten on rerun (VCs are deterministic). Race-check benchmarks
//!   have no VC and are skipped with a console note.
//! - `results/<unix-seconds>-<pid>-<command>.json` - the machine-readable
//!   results for every run command, timestamped like
//!   `volta_common::run_log` files so runs never clobber each other. The
//!   `--json <path>` flag writes the same document to an explicit path in
//!   addition.

use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use serde_json::{Value, json};

use crate::runner::BenchmarkResult;
use crate::z3_compare::Z3CompareRow;

/// Reduce a display name like "(Attention, FA1)" to a filesystem-safe
/// slug like "attention-fa1": lowercase alphanumerics, every other run of
/// characters collapsed to a single '-', no leading/trailing '-'.
pub fn sanitize_name(name: &str) -> String {
    let mut out = String::with_capacity(name.len());
    for c in name.chars() {
        if c.is_ascii_alphanumeric() {
            out.push(c.to_ascii_lowercase());
        } else if !out.is_empty() && !out.ends_with('-') {
            out.push('-');
        }
    }
    let out = out.trim_end_matches('-').to_string();
    if out.is_empty() {
        "benchmark".to_string()
    } else {
        out
    }
}

/// Where a benchmark's VC dump lives under the vcs directory.
pub fn vc_dump_path(vcs_dir: &Path, benchmark_name: &str) -> PathBuf {
    vcs_dir.join(format!("{}.vcdump", sanitize_name(benchmark_name)))
}

/// Reject a benchmark set whose names collide under [`sanitize_name`]:
/// sanitization is many-to-one ("(Red-1, Red-2)" and "Red 1 Red 2" both
/// slug to "red-1-red-2"), and VC dumps are keyed by slug and overwrite
/// silently, so a colliding set would clobber one benchmark's dump with
/// another's. The error names both offenders. Run this wherever a run's
/// benchmark set is known, before anything is written.
pub fn check_slug_collisions<'a>(names: impl IntoIterator<Item = &'a str>) -> Result<()> {
    let mut by_slug: std::collections::HashMap<String, &str> = std::collections::HashMap::new();
    for name in names {
        match by_slug.entry(sanitize_name(name)) {
            std::collections::hash_map::Entry::Occupied(entry) => anyhow::bail!(
                "benchmark names '{}' and '{}' both sanitize to VC-dump slug '{}'; \
                 their dumps would overwrite each other - rename one of them",
                entry.get(),
                name,
                entry.key(),
            ),
            std::collections::hash_map::Entry::Vacant(entry) => {
                entry.insert(name);
            }
        }
    }
    Ok(())
}

/// Median of the samples (mean of the two middle values for even counts);
/// `None` for an empty slice.
pub fn median(xs: &[f64]) -> Option<f64> {
    if xs.is_empty() {
        return None;
    }
    let mut sorted = xs.to_vec();
    sorted.sort_by(f64::total_cmp);
    let mid = sorted.len() / 2;
    Some(if sorted.len().is_multiple_of(2) {
        (sorted[mid - 1] + sorted[mid]) / 2.0
    } else {
        sorted[mid]
    })
}

/// Arithmetic mean; `None` for an empty slice.
pub fn mean(xs: &[f64]) -> Option<f64> {
    if xs.is_empty() {
        return None;
    }
    Some(xs.iter().sum::<f64>() / xs.len() as f64)
}

/// Minimum; `None` for an empty slice.
pub fn min(xs: &[f64]) -> Option<f64> {
    xs.iter().copied().min_by(f64::total_cmp)
}

/// The run-wide header of a results document: what was asked for, so a
/// results file is interpretable without the shell history.
pub struct RunMeta {
    pub command: &'static str,
    pub iterations: usize,
    pub sample: u64,
    pub verify_numeric: bool,
    pub recycle_terms: usize,
    /// Set for z3-compare runs only.
    pub z3_timeout_secs: Option<u64>,
}

/// Insert `<prefix>_iters_secs` / `<prefix>_median_secs` /
/// `<prefix>_min_secs` / `<prefix>_mean_secs` (nulls when no solve ran).
fn insert_solve_stats(obj: &mut serde_json::Map<String, Value>, prefix: &str, iters: &[f64]) {
    obj.insert(format!("{prefix}_iters_secs"), json!(iters));
    obj.insert(format!("{prefix}_median_secs"), json!(median(iters)));
    obj.insert(format!("{prefix}_min_secs"), json!(min(iters)));
    obj.insert(format!("{prefix}_mean_secs"), json!(mean(iters)));
}

/// One `all`/`category`/`single` benchmark as a results-JSON record.
pub fn benchmark_record(r: &BenchmarkResult) -> Value {
    let mut obj = serde_json::Map::new();
    obj.insert("name".into(), json!(r.name));
    obj.insert("category".into(), json!(r.category.name()));
    obj.insert("status".into(), json!(r.outcome.status()));
    obj.insert(
        "detail".into(),
        json!(crate::reporter::describe(&r.outcome)),
    );
    obj.insert("passed".into(), json!(r.passed));
    obj.insert("elements_checked".into(), json!(r.stats.elements_checked));
    obj.insert("elements_total".into(), json!(r.stats.elements_total));
    obj.insert("exec_secs".into(), json!(r.stats.exec_secs));
    obj.insert("vc_gen_secs".into(), json!(r.stats.vc_gen_secs));
    obj.insert("dump_write_secs".into(), json!(r.stats.dump_write_secs));
    insert_solve_stats(&mut obj, "solve", &r.stats.solve_iters_secs);
    // The `--verify-numeric` oracle's time (iteration 1 only); null when
    // the flag was off. Kept out of the solve stats so those don't move
    // when the oracle is toggled.
    obj.insert(
        "verify_numeric_secs".into(),
        json!(r.stats.verify_numeric_secs),
    );
    obj.insert("instructions".into(), json!(r.stats.instructions));
    obj.insert("block_syncs".into(), json!(r.stats.block_syncs));
    obj.insert("warp_syncs".into(), json!(r.stats.warp_syncs));
    obj.insert("dump_path".into(), json!(r.dump_path));
    Value::Object(obj)
}

fn z3_counts_json(c: &volta_z3::Z3Counts) -> Value {
    json!({
        "equivalent": c.equivalent,
        "not_equivalent": c.not_equivalent,
        "unknown": c.unknown,
        "timeout": c.timeout,
        "unsupported": c.unsupported,
        "error": c.error,
    })
}

/// One z3-compare row as a results-JSON record.
pub fn z3_row_record(r: &Z3CompareRow) -> Value {
    let mut obj = serde_json::Map::new();
    obj.insert("name".into(), json!(r.name));
    obj.insert("category".into(), json!(r.category.name()));
    obj.insert("vc_gen_secs".into(), json!(r.vc_gen_secs));
    obj.insert("decision_status".into(), json!(r.decision_status));
    insert_solve_stats(&mut obj, "decision_solve", &r.decision_iters_secs);
    insert_solve_stats(&mut obj, "z3_solve", &r.z3_iters_secs);
    obj.insert("z3_counts".into(), z3_counts_json(&r.z3));
    obj.insert(
        "z3_axiom".into(),
        match &r.z3_axiom {
            Some(rerun) => {
                let mut axiom = serde_json::Map::new();
                insert_solve_stats(&mut axiom, "z3_solve", &rerun.iters_secs);
                axiom.insert("z3_counts".into(), z3_counts_json(&rerun.counts));
                Value::Object(axiom)
            }
            None => Value::Null,
        },
    );
    obj.insert("dump_path".into(), json!(r.dump_path));
    obj.insert("error".into(), json!(r.error));
    Value::Object(obj)
}

/// Assemble the full results document: run header + per-benchmark records.
pub fn results_doc(meta: &RunMeta, benchmarks: Vec<Value>) -> Value {
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let mut doc = serde_json::Map::new();
    doc.insert("command".into(), json!(meta.command));
    doc.insert(
        "argv".into(),
        json!(std::env::args().collect::<Vec<String>>()),
    );
    doc.insert("timestamp_unix".into(), json!(timestamp));
    doc.insert("iterations".into(), json!(meta.iterations));
    doc.insert("sample".into(), json!(meta.sample));
    doc.insert("verify_numeric".into(), json!(meta.verify_numeric));
    doc.insert("recycle_terms".into(), json!(meta.recycle_terms));
    if let Some(timeout) = meta.z3_timeout_secs {
        doc.insert("z3_timeout_secs".into(), json!(timeout));
        // The carve-out convention (see `z3_compare`): re-solving a
        // timeout would multiply its full budget into every iteration,
        // and unsupported/error elements never reach the solver.
        doc.insert(
            "z3_iteration_convention".into(),
            json!(
                "elements whose iteration-1 z3 outcome is timeout/unsupported/error \
                 are solved once; their iteration-1 solve time is charged to every \
                 iteration's total"
            ),
        );
    }
    doc.insert("benchmarks".into(), Value::Array(benchmarks));
    Value::Object(doc)
}

/// Write `doc` to `<out_dir>/results/<unix-seconds>-<pid>-<command>.json`
/// (the `run_log` naming convention, so concurrent or scripted runs never
/// clobber each other) and return the path.
pub fn write_results_file(out_dir: &Path, command: &str, doc: &Value) -> Result<PathBuf> {
    let dir = out_dir.join("results");
    std::fs::create_dir_all(&dir)
        .with_context(|| format!("creating results directory {}", dir.display()))?;
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let path = dir.join(format!("{stamp}-{}-{command}.json", std::process::id()));
    write_results_to(&path, doc)?;
    Ok(path)
}

/// Write `doc` to an explicit path (the `--json` flag).
pub fn write_results_to(path: &Path, doc: &Value) -> Result<()> {
    std::fs::write(path, serde_json::to_string_pretty(doc)?)
        .with_context(|| format!("writing results to {}", path.display()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sanitize_name_makes_filesystem_safe_slugs() {
        assert_eq!(sanitize_name("(Attention, FA1)"), "attention-fa1");
        assert_eq!(sanitize_name("(Red-1, Red-2)"), "red-1-red-2");
        assert_eq!(sanitize_name("plain"), "plain");
        assert_eq!(sanitize_name("  spaced   out  "), "spaced-out");
        assert_eq!(sanitize_name("A/B\\C:D"), "a-b-c-d");
        // Nothing usable left: still a nonempty filename.
        assert_eq!(sanitize_name("()"), "benchmark");
        assert_eq!(sanitize_name(""), "benchmark");
    }

    #[test]
    fn solve_stats_handle_odd_even_and_empty() {
        assert_eq!(median(&[]), None);
        assert_eq!(mean(&[]), None);
        assert_eq!(min(&[]), None);

        // Odd count: the middle element; order must not matter.
        assert_eq!(median(&[3.0, 1.0, 2.0]), Some(2.0));
        // Even count: mean of the two middle elements.
        assert_eq!(median(&[4.0, 1.0, 3.0, 2.0]), Some(2.5));
        assert_eq!(median(&[5.0]), Some(5.0));

        assert_eq!(mean(&[1.0, 2.0, 6.0]), Some(3.0));
        assert_eq!(min(&[3.0, 1.0, 2.0]), Some(1.0));
    }

    /// The vcdump path is deterministic per benchmark name, so reruns
    /// overwrite instead of accumulating.
    #[test]
    fn vc_dump_path_is_stable() {
        let dir = Path::new("bench-out/vcs");
        assert_eq!(
            vc_dump_path(dir, "(Attention, FA1)"),
            Path::new("bench-out/vcs/attention-fa1.vcdump")
        );
        assert_eq!(
            vc_dump_path(dir, "(Attention, FA1)"),
            vc_dump_path(dir, "(Attention, FA1)"),
        );
    }

    /// `sanitize_name` is many-to-one; a set containing two names with
    /// the same slug must be rejected up front, naming both offenders.
    #[test]
    fn slug_collisions_fail_with_both_names() {
        check_slug_collisions(["(Red-1, Red-2)", "(Red-1, Red-3)"]).unwrap();
        check_slug_collisions(std::iter::empty::<&str>()).unwrap();

        let err = check_slug_collisions(["(Red-1, Red-2)", "Red 1 Red: 2"]).unwrap_err();
        let message = format!("{err:#}");
        assert!(message.contains("(Red-1, Red-2)"), "{message}");
        assert!(message.contains("Red 1 Red: 2"), "{message}");
        assert!(message.contains("'red-1-red-2'"), "{message}");
    }

    /// Guardrail over the real corpus: every name in `all_benchmarks()`
    /// must slug uniquely, so a future benchmark whose name collides with
    /// an existing one fails here (in CI) instead of silently overwriting
    /// that benchmark's VC dump.
    #[test]
    fn full_corpus_has_no_slug_collisions() {
        let suite = crate::all_benchmarks();
        check_slug_collisions(suite.benchmarks.iter().map(|b| b.name.as_str())).unwrap();
    }

    /// A benchmark that fails *after* its VC dump is written still has
    /// the dump on disk: the record builders must keep a `Some(dump_path)`
    /// alongside an error outcome rather than nulling it.
    #[test]
    fn error_records_keep_the_dump_path() {
        let dump = PathBuf::from("bench-out/vcs/x.vcdump");

        let result = crate::runner::BenchmarkResult {
            name: "x".to_string(),
            category: crate::config::BenchmarkCategory::Reduction,
            elapsed_secs: 0.0,
            outcome: crate::runner::ActualOutcome::Error {
                message: "post-dump failure".to_string(),
            },
            stats: Default::default(),
            passed: false,
            dump_path: Some(dump.clone()),
        };
        let record = benchmark_record(&result);
        assert_eq!(record["status"], json!("ERR"));
        assert_eq!(record["dump_path"], json!(dump));

        let row = Z3CompareRow {
            name: "x".to_string(),
            category: crate::config::BenchmarkCategory::Reduction,
            vc_gen_secs: 0.0,
            decision_iters_secs: Vec::new(),
            decision_status: "N/A".to_string(),
            z3_iters_secs: Vec::new(),
            z3: volta_z3::Z3Counts::default(),
            z3_axiom: None,
            dump_path: Some(dump.clone()),
            error: Some("post-dump failure".to_string()),
        };
        let record = z3_row_record(&row);
        assert!(record["error"].is_string());
        assert_eq!(record["dump_path"], json!(dump));
    }

    /// The oracle-time bucket is `Some` exactly when `--verify-numeric`
    /// ran (JSON: a number vs null), so its absence is distinguishable
    /// from a fast oracle.
    #[test]
    fn verify_numeric_secs_round_trips() {
        let mut result = crate::runner::BenchmarkResult {
            name: "x".to_string(),
            category: crate::config::BenchmarkCategory::Reduction,
            elapsed_secs: 0.0,
            outcome: crate::runner::ActualOutcome::Equivalent,
            stats: Default::default(),
            passed: true,
            dump_path: None,
        };
        assert!(benchmark_record(&result)["verify_numeric_secs"].is_null());

        result.stats.verify_numeric_secs = Some(0.25);
        assert_eq!(
            benchmark_record(&result)["verify_numeric_secs"],
            json!(0.25)
        );
    }
}

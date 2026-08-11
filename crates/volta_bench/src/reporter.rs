//! Result reporting: console tables and summaries. The machine-readable
//! results JSON lives in `crate::results`.

use std::io::Write;

use anyhow::Result;

use crate::config::BenchmarkCategory;
use crate::runner::{ActualOutcome, BenchmarkResult};

/// Print one category's results as a table (paper-style columns). The
/// "VC (s)" column is the *median* solve time across the run's
/// `iterations` (noted on the header line); "Exec (s)" is symbolic
/// execution alone.
pub fn print_results_table(
    out: &mut impl Write,
    results: &[BenchmarkResult],
    category: BenchmarkCategory,
    iterations: usize,
) -> Result<()> {
    writeln!(
        out,
        "\n{} ({}) [VC (s): median of {} solve iteration(s)]",
        category.name(),
        category.table_ref(),
        iterations
    )?;
    writeln!(
        out,
        "{:<28} {:>7} {:>9} {:>9} {:>11} {:>11} {:>9}",
        "Benchmark", "Status", "Exec (s)", "VC (s)", "#BlockSync", "#WarpSync", "Elems"
    )?;
    writeln!(out, "{}", "-".repeat(90))?;
    for r in results.iter().filter(|r| r.category == category) {
        let elems = if r.stats.elements_checked == r.stats.elements_total {
            format!("{}", r.stats.elements_total)
        } else {
            format!("{}/{}", r.stats.elements_checked, r.stats.elements_total)
        };
        writeln!(
            out,
            "{:<28} {:>7} {:>9.2} {:>9.2} {:>11} {:>11} {:>9}",
            r.name,
            r.outcome.status(),
            r.stats.exec_secs,
            r.stats.vc_secs,
            r.stats.block_syncs,
            r.stats.warp_syncs,
            elems,
        )?;
        if !r.passed {
            writeln!(out, "    UNEXPECTED: {}", describe(&r.outcome))?;
        }
    }
    Ok(())
}

/// Print all results grouped by category.
pub fn print_all_results(
    out: &mut impl Write,
    results: &[BenchmarkResult],
    iterations: usize,
) -> Result<()> {
    for category in BenchmarkCategory::all() {
        if results.iter().any(|r| r.category == category) {
            print_results_table(out, results, category, iterations)?;
        }
    }
    print_summary(out, results)
}

pub fn print_summary(out: &mut impl Write, results: &[BenchmarkResult]) -> Result<()> {
    let passed = results.iter().filter(|r| r.passed).count();
    writeln!(out, "\n{}/{} benchmarks passed", passed, results.len())?;
    for r in results.iter().filter(|r| !r.passed) {
        writeln!(out, "  FAILED {}: {}", r.name, describe(&r.outcome))?;
    }
    Ok(())
}

pub fn describe(outcome: &ActualOutcome) -> String {
    match outcome {
        ActualOutcome::Equivalent => "equivalent".to_string(),
        ActualOutcome::NotEquivalent { mismatches, first } => {
            format!("{} mismatched elements (first: {})", mismatches, first)
        }
        ActualOutcome::Rejected { description, .. } => description.clone(),
        ActualOutcome::RaceFree => "race-free".to_string(),
        ActualOutcome::Error { message } => message.clone(),
    }
}

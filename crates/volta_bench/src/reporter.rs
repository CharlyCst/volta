//! Result reporting: console tables and summaries. The machine-readable
//! results JSON lives in `crate::results`.

use std::io::Write;

use anyhow::Result;

use crate::config::BenchmarkCategory;
use crate::runner::{ActualOutcome, BenchmarkResult};
use crate::z3_phase::Z3PhaseOutcome;

/// Print one category's results as a table (paper-style columns, plus Z3
/// columns when the run had a Z3 phase). "Gen (s)" and "Solve (s)" (and
/// "Z3 (s)") are *medians* across the run's `iterations` (noted on the
/// header line).
pub fn print_results_table(
    out: &mut impl Write,
    results: &[BenchmarkResult],
    category: BenchmarkCategory,
    iterations: usize,
    z3: bool,
) -> Result<()> {
    writeln!(
        out,
        "\n{} ({}) [Gen/Solve (s): median of {} iteration(s)]",
        category.name(),
        category.table_ref(),
        iterations
    )?;
    write!(
        out,
        "{:<28} {:>7} {:>9} {:>9} {:>11} {:>11} {:>9}",
        "Benchmark", "Status", "Gen (s)", "Solve (s)", "#BlockSync", "#WarpSync", "Elems"
    )?;
    if z3 {
        write!(
            out,
            " {:>9}  {}",
            "Z3 (s)", "Z3: equiv/diff/unk/to/unsup/err"
        )?;
    }
    writeln!(out)?;
    writeln!(out, "{}", "-".repeat(if z3 { 133 } else { 90 }))?;
    for r in results.iter().filter(|r| r.category == category) {
        let elems = if r.stats.elements_checked == r.stats.elements_total {
            format!("{}", r.stats.elements_total)
        } else {
            format!("{}/{}", r.stats.elements_checked, r.stats.elements_total)
        };
        write!(
            out,
            "{:<28} {:>7} {:>9.2} {:>9.3} {:>11} {:>11} {:>9}",
            r.name,
            r.outcome.status(),
            r.stats.vc_gen_median_secs(),
            r.stats.solve_median_secs(),
            r.stats.block_syncs,
            r.stats.warp_syncs,
            elems,
        )?;
        if z3 {
            match &r.z3 {
                Some(Z3PhaseOutcome::Ran(phase)) => write!(
                    out,
                    " {:>9.3}  {}",
                    phase.plain.median_secs(),
                    phase.plain.counts.compact()
                )?,
                // Failed phase: flagged on its own line below.
                // Race-check/rejected/errored benchmarks: no Z3 phase.
                Some(Z3PhaseOutcome::Failed(_)) | None => {}
            }
        }
        writeln!(out)?;
        if let Some(Z3PhaseOutcome::Ran(phase)) = &r.z3
            && let Some(axiom) = &phase.axiom
        {
            writeln!(
                out,
                "{:<28} {:>7} {:>9} {:>9} {:>11} {:>11} {:>9} {:>9.3}  {}",
                "  +exp-axiom",
                "",
                "",
                "",
                "",
                "",
                "",
                axiom.median_secs(),
                axiom.counts.compact(),
            )?;
        }
        if let Some(Z3PhaseOutcome::Failed(message)) = &r.z3 {
            writeln!(out, "    Z3 PHASE FAILED: {}", message)?;
        }
        if !r.outcome_matched {
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
    z3: bool,
) -> Result<()> {
    for category in BenchmarkCategory::all() {
        if results.iter().any(|r| r.category == category) {
            print_results_table(out, results, category, iterations, z3)?;
        }
    }
    print_summary(out, results)
}

pub fn print_summary(out: &mut impl Write, results: &[BenchmarkResult]) -> Result<()> {
    let passed = results.iter().filter(|r| r.passed).count();
    writeln!(out, "\n{}/{} benchmarks passed", passed, results.len())?;
    for r in results.iter().filter(|r| !r.passed) {
        // Name every reason: an unexpected outcome, a failed Z3 phase,
        // or both.
        let mut reasons = Vec::new();
        if !r.outcome_matched {
            reasons.push(describe(&r.outcome));
        }
        if let Some(Z3PhaseOutcome::Failed(message)) = &r.z3 {
            reasons.push(format!("z3 phase failed: {}", message));
        }
        writeln!(out, "  FAILED {}: {}", r.name, reasons.join("; "))?;
    }
    Ok(())
}

/// The `single` command's detailed report for one benchmark.
pub fn print_single_result(out: &mut impl Write, result: &BenchmarkResult) -> Result<()> {
    writeln!(out, "Status:  {}", result.outcome.status())?;
    writeln!(out, "Detail:  {}", describe(&result.outcome))?;
    writeln!(out, "Passed:  {}", if result.passed { "yes" } else { "no" })?;
    writeln!(
        out,
        "VC gen:  {:.3}s median of {} iteration(s) (lowering + exec + footprint pairing)",
        result.stats.vc_gen_median_secs(),
        result.stats.vc_gen_iters_secs.len(),
    )?;
    // Race-check benchmarks (and failed runs) have no solve phase; don't
    // print a "0.000s median of 0 iteration(s)" line.
    if !result.stats.solve_iters_secs.is_empty() {
        writeln!(
            out,
            "Solve:   {:.3}s median of {} iteration(s)",
            result.stats.solve_median_secs(),
            result.stats.solve_iters_secs.len()
        )?;
    }
    match &result.z3 {
        None => {}
        Some(Z3PhaseOutcome::Failed(message)) => {
            writeln!(out, "Z3:      PHASE FAILED: {}", message)?
        }
        Some(Z3PhaseOutcome::Ran(phase)) => {
            writeln!(
                out,
                "Z3:      {:.3}s median of {} iteration(s)  [{}]",
                phase.plain.median_secs(),
                phase.plain.iters_secs.len(),
                phase.plain.counts.compact(),
            )?;
            if let Some(axiom) = &phase.axiom {
                writeln!(
                    out,
                    "Z3+ax:   {:.3}s median of {} iteration(s)  [{}]",
                    axiom.median_secs(),
                    axiom.iters_secs.len(),
                    axiom.counts.compact(),
                )?;
            }
        }
    }
    if let Some(path) = &result.dump_path {
        writeln!(out, "VC dump: {}", path.display())?;
    }
    writeln!(out, "Instrs:  {}", result.stats.instructions)?;
    writeln!(
        out,
        "Syncs:   {} block, {} warp",
        result.stats.block_syncs, result.stats.warp_syncs
    )?;
    writeln!(
        out,
        "Elems:   {} checked of {}",
        result.stats.elements_checked, result.stats.elements_total
    )?;
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

//! End-to-end check of the phase-decoupled workflow: `generate` then
//! `solve` must reach exactly the verdict the one-shot pipeline reaches,
//! from a dump plus manifest that `generate` really wrote. Runs the
//! smallest real equivalence benchmark, so this exercises the actual
//! kernels, symbolic execution, dump format, manifest, and decision
//! procedure (one iteration each - the timing loop is not under test).

use std::num::NonZeroUsize;
use std::path::PathBuf;

use volta_bench::{
    BenchmarkDef, BenchmarkRunner, KERNELS_DIR, RunnerConfig, SolveBackend, SolveItem,
    all_benchmarks, manifest, results,
};

fn benchmark(name: &str) -> BenchmarkDef {
    all_benchmarks()
        .benchmarks
        .iter()
        .find(|b| b.name == name)
        .unwrap_or_else(|| panic!("benchmark {} not found", name))
        .clone()
}

fn runner(vcs_dir: Option<PathBuf>) -> BenchmarkRunner {
    BenchmarkRunner::new(RunnerConfig {
        kernels_dir: PathBuf::from(KERNELS_DIR),
        vcs_dir,
        iterations: NonZeroUsize::MIN,
        ..RunnerConfig::default()
    })
}

#[test]
fn generate_then_solve_matches_the_one_shot_verdict() {
    let def = benchmark("(Red-1, Red-2)");
    let out = std::env::temp_dir().join(format!("volta_decoupled_{}", std::process::id()));
    let vcs_dir = out.join("vcs");

    // The one-shot pipeline (no dump involved): the reference verdict.
    let one_shot = runner(None).run(&def);
    assert_eq!(one_shot.outcome.status(), "EQUIV");
    assert!(one_shot.passed);

    // generate: dump + manifest written, gen stats only, no solve.
    let generated = runner(Some(vcs_dir.clone())).run_generate(&def);
    assert_eq!(generated.outcome.status(), "GEN");
    assert!(
        generated.passed,
        "generate must pass: {:?}",
        generated.outcome
    );
    assert!(!generated.stats.vc_gen_iters_secs.is_empty());
    assert!(
        generated.stats.solve_iters_secs.is_empty(),
        "generate must not solve"
    );
    let dump_path = generated
        .dump_path
        .as_ref()
        .expect("generate writes the dump");
    assert!(dump_path.exists());
    let manifest = manifest::read_manifest(&vcs_dir)
        .expect("manifest is readable")
        .expect("generate writes the manifest");
    let entry = &manifest.entries[&results::sanitize_name(&def.name)];
    assert_eq!(entry.benchmark, def.name);
    // Red benchmarks write out[0] only.
    assert_eq!(entry.reference_elements["out"], 1);

    // The record shapes: gen-only for generate.
    let record = results::generate_record(&generated);
    assert!(record.get("vc_gen_iters_secs").is_some());
    assert!(record.get("solve_iters_secs").is_none());

    // solve: from the dump, decision backend - the same verdict as the
    // one-shot run, same element coverage, and the load time recorded.
    let items = runner(Some(vcs_dir.clone()))
        .solve_suite(std::slice::from_ref(&def), SolveBackend::Decision)
        .expect("manifest is present and valid");
    assert_eq!(items.len(), 1);
    let SolveItem::Solved(solved) = &items[0] else {
        panic!("equivalence benchmarks are solved, not skipped");
    };
    assert_eq!(solved.outcome.status(), one_shot.outcome.status());
    assert!(solved.passed);
    assert_eq!(
        solved.stats.elements_checked,
        one_shot.stats.elements_checked
    );
    assert_eq!(solved.stats.elements_total, one_shot.stats.elements_total);
    assert!(solved.stats.dump_load_secs.is_some());
    assert!(
        solved.stats.vc_gen_iters_secs.is_empty(),
        "solve must not generate"
    );
    let record = results::solve_record(solved);
    assert!(record.get("dump_load_secs").is_some());
    assert!(record.get("vc_gen_iters_secs").is_none());

    // A race-check benchmark under solve: skipped, its verdict belongs
    // to generate (where it runs fully - covered by the smoke runs; the
    // interpreter work is identical to the one-shot path by
    // construction, as both call the same generation phase).
    let race = benchmark("Red-5 (racy)");
    let items = runner(Some(vcs_dir))
        .solve_suite(std::slice::from_ref(&race), SolveBackend::Decision)
        .unwrap();
    assert!(matches!(&items[0], SolveItem::Skipped { name, .. } if name == "Red-5 (racy)"));

    let _ = std::fs::remove_dir_all(&out);
}

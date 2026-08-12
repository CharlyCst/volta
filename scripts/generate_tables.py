#!/usr/bin/env python3
"""Generate the paper's evaluation tables from volta-bench results files.

Usage:
    python3 scripts/generate_tables.py <results-dir> [-o tables.md]
        [--kernels-dir crates/volta_bench/kernels]

<results-dir> holds the results JSONs written by the four-command
reproduction workflow (any file layout; runs are identified by header):

    generate all                                   -> generation times, sync
                                                      counters, race verdicts
    solve all (backend decision, sample 0)         -> full-footprint VC times
    solve all --sample 1 (backend decision)        -> Table 8 Volta column
    solve all --sample 1 --backend z3              -> Table 8 Z3 columns

If several files match one kind, the newest (by header timestamp) wins.
Missing runs degrade gracefully: the affected columns show "-" and the
Z3 table is omitted without a z3 run. PTX LOC is computed from the
kernel corpus when --kernels-dir exists, else shown as "-".

Values are medians over each run's iterations; a value whose phase had a
coefficient of variation above 0.10 is marked with "*". Warp syncs are
reported as Volta counts them: per fired group (multiply by 32 for the
per-thread convention used by the paper's Table 3).
"""

import argparse
import json
import statistics
import sys
from pathlib import Path

# --------------------------------------------------------------------------
# Table specifications: (paper row label, benchmark name in the results,
# PTX file(s) for the LOC column, threads-column entry or None).
# The benchmark names are the registry names from volta_bench.

RED = [(f"Red-{i}", f"(Red-1, Red-{i})", [f"01_reduction/Red-{i}.ptx"], None) for i in (1, 2, 3, 4)]
MATMUL = [
    (f"MatMul-{i}", f"(MatMul-1, MatMul-{i})", [f"02_matmul/MatMul-{i}.ptx"], None)
    for i in range(1, 8)
]
ATTENTION = [
    ("Attention", "(Attention, Attention)", ["03_attention/Attention.ptx"], "1"),
    ("FA1", "(Attention, FA1)", ["03_attention/FA1.ptx"], "128"),
    ("FA1-TC", "(Attention, FA1-TC)", ["03_attention/FA1-TC.ptx"], "128"),
    ("FA2-TC", "(Attention, FA2-TC)", ["03_attention/FA2-TC.ptx"], "128"),
]
CAUSAL = [
    (
        f"Causal-{k}",
        f"(Causal-{k}-naive, Causal-{k}-fused)",
        [f"04_causal_attention/Causal-{k}-fused.ptx", f"04_causal_attention/Causal-{k}-naive.ptx"],
        "1" if k == "Attention" else "128",
    )
    for k in ("Attention", "FA1", "FA1-TC", "FA2-TC")
]
CONV = [("Conv2D", "(Conv2D-ref, Conv2D-opt)", ["05_conv2d_llm/Conv2D-opt.ptx"], "256")]
GEMM = [
    (f"GEMM-{i}", f"(MatMul-1-32x32, GEMM-{i})", [f"06_agent_gemm/GEMM-{i}.ptx"], "512")
    for i in (1, 2, 3)
]
TILELANG = [
    (
        size,
        f"(TL-{size}-ref, TL-{size}-opt)",
        [f"07_tilelang/{size}-ref.ptx", f"07_tilelang/{size}-opt.ptx"],
        "(128, 128)",
    )
    for size in ("32x32x32", "64x32x32", "64x64x32")
]
RACES = [
    ("BucketPositions", "OpenMM", "08_races/BucketPositions-pre_racy.ptx"),
    ("ComputeRange", "OpenMM", "08_races/ComputeRange-pre_racy.ptx"),
    ("ReduceValue", "OpenMM", "08_races/ReduceValue-pre_racy.ptx"),
    ("LayerNorm", "MegatronLM", "08_races/LayerNorm-pre_racy.ptx"),
    ("GradInput", "MegatronLM", "08_races/GradInput-pre_racy.ptx"),
]
# Table 8 rows in paper order: label -> benchmark name.
TABLE8 = RED + MATMUL + CONV + GEMM + TILELANG + ATTENTION + CAUSAL


def load_runs(results_dir: Path):
    """Discover the four runs by header; newest timestamp wins per kind."""
    runs = {}
    for path in sorted(results_dir.glob("*.json")):
        try:
            doc = json.loads(path.read_text())
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if "benchmarks" not in doc or "command" not in doc:
            continue
        cmd, backend, sample = doc["command"], doc.get("backend"), doc.get("sample")
        if cmd == "generate":
            kind = "generate"
        elif cmd == "solve" and backend in ("decision", "both") and sample == 0:
            kind = "full"
        elif cmd == "solve" and backend in ("decision", "both") and sample == 1:
            kind = "sample1"
        elif cmd == "solve" and backend in ("z3", "both") and sample == 1:
            kind = "z3"
        else:
            continue
        stamp = doc.get("timestamp_unix", 0)
        if kind not in runs or stamp > runs[kind][0]:
            runs[kind] = (stamp, path.name, doc)
    return {k: (name, {b["name"]: b for b in doc["benchmarks"]}, doc) for k, (_, name, doc) in runs.items()}


def fmt_secs(v):
    if v is None:
        return "-"
    if v < 0.0001:
        return f"{v*1000:.2f} ms"
    if v < 1:
        return f"{v:.4f}"
    return f"{v:.2f}" if v < 100 else f"{v:.1f}"


def median_cell(record, prefix):
    """Median of a timed phase, '*'-marked when its CV exceeds 0.10."""
    if record is None:
        return "-"
    med = record.get(f"{prefix}_median_secs")
    if med is None:
        return "-"
    cv = record.get(f"{prefix}_cv")
    return fmt_secs(med) + ("*" if cv is not None and cv > 0.10 else "")


def loc_cell(kernels_dir, files):
    if kernels_dir is None:
        return "-"
    counts = []
    for f in files:
        p = kernels_dir / f
        if not p.exists():
            return "-"
        counts.append(sum(1 for _ in p.open()))
    return str(counts[0]) if len(counts) == 1 else "(" + ", ".join(map(str, counts)) + ")"


def z3_cell(record, section):
    """Render one z3 mode: outcome@time of the first element, or counts."""
    z = (record or {}).get("z3")
    sec = z if section == "plain" else (z or {}).get("axiom")
    if not sec:
        return "-"
    els = sec.get("elements") or []
    if len(els) == 1:
        el = els[0]
        t = el["solve_secs"]
        when = f"{t*1000:.1f} ms" if t < 1 else f"{t:.1f} s"
        return f"{el['outcome']} @ {when}"
    c = sec["counts"]
    total = sec.get("solve_median_secs")
    return (
        f"eq {c['equivalent']} / df {c['not_equivalent']} / unk {c['unknown']} / to {c['timeout']}"
        + (f" @ {fmt_secs(total)} s" if total is not None else "")
    )


def equivalence_table(out, title, rows, columns, gen, full, kernels_dir):
    """One paper-shaped table over equivalence benchmarks.

    columns is a subset of {"warp", "loc", "threads"}; the VC-generation
    column (the addition over the paper's shape) is always present.
    """
    header = ["Kernel", "VC Gen (s)", "VC Time (s)", "#Block Sync"]
    if "warp" in columns:
        header.append("#Warp Sync (groups)")
    if "loc" in columns:
        header.append("PTX LOC")
    if "threads" in columns:
        header.append("#Threads")
    out.append(f"## {title}\n")
    out.append("| " + " | ".join(header) + " |")
    out.append("|" + "---|" * len(header))
    for label, name, files, threads in rows:
        g, f = gen.get(name), full.get(name)
        cells = [
            label,
            median_cell(g, "vc_gen"),
            median_cell(f, "solve"),
            f"{g['block_syncs']:,}" if g else "-",
        ]
        if "warp" in columns:
            cells.append(f"{g['warp_syncs']:,}" if g else "-")
        if "loc" in columns:
            cells.append(loc_cell(kernels_dir, files))
        if "threads" in columns:
            cells.append(threads or "-")
        out.append("| " + " | ".join(cells) + " |")
    out.append("")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results_dir", type=Path)
    ap.add_argument("-o", "--output", type=Path, help="write markdown here (default stdout)")
    ap.add_argument(
        "--kernels-dir",
        type=Path,
        default=Path("crates/volta_bench/kernels"),
        help="kernel corpus for the PTX LOC columns",
    )
    args = ap.parse_args()

    runs = load_runs(args.results_dir)
    if not runs:
        sys.exit(f"no volta-bench results files found in {args.results_dir}")
    gen = runs.get("generate", (None, {}, None))[1]
    full = runs.get("full", (None, {}, None))[1]
    sample1 = runs.get("sample1", (None, {}, None))[1]
    z3 = runs.get("z3", (None, {}, None))[1]
    kernels_dir = args.kernels_dir if args.kernels_dir.is_dir() else None

    out = []
    out.append("# Volta evaluation tables\n")
    out.append("Sources (newest per kind):\n")
    for kind, label in [("generate", "generation"), ("full", "full solve"), ("sample1", "sample-1 solve"), ("z3", "z3 solve")]:
        if kind in runs:
            name, _, doc = runs[kind]
            extra = f", z3 {doc['z3_version']}" if doc.get("z3_version") else ""
            out.append(f"- {label}: `{name}` (iterations {doc.get('iterations')}{extra})")
        else:
            out.append(f"- {label}: MISSING")
    out.append("")
    failed = [
        (kind, b["name"])
        for kind, (_, by_name, _) in runs.items()
        for b in by_name.values()
        if not b.get("passed", True)
    ]
    out.append(
        "All benchmark rows passed in every run.\n"
        if not failed
        else "**FAILED ROWS:** " + ", ".join(f"{n} ({k})" for k, n in failed) + "\n"
    )
    out.append(
        "Times are medians over each run's iterations (`*` = CV > 0.10). VC Gen = "
        "lowering + both symbolic executions + footprint pairing; VC Time = the "
        "decision procedure over all footprint elements. Warp syncs are per fired "
        "group (x32 for the per-thread convention).\n"
    )

    equivalence_table(out, "Table 1: reduction (vs Red-1)", RED, {"loc"}, gen, full, kernels_dir)
    equivalence_table(out, "Table 2: matmul (vs MatMul-1)", MATMUL, {"loc"}, gen, full, kernels_dir)
    equivalence_table(out, "Table 3: attention (vs Attention)", ATTENTION, {"warp", "loc", "threads"}, gen, full, kernels_dir)
    equivalence_table(out, "Table 4: causal attention (fused vs naive); LOC = (fused, naive)", CAUSAL, {"warp", "loc", "threads"}, gen, full, kernels_dir)
    equivalence_table(out, "Conv2D (paper section 6.2)", CONV, {"warp", "loc", "threads"}, gen, full, kernels_dir)
    equivalence_table(out, "Table 5: Claude-generated GEMM (vs MatMul-1 32x32)", GEMM, {"warp", "loc", "threads"}, gen, full, kernels_dir)
    equivalence_table(out, "Table 6: TileLang (ref vs opt); LOC = (ref, opt)", TILELANG, {"warp", "loc", "threads"}, gen, full, kernels_dir)

    out.append("## Table 7: racy FaialAA benchmarks\n")
    out.append("| Kernel | Library | PTX LOC | pre-fix | Check (s) | post-fix | Check (s) |")
    out.append("|---|---|---|---|---|---|---|")
    for label, lib, loc_file in RACES:
        pre, post = gen.get(f"{label} (pre-fix)"), gen.get(f"{label} (post-fix)")
        out.append(
            f"| {label} | {lib} | {loc_cell(kernels_dir, [loc_file])} "
            f"| {pre['status'] if pre else '-'} | {median_cell(pre, 'vc_gen')} "
            f"| {post['status'] if post else '-'} | {median_cell(post, 'vc_gen')} |"
        )
    out.append("")

    if z3:
        doc = runs["z3"][2]
        out.append(
            f"## Table 8: single-element VC, Volta decision procedure vs Z3 "
            f"(budget {doc.get('z3_timeout_secs', '?')} s)\n"
        )
        out.append("| Kernel | Volta (ms) | Z3 | Z3 + exp axiom |")
        out.append("|---|---|---|---|")
        for label, name, _, _ in TABLE8:
            s, zr = sample1.get(name), z3.get(name)
            volta = f"{s['solve_median_secs']*1000:.2f}" if s else "-"
            out.append(f"| {label} | {volta} | {z3_cell(zr, 'plain')} | {z3_cell(zr, 'axiom')} |")
        out.append("")

    known = {name for _, name, *_ in TABLE8} | {f"{l} ({s}-fix)" for l, _, _ in RACES for s in ("pre", "post")}
    # The racy tutorial reductions appear in the paper's prose, not a table.
    known |= {f"Red-{i} (racy)" for i in (5, 6, 7)}
    for kind, (_, by_name, _) in runs.items():
        extra = [n for n, b in by_name.items() if n not in known and b.get("status") != "SKIP"]
        if extra:
            out.append(f"Benchmarks in the {kind} run without a table row: {', '.join(sorted(extra))}\n")

    text = "\n".join(out)
    if args.output:
        args.output.write_text(text)
        print(f"wrote {args.output}", file=sys.stderr)
    else:
        print(text)


if __name__ == "__main__":
    main()

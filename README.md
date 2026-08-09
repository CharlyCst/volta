# Volta

> ⚠️ This is not the code that was used for the paper. That code was written by Kshitij Dubey at MSR and they have not agreed to released it. Hopefully, this reconstruction is useful; it appears to work on all the benchmarks from the paper. I've been quite busy, so some parts were written almost entirely by Fable 5 and I did not have time to do extensive quality assurance. ⚠️

Volta is a data race and equivalence checker for NVIDIA GPU kernels, implementing the approach from ["Equivalence Checking of ML GPU Kernels"](https://arxiv.org/pdf/2511.12638). Given a reference kernel implementation and an optimized counterpart, Volta proves their semantic equivalence over the reals, i.e., that they produce identical outputs for all valid inputs modulo floating point error, thereby verifying the correctness of the optimized kernel.

## Features

- **Deadlock Detection**: Identify deadlocks arising from over synchronization
- **Data Race Detection**: Identify races arising from under synchronization
- **Equivalence Checking**: Verify that optimized GPU kernels are semantically equivalent to their reference implementations
- **Two-kernel equivalence from the CLI**: `volta compare` checks a reference/optimized pair directly, without going through `volta_bench`
- **VC dump/replay**: persist the verification conditions from one run and rerun just the equivalence check later, skipping parse/lower/symbolic-execution entirely
- **Per-run logging and execution profiling**: every run gets a log file, and a per-instruction-kind execution profile is shown by default
- **Z3 backend**: check the same verification conditions with Z3 instead of the built-in decision procedure, for a "decides vs. cannot decide" timing/capability comparison

## How It Works

Volta has two phases:

1. **Symbolic Execution**: Executes both kernels symbolically (round-robin over all threads of CTA 0), tracking memory accesses and synchronization to detect data races and deadlocks and producing symbolic expressions representing output values as functions of input tensors.

2. **Equivalence Checking**: Verifies that the symbolic expressions from both kernels are mathematically equal over the reals. Each output element canonicalizes to a rational function whose polynomials are sums `c * monomial * e^{poly}` terms with exact rational coefficients. An optional `f64` oracle (`--verify-numeric`) re-checks every verdict at seeded random inputs.

## Soundness and Completeness

Equivalence checking treats floating-point values as reals. Within that model:

- Race and deadlock detection is sound and complete for structured-CTAs (see
  [requirements](#requirements)) using `+`, `-`, `*`, `/`, `exp`, and `max`/`min`
  with symmetric CTAs (only CTA 0 is checked, but note that the grid size still
  matters for index computations).

- `sqrt`, `log`, `abs`, `rem`, floor, bitwise ops, shifts, comparisons, boolean
  ops, `select`, and data-dependent array reads are carried as uninterpreted
  atoms, equal only when syntactically identical after canonicalizing their
  arguments. We lose completeness but not soundness.

## Requirements

The input to Volta is PTX code (the lowest level of the public-facing language stack for NVIDIA GPUs). PTX files can be generated from CUDA or CUTLASS code using `nvcc`.

We require that kernels are _structured-CTAs_. That is:

- Tensor/array sizes are statically known
- Branch targets and memory addresses can be resolved statically given the grid dimensions and input arrays
- There is no recursion

The only synchronization primitives we currently support are barriers, such as `syncwarp`, `syncthreads`, and the implicit warp-level barriers of tensor core operations (`mma.sync`, `wmma.*`, `ldmatrix`, `shfl.sync`). We do not support asynchronous primitives such as `arrive` and `wgmma`.

## Building

```bash
cargo build --release   # release mode matters: ~20x faster analysis
cargo test --workspace  # run the test suite
```

## Usage

### Parse a PTX file (syntax check)

```bash
cargo run --release -- parse <file.ptx>
```

### Analyze one kernel

Symbolically executes a kernel: reports data races and deadlocks, and prints
the symbolic expressions for each output array element.

```bash
cargo run --release -- analyze <file.ptx> -k <kernel> -b 32,4 -g 1 \
    --array "vals:0x100000000:4:2048:in" \
    --array "out:0x200000000:4:2048:out" \
    --param ptr:out --param ptr:vals --param int:2048 \
    --dyn-shared 1024
```

- `-k, --kernel`: kernel entry name (defaults to the first kernel in the module)
- `-b, --block` / `-g, --grid`: launch dimensions, e.g. `128` or `32,4,1`
- `--array "name:base:elem_width:len:kind"` (repeatable): declares a global
  array at address `base` with `len` elements of `elem_width` bytes; `kind`
  is `in` (symbolic input), `out`, `inout`, or `index` (concrete
  `arr[i] = i`, for index/permutation inputs)
- `--param` (repeatable, in declaration order): `int:N`, `float:X`,
  `sym:name` (a named symbolic float), or `ptr:array_name`
- `--global NAME=value` (repeatable): module-scope `.global` variable values
- `--dyn-shared N`: dynamic (extern) shared memory bytes
- `--print-outputs N`: print up to N elements per output array (default 8)
- `--no-profile`: skip the per-instruction-kind execution profile (shown by default)

### Compare two kernels

`volta compare` checks a reference/optimized pair for equivalence directly
from the CLI (races/deadlocks are still checked for each kernel individually).
Arrays/params/globals are shared by both kernels by default; give `--block2`/
`--grid2` if the optimized kernel's launch config differs (e.g. a
single-thread reference vs. a 128-thread optimized kernel computing the
same tile). Comparison follows the paper's CTA-to-CTA model: it runs
along the declared output arrays (`out`/`inout` kinds), and both CTA-0
runs must write each of them with identical per-array footprints.
Arrays not declared as outputs are not compared - which is how
auxiliary exports like FlashAttention's softmax statistics stay out of
a comparison against a reference that never computes them.

```bash
cargo run --release -- compare <ref.ptx> <opt.ptx> \
    --kernel1 <ref_kernel> --kernel2 <opt_kernel> -b 128 \
    --array "in:0x10000:4:128:in" --array "out:0x20000:4:1:out" \
    --param ptr:in --param ptr:out
```

- `--sample N`, `--verify-numeric`, `--recycle-terms N`: same meaning as the
  `volta_bench` flags below
- `--no-profile`: skip the per-instruction-kind execution profile (shown by default)
- `--backend decision|z3` (default `decision`): which decision procedure to
  check equivalence with - see [Z3 backend](#z3-backend)
- `--exp-axiom` (with `--backend z3`): the paper's "with axiom" exp
  encoding - see [Z3 backend](#z3-backend)

Exit code: `compare` exits 0 only when every checked element was proved
equivalent (with `--backend z3` that excludes `unknown`/`timeout`/
`unsupported`/error elements, not just mismatches - a run that verified
nothing does not exit 0).

**VC dump/replay**: after symbolic execution, persist both kernels'
verification conditions (the expression arena + output footprint) to disk,
then rerun just the equivalence check from that dump later - no PTX parsing,
lowering, or symbolic execution involved on replay.

```bash
cargo run --release -- compare <ref.ptx> <opt.ptx> ... --dump-vcs pair.vcdump
cargo run --release -- compare --from-dump pair.vcdump   # rerun later, instantly
```

Dump files carry a magic/version header and are validated on load, so a
truncated, corrupted, or version-skewed dump fails with a clean error
rather than a crash.

### Logging

Every `volta`/`volta-bench` run writes a log file under `volta-logs/`
(`<unix-seconds>-<pid>-<command>.log`; the pid keeps two runs in the same
second from clobbering each other), recording the exact command line and a
one-line outcome summary - independent of the `logging` feature, so it
works in a plain build. Pass `--log-dir <path>` to change the directory or
`--no-log-file` to disable it. Building with `--features logging` also
mirrors the `log` crate's trace/debug/info/warn output into the same file
(`--log-level`), in addition to stderr:

```bash
cargo run --release --features logging -- --log-level info analyze ...
```

### Z3 backend

Checks the same verification conditions with Z3 instead of Volta's own
decision procedure, for a timing/capability comparison. Queries are
generated as SMT-LIB2 text (auditable: any query can be replayed against
a standalone z3) and evaluated through libz3's C API - a hand-written
eight-function binding, no `z3-sys`/bindgen/libclang, no temp files, and
no `z3` binary needed at runtime (each query runs in a worker subprocess,
but that worker is this same binary re-invoked - see the timeout note
below). The one prerequisite is the Z3 shared library at build time:

```bash
sudo apt-get install -y libz3-dev
cargo run --release -- compare <ref.ptx> <opt.ptx> ... --backend z3
```

The expected shape of the results reproduces the paper's section 6.5 and
Table 8. Z3 decides the polynomial fragment (reduction/matmul/conv) in
milliseconds. On the exponential fragment (softmax/attention) there are
two documented baselines, and the backend implements both:

- **Default encoding**: the exponential is a nonlinear power term with a
  strictly-bounded base. Z3 returns `unknown` - no decision procedure
  covers symbolic real exponents. This is the paper's no-intervention
  baseline.
- **`--exp-axiom`**: the exponential becomes an uninterpreted function
  plus the addition-law axiom `forall x y. e^x e^y = e^(x+y)` (the
  paper's "Z3 with axiom" setup). The axiom sends Z3 into an unbounded
  quantifier-instantiation loop on softmax-shaped VCs, so instead of a
  fast `unknown` the query runs until the time budget kills it -
  reported as `timeout`. The paper used a 10-minute budget.

`--z3-timeout` is a *hard* per-query bound: z3 4.8.12 does not reliably
honor its own soft timeout in the axiom-induced loop (measured: a
3-second soft timeout still running after 90 seconds), so each query
evaluates in a worker subprocess (the binary re-invoking itself; no
separate executable) that is killed on expiry. `timeout` in the
element counts means the budget expired; `unknown` means z3 itself gave
up with budget to spare.

Because the translation deliberately does no reasoning, *every* element
costs a genuine spawn+solve - trivially identical sides included, at
tens of milliseconds each. A full-footprint run over a large output
(tens of thousands of elements) therefore takes hours where the decision
procedure takes seconds; that gap is a result, not an inefficiency. Use
`--sample` to bound the element count (Table 8 uses `--sample 1`).

Covers the arithmetic + `Exp` + `Max`/`Min` fragment as a **direct
semantic image**: every expression node maps to its defining SMT term
and all algebraic reasoning (commutativity, cancellation, distribution,
max/min case analysis - `max`/`min` render as `ite` over real
comparisons) is left to the solver, so the timings measure Z3, not the
translator. What the translation owns is fidelity and transport: float
constants as their exact binary values (the same reading as the decision
procedure and the numeric oracle), user symbol names in reserved
namespaces so they cannot collide with generated solver names, and
`let`-bound DAG sharing so query text stays linear in the expression
arena. One inherited SMT semantic: real division is total but
underspecified at zero, so field identities like `x/x = 1` are
falsifiable (countermodel `x = 0`) - corpus VCs only divide inside
exp-laden softmax terms, where the verdict is `unknown` regardless.
Anything outside the fragment (`Select`, comparisons, bitwise ops,
data-dependent array reads, ...) is reported `unsupported` for that
element rather than guessed at unsoundly. See
`crates/volta_z3/src/translate.rs` for the exact boundary.

### Reproduce the paper's evaluation

`volta_bench` runs every benchmark from the paper (39 in total) over the PTX
collected in `crates/volta_bench/kernels/`.

```bash
cargo run --release -p volta_bench -- list
cargo run --release -p volta_bench -- all
cargo run --release -p volta_bench -- category <reduction|matmul|attention|causal|conv|agent|tilelang|race>
cargo run --release -p volta_bench -- single "(Attention, FA1)"
```

Useful flags (global):

- `--sample N`: check at most N output elements per array (0 = all)
- `--verify-numeric`: confirm every verdict with the f64 oracle
- `--recycle-terms N`: recycle the VC intern tables past N interned terms
  (0 = never). Lower values bound memory at the cost of re-canonicalizing
  shared structure
- `--json <path>` (on `all`/`category`): export results as JSON

`single` also prints a per-instruction-kind execution profile for both
kernels automatically (matching `volta compare`'s default); `all`/`category`
stay compact and don't, to avoid flooding the table with one profile per
benchmark row.

To compare against Z3 instead of (or alongside) the decision procedure, use
`z3-compare` (needs `z3` on `PATH` - see [Z3 backend](#z3-backend)):

```bash
cargo run --release -p volta_bench -- z3-compare all --json results.json
cargo run --release -p volta_bench -- z3-compare reduction
cargo run --release -p volta_bench -- z3-compare "(Attention, FA1)"
```

For every equivalence benchmark matched by the selector (`all`, a category,
or an exact benchmark name), this runs *both* backends and prints exec/
decision/Z3 timing side by side, plus Z3's per-element equivalent/
not-equivalent/unknown/timeout/unsupported/error breakdown. Benchmarks
whose VCs contain exponentials additionally get a `+exp-axiom` sub-row:
the same elements rerun under the paper's addition-law-axiom encoding
(expected outcome: `timeout`, versus `unknown` on the default row - see
[Z3 backend](#z3-backend)). Race-check benchmarks (no optimized kernel)
are skipped with a note when matched by `all` or a category.
`--z3-timeout N` hard-bounds each Z3 query in seconds (default 30, `0` =
no limit); the global `--sample` flag applies to both backends, and
`--recycle-terms`/`--verify-numeric` to the decision-procedure column,
exactly as in the default commands. Exits nonzero if any benchmark row
failed outright. The default `all`/`category`/`single` commands never
invoke Z3 - `z3-compare` is opt-in.

To reproduce the paper's Table 8 exactly - one element per output tensor,
a 10-minute budget per query:

```bash
cargo run --release -p volta_bench -- --sample 1 z3-compare all --z3-timeout 600
```

(Expect the attention/causal rows to spend the full budget per element on
their `+exp-axiom` sub-rows.) One caveat on small machines: the
axiom-induced grind also eats memory, and if z3 exhausts memory before
the deadline it gives up with `unknown` instead of surviving to the kill
(measured on a 15 GiB box: the memory wall arrives after roughly half a
minute of grinding; the paper's 10-minute timeouts assume its 220 GB
machine). Pick a budget below the memory wall - e.g. `--z3-timeout 20` -
to see the `timeout` outcome on constrained hardware, and run one
category at a time.

**Memory note**: symbolic execution plus a warm VC session can use tens of
GiB on the attention benchmarks (each output row retains a large shared
softmax denominator). On machines with limited RAM, run one category at a
time and bound the VC tables, e.g.:

```bash
bash -c 'ulimit -v 12582912; exec cargo run --release -p volta_bench -- \
    --recycle-terms 250000 category attention'
```

which holds peak memory near the symbolic-execution floor (~5 GiB) in
exchange for slower VC checking. The other categories are far lighter
(full matmul: ~2 GiB).

## Citation

```bibtex
@misc{dubey2025equivalencecheckingmlgpu,
      title={Equivalence Checking of ML GPU Kernels},
      author={Kshitij Dubey and Benjamin Driscoll and Anjiang Wei and Neeraj Kayal and Rahul Sharma and Alex Aiken},
      year={2025},
      eprint={2511.12638},
      archivePrefix={arXiv},
      primaryClass={cs.PL},
      url={https://arxiv.org/abs/2511.12638},
}
```

## License

This repository is licensed under [LICENSE](LICENSE). This implementation of Volta is completely independent from the Python implementation mentioned in the evaluation section of the arxiv paper, which was written by Kshitij Dubey and is owned by Microsoft Research. While I was a co-author of that paper, I never viewed Kshitij's implementation, nor did I discuss any details of it not presented in the arxiv paper.

# Volta Architecture Documentation

**Purpose**: Codebase context for AI-assisted development

## Overview

Volta is an abstract interpreter for NVIDIA PTX kernels, implementing the approach from "Equivalence Checking of ML GPU Kernels" (arXiv:2511.12638). It detects data races and verifies kernel equivalence.

## Coding practices

- Take advantage of the type system to ensure correctness. E.g., rather than using a `u32`, to avoid mixing up different kinds of indices, consider using a new type that wraps a `u32`. Likewise, consider where it is better to use a custom, two variant enum in place of a `bool`.
- Shared state makes reasoning about code complex. Err on the side of slightly less efficient but pure implementations.

## Crate Structure

```
crates/
├── volta_common/     # Base utilities (spans, file caching, error reporting, run logs)
├── volta_frontend/   # PTX lexer and parser
├── volta_analysis/   # Abstract interpreter
├── volta_z3/         # Z3 comparison backend (SMT-LIB2 via linked libz3)
├── volta_bench/      # Paper-evaluation benchmark harness
└── volta_cli/        # Command-line interface
```

### Dependency Graph

Direct dependencies of each crate:

```
volta_cli      → volta_z3, volta_analysis, volta_frontend, volta_common
volta_bench    → volta_z3, volta_analysis, volta_frontend, volta_common
volta_z3       → volta_analysis
volta_analysis → volta_frontend, volta_common
volta_frontend → volta_common
volta_common   → (nothing)
```

## Crate: volta_common

**Path**: `crates/volta_common/`

- `Span` - Source location (low + high byte offset)
- `FileCache` - Caches file content to make sure we always use a consistent version of each file
- `Locate<E>` - Error wrapper with optional location info (span + file path)
- `report_error` - Produces an error message from a title, message, span, and file content. Extracts out and includes the code snippet at the given span in the given file content
- `run_log::RunLog` - Per-invocation log file (`<unix-seconds>-<pid>-<command>.log` under `--log-dir`, default `volta-logs/`), shared by the `volta` and `volta-bench` binaries; `tee` mirrors an `env_logger` target into it under the binaries' `logging` features

The pattern is to create an error kind type, and then an alias for `Locate` of that error kind. `locate_span` can be used to tag a `Locate` with a span if it does not already have one. `locate_path` can be used to tag a `Locate` with a path if it does not already have one.

## Crate: volta_frontend

**Path**: `crates/volta_frontend/`

### Lexer (`lex.rs`)

Tokenizes PTX source. Key methods: `next()`, `peek()`, `expect(kind)`.

### Parser (`parse.rs`)

Pratt parser producing AST. Entry point: `parse_module()`.

### AST (`ast.rs`)

- `Module` - Top-level: version, target, address_size, directives
- `Function` - Kernel/device function with params and body
- `Instruction` - Generic instruction with mnemonic string and operands
- `ScalarType` - Pred, Signed/Unsigned/Float/Bits with width

### Instruction Parsing (`instr.rs`, `instr_parse.rs`)

- `InstrTrie` - O(n) lookup for PTX mnemonics → `InstrKind`
- `ParsedInstruction` - Strongly-typed enum (~80 instruction variants)
- Converts generic `Instruction` to typed variants with validated modifiers

## Crate: volta_analysis

**Path**: `crates/volta_analysis/`

### ID Types

Strongly-typed IDs (`#[id_type]` from `id_collections`), each declared next
to its subsystem:

- `InstrId` (`lowered.rs`), `ThreadId` (`eval/mod.rs`), `ParamId` and
  `RegId` = `RegClass` + `RegIndex` (`symbols.rs`; class: Pred,
  Bits8/16/32/64/128), `ExprId`/`StringId`/`SymbolId` (`symbolic.rs`;
  `SymbolId::fresh()` draws from a process-global counter)
- `types.rs` holds `ScalarTypeExt` (width/kind helpers over the AST's
  `ScalarType`), not IDs

### Symbolic Expressions (`symbolic.rs`)

Arena-allocated: nodes live in an `ExprArena`, referenced by copyable `ExprId`
handles. Constructors constant-fold eagerly - **exactly**: float constants
are `Real`s (arbitrary-precision rationals via `rug`, boxed, plus
`NegInf`/`PosInf`; NaN is rejected at every f64 ingestion point -
`Real::from_f64`/`arena.float_from_f64` are fallible, NaN literals are a
lowering error, NaN params a config validation error), so the fold algebra
and canon's rational algebra coincide by construction. Folds are exact on
ℚ (div/rcp fold only fully-concrete quotients with a nonzero divisor -
`x/0` and `0/symbolic` both stay unfolded, so a formally-zero
denominator always reaches canon's loud division error); ±inf folds
only the unambiguous extended-real forms (max/min absorption, neg,
inf±finite, inf·nonzero), undefined forms (inf−inf, 0·inf - integer or
real zero alike - anything/0) build unfolded nodes.

- **Atoms**: `IntConst`, `RealConst(Real)`, `BoolConst`, `Symbol(SymbolId)`, `ParamSymbol(StringId)`, `InputElement { array, index }`, `Undefined`
- Symbol identity is typed (`SymbolRef`: `Param`/`Element`/`Machine`,
  disjoint namespaces; one mapping in `ExprNode::symbol_ref`). Identity
  comes only from launch-config names - PTX-source names are scoped and
  must not carry identity; values without a config binding are fresh
  machine `Symbol`s. `AnalysisConfig::validate` rejects ambiguous configs.
- **Arithmetic**: `Add`, `Sub`, `Mul`, `Div`, `Rem`, `Neg`, `Fma`
- **Transcendental**: `Exp`, `Log`, `Sqrt`, `Rcp`
- **Bitwise**: `BitAnd`, `BitOr`, `BitXor`, `BitNot`, `Shl`, `Shr`, `LShr`
- **Comparison**: `Eq`, `Ne`, `Lt`, `Le`, `Gt`, `Ge` (return boolean)
- **Boolean**: `And`, `Or`, `Not`
- **Other**: `Select` (ternary), `Min`, `Max`, `Abs`, type conversions

### Lowering (`lowering.rs`, `lowered.rs`)

Converts AST to linear instruction format:

- `LoweredProgram` - `IdVec<InstrId, LoweredInstr>` + `SymbolTable` + `SourceMap`
- `LoweredInstr` variants: `LoadParam`, `Load`, `Store`, `Mov`, `BinOp`, `UnaryOp`, `Fma`, `Mad`, `MulWide`, `MulHi`, `Setp`, `Selp`, `Cvt`, `Bra`, `Ret`, `Exit`, `BarSync`, `BarWarpSync`, `ShflSync`, `Ldmatrix`, `Mma`, `WmmaLoad/Store/Mma`, `Activemask`, `Trap`, etc.
- `SymbolTable` - Register/param/label name → ID resolution; assigns addresses to shared/local/module-global variables
- `SourceMap` - Maps lowered elements back to source spans
- The nvcc callseq idiom for `call __symexpf` (the paper's symbolic-exp hook) collapses to `UnaryOp::Exp` at lowering time
- `define_instr_kinds!` generates the profiling table from one variant
  list: `KIND_COUNT`, `KIND_NAMES`, `kind_index()` (dense index for the
  interpreter's fixed-size counters), `kind_name()`

### Special Registers (`symbols.rs`)

`SpecialRegKind`: `TidX/Y/Z`, `NtidX/Y/Z`, `CtaidX/Y/Z`, `LaneId`, `WarpId`, etc.

### Evaluator (`eval/`)

The interpreter from the paper (per-thread round-robin symbolic execution):

- `eval/interp.rs` - `Interpreter`: scheduler (run a thread until it blocks or exits), instruction evaluation into the arena, barrier firing per the paper's Sync rule (exited threads count as arrived), deadlock detection, structured-CTA concreteness checks
- `eval/value.rs` - `Value::{Scalar, Pair}` (`Pair` = packed f16 halves in a 32-bit register) and per-thread `RegFile`
- `eval/memory.rs` - byte-addressed granule memory; 4-byte reads combine two 2-byte granules into a `Pair`, 2-byte accesses split `Pair` granules; program writes are `dirty` (the output footprint)
- `eval/race.rs` - χ-context race detection per byte (paper Section 3.2); full-CTA barrier sync is a wholesale clear
- `eval/warp.rs` - warp-cooperative ops (`shfl.sync`, `ldmatrix`, `mma.sync`, `wmma.*`): block until all mask lanes converge at the pc, sync χ, execute via the `tensor_core.rs` fragment tables with exact per-lane access attribution
- `eval/config.rs` - `AnalysisConfig`: launch dims, positional `ParamValue`s (int/float/symbolic-float/array-pointer), `ArrayDef`s (`Input`/`Output`/`InputOutput`/`IndexInput`), module-global values, dynamic shared size
- `eval/error.rs` - `EvalError`: `DataRace`, `Deadlock`, `NotConcrete`, `OutOfBounds`, `UndefinedOutput`, `TrapReached`, etc.

Key semantics: input-array symbols materialize lazily on first read; reads of
never-written registers/shared bytes yield `Undefined` (an error only if it
reaches an output or a concreteness point) - the paper's race example and
nvcc's `selp` accumulator-init idiom both rely on this.

### Driver (`driver.rs`)

- `analyze_kernel(module, kernel_name, config) -> Result<AnalysisOutput, AnalysisError>`
- `AnalysisOutput`: per-output-array written elements as `(index, ExprId)` + `Stats` (instructions, block syncs; warp syncs counted per fired group, not per thread) + `op_counts` (per-instruction-kind execution counts; the interpreter tallies into a fixed `[u64; KIND_COUNT]` and folds to this `BTreeMap` at the end)
- `paired_elements(ref, opt, arrays)` - pairs the two outputs' written
  elements for each array the caller names (both sides must have each
  named array with identical index sets; unnamed arrays are not compared
  - FlashAttention's optimized-only `l`/`m` exports rely on this; an
  empty list is an error). Callers derive the list explicitly: the bench
  harness and CLI use the reference config's declared output arrays.
  Shared by the decision procedure and `volta_z3` so both backends check
  exactly the same elements
- `check_output_equivalence_with(ref, opt, options)` - the per-element
  check via one shared `EquivSession`. `EquivCheckOptions`: `sample`,
  `verify_numeric` (f64 oracle per element), `recycle_terms`. Returns a
  report with the outcome, checked/total element counts, and
  `check_time` (summed `EquivSession::check` durations only - pairing
  and the oracle excluded; the decision-procedure time the bench and
  CLI report).
- `check_output_equivalence(ref, opt)` - the Default-options wrapper
  (all elements, no oracle)
- `VcSnapshot`/`VcDump` - serde-serializable arena + output footprint, the
  payload of `volta compare --dump-vcs`/`--from-dump`; `validate()` checks
  every id (in bounds and children-before-parents) so a corrupt dump errors
  instead of panicking. The arena's serde impls are plain derives over
  `IdVec` via `id_collections`'s `serde` feature (wire-identical to `Vec`,
  serialized in place - no clone of GiB-scale arenas at dump time).
- `write_op_counts(out, label, counts)` - the one profile-table formatter,
  used by both `volta` and `volta-bench`

### Decision procedure (`canon/`, `equiv.rs`, `numeric.rs`)

The paper's canonicalizer, in Rust:

- `canon/` - expressions canonicalize to interned `Σ c·monomial·e^{poly}`
  rationals in one memoized bottom-up pass per `Session` (both kernels, all
  VC elements share intern tables). Exact i128 rational coefficients;
  `e^a·e^b` fuses at term multiplication; max/min flatten into sorted atoms;
  ops outside the fragment (sqrt/log/bitwise/comparisons/select/symbolic
  array reads) become opaque `Atom::Uninterp` atoms over an `UninterpOp`
  enum - sound, incomplete. Fraction equality goes id-compare →
  monomial-quotient (softmax rescaling) → cross-multiplication under a term
  budget. Two load-bearing invariants: single-use chain intermediates stay
  *transient* owned vectors (interning everything retains O(K²) per
  accumulator), and polys sort by *descending* TermId so chain unwinding
  appends in O(1).
- `equiv.rs` - thin wrapper: `EquivSession` (reuse across elements;
  recycles its intern tables past a configurable term bound -
  `with_recycle_terms`, default `DEFAULT_RECYCLE_TERMS` = 4M) and one-shot
  `check_equivalent`. Memory scale: exp-heavy attention terms run 2-4 KB
  each, so one warm FlashAttention output row retains several GiB; small
  bounds trade re-canonicalization time for bounded memory.
- `numeric.rs` - the f64 oracle: seeded random inputs, memoized DAG eval;
  `verify_verdict` confirms EQUIV/DIFF claims (volta-bench
  `--verify-numeric`). Agreement at random points ⇒ equality almost surely
  for this fragment (the paper's own Schwartz-Zippel argument).

### Logging (`logging.rs`)

Gated by the `logging` feature (`volta_analysis`, passed through by
`volta_cli`); without it the macros are no-op stubs. Wired at the decision
points: barrier/warp-group fires (trace), deadlock (warn), launch config,
completion stats, and VC session recycles (info), fraction-equality
escalation (debug). `cargo run -p volta_cli --features logging --
--log-level info analyze ...` narrates a run.

## Crate: volta_z3

**Path**: `crates/volta_z3/`

Z3 comparison backend for the same verification conditions: generates
SMT-LIB2 text and evaluates it through libz3's C API (`ffi.rs`, a
hand-written eight-function binding - no `z3-sys`/bindgen; building
requires `libz3-dev`). Each query runs in a **worker subprocess** (the
binary re-invokes itself via `std::process::Command`; thread-safe, no
separate executable) killed on timeout expiry: z3 4.8.12 does not
reliably honor its soft timeout or `Z3_interrupt` in the quantifier
loop the exp-axiom mode provokes (measured), so a hard kill is the only
real bound - which also gives per-element crash containment. Contract:
any binary that evaluates queries through this crate calls
`volta_z3::init_worker()` as the first statement of `main` (loudly
checked via a handshake). A capability/timing comparison point against
`canon`, not a replacement.

Two `ExpMode`s reproduce the paper's section 6.5 baselines: the default
`PowerBounded` (`(^ e a)`, bounded free `e`; attention VCs come back
`unknown`) and `AdditionAxiom` (uninterpreted `uexp` plus
`forall x y. uexp(x) uexp(y) = uexp(x+y)`; attention VCs run until the
budget kills them, reported `Timeout` - Table 8's "with axiom" column,
10-minute budget in the paper).

- `translate.rs` - a *direct semantic image* of the fragment
  (arithmetic + `Exp` + `Max`/`Min`; everything else `Unsupported`):
  every node maps to its defining SMT term and ALL algebraic reasoning
  is left to the solver, so timings measure Z3, not the translator
  (`max`/`min` are `ite` case splits, not opaque atoms; no
  canonicalization, no structural short-circuit). The translation owns
  fidelity/transport only: exact rational literals straight from
  `RealConst` (same reading as `canon`/`numeric`; the infinities are
  loud `Unsupported`), user symbols as an injection of the
  typed `SymbolRef` namespaces (`|p!name|` params, `|e!array[i]|`
  elements - a param named `t0`/`e` cannot capture generated names),
  memoized `let`-bound DAG sharing (query text linear in the arena;
  deeply nested `let`s are fine for z3, `define-fun` chains are not -
  measured), `stacker`-guarded recursion for deep accumulator spines,
  linear `let`-chain assembly, and the exp base as a strictly-bounded
  free constant (a definite rational base proved false equivalences).
  Inherited SMT semantic: division is underspecified at zero, so
  `x/x = 1` is falsifiable - unlike canon's field model (moot on the
  corpus: division only occurs inside exp-laden VCs).
- `ffi.rs` - `init_worker` (the host-binary hook that turns the
  re-invoked process into a solver worker), `eval_smtlib2` (spawns the
  worker, writes the query to its stdin, enforces the hard deadline via
  kill; inside the worker: fresh context per query, no-op error handler
  so z3 API errors surface as `(error ...)` text instead of aborting,
  soft timeout via the process-global `timeout` param) and `z3_version`.
  The worker times the libz3 evaluation itself (empty-script warmup
  first, so z3's lazy per-context frontend setup stays outside the
  span) and reports it in-band (a `t:<nanoseconds>` line after the
  handshake); `eval_smtlib2` returns it as
  `EvalOutcome::Output { text, solve }`. Solver time is measured there
  because the worker's fixed scaffolding - process
  spawn/exec/link/pipes ~1.6ms plus z3 context create/destroy and
  frontend setup ~9ms (all measured) - is several times an entire
  polynomial-fragment solve; an outer timer measures scaffolding, not
  z3.
- `lib.rs` - per-element querying (`check_equivalent`; every element is
  a genuine solver query - no structural short-circuit, so identical
  sides cost a full spawn+solve, which is the point), verdict
  parsing, `Z3Counts`, `check_output_equivalence` over
  `driver::paired_elements` (the same element pairing as the decision
  procedure), and the regression tests for every invariant above.
  Reported solve time (`Z3CheckResult::solve`) is the in-worker
  measurement - process spawn and translation excluded; Timeout
  verdicts report the budget itself rather than a measurement (the
  paper's convention for timeout rows), under either delivery
  mechanism (hard kill or z3's in-band soft cancel).

## Crate: volta_bench

**Path**: `crates/volta_bench/`

Reproduces the paper's evaluation over `kernels/` (the `.cu` + `.ptx` for
every benchmark in the paper, organized by table/section).
Benchmark definitions with full launch/param configs live in
`src/benchmarks/*.rs`. Run with `cargo run --release -p volta_bench --
category <reduction|matmul|attention|causal|conv|agent|tilelang|race>
[--sample N] [--verify-numeric] [--recycle-terms N]` (also `all`, `single
<name>`, `list`; release mode matters: ~20x). The element loop is
`driver::check_output_equivalence_with` (exact per-array footprints
against the reference; every corpus pair is footprint-identical).
`z3-compare <all|category|name>` runs equivalence benchmarks through both
the decision procedure and `volta_z3` side by side (skips race-check
benchmarks; exits nonzero if any row fails outright); benchmarks whose
VCs contain exponentials get a second `+exp-axiom` sub-row rerun under
`ExpMode::AdditionAxiom`. Its two timing columns cover the deciding
work only: `Dec(s)` = summed `EquivSession::check` (pairing and the
optional numeric oracle excluded), `Z3(s)` = in-worker libz3 solve
time (worker spawn/exec and translation excluded; timeout elements
count the full budget). The default commands' `VC (s)` column is the
same `check_time` quantity. Paper Table 8 reproduction:
`--sample 1 z3-compare all --z3-timeout 600`. Every run writes a
`volta_common::run_log` file (`--log-dir`/`--no-log-file`).
Memory: full-element attention wants tens of GiB warm - on small machines
run one category at a time under `ulimit -v` with `--recycle-terms 250000`
(bounded at ~5 GiB, slower VCs).

## Crate: volta_cli

**Path**: `crates/volta_cli/`

Commands:

- `volta parse <file>` - Check syntax
- `volta analyze <file> -k <kernel> -b 32,4 -g 1,2 --array name:base:width:len:kind --param ptr:name ...` - Run symbolic execution, report races/deadlocks, print output expressions (+ a per-instruction-kind profile; `--no-profile` to skip)
- `volta compare <ref.ptx> <opt.ptx> --kernel1 .. --kernel2 ..` - Two-kernel
  equivalence check (launch flags shared with `analyze` via a flattened
  `LaunchArgs`; `--block2`/`--grid2` override the optimized kernel's dims).
  `--backend decision|z3`, `--dump-vcs`/
  `--from-dump` (validated, versioned dump files). Exits 0 only when every
  checked element is proved equivalent.

Every run writes a `volta_common::run_log` file (`--log-dir`/
`--no-log-file`).

## Data Flow

```
PTX Source
    │
    ▼
Lexer (lex.rs) ──► Tokenizes source
    │
    ▼
Parser (parse.rs) ──► Builds AST
    │
    ▼
Instruction Parser (instr_parse.rs) ──► Strongly-typed ParsedInstruction
    │
    ▼
Lowering (lowering.rs) ──► LoweredProgram (resolved RegIds, InstrIds)
    │
    ▼
Evaluator (eval/interp.rs) ──► Symbolic execution with N threads
    │                           (χ race detection, warp/tensor-core ops)
    ▼
AnalysisOutput ──► per-element output expressions + statistics
    │
    ▼
Decision procedure (canon/ via equiv.rs) ──► per-element VC checking
                                             (+ numeric.rs oracle)
```

## Key Design Decisions

### 1. Symbolic Execution over Concrete Values

All values are `Expr` (symbolic expressions). This allows analyzing behavior for arbitrary thread indices and detecting races without enumerating all inputs.

### 2. Concrete Addresses for Race Detection

Memory addresses must be concrete (`u64`) for race detection. Thread indices are concrete (specific block configuration). Symbolic address accesses produce `SymbolicAddress` error.

### 3. χ-Context for Race Detection

From the paper (Section 4.2): Track which threads haven't synchronized since each memory access. After barrier, threads in sync set remove each other from "needs sync" sets. Race detected when accessing thread is in the "needs sync" set.

### 4. Round-Robin Scheduling

Simple, deterministic interleaving. Sufficient for race detection (any interleaving that produces a race proves the race exists).

### 5. Strongly-Typed IDs

Newtype pattern for IDs (`RegId`, `InstrId`, `ThreadId`) prevents mixing at compile time. Uses `IdVec<K, V>` for type-safe indexed collections.

### 6. Two-Phase Instruction Parsing

1. Lexer/Parser → generic `Instruction { op: String, operands }`
2. Instruction Parser → strongly-typed `ParsedInstruction` variants

Enables robust modifier validation and better error messages.

### 7. Separate Memory Spaces

Global, shared, local, param memories are separate `Memory` instances, matching PTX memory model.

## Adding a New Instruction

1. **Frontend**: Add `InstrKind` variant in `instr.rs`, add to trie
2. **Instruction Parsing**: Add `ParsedInstruction` variant, parser function
   in `instr_parse.rs` (use `expect_operands` for exact-arity operand lists)
3. **Lowering**: Add `LoweredInstr` variant, lowering case
4. **Evaluation**: Add evaluation case in `eval/interp.rs`

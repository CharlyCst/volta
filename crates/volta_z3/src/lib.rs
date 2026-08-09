//! Z3 backend: translate a verification-condition pair to SMT-LIB2,
//! evaluate it through the linked libz3 (in a forked, killable worker -
//! see `ffi`), and interpret unsat/sat/unknown/timeout. This is a
//! timing/capability comparison point against `volta_analysis::canon`'s
//! own decision procedure - not a replacement for it. See the `translate`
//! module for exactly which fragment of `ExprNode` this backend covers,
//! why, and how DAG sharing is preserved (structural interning into
//! SMT-LIB2 `let` bindings), plus the two exponential encodings
//! ([`ExpMode`]) that reproduce the paper's section 6.5 baselines.
//!
//! The query still goes through SMT-LIB2 *text* even though the solver is
//! linked: the textual form is what keeps the translation auditable (any
//! query can be dumped and replayed against a standalone z3), and the
//! whole soundness argument lives in the translation, not the transport.
//! `libz3` is a build/link-time requirement (Debian/Ubuntu: `libz3-dev`);
//! no z3 binary or temp files are involved at runtime.

mod ffi;
mod translate;

pub use ffi::z3_version;
pub use translate::{Builder, ExpMode, Unsupported, translate_root};

use std::fmt;
use std::time::{Duration, Instant};

use volta_analysis::driver::EquivCheckError;
use volta_analysis::symbolic::{ExprArena, ExprId};

/// Outcome of one Z3 query.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Z3Verdict {
    /// `unsat` on `(not (= a b))`: the expressions are proved equal.
    Equivalent,
    /// `sat`: Z3 found a model where the two sides differ. The
    /// translation is a direct semantic image (max/min are real `ite`
    /// case splits, not opaque atoms), so this is a genuine countermodel
    /// for exp-free queries; when the query contains the exponential it
    /// is not definitive (the bounded-free `e`, or `uexp` under
    /// [`ExpMode::AdditionAxiom`], underconstrain the real exp).
    NotEquivalent,
    /// `unknown` - Z3 gave up on the query without exhausting its time
    /// budget (no decision procedure applies). The expected result for
    /// exponential-heavy (softmax/attention) VCs under the default
    /// [`ExpMode::PowerBounded`] encoding, since Z3's decidable theories
    /// do not cover symbolic real exponents.
    Unknown,
    /// The time budget expired while the solver was still working (the
    /// worker was killed, or z3 reported an in-band cancellation). The
    /// expected result for the attention VCs under
    /// [`ExpMode::AdditionAxiom`] - the paper's "with axiom" baseline,
    /// where every attention benchmark exceeds a 10-minute budget.
    Timeout,
}

#[derive(Debug, thiserror::Error)]
pub enum Z3Error {
    #[error(transparent)]
    Unsupported(#[from] Unsupported),
    #[error("z3 produced unexpected output: {0:?}")]
    UnexpectedOutput(String),
    #[error("z3 worker failed: {0}")]
    Worker(String),
}

/// One element's check: the verdict and how long the solver evaluation
/// took. Translation time isn't included - that front-end cost is the
/// same for both backends and isn't the thing being compared.
#[derive(Debug, Clone)]
pub struct Z3CheckResult {
    pub verdict: Z3Verdict,
    pub solve_secs: f64,
}

/// An `unknown` whose `(get-info :reason-unknown)` blames the time or
/// resource budget rather than genuine incompleteness. z3 4.8 spells
/// budget exhaustion "canceled" (or "timeout"/"resource" depending on
/// path); genuine give-ups read like "incomplete (theory arithmetic)".
fn unknown_reason_is_budget(output: &str) -> bool {
    output.lines().any(|l| {
        let l = l.trim();
        l.contains(":reason-unknown")
            && (l.contains("timeout") || l.contains("canceled") || l.contains("resource"))
    })
}

/// Check whether `a` (in `arena_a`) and `b` (in `arena_b`) are equal over
/// the reals, using Z3 instead of `volta_analysis::canon` as the decision
/// procedure. `timeout` bounds the solver call (`None` = no limit) - a
/// hard bound: the query runs in a forked worker that is killed on
/// expiry, reported as [`Z3Verdict::Timeout`]. `mode` selects the
/// exponential encoding (see [`ExpMode`]).
pub fn check_equivalent(
    arena_a: &ExprArena,
    a: ExprId,
    arena_b: &ExprArena,
    b: ExprId,
    timeout: Option<Duration>,
    mode: ExpMode,
) -> Result<Z3CheckResult, Z3Error> {
    let mut builder = Builder::with_exp_mode(mode);
    let ta = translate_root(&mut builder, arena_a, a)?;
    let tb = translate_root(&mut builder, arena_b, b)?;

    let body = builder.wrap_in_lets(&format!("(not (= {} {}))", ta, tb));
    let mut query = builder.preamble();
    query.push_str(&format!("(assert {})\n", body));
    query.push_str("(check-sat)\n");
    // Harmless on decided queries (empty reason, no error); on `unknown`
    // it distinguishes a budget cancellation from a genuine give-up.
    query.push_str("(get-info :reason-unknown)\n");

    let start = Instant::now();
    let outcome = ffi::eval_smtlib2(&query, timeout);
    let solve_secs = start.elapsed().as_secs_f64();

    let verdict = match outcome {
        ffi::EvalOutcome::HardTimeout => Z3Verdict::Timeout,
        ffi::EvalOutcome::ChildDied(how) => return Err(Z3Error::Worker(how)),
        ffi::EvalOutcome::Output(output) => {
            match output.lines().map(str::trim).find(|l| !l.is_empty()) {
                Some("unsat") => Z3Verdict::Equivalent,
                Some("sat") => Z3Verdict::NotEquivalent,
                Some("unknown") | Some("timeout") => {
                    if unknown_reason_is_budget(&output) {
                        Z3Verdict::Timeout
                    } else {
                        Z3Verdict::Unknown
                    }
                }
                _ => return Err(Z3Error::UnexpectedOutput(output)),
            }
        }
    };

    Ok(Z3CheckResult {
        verdict,
        solve_secs,
    })
}

/// The CLI convention for Z3 timeouts: `0` means no limit.
pub fn timeout_from_secs(secs: u64) -> Option<Duration> {
    (secs != 0).then(|| Duration::from_secs(secs))
}

/// Per-element outcome from the Z3 backend - a finer-grained result than
/// the decision procedure's binary equivalent/not, since Z3 can also fail
/// to decide (`Unknown`) or refuse a VC outright (`Unsupported`).
#[derive(Debug, Clone)]
pub enum ElementOutcome {
    Equivalent,
    NotEquivalent,
    Unknown,
    Timeout,
    Unsupported(String),
    /// The `z3` call itself failed for a reason other than an unsupported
    /// fragment (e.g. malformed output, worker crash) - recorded per
    /// element rather than aborting the whole comparison.
    Error(String),
}

#[derive(Debug, Clone)]
pub struct ElementResult {
    pub array: String,
    pub index: u64,
    pub outcome: ElementOutcome,
    pub solve_secs: f64,
}

/// Per-outcome element counts. A named struct rather than a positional
/// tuple: the six counts are all `usize`, and a transposition anywhere in
/// the report/JSON/print chain would compile silently.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Z3Counts {
    pub equivalent: usize,
    pub not_equivalent: usize,
    pub unknown: usize,
    pub timeout: usize,
    pub unsupported: usize,
    pub error: usize,
}

impl Z3Counts {
    pub fn total(&self) -> usize {
        self.equivalent
            + self.not_equivalent
            + self.unknown
            + self.timeout
            + self.unsupported
            + self.error
    }

    /// Every checked element was proved equivalent (vacuously true for an
    /// empty footprint, matching the decision backend's outcome there).
    pub fn all_equivalent(&self) -> bool {
        self.equivalent == self.total()
    }

    /// Compact `equiv/diff/unknown/timeout/unsupported/error` form for
    /// tables.
    pub fn compact(&self) -> String {
        format!(
            "{}/{}/{}/{}/{}/{}",
            self.equivalent,
            self.not_equivalent,
            self.unknown,
            self.timeout,
            self.unsupported,
            self.error
        )
    }
}

impl fmt::Display for Z3Counts {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "equivalent: {}  not-equivalent: {}  unknown: {}  timeout: {}  unsupported: {}  error: {}",
            self.equivalent,
            self.not_equivalent,
            self.unknown,
            self.timeout,
            self.unsupported,
            self.error
        )
    }
}

#[derive(Debug, Clone, Default)]
pub struct Z3EquivReport {
    pub elements: Vec<ElementResult>,
}

impl Z3EquivReport {
    pub fn counts(&self) -> Z3Counts {
        let mut c = Z3Counts::default();
        for e in &self.elements {
            match &e.outcome {
                ElementOutcome::Equivalent => c.equivalent += 1,
                ElementOutcome::NotEquivalent => c.not_equivalent += 1,
                ElementOutcome::Unknown => c.unknown += 1,
                ElementOutcome::Timeout => c.timeout += 1,
                ElementOutcome::Unsupported(_) => c.unsupported += 1,
                ElementOutcome::Error(_) => c.error += 1,
            }
        }
        c
    }

    pub fn total_solve_secs(&self) -> f64 {
        self.elements.iter().map(|e| e.solve_secs).sum()
    }
}

/// Check every paired output element (exactly like
/// `volta_analysis::driver::check_output_equivalence_with`, against the
/// reference's footprint) with Z3 instead of the decision procedure. Unlike the decision procedure, this never
/// aborts partway through a run over a single element's failure - each
/// element's outcome (including "unsupported" or a solver error) is
/// recorded independently, since the whole point is comparing coverage as
/// well as speed. (`Err` is only the up-front pairing failing; the solver
/// itself is linked in and cannot be missing at runtime.)
pub fn check_output_equivalence(
    reference: &volta_analysis::eval::AnalysisOutput,
    optimized: &volta_analysis::eval::AnalysisOutput,
    arrays: &[String],
    sample: u64,
    timeout: Option<Duration>,
    mode: ExpMode,
) -> Result<Z3EquivReport, EquivCheckError> {
    let paired = volta_analysis::driver::paired_elements(reference, optimized, arrays)?;
    let mut elements = Vec::new();
    for (name, common) in paired {
        let limit = match sample {
            0 => common.len(),
            n => common.len().min(n as usize),
        };
        for (index, r, o) in common.into_iter().take(limit) {
            let (outcome, solve_secs) =
                match check_equivalent(&reference.arena, r, &optimized.arena, o, timeout, mode) {
                    Ok(res) => (
                        match res.verdict {
                            Z3Verdict::Equivalent => ElementOutcome::Equivalent,
                            Z3Verdict::NotEquivalent => ElementOutcome::NotEquivalent,
                            Z3Verdict::Unknown => ElementOutcome::Unknown,
                            Z3Verdict::Timeout => ElementOutcome::Timeout,
                        },
                        res.solve_secs,
                    ),
                    Err(Z3Error::Unsupported(u)) => (ElementOutcome::Unsupported(u.0), 0.0),
                    Err(e) => (ElementOutcome::Error(e.to_string()), 0.0),
                };
            elements.push(ElementResult {
                array: name.clone(),
                index,
                outcome,
                solve_secs,
            });
        }
    }
    Ok(Z3EquivReport { elements })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn check(arena_a: &ExprArena, a: ExprId, arena_b: &ExprArena, b: ExprId) -> Z3Verdict {
        check_mode(arena_a, a, arena_b, b, ExpMode::PowerBounded)
    }

    fn check_mode(
        arena_a: &ExprArena,
        a: ExprId,
        arena_b: &ExprArena,
        b: ExprId,
        mode: ExpMode,
    ) -> Z3Verdict {
        // Generous: solver needs only ms uncontended, but the whole suite
        // shares the cores and every test really runs z3 (no structural
        // short-circuits by design).
        check_equivalent(arena_a, a, arena_b, b, Some(Duration::from_secs(60)), mode)
            .unwrap()
            .verdict
    }

    #[test]
    fn commutative_add_is_equivalent() {
        // x + 1 vs 1 + x: commutativity is the solver's job now - Z3
        // proves it, the translation does not pre-normalize it away.
        let mut ar = ExprArena::new();
        let x = ar.param_symbol("x");
        let one = ar.int(1);
        let lhs = ar.add(x, one);
        let rhs = ar.add(one, x);
        assert_eq!(check(&ar, lhs, &ar, rhs), Z3Verdict::Equivalent);
    }

    #[test]
    fn cross_arena_named_symbol_correlates_by_string() {
        // Two independent arenas, each with its own "x" - correlated by
        // name, exactly like `canon`'s own convention.
        let mut ar_a = ExprArena::new();
        let x_a = ar_a.param_symbol("x");
        let two_a = ar_a.int(2);
        let lhs = ar_a.mul(x_a, two_a);

        let mut ar_b = ExprArena::new();
        let x_b = ar_b.param_symbol("x");
        let rhs = ar_b.add(x_b, x_b);

        assert_eq!(check(&ar_a, lhs, &ar_b, rhs), Z3Verdict::Equivalent);
    }

    #[test]
    fn distinct_symbols_are_not_equivalent() {
        let mut ar = ExprArena::new();
        let x = ar.param_symbol("x");
        let y = ar.param_symbol("y");
        assert_eq!(check(&ar, x, &ar, y), Z3Verdict::NotEquivalent);
    }

    #[test]
    fn exp_addition_law_is_unknown_without_an_axiom() {
        // exp(a)*exp(b) vs exp(a+b): true over the reals, but Z3 alone
        // (native `^`, no axiom) has no decision procedure for symbolic
        // real exponents - the paper's no-intervention baseline.
        let mut ar = ExprArena::new();
        let a = ar.param_symbol("a");
        let b = ar.param_symbol("b");
        let ea = ar.exp(a);
        let eb = ar.exp(b);
        let lhs = ar.mul(ea, eb);
        let sum = ar.add(a, b);
        let rhs = ar.exp(sum);
        assert_eq!(check(&ar, lhs, &ar, rhs), Z3Verdict::Unknown);

        // With the paper's addition-law axiom the toy law itself is a
        // direct instantiation: proved.
        assert_eq!(
            check_mode(&ar, lhs, &ar, rhs, ExpMode::AdditionAxiom),
            Z3Verdict::Equivalent
        );
    }

    /// The paper's Table 8 behavior in miniature: a softmax-style
    /// rescaling identity under the addition-law axiom sends z3 into an
    /// E-matching loop that ignores its own soft timeout - the forked
    /// worker's hard deadline must fire and report Timeout.
    #[test]
    fn axiom_mode_softmax_rescaling_hard_times_out() {
        let n = 8;
        let mut ar = ExprArena::new();
        let m = ar.param_symbol("m");
        // lhs: sum_i exp(s_i - m); rhs: exp(-m) * sum_i exp(s_i). Equal
        // over the reals via the addition law, but a Sum-of-Exps vs a
        // Prod structurally - no short-circuit, the solver must run.
        let mut lhs_terms = Vec::new();
        let mut rhs_terms = Vec::new();
        for i in 0..n {
            let s = ar.param_symbol(format!("s{}", i));
            let shifted = ar.sub(s, m);
            lhs_terms.push(ar.exp(shifted));
            rhs_terms.push(ar.exp(s));
        }
        let mut lhs = lhs_terms[0];
        for &t in &lhs_terms[1..] {
            lhs = ar.add(lhs, t);
        }
        let neg_m = ar.neg(m);
        let e_neg_m = ar.exp(neg_m);
        let mut rhs_sum = rhs_terms[0];
        for &t in &rhs_terms[1..] {
            rhs_sum = ar.add(rhs_sum, t);
        }
        let rhs = ar.mul(e_neg_m, rhs_sum);

        let res = check_equivalent(
            &ar,
            lhs,
            &ar,
            rhs,
            Some(Duration::from_secs(2)),
            ExpMode::AdditionAxiom,
        )
        .unwrap();
        assert_eq!(res.verdict, Z3Verdict::Timeout);
        assert!(res.solve_secs >= 2.0, "must have run to the deadline");
    }

    /// Budget-vs-incompleteness classification of `unknown` reasons.
    #[test]
    fn unknown_reason_classification() {
        assert!(unknown_reason_is_budget(
            "unknown\n(:reason-unknown \"canceled\")\n"
        ));
        assert!(unknown_reason_is_budget(
            "unknown\n(:reason-unknown \"timeout\")\n"
        ));
        assert!(!unknown_reason_is_budget(
            "unknown\n(:reason-unknown \"incomplete\")\n"
        ));
        assert!(!unknown_reason_is_budget(
            "unknown\n(:reason-unknown \"smt tactic failed to show goal to be sat/unsat (incomplete (theory arithmetic))\")\n"
        ));
    }

    #[test]
    fn select_is_unsupported() {
        // A concrete condition constant-folds away (see `ExprArena`'s
        // "constructors constant-fold eagerly"), so use a symbolic
        // predicate to force an actual `Select` node.
        let mut ar = ExprArena::new();
        let c = ar.param_symbol("cond");
        let t = ar.int(1);
        let f = ar.int(0);
        let id = ar.select(c, t, f);
        let result = check_equivalent(&ar, id, &ar, id, None, ExpMode::PowerBounded);
        assert!(matches!(result, Err(Z3Error::Unsupported(_))));
    }

    /// max/min are real ite case splits, so structurally different but
    /// equal maxes across the two arenas are proved equal by the solver.
    /// (Two earlier opaque-atom designs got this wrong or incomplete.)
    #[test]
    fn maxmin_compound_args_unify_across_arenas() {
        let build = |flip: bool| {
            let mut ar = ExprArena::new();
            let x = ar.param_symbol("x");
            let y = ar.param_symbol("y");
            let z = ar.param_symbol("z");
            let sum = if flip { ar.add(y, x) } else { ar.add(x, y) };
            let m = ar.max(sum, z);
            (ar, m)
        };
        let (ar_a, ma) = build(false);
        let (ar_b, mb) = build(true);
        assert_eq!(check(&ar_a, ma, &ar_b, mb), Z3Verdict::Equivalent);
    }

    /// Nested running-max chains reassociated differently across the two
    /// sides are proved equal via the ite semantics.
    #[test]
    fn maxmin_key_is_traversal_order_independent() {
        let build = |outer_first: bool| {
            let mut ar = ExprArena::new();
            let s0 = ar.param_symbol("s0");
            let s1 = ar.param_symbol("s1");
            let s2 = ar.param_symbol("s2");
            let m1 = ar.max(s0, s1);
            let m2 = ar.max(m1, s2);
            let root = if outer_first {
                ar.add(m2, m1)
            } else {
                ar.add(m1, m2)
            };
            (ar, root)
        };
        let (ar_a, ra) = build(false);
        let (ar_b, rb) = build(true);
        assert_eq!(check(&ar_a, ra, &ar_b, rb), Z3Verdict::Equivalent);
    }

    /// Named and machine symbols are disjoint namespaces (as in canon and
    /// the numeric oracle): a symbol literally named "s{N}" is not the
    /// machine `Symbol(N)`.
    #[test]
    fn named_symbol_does_not_alias_machine_symbol() {
        use volta_analysis::symbolic::ExprNode;

        let mut ar = ExprArena::new();
        let machine = ar.symbol();
        let ExprNode::Symbol(sym) = *ar.node(machine) else {
            panic!("arena.symbol() must produce ExprNode::Symbol");
        };
        let named = ar.param_symbol(sym.to_string());
        assert_eq!(check(&ar, machine, &ar, named), Z3Verdict::NotEquivalent);
    }

    /// Regression: a user symbol literally named `t0` used to be captured
    /// by the generated `|t0|` let binding, proving `a*b` "equivalent" to
    /// an unrelated parameter - a false EQUIVALENT. User symbols now live
    /// in reserved namespaces (`p!`/`e!`).
    #[test]
    fn user_symbol_named_t0_is_not_captured() {
        let mut ar_a = ExprArena::new();
        let a = ar_a.param_symbol("a");
        let b = ar_a.param_symbol("b");
        let lhs = ar_a.mul(a, b);

        let mut ar_b = ExprArena::new();
        let rhs = ar_b.param_symbol("t0");

        assert_eq!(check(&ar_a, lhs, &ar_b, rhs), Z3Verdict::NotEquivalent);
    }

    /// Regression: a user symbol named `e` used to collide with the
    /// `(define-fun e ...)` exp-base constant, making z3 error out on the
    /// duplicate declaration.
    #[test]
    fn user_symbol_named_e_is_distinct_from_exp_base() {
        let mut ar = ExprArena::new();
        let e_sym = ar.param_symbol("e");
        let other = ar.param_symbol("other");
        assert_eq!(check(&ar, e_sym, &ar, e_sym), Z3Verdict::Equivalent);
        assert_eq!(check(&ar, e_sym, &ar, other), Z3Verdict::NotEquivalent);
        // And it is a free symbol, not the constant 2.718...:
        let mut ar_b = ExprArena::new();
        let approx = ar_b.float(2.718281828459045);
        assert_eq!(check(&ar, e_sym, &ar_b, approx), Z3Verdict::NotEquivalent);
    }

    /// Regression: float constants used to be rendered as their shortest
    /// decimal, which SMT reads as an exact decimal rational - so z3 saw
    /// `0.1f64` as 1/10 while canon and the numeric oracle use the exact
    /// binary value, and the backends could reach opposite verdicts.
    #[test]
    fn float_literals_use_exact_binary_semantics() {
        // The divisions stay symbolic (`x / c`): concrete int/int division
        // would constant-fold with integer semantics before translation.
        //
        // 0.5f64 is a dyadic rational, exactly 1/2 in both readings:
        // x * 0.5 == x / 2 over the reals.
        let mut ar_a = ExprArena::new();
        let x = ar_a.param_symbol("x");
        let half = ar_a.float(0.5);
        let lhs = ar_a.mul(x, half);
        let mut ar_b = ExprArena::new();
        let x = ar_b.param_symbol("x");
        let two = ar_b.int(2);
        let rhs = ar_b.div(x, two);
        assert_eq!(check(&ar_a, lhs, &ar_b, rhs), Z3Verdict::Equivalent);

        // 0.1f64 is NOT 1/10 (its exact binary value is
        // 3602879701896397/2^55): canon reports x*0.1 != x/10, and now so
        // does the Z3 backend. The old shortest-decimal rendering read
        // 0.1f64 as exactly 1/10 and proved these "equivalent".
        let mut ar_c = ExprArena::new();
        let x = ar_c.param_symbol("x");
        let tenth = ar_c.float(0.1);
        let lhs = ar_c.mul(x, tenth);
        let mut ar_d = ExprArena::new();
        let x = ar_d.param_symbol("x");
        let ten = ar_d.int(10);
        let rhs = ar_d.div(x, ten);
        assert_eq!(check(&ar_c, lhs, &ar_d, rhs), Z3Verdict::NotEquivalent);
    }

    #[test]
    fn negative_zero_equals_zero() {
        let mut ar_a = ExprArena::new();
        let nz = ar_a.float(-0.0);
        let mut ar_b = ExprArena::new();
        let z = ar_b.float(0.0);
        assert_eq!(check(&ar_a, nz, &ar_b, z), Z3Verdict::Equivalent);
    }

    /// Regression: translation used to recurse once per chain element and
    /// overflow the stack at ~16-20k-deep Fma spines (a matmul
    /// accumulator's shape). Translation-only: no solver involved.
    #[test]
    fn deep_fma_chain_translates_without_stack_overflow() {
        let mut ar = ExprArena::new();
        let a = ar.param_symbol("a");
        let b = ar.param_symbol("b");
        let mut acc = ar.param_symbol("acc");
        for _ in 0..100_000 {
            acc = ar.fma(a, b, acc);
        }
        let mut bld = Builder::new();
        assert!(translate_root(&mut bld, &ar, acc).is_ok());
    }

    /// Regression: `flatten` used to re-expand a node reachable twice
    /// within one chain (e.g. `Add(x, x)` doubling chains from
    /// `add f,f,f`), taking 2^n time on an n-node arena. 64 levels would
    /// be ~10^19 leaf visits without the visited cutoff.
    #[test]
    fn self_sharing_doubling_chain_translates_linearly() {
        let mut ar = ExprArena::new();
        let mut x = ar.param_symbol("x");
        for _ in 0..64 {
            x = ar.add(x, x);
        }
        let mut bld = Builder::new();
        assert!(translate_root(&mut bld, &ar, x).is_ok());

        let mut ar2 = ExprArena::new();
        let mut y = ar2.param_symbol("y");
        for _ in 0..64 {
            y = ar2.mul(y, y);
        }
        let mut bld2 = Builder::new();
        assert!(translate_root(&mut bld2, &ar2, y).is_ok());
    }

    /// The self-compare case goes to the solver like everything else
    /// (there is deliberately no structural short-circuit - the point is
    /// to measure Z3) and is proved.
    #[test]
    fn self_compare_is_proved_by_the_solver() {
        let mut ar = ExprArena::new();
        let x = ar.param_symbol("x");
        let y = ar.param_symbol("y");
        let sum = ar.add(x, y);
        let m = ar.max(sum, x);
        let res = check_equivalent(&ar, m, &ar, m, None, ExpMode::PowerBounded).unwrap();
        assert_eq!(res.verdict, Z3Verdict::Equivalent);
    }

    /// Regression (adversarial find): `exp(1)` used to prove equal to the
    /// rational 2718281828459045/10^15 because the exp base was DEFINED as
    /// that rational - a false EQUIVALENT the decision procedure rejects.
    /// The base is now a free constant strictly bounded around Euler's e.
    #[test]
    fn exp_of_one_is_not_a_rational() {
        let mut ar_a = ExprArena::new();
        let one = ar_a.int(1);
        let lhs = ar_a.exp(one);
        let mut ar_b = ExprArena::new();
        let rhs = ar_b.float(2.718281828459045);
        assert_eq!(check(&ar_a, lhs, &ar_b, rhs), Z3Verdict::NotEquivalent);
    }

    /// Regression (adversarial find): a `+` chain feeding a max atom used
    /// to key differently depending on whether a subterm was DAG-shared
    /// (memoized) on one side - identical trees, different atoms, false
    /// NOT-EQUIVALENT. Multiset splicing makes the key sharing-independent.
    #[test]
    fn maxmin_key_is_sharing_independent() {
        // Side A: `xy = x + y` is one shared node, used by the subtrahend
        // AND inside the max argument. Side B: two separate x+y nodes.
        let mut ar_a = ExprArena::new();
        let x = ar_a.param_symbol("x");
        let y = ar_a.param_symbol("y");
        let w = ar_a.param_symbol("w");
        let z = ar_a.param_symbol("z");
        let p = ar_a.param_symbol("p");
        let xy = ar_a.add(x, y);
        let sub = ar_a.sub(p, xy);
        let arg = ar_a.add(xy, w);
        let m = ar_a.max(arg, z);
        let root_a = ar_a.add(sub, m);

        let mut ar_b = ExprArena::new();
        let x = ar_b.param_symbol("x");
        let y = ar_b.param_symbol("y");
        let w = ar_b.param_symbol("w");
        let z = ar_b.param_symbol("z");
        let p = ar_b.param_symbol("p");
        let xy1 = ar_b.add(x, y);
        let sub = ar_b.sub(p, xy1);
        let xy2 = ar_b.add(y, x);
        let arg = ar_b.add(xy2, w);
        let m = ar_b.max(arg, z);
        let root_b = ar_b.add(sub, m);

        assert_eq!(check(&ar_a, root_a, &ar_b, root_b), Z3Verdict::Equivalent);
    }

    /// `min(y - max(x,z), w)` vs `min(y + (-max(x,z)), w)`: Sub vs
    /// Add-of-Neg inside min/max arguments - the solver proves it (this
    /// was a measured false-DIFF class under the old opaque-atom design).
    #[test]
    fn sub_and_add_neg_are_one_term_inside_atoms() {
        let mut ar_a = ExprArena::new();
        let x = ar_a.param_symbol("x");
        let y = ar_a.param_symbol("y");
        let z = ar_a.param_symbol("z");
        let w = ar_a.param_symbol("w");
        let m = ar_a.max(x, z);
        let diff = ar_a.sub(y, m);
        let root_a = ar_a.min(diff, w);

        let mut ar_b = ExprArena::new();
        let x = ar_b.param_symbol("x");
        let y = ar_b.param_symbol("y");
        let z = ar_b.param_symbol("z");
        let w = ar_b.param_symbol("w");
        let m = ar_b.max(x, z);
        let neg = ar_b.neg(m);
        let sum = ar_b.add(y, neg);
        let root_b = ar_b.min(sum, w);

        assert_eq!(check(&ar_a, root_a, &ar_b, root_b), Z3Verdict::Equivalent);
    }

    /// Fma(a, b, c) renders as its definition a*b + c, so it is proved
    /// equal to the written-out form (common compiler variance).
    #[test]
    fn fma_desugars_to_mul_add() {
        let mut ar_a = ExprArena::new();
        let a = ar_a.param_symbol("a");
        let b = ar_a.param_symbol("b");
        let c = ar_a.param_symbol("c");
        let lhs = ar_a.fma(a, b, c);

        let mut ar_b = ExprArena::new();
        let a = ar_b.param_symbol("a");
        let b = ar_b.param_symbol("b");
        let c = ar_b.param_symbol("c");
        let prod = ar_b.mul(a, b);
        let rhs = ar_b.add(prod, c);

        let res = check_equivalent(&ar_a, lhs, &ar_b, rhs, None, ExpMode::PowerBounded).unwrap();
        assert_eq!(res.verdict, Z3Verdict::Equivalent);
    }

    /// Exact cancellation: x - x is the literal 0,
    /// x / x is the literal 1, -(-x) is x.
    #[test]
    fn cancellation_and_involution() {
        let mut ar = ExprArena::new();
        let x = ar.param_symbol("x");
        let d = ar.sub(x, x);
        let mut zr = ExprArena::new();
        let zero = zr.float(0.0);
        assert_eq!(
            check_equivalent(&ar, d, &zr, zero, None, ExpMode::PowerBounded)
                .unwrap()
                .verdict,
            Z3Verdict::Equivalent
        );

        // x / x vs 1 is NOT valid over SMT reals: division is total but
        // underspecified at zero, so x = 0 is a genuine countermodel.
        // (canon's rational-field semantics formally cancels x/x to 1 - a
        // documented modeling divergence of the two backends; corpus VCs
        // only divide inside exp-laden softmax terms, where z3 answers
        // unknown regardless.)
        let q = ar.div(x, x);
        let mut on = ExprArena::new();
        let one = on.float(1.0);
        assert_eq!(
            check_equivalent(&ar, q, &on, one, None, ExpMode::PowerBounded)
                .unwrap()
                .verdict,
            Z3Verdict::NotEquivalent
        );

        let n = ar.neg(x);
        let nn = ar.neg(n);
        let res = check_equivalent(&ar, nn, &ar, x, None, ExpMode::PowerBounded).unwrap();
        assert_eq!(res.verdict, Z3Verdict::Equivalent);
    }

    /// `x + (-0.1)` vs `x - 0.1`: negative literals vs negated positives
    /// are the solver's problem now, and it proves them equal.
    #[test]
    fn negative_literals_unify_with_negated_positives() {
        let mut ar_a = ExprArena::new();
        let x = ar_a.param_symbol("x");
        let neg_tenth = ar_a.float(-0.1);
        let lhs = ar_a.add(x, neg_tenth);

        let mut ar_b = ExprArena::new();
        let x = ar_b.param_symbol("x");
        let tenth = ar_b.float(0.1);
        let rhs = ar_b.sub(x, tenth);

        let res = check_equivalent(&ar_a, lhs, &ar_b, rhs, None, ExpMode::PowerBounded).unwrap();
        assert_eq!(res.verdict, Z3Verdict::Equivalent);

        // And exact cancellation across the two spellings: (x + -0.1) -
        // (x - 0.1) is the literal 0.
        let mut ar_c = ExprArena::new();
        let x = ar_c.param_symbol("x");
        let neg_tenth = ar_c.float(-0.1);
        let sum = ar_c.add(x, neg_tenth);
        let tenth = ar_c.float(0.1);
        let diff = ar_c.sub(x, tenth);
        let zero_expr = ar_c.sub(sum, diff);
        let mut zr = ExprArena::new();
        let zero = zr.float(0.0);
        assert_eq!(
            check_equivalent(&ar_c, zero_expr, &zr, zero, None, ExpMode::PowerBounded)
                .unwrap()
                .verdict,
            Z3Verdict::Equivalent
        );
    }
}

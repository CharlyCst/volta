//! Translates the arithmetic + Exp + Max/Min fragment of `ExprArena` into
//! SMT-LIB2 text for Z3, mirroring `volta_analysis::canon`'s atom boundary
//! and symbol-naming convention so the two backends are compared on equal
//! footing (cross-arena symbol correlation: `NamedSymbol` by its string,
//! `Symbol` by its `Display` name - the same names canon interns).
//!
//! Ops outside this fragment (`Select`, comparisons, bitwise ops, `Rem`,
//! `Log`, `Sqrt`, `Abs`, `SymbolicRead`, ...) are refused outright
//! (`Unsupported`) rather than modeled as opaque atoms: an opaque atom is
//! only sound if syntactically-distinct-but-equal occurrences share a key,
//! and getting that right in general needs `canon`'s own `Rat`-based
//! normalization. `Max`/`Min` are the one exception - needed for every
//! softmax/attention benchmark - modeled as opaque atoms over a canonical
//! argument list (see below).
//!
//! # Structural interning over signed multisets
//!
//! `ExprArena` is a DAG, not a tree - shared subexpressions (e.g. the
//! softmax row-max, reused by every term) are the norm. Every distinct
//! *structure* is therefore translated once per `Builder`, across both
//! arenas, and structure is put in a canonical form first:
//!
//! - The additive family (`Add`, `Sub`, `Neg`, and `Fma`, which desugars
//!   to product-plus-addend) flattens into one *sum term*: a sorted
//!   multiset of operand terms with signed integer counts. `x - x`
//!   cancels to `0.0`, `-(-x)` is `x`, `a*b + c` and `Fma(a, b, c)` are
//!   the same term, and the multiset is independent of association order,
//!   DAG sharing (an operand already translated splices its stored
//!   multiset back in), and operand multiplicity (`Add(t, t)` doubling
//!   chains become one entry with count 2^k, not 2^k entries).
//! - The multiplicative family (`Mul`, `Div`, `Rcp`) flattens the same
//!   way into a *product term* with signed exponents; `x / x` is `1.0`,
//!   and a pure negation inside a product hoists out as a sign on the
//!   surrounding sum (`(-a)*b` keys as `-(a*b)`).
//! - `Max`/`Min` atoms key on their fully spliced, sorted, deduplicated
//!   argument-term list (nested same-op nodes always splice in, even when
//!   already translated elsewhere - the atom's identity must not depend
//!   on traversal order), so structurally equal max/min expressions map
//!   to one atom in the same arena or across the two kernels' arenas.
//!   `max(x, x)` collapses to `x`.
//!
//! What is deliberately NOT normalized: distribution (`x*(a+b)` vs
//! `x*a + x*b`) and constant arithmetic (`(* 2.0 0.5 x)` vs `x`). Between
//! interpreted operators z3 proves those equal anyway; *inside* a
//! `Max`/`Min` atom argument they produce distinct atoms, which can only
//! yield the (documented, non-definitive) sat direction - full semantic
//! normalization of atom arguments is exactly `canon`'s job, not this
//! backend's. Counts/exponents saturate at i128: a DAG whose
//! multiplicities exceed 2^127 (a ~185-level Fibonacci-sharing cascade -
//! numerically far past f64 overflow in the kernel itself) is refused as
//! `Unsupported` rather than silently wrapped, mirroring `canon`'s own
//! `Coeff` overflow errors. Scale note: a chain whose every prefix is separately
//! referenced (a running max, or running sums each stored to memory)
//! interns one multiset per prefix - quadratic in the chain length, the
//! same shape `canon`'s flatten-to-sorted-atoms takes; benchmark-scale
//! chains (~512) are microseconds, 100k-prefix chains are not.
//!
//! # The exponential (two modes)
//!
//! In the default [`ExpMode::PowerBounded`], `Exp(a)` renders as
//! `(^ e a)` where `e` is a *free* constant pinned by strict rational
//! bounds `2.718281828459045 < e < 2.7182818284590455`. Defining `e` as a
//! rational (an earlier design) let z3 prove
//! `exp(1) = 2718281828459045/10^15` - a false EQUIVALENT the decision
//! procedure rejects, since exp(1) is irrational. With bounded-free `e`,
//! no concrete rational ever equals a power of `e`, while `(^ e x)` stays
//! a nonlinear term z3 answers `unknown` on for the symbolic-exponent
//! identities - reproducing the paper's no-intervention baseline
//! ("Z3 returns unknown" on the attention benchmarks, section 6.5).
//!
//! [`ExpMode::AdditionAxiom`] reproduces the paper's other baseline:
//! `Exp(a)` becomes an *uninterpreted* function application `(uexp a)`
//! and the preamble asserts the addition law
//! `forall x y. uexp(x)*uexp(y) = uexp(x+y)` (Table 8's
//! "Z3 with axiom" column). The axiom gives E-matching a trigger it
//! instantiates without bound on softmax-shaped VCs, so z3 neither
//! decides nor gives up - measured on 4.8.12, it ignores its own soft
//! timeout and explicit interrupts, which is why query evaluation runs
//! in a killable forked child (see `ffi`). Verdicts under this mode are
//! for benchmarking: `unsat` (Equivalent) is still sound (the axiom is
//! true of the real exponential), but `sat` says even less than in the
//! default mode, since `uexp` carries no semantics beyond the one law.
//!
//! # Namespaces
//!
//! Generated names (`|tN|` let binders, `max_N`/`min_N` atoms, `e`, `sN`
//! machine symbols) and user-controlled names must never collide - `|t0|`
//! and `t0` are the *same* SMT symbol, so an unprefixed user symbol named
//! `t0` would be captured by a let binding and corrupt the query. All
//! user symbols are therefore quoted into a reserved `u!` namespace
//! (`|u!name|`), which no generated name inhabits. Note one deliberate
//! divergence from `canon`: canon conflates `NamedSymbol("s5")` with
//! `Symbol(5)` (both intern as `s5`); here the named symbol is `|u!s5|`,
//! distinct from `s5` - the conflation would be unsound and only bites
//! kernels whose params are adversarially named `sN`.

use std::collections::{BTreeSet, HashMap, HashSet};
use std::rc::Rc;

use volta_analysis::symbolic::{ExprArena, ExprId, ExprNode};

#[derive(Debug, Clone, thiserror::Error)]
#[error("unsupported by the Z3 backend: {0}")]
pub struct Unsupported(pub String);

/// Which of the two opaque-atom families a node belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MaxMin {
    Max,
    Min,
}

/// How `Exp` nodes are encoded for the solver (see the module docs).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ExpMode {
    /// `(^ e a)` with `e` a free constant strictly bounded around Euler's
    /// number: sound, and z3 answers `unknown` on the exp identities -
    /// the paper's no-intervention baseline.
    #[default]
    PowerBounded,
    /// Uninterpreted `(uexp a)` plus the quantified addition law
    /// `forall x y. uexp(x)*uexp(y) = uexp(x+y)` - the paper's
    /// "with axiom" baseline, which drives z3 into an unbounded
    /// E-matching loop on softmax-shaped VCs (reported as Timeout).
    AdditionAxiom,
}

/// Handle to one interned term. Ordered so canonical operand lists can be
/// sorted.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
struct TermId(u32);

/// Structural identity of a translated term, shared across both arenas.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum StructKey {
    /// Literals and symbols, keyed by their rendered SMT text.
    Atom(String),
    /// Sum of operand terms with signed counts (sorted, no zero counts).
    Sum(Vec<(TermId, i128)>),
    /// Product of operand terms with signed exponents (sorted, no zeros).
    Prod(Vec<(TermId, i128)>),
    /// `e` raised to an operand term.
    Exp(TermId),
    /// An opaque max/min atom over sorted, deduplicated operands.
    MaxMin(MaxMin, Vec<TermId>),
}

/// Accumulates the structural intern table, declared symbols, opaque
/// atoms, and `let` bindings for one query. The reference and optimized
/// sides of one VC element share a `Builder`, so identical structure
/// (including a shared input symbol, e.g. both kernels reading `in[5]`)
/// collapses to the same term. `bindings` is filled in dependency order
/// (a binding's definition only references earlier bindings), so wrapping
/// the final query body in the bindings in order is always well-scoped.
#[derive(Default)]
pub struct Builder {
    terms: HashMap<StructKey, TermId>,
    /// Rendered reference text per term: the literal/symbol itself for
    /// atoms, the `|tN|` binder for compounds.
    term_text: Vec<String>,
    /// For sum terms, the canonical signed multiset - so an operand that
    /// is itself a sum splices into its parent instead of nesting, making
    /// the parent's key independent of grouping and DAG sharing.
    term_sum: Vec<Option<Rc<Vec<(TermId, i128)>>>>,
    /// Likewise for product terms (signed exponents).
    term_prod: Vec<Option<Rc<Vec<(TermId, i128)>>>>,
    /// Names needing `(declare-const _ Real)`: user symbols, machine
    /// symbols, and opaque max/min atoms. Sorted for deterministic output.
    reals: BTreeSet<String>,
    bindings: Vec<(String, String)>,
    next_atom: u32,
    uses_exp: bool,
    exp_mode: ExpMode,
}

impl Builder {
    pub fn new() -> Self {
        Self::default()
    }

    /// A builder whose `Exp` encoding follows `mode` (see [`ExpMode`]).
    pub fn with_exp_mode(mode: ExpMode) -> Self {
        Self {
            exp_mode: mode,
            ..Self::default()
        }
    }

    fn text(&self, t: TermId) -> &str {
        &self.term_text[t.0 as usize]
    }

    fn fresh_term(
        &mut self,
        key: StructKey,
        text: String,
        sum: Option<Rc<Vec<(TermId, i128)>>>,
        prod: Option<Rc<Vec<(TermId, i128)>>>,
    ) -> TermId {
        let t = TermId(self.term_text.len() as u32);
        self.terms.insert(key, t);
        self.term_text.push(text);
        self.term_sum.push(sum);
        self.term_prod.push(prod);
        t
    }

    /// Intern a literal or symbol by its rendered text, declaring it as a
    /// Real constant when `declare` is set (symbols yes, literals no).
    fn atom(&mut self, text: String, declare: bool) -> TermId {
        if let Some(&t) = self.terms.get(&StructKey::Atom(text.clone())) {
            return t;
        }
        if declare {
            self.reals.insert(text.clone());
        }
        self.fresh_term(StructKey::Atom(text.clone()), text, None, None)
    }

    fn bind(&mut self, def: String) -> String {
        let name = format!("|t{}|", self.bindings.len());
        self.bindings.push((name.clone(), def));
        name
    }

    /// The term is a pure negation `Sum{u: -1}`: return `u`.
    fn as_pure_negation(&self, t: TermId) -> Option<TermId> {
        match self.term_sum[t.0 as usize].as_ref().map(|v| v.as_slice()) {
            Some([(u, -1)]) => Some(*u),
            _ => None,
        }
    }

    /// Sort by term, merge counts, drop zeros. `None` on count overflow
    /// (only reachable via absurd ~2^127-fold multiplicities).
    fn canonicalize(mut parts: Vec<(TermId, i128)>) -> Option<Vec<(TermId, i128)>> {
        parts.sort_unstable_by_key(|&(t, _)| t);
        let mut merged: Vec<(TermId, i128)> = Vec::with_capacity(parts.len());
        for (t, c) in parts {
            match merged.last_mut() {
                Some((last, acc)) if *last == t => *acc = acc.checked_add(c)?,
                _ => merged.push((t, c)),
            }
        }
        merged.retain(|&(_, c)| c != 0);
        Some(merged)
    }

    /// Intern the sum of `raw` operand-terms with signed counts. Operands
    /// that are themselves sum terms splice in (scaled), so the key is
    /// independent of grouping and sharing. Cancellation is exact: an
    /// empty multiset is the literal `0.0`, a single `(t, 1)` is `t`.
    fn sum_term(&mut self, raw: Vec<(TermId, i128)>) -> Result<TermId, Unsupported> {
        let mut parts = Vec::with_capacity(raw.len());
        for (t, scale) in raw {
            match self.term_sum[t.0 as usize].clone() {
                Some(list) => {
                    for &(u, c) in list.iter() {
                        let c = c
                            .checked_mul(scale)
                            .ok_or_else(|| Unsupported("sum coefficient overflow".into()))?;
                        parts.push((u, c));
                    }
                }
                None => parts.push((t, scale)),
            }
        }
        let parts = Self::canonicalize(parts)
            .ok_or_else(|| Unsupported("sum coefficient overflow".into()))?;
        match parts[..] {
            [] => return Ok(self.atom("0.0".to_string(), false)),
            [(t, 1)] => return Ok(t),
            _ => {}
        }
        if let Some(&t) = self.terms.get(&StructKey::Sum(parts.clone())) {
            return Ok(t);
        }
        let rendered: Vec<String> = parts
            .iter()
            .map(|&(t, c)| {
                let text = self.text(t);
                match c {
                    1 => text.to_string(),
                    -1 => format!("(- {})", text),
                    c if c > 1 => format!("(* {}.0 {})", c, text),
                    c => format!("(- (* {}.0 {}))", c.unsigned_abs(), text),
                }
            })
            .collect();
        let def = if rendered.len() == 1 {
            rendered.into_iter().next().unwrap()
        } else {
            format!("(+ {})", rendered.join(" "))
        };
        let name = self.bind(def);
        Ok(self.fresh_term(
            StructKey::Sum(parts.clone()),
            name,
            Some(Rc::new(parts)),
            None,
        ))
    }

    /// Intern the product of `raw` operand-terms with signed exponents.
    /// Operands that are product terms splice in; a pure-negation operand
    /// hoists its sign out of the product, so sign placement doesn't
    /// split atom keys. An empty multiset is the literal `1.0`, a single
    /// `(t, 1)` is `t`.
    fn prod_term(&mut self, raw: Vec<(TermId, i128)>) -> Result<TermId, Unsupported> {
        let mut parts = Vec::with_capacity(raw.len());
        let mut odd_negations = false;
        for (mut t, scale) in raw {
            if let Some(u) = self.as_pure_negation(t) {
                // (-u)^scale = u^scale * (-1)^scale.
                if scale.rem_euclid(2) == 1 {
                    odd_negations = !odd_negations;
                }
                t = u;
            }
            match self.term_prod[t.0 as usize].clone() {
                Some(list) => {
                    for &(u, c) in list.iter() {
                        let c = c
                            .checked_mul(scale)
                            .ok_or_else(|| Unsupported("product exponent overflow".into()))?;
                        parts.push((u, c));
                    }
                }
                None => parts.push((t, scale)),
            }
        }
        let parts = Self::canonicalize(parts)
            .ok_or_else(|| Unsupported("product exponent overflow".into()))?;
        let base = match parts[..] {
            [] => self.atom("1.0".to_string(), false),
            [(t, 1)] => t,
            _ => {
                if let Some(&t) = self.terms.get(&StructKey::Prod(parts.clone())) {
                    t
                } else {
                    let render_side = |bld: &Self, side: &[(TermId, i128)]| -> String {
                        let rendered: Vec<String> = side
                            .iter()
                            .map(|&(t, e)| {
                                let text = bld.text(t);
                                if e == 1 {
                                    text.to_string()
                                } else {
                                    format!("(^ {} {}.0)", text, e)
                                }
                            })
                            .collect();
                        match rendered.len() {
                            0 => "1.0".to_string(),
                            1 => rendered.into_iter().next().unwrap(),
                            _ => format!("(* {})", rendered.join(" ")),
                        }
                    };
                    let num: Vec<(TermId, i128)> =
                        parts.iter().filter(|&&(_, e)| e > 0).copied().collect();
                    let den: Vec<(TermId, i128)> = parts
                        .iter()
                        .filter(|&&(_, e)| e < 0)
                        .map(|&(t, e)| (t, -e))
                        .collect();
                    let def = if den.is_empty() {
                        render_side(self, &num)
                    } else {
                        format!(
                            "(/ {} {})",
                            render_side(self, &num),
                            render_side(self, &den)
                        )
                    };
                    let name = self.bind(def);
                    self.fresh_term(
                        StructKey::Prod(parts.clone()),
                        name,
                        None,
                        Some(Rc::new(parts)),
                    )
                }
            }
        };
        if odd_negations {
            self.sum_term(vec![(base, -1)])
        } else {
            Ok(base)
        }
    }

    /// Intern the exponential of an operand, rendered per `exp_mode`.
    fn exp_term(&mut self, arg: TermId) -> TermId {
        self.uses_exp = true;
        if let Some(&t) = self.terms.get(&StructKey::Exp(arg)) {
            return t;
        }
        let def = match self.exp_mode {
            ExpMode::PowerBounded => format!("(^ e {})", self.text(arg)),
            ExpMode::AdditionAxiom => format!("(uexp {})", self.text(arg)),
        };
        let name = self.bind(def);
        self.fresh_term(StructKey::Exp(arg), name, None, None)
    }

    /// Intern an opaque max/min atom. Operands are sorted and deduplicated
    /// so the key is canonical; a singleton collapses to its operand
    /// (`max(x, x) = x`).
    fn maxmin_atom(&mut self, mm: MaxMin, mut args: Vec<TermId>) -> TermId {
        args.sort();
        args.dedup();
        if let [only] = args[..] {
            return only;
        }
        let key = StructKey::MaxMin(mm, args);
        if let Some(&t) = self.terms.get(&key) {
            return t;
        }
        let prefix = match mm {
            MaxMin::Max => "max_",
            MaxMin::Min => "min_",
        };
        let name = format!("{}{}", prefix, self.next_atom);
        self.next_atom += 1;
        self.reals.insert(name.clone());
        self.fresh_term(key, name, None, None)
    }

    /// SMT-LIB2 preamble: every declared symbol and opaque atom as a
    /// `declare-const`, plus - when the query uses `Exp` - the encoding
    /// of the exponential per `exp_mode`: the strictly-bounded free `e`
    /// constant, or the uninterpreted `uexp` with the addition-law axiom
    /// (see the module docs).
    pub fn preamble(&self) -> String {
        use std::fmt::Write as _;
        let mut out = String::new();
        if self.uses_exp {
            match self.exp_mode {
                ExpMode::PowerBounded => {
                    out.push_str("(declare-const e Real)\n");
                    out.push_str("(assert (< 2.718281828459045 e))\n");
                    out.push_str("(assert (< e 2.7182818284590455))\n");
                }
                ExpMode::AdditionAxiom => {
                    out.push_str("(declare-fun uexp (Real) Real)\n");
                    out.push_str(
                        "(assert (forall ((x Real) (y Real)) \
                         (= (* (uexp x) (uexp y)) (uexp (+ x y)))))\n",
                    );
                }
            }
        }
        for name in &self.reals {
            let _ = writeln!(out, "(declare-const {} Real)", name);
        }
        out
    }

    /// Wrap `body` in every accumulated `let` binding, first binding
    /// outermost - so each definition's free variables are already in
    /// scope. Built linearly (prefixes, body, closing parens) rather than
    /// by repeated re-wrapping: attention-scale queries have tens of
    /// thousands of bindings, and re-copying the accumulated string per
    /// binding is quadratic in the query size. (Deeply *nested* lets are
    /// fine: z3 parses 100k-deep nesting instantly, while flat
    /// `define-fun` chains at that scale hang its macro expansion.)
    pub fn wrap_in_lets(&self, body: &str) -> String {
        let bindings_len: usize = self
            .bindings
            .iter()
            .map(|(n, d)| n.len() + d.len() + 12)
            .sum();
        let mut out = String::with_capacity(bindings_len + body.len() + self.bindings.len());
        for (name, def) in &self.bindings {
            out.push_str("(let ((");
            out.push_str(name);
            out.push(' ');
            out.push_str(def);
            out.push_str(")) ");
        }
        out.push_str(body);
        out.extend(std::iter::repeat(')').take(self.bindings.len()));
        out
    }
}

/// Quote a user-controlled symbol name into the reserved `u!` namespace.
/// The prefix keeps user names disjoint from every generated name (`tN`
/// binders, `max_N`/`min_N` atoms, `e`, `sN`); the escaping keeps the
/// mapping injective (z3 accepts `\|`/`\\` inside quoted symbols).
fn quote_user(name: &str) -> String {
    format!("|u!{}|", name.replace('\\', "\\\\").replace('|', "\\|"))
}

/// Exact decimal rendering of `m * 2^k` (little-endian digit doubling; k
/// is at most 1074, so this is at most ~350k digit operations).
fn shift_decimal(m: u64, k: u32) -> String {
    let mut digits: Vec<u8> = m.to_string().bytes().rev().map(|b| b - b'0').collect();
    for _ in 0..k {
        let mut carry = 0u8;
        for d in digits.iter_mut() {
            let v = *d * 2 + carry;
            *d = v % 10;
            carry = v / 10;
        }
        if carry > 0 {
            digits.push(carry);
        }
    }
    digits.iter().rev().map(|d| (d + b'0') as char).collect()
}

/// Exact SMT real literal for the MAGNITUDE of a finite, nonzero f64: the
/// binary value `m * 2^e` from the bit pattern, rendered as an integer or
/// an exact fraction. This matches how `canon` (`Coeff::from_f64`) and
/// the numeric oracle interpret the same constant - rendering the
/// shortest decimal instead would silently reinterpret e.g. `0.1f64` as
/// the different real 1/10 and let the two backends reach opposite
/// verdicts on the same VC. The sign is handled by the caller: a negative
/// constant becomes a negated-positive SUM term, so `FloatConst(-c)` and
/// `Neg(FloatConst(c))` are one canonical term and cancellation across
/// them is exact.
fn real_literal_magnitude(v: f64) -> String {
    debug_assert!(v.is_finite() && v != 0.0);
    let bits = v.to_bits();
    let biased = ((bits >> 52) & 0x7ff) as i64;
    let frac = bits & ((1u64 << 52) - 1);
    let (mut m, mut e) = if biased == 0 {
        (frac, -1074i64)
    } else {
        (frac | (1u64 << 52), biased - 1075)
    };
    while m & 1 == 0 && e < 0 {
        m >>= 1;
        e += 1;
    }
    if e >= 0 {
        format!("{}.0", shift_decimal(m, e as u32))
    } else {
        format!("(/ {}.0 {}.0)", m, shift_decimal(1, (-e) as u32))
    }
}

type Memo = HashMap<ExprId, TermId>;

/// A leaf of a flattened additive chain: an operand with a signed count,
/// or an `Fma`'s product half (translated later, so flattening itself
/// stays allocation-only).
enum SumLeaf {
    Term(ExprId, i128),
    FmaProd(ExprId, ExprId, i128),
}

/// Flatten a chain of additive-family nodes (`Add`/`Sub`/`Neg`/`Fma`)
/// into leaves with signs, iteratively (an accumulator loop of N
/// iterations produces an N-deep spine; recursing here would overflow the
/// stack). Stops at nodes already translated (`memo`) and at nodes
/// reached a second time within this flatten (`visited`): a node whose
/// parent chain references it twice, e.g. `Add(x, x)` from `add f,f,f`,
/// becomes a leaf on later visits and its multiset splices back in at
/// `sum_term` - preserving multiplicity while keeping the expansion
/// linear (without the cutoff a self-referential doubling chain
/// re-expands exponentially).
fn flatten_sum(arena: &ExprArena, memo: &Memo, id: ExprId) -> Vec<SumLeaf> {
    let mut leaves = Vec::new();
    let mut visited = HashSet::new();
    let mut stack: Vec<(ExprId, i128)> = vec![(id, 1)];
    while let Some((cur, sign)) = stack.pop() {
        if memo.contains_key(&cur) || !visited.insert(cur) {
            leaves.push(SumLeaf::Term(cur, sign));
            continue;
        }
        match arena.node(cur) {
            ExprNode::Add(a, b) => {
                stack.push((*b, sign));
                stack.push((*a, sign));
            }
            ExprNode::Sub(a, b) => {
                stack.push((*b, -sign));
                stack.push((*a, sign));
            }
            ExprNode::Neg(a) => stack.push((*a, -sign)),
            ExprNode::Fma(a, b, c) => {
                leaves.push(SumLeaf::FmaProd(*a, *b, sign));
                stack.push((*c, sign));
            }
            _ => leaves.push(SumLeaf::Term(cur, sign)),
        }
    }
    leaves
}

/// Flatten a chain of multiplicative-family nodes (`Mul`/`Div`/`Rcp`)
/// into leaves with signed exponents; same memo/visited cutoffs as
/// `flatten_sum`.
fn flatten_prod(arena: &ExprArena, memo: &Memo, id: ExprId) -> Vec<(ExprId, i128)> {
    let mut leaves = Vec::new();
    let mut visited = HashSet::new();
    let mut stack: Vec<(ExprId, i128)> = vec![(id, 1)];
    while let Some((cur, exp)) = stack.pop() {
        if memo.contains_key(&cur) || !visited.insert(cur) {
            leaves.push((cur, exp));
            continue;
        }
        match arena.node(cur) {
            ExprNode::Mul(a, b) => {
                stack.push((*b, exp));
                stack.push((*a, exp));
            }
            ExprNode::Div(a, b) => {
                stack.push((*b, -exp));
                stack.push((*a, exp));
            }
            ExprNode::Rcp(a) => stack.push((*a, -exp)),
            _ => leaves.push((cur, exp)),
        }
    }
    leaves
}

fn translate_sum(
    bld: &mut Builder,
    memo: &mut Memo,
    arena: &ExprArena,
    id: ExprId,
) -> Result<TermId, Unsupported> {
    let leaves = flatten_sum(arena, memo, id);
    let mut raw = Vec::with_capacity(leaves.len());
    for leaf in leaves {
        match leaf {
            SumLeaf::Term(x, sign) => raw.push((translate(bld, memo, arena, x)?, sign)),
            SumLeaf::FmaProd(a, b, sign) => {
                let ta = translate(bld, memo, arena, a)?;
                let tb = translate(bld, memo, arena, b)?;
                let t = bld.prod_term(vec![(ta, 1), (tb, 1)])?;
                raw.push((t, sign));
            }
        }
    }
    bld.sum_term(raw)
}

fn translate_prod(
    bld: &mut Builder,
    memo: &mut Memo,
    arena: &ExprArena,
    id: ExprId,
) -> Result<TermId, Unsupported> {
    let leaves = flatten_prod(arena, memo, id);
    let mut raw = Vec::with_capacity(leaves.len());
    for (x, exp) in leaves {
        raw.push((translate(bld, memo, arena, x)?, exp));
    }
    bld.prod_term(raw)
}

fn is_max(node: &ExprNode) -> Option<(ExprId, ExprId)> {
    match node {
        ExprNode::Max(a, b) => Some((*a, *b)),
        _ => None,
    }
}

fn is_min(node: &ExprNode) -> Option<(ExprId, ExprId)> {
    match node {
        ExprNode::Min(a, b) => Some((*a, *b)),
        _ => None,
    }
}

fn translate_maxmin(
    bld: &mut Builder,
    memo: &mut Memo,
    arena: &ExprArena,
    id: ExprId,
    mm: MaxMin,
) -> Result<TermId, Unsupported> {
    let split = match mm {
        MaxMin::Max => is_max,
        MaxMin::Min => is_min,
    };
    // Unlike the sum/product families, nested same-op nodes are spliced
    // in even when already translated: the atom's identity IS its full
    // argument list, so it must not depend on whether an inner max was
    // reached first through some other parent. Max/min are idempotent, so
    // repeats are skipped outright rather than kept for multiplicity.
    let mut leaves = Vec::new();
    let mut visited = HashSet::new();
    let mut stack = vec![id];
    while let Some(cur) = stack.pop() {
        if !visited.insert(cur) {
            continue;
        }
        match split(arena.node(cur)) {
            Some((a, b)) => {
                stack.push(b);
                stack.push(a);
            }
            None => leaves.push(cur),
        }
    }
    let mut args = Vec::with_capacity(leaves.len());
    for leaf in leaves {
        args.push(translate(bld, memo, arena, leaf)?);
    }
    Ok(bld.maxmin_atom(mm, args))
}

/// Short, best-effort name for an unsupported node - the exact variant
/// name via `Debug`, not the full (potentially huge) subtree.
fn describe(node: &ExprNode) -> String {
    let s = format!("{:?}", node);
    s.split(|c: char| c == ' ' || c == '{' || c == '(')
        .next()
        .unwrap_or(&s)
        .to_string()
}

/// Translate one expression, interning through `bld`. `memo` caches this
/// arena's `ExprId -> TermId` mapping (ids from different arenas must not
/// share a memo; structural sharing across arenas happens in `bld`).
/// Guarded by `stacker::maybe_grow` like every other deep recursion over
/// expression chains in this workspace (`canon`, the numeric oracle):
/// chains of one family flatten iteratively, but deep spines that
/// alternate families recurse once per boundary.
fn translate(
    bld: &mut Builder,
    memo: &mut Memo,
    arena: &ExprArena,
    id: ExprId,
) -> Result<TermId, Unsupported> {
    if let Some(&t) = memo.get(&id) {
        return Ok(t);
    }
    let t = stacker::maybe_grow(64 * 1024, 8 * 1024 * 1024, || {
        translate_uncached(bld, memo, arena, id)
    })?;
    memo.insert(id, t);
    Ok(t)
}

fn translate_uncached(
    bld: &mut Builder,
    memo: &mut Memo,
    arena: &ExprArena,
    id: ExprId,
) -> Result<TermId, Unsupported> {
    match arena.node(id) {
        ExprNode::IntConst(v) => {
            let text = format!("{}.0", (*v as i128).unsigned_abs());
            let mag = bld.atom(text, false);
            if *v < 0 {
                bld.sum_term(vec![(mag, -1)])
            } else {
                Ok(mag)
            }
        }
        ExprNode::FloatConst(v) => {
            if !v.is_finite() {
                return Err(Unsupported(format!("non-finite float constant {}", v)));
            }
            if *v == 0.0 {
                // Covers -0.0 too: over the reals they are the same number.
                return Ok(bld.atom("0.0".to_string(), false));
            }
            let mag = bld.atom(real_literal_magnitude(v.abs()), false);
            if v.is_sign_negative() {
                bld.sum_term(vec![(mag, -1)])
            } else {
                Ok(mag)
            }
        }
        ExprNode::BoolConst(v) => Ok(bld.atom(if *v { "1.0" } else { "0.0" }.to_string(), false)),

        ExprNode::NamedSymbol(sid) => Ok(bld.atom(quote_user(arena.string(*sid)), true)),
        ExprNode::Symbol(sym) => Ok(bld.atom(sym.to_string(), true)),

        ExprNode::Add(..) | ExprNode::Sub(..) | ExprNode::Neg(..) | ExprNode::Fma(..) => {
            translate_sum(bld, memo, arena, id)
        }
        ExprNode::Mul(..) | ExprNode::Div(..) | ExprNode::Rcp(..) => {
            translate_prod(bld, memo, arena, id)
        }

        ExprNode::Exp(a) => {
            let ta = translate(bld, memo, arena, *a)?;
            Ok(bld.exp_term(ta))
        }

        ExprNode::Max(..) => translate_maxmin(bld, memo, arena, id, MaxMin::Max),
        ExprNode::Min(..) => translate_maxmin(bld, memo, arena, id, MaxMin::Min),

        // Conversions: identity over the reals (matches
        // `canon::canonicalize`'s treatment of the same nodes exactly).
        // Not interned themselves - they forward straight to the child.
        ExprNode::ToFloat(a)
        | ExprNode::SignExtend { value: a, .. }
        | ExprNode::ZeroExtend { value: a, .. }
        | ExprNode::Truncate { value: a, .. } => translate(bld, memo, arena, *a),

        other => Err(Unsupported(describe(other))),
    }
}

/// Translate a whole root expression, returning its rendered term text.
/// Starts a fresh per-arena memo; use one call per side (reference and
/// optimized) sharing the same `Builder` so identical structure - and
/// max/min atom identity - is shared across the two sides.
pub fn translate_root(
    bld: &mut Builder,
    arena: &ExprArena,
    id: ExprId,
) -> Result<String, Unsupported> {
    let mut memo = Memo::new();
    let t = translate(bld, &mut memo, arena, id)?;
    Ok(bld.text(t).to_string())
}

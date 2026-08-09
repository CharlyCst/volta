//! Symbolic expressions for abstract interpretation
//!
//! This module defines symbolic expressions that represent values during
//! abstract interpretation. These expressions are over the mathematical reals,
//! as the paper treats floating-point values as reals for equivalence checking.
//!
//! Expressions are arena-allocated: each expression node lives in an `ExprArena`,
//! and is referred to by a lightweight, copyable `ExprId` handle.

use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};

use id_collections::{IdVec, id_type};

/// Global counter for generating fresh symbol IDs
static SYMBOL_COUNTER: AtomicU64 = AtomicU64::new(0);

/// A unique identifier for a symbolic variable
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct SymbolId(pub u64);

impl SymbolId {
    /// Generate a fresh symbol ID
    pub fn fresh() -> Self {
        Self(SYMBOL_COUNTER.fetch_add(1, Ordering::SeqCst))
    }
}

impl fmt::Display for SymbolId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "s{}", self.0)
    }
}

/// The identity of a symbolic atom, shared by every backend that must
/// agree on which symbols are equal (canon, the numeric oracle,
/// `volta_z3`). Only launch-config names carry identity - PTX-source
/// names are scoped and must not (bind those to fresh machine
/// [`Symbol`](ExprNode::Symbol)s instead) - and the three namespaces are
/// disjoint by construction rather than by string formatting.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SymbolRef<'a> {
    /// A launch-config symbolic parameter (`sym:NAME`), by config name.
    Param(&'a str),
    /// One element of a launch-config input array, by config array name
    /// and element index.
    Element { array: &'a str, index: u64 },
    /// A machine-generated `Symbol`, identified by its `SymbolId`.
    Machine(SymbolId),
}

// =========================================================================
// Arena IDs
// =========================================================================

/// A lightweight handle to an expression node in an `ExprArena`.
///
/// The serde impls (via `id_collections`) encode the bare index; an id is
/// only meaningful when paired with the `ExprArena` that produced it -
/// see `ExprArena`'s serialization below.
#[id_type(serde = true)]
pub struct ExprId(pub u32);

/// A handle to a string stored in the arena's string table.
#[id_type(serde = true)]
pub struct StringId(pub u32);

// =========================================================================
// Expression node
// =========================================================================

/// A single expression node. Children are referenced by `ExprId`.
///
/// Following the paper, we model tensor values as real numbers.
/// The decision procedure will later check equality of these expressions.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum ExprNode {
    // =====================================================================
    // Atoms
    // =====================================================================
    /// Integer constant
    IntConst(i64),
    /// Floating-point constant (treated as real)
    FloatConst(f64),
    /// Boolean constant (for predicates)
    BoolConst(bool),
    /// Machine-generated symbolic variable: a fresh unknown with no
    /// launch-config identity, correlated with nothing (see [`SymbolRef`]).
    Symbol(SymbolId),
    /// A launch-config symbolic parameter (`sym:NAME`), by config name.
    ParamSymbol(StringId),
    /// Element `index` of the launch-config input array named `array`.
    InputElement { array: StringId, index: u64 },

    // =====================================================================
    // Arithmetic (over reals)
    // =====================================================================
    /// Addition: a + b
    Add(ExprId, ExprId),
    /// Subtraction: a - b
    Sub(ExprId, ExprId),
    /// Multiplication: a * b
    Mul(ExprId, ExprId),
    /// Division: a / b
    Div(ExprId, ExprId),
    /// Remainder: a % b (integer remainder; uninterpreted for equivalence)
    Rem(ExprId, ExprId),
    /// Negation: -a
    Neg(ExprId),

    // =====================================================================
    // Transcendental functions
    // =====================================================================
    /// Exponential: e^a
    Exp(ExprId),
    /// Natural logarithm: ln(a)
    Log(ExprId),
    /// Square root: sqrt(a)
    Sqrt(ExprId),
    /// Reciprocal: 1/a
    Rcp(ExprId),

    // =====================================================================
    // Min/Max
    // =====================================================================
    /// Maximum: max(a, b)
    Max(ExprId, ExprId),
    /// Minimum: min(a, b)
    Min(ExprId, ExprId),
    /// Absolute value: |a|
    Abs(ExprId),

    // =====================================================================
    // Bitwise operations
    // =====================================================================
    /// Bitwise AND: a & b
    BitAnd(ExprId, ExprId),
    /// Bitwise OR: a | b
    BitOr(ExprId, ExprId),
    /// Bitwise XOR: a ^ b
    BitXor(ExprId, ExprId),
    /// Bitwise NOT: ~a
    BitNot(ExprId),
    /// Left shift: a << b
    Shl(ExprId, ExprId),
    /// Arithmetic right shift: a >> b (sign-extending)
    Shr(ExprId, ExprId),
    /// Logical right shift: a >>> b (zero-extending)
    LShr(ExprId, ExprId),

    // =====================================================================
    // Comparisons (return boolean expressions)
    // =====================================================================
    /// Equal: a == b
    Eq(ExprId, ExprId),
    /// Not equal: a != b
    Ne(ExprId, ExprId),
    /// Less than: a < b
    Lt(ExprId, ExprId),
    /// Less than or equal: a <= b
    Le(ExprId, ExprId),
    /// Greater than: a > b
    Gt(ExprId, ExprId),
    /// Greater than or equal: a >= b
    Ge(ExprId, ExprId),

    // =====================================================================
    // Boolean operations
    // =====================================================================
    /// Logical AND: a && b
    And(ExprId, ExprId),
    /// Logical OR: a || b
    Or(ExprId, ExprId),
    /// Logical NOT: !a
    Not(ExprId),

    // =====================================================================
    // Conditional
    // =====================================================================
    /// Select: cond ? then_val : else_val
    Select(ExprId, ExprId, ExprId),

    // =====================================================================
    // Type conversions
    // =====================================================================
    /// Convert to float (from int)
    ToFloat(ExprId),
    /// Convert to int (from float, truncating)
    ToInt(ExprId),
    /// Sign extend from narrower int
    SignExtend {
        value: ExprId,
        from_bits: u32,
        to_bits: u32,
    },
    /// Zero extend from narrower int
    ZeroExtend {
        value: ExprId,
        from_bits: u32,
        to_bits: u32,
    },
    /// Truncate to narrower int
    Truncate { value: ExprId, to_bits: u32 },

    // =====================================================================
    // Special
    // =====================================================================
    /// Fused multiply-add: a * b + c
    Fma(ExprId, ExprId, ExprId),
    /// Symbolic read from a launch-config input array at a symbolic index.
    ///
    /// Represents `array[index]`. When `index` is substituted to a
    /// concrete integer `i`, this resolves to `InputElement { array, i }`,
    /// the same identity lazy input materialization produces.
    SymbolicRead { array: StringId, index: ExprId },
    /// Discarded per-thread value. Set during re-aggregation when static
    /// liveness analysis proves the register will be overwritten before being
    /// read. If the evaluator reads this, the liveness analysis has a bug.
    Discarded,
    /// Undefined value (for detecting use of uninitialized data)
    Undefined,
}

impl ExprNode {
    /// Call `f` on every child `ExprId` of this node. An exhaustive match
    /// (no wildcard) so a new variant fails to compile here instead of
    /// silently having its children skipped.
    pub fn for_each_child(&self, mut f: impl FnMut(ExprId)) {
        match self {
            ExprNode::IntConst(_)
            | ExprNode::FloatConst(_)
            | ExprNode::BoolConst(_)
            | ExprNode::Symbol(_)
            | ExprNode::ParamSymbol(_)
            | ExprNode::InputElement { .. }
            | ExprNode::Discarded
            | ExprNode::Undefined => {}

            ExprNode::Neg(a)
            | ExprNode::Exp(a)
            | ExprNode::Log(a)
            | ExprNode::Sqrt(a)
            | ExprNode::Rcp(a)
            | ExprNode::Abs(a)
            | ExprNode::BitNot(a)
            | ExprNode::Not(a)
            | ExprNode::ToFloat(a)
            | ExprNode::ToInt(a)
            | ExprNode::SignExtend { value: a, .. }
            | ExprNode::ZeroExtend { value: a, .. }
            | ExprNode::Truncate { value: a, .. }
            | ExprNode::SymbolicRead { index: a, .. } => f(*a),

            ExprNode::Add(a, b)
            | ExprNode::Sub(a, b)
            | ExprNode::Mul(a, b)
            | ExprNode::Div(a, b)
            | ExprNode::Rem(a, b)
            | ExprNode::Max(a, b)
            | ExprNode::Min(a, b)
            | ExprNode::BitAnd(a, b)
            | ExprNode::BitOr(a, b)
            | ExprNode::BitXor(a, b)
            | ExprNode::Shl(a, b)
            | ExprNode::Shr(a, b)
            | ExprNode::LShr(a, b)
            | ExprNode::Eq(a, b)
            | ExprNode::Ne(a, b)
            | ExprNode::Lt(a, b)
            | ExprNode::Le(a, b)
            | ExprNode::Gt(a, b)
            | ExprNode::Ge(a, b)
            | ExprNode::And(a, b)
            | ExprNode::Or(a, b) => {
                f(*a);
                f(*b);
            }

            ExprNode::Select(a, b, c) | ExprNode::Fma(a, b, c) => {
                f(*a);
                f(*b);
                f(*c);
            }
        }
    }

    /// The `StringId` this node references, if any.
    pub fn string_id(&self) -> Option<StringId> {
        match self {
            ExprNode::ParamSymbol(sid) => Some(*sid),
            ExprNode::InputElement { array, .. } => Some(*array),
            ExprNode::SymbolicRead { array, .. } => Some(*array),
            _ => None,
        }
    }

    /// The symbol identity this node denotes, if it is a symbolic atom -
    /// the single mapping from node representation to [`SymbolRef`].
    pub fn symbol_ref<'a>(&self, arena: &'a ExprArena) -> Option<SymbolRef<'a>> {
        match self {
            ExprNode::ParamSymbol(sid) => Some(SymbolRef::Param(arena.string(*sid))),
            ExprNode::InputElement { array, index } => Some(SymbolRef::Element {
                array: arena.string(*array),
                index: *index,
            }),
            ExprNode::Symbol(sym) => Some(SymbolRef::Machine(*sym)),
            _ => None,
        }
    }
}

// =========================================================================
// Arena
// =========================================================================

/// Arena-based storage for expression nodes.
///
/// All expression nodes live here. Callers manipulate expressions via
/// lightweight, copyable `ExprId` handles.
///
/// The serde impls are the `volta compare --dump-vcs`/`--from-dump`
/// persistence format: `IdVec` (via `id_collections`'s `serde` feature)
/// encodes exactly like a `Vec`, serialized in place - no transient clone
/// of GiB-scale arenas at dump time.
#[derive(serde::Serialize, serde::Deserialize)]
pub struct ExprArena {
    nodes: IdVec<ExprId, ExprNode>,
    strings: IdVec<StringId, String>,
}

impl ExprArena {
    /// Create an empty arena.
    pub fn new() -> Self {
        Self {
            nodes: IdVec::new(),
            strings: IdVec::new(),
        }
    }

    // -----------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------

    /// Push a node into the arena and return its id.
    fn push(&mut self, node: ExprNode) -> ExprId {
        self.nodes.push(node)
    }

    // -----------------------------------------------------------------
    // Lookup
    // -----------------------------------------------------------------

    /// Look up the node for a given id.
    pub fn node(&self, id: ExprId) -> &ExprNode {
        &self.nodes[id]
    }

    /// Number of nodes in the arena.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Look up a string by its id.
    pub fn string(&self, id: StringId) -> &str {
        &self.strings[id]
    }

    /// Check that every node's children were pushed strictly before it
    /// (which also bounds them inside the arena) and every `StringId` is
    /// in range. Construction guarantees children-before-parents, so a
    /// freshly built arena always passes; this exists for arenas rebuilt
    /// from external data (`--from-dump`), where a corrupt or
    /// version-skewed file could otherwise smuggle in a forward or
    /// self-reference and send a later consumer into unbounded recursion,
    /// or an out-of-bounds id into a panic deep inside a check.
    pub fn validate(&self) -> Result<(), String> {
        let n_strings = self.strings.len();
        for (id, node) in self.nodes.as_slice().iter().enumerate() {
            let mut bad_child = None;
            node.for_each_child(|child| {
                if id_collections::Id::to_index(child) as usize >= id && bad_child.is_none() {
                    bad_child = Some(child);
                }
            });
            if let Some(child) = bad_child {
                return Err(format!(
                    "node {} references child expression {}, but children must \
                     precede their parents (forward, self, or out-of-bounds \
                     reference)",
                    id,
                    id_collections::Id::to_index(child),
                ));
            }
            if let Some(sid) = node.string_id() {
                if id_collections::Id::to_index(sid) as usize >= n_strings {
                    return Err(format!(
                        "node {} references string {} but the arena has {} strings",
                        id,
                        id_collections::Id::to_index(sid),
                        n_strings
                    ));
                }
            }
        }
        Ok(())
    }

    /// Test-only back door for building arenas that violate construction
    /// invariants, to exercise `validate`.
    #[cfg(test)]
    pub(crate) fn from_raw_parts_for_test(nodes: Vec<ExprNode>, strings: Vec<String>) -> Self {
        Self {
            nodes: IdVec::from_vec(nodes),
            strings: IdVec::from_vec(strings),
        }
    }

    // -----------------------------------------------------------------
    // Atom constructors
    // -----------------------------------------------------------------

    /// Create an integer constant node.
    pub fn int(&mut self, v: i64) -> ExprId {
        self.push(ExprNode::IntConst(v))
    }

    /// Create a floating-point constant node.
    pub fn float(&mut self, v: f64) -> ExprId {
        self.push(ExprNode::FloatConst(v))
    }

    /// Create a boolean constant node.
    pub fn bool_val(&mut self, v: bool) -> ExprId {
        self.push(ExprNode::BoolConst(v))
    }

    /// Create a fresh machine symbolic variable - the constructor for any
    /// value without a launch-config identity (see [`SymbolRef`]).
    pub fn symbol(&mut self) -> ExprId {
        self.push(ExprNode::Symbol(SymbolId::fresh()))
    }

    /// Intern a string, returning its id. Does not deduplicate; callers
    /// creating many nodes that share a string (e.g. one array's elements)
    /// should intern once and reuse the id.
    pub fn intern_string(&mut self, s: impl Into<String>) -> StringId {
        self.strings.push(s.into())
    }

    /// Create a launch-config symbolic parameter (`sym:NAME`); the name
    /// must come from the launch config (see [`SymbolRef`]).
    pub fn param_symbol(&mut self, name: impl Into<String>) -> ExprId {
        let sid = self.strings.push(name.into());
        self.push(ExprNode::ParamSymbol(sid))
    }

    /// Create the symbol for element `index` of the launch-config input
    /// array whose name is interned at `array`.
    pub fn input_element(&mut self, array: StringId, index: u64) -> ExprId {
        self.push(ExprNode::InputElement { array, index })
    }

    /// Create an undefined-value node.
    pub fn undefined(&mut self) -> ExprId {
        self.push(ExprNode::Undefined)
    }

    // =================================================================
    // Arithmetic builders (with eager constant folding)
    // =================================================================

    /// Addition with constant folding.
    pub fn add(&mut self, a: ExprId, b: ExprId) -> ExprId {
        match (self.node(a), self.node(b)) {
            (ExprNode::IntConst(x), ExprNode::IntConst(y)) => {
                let r = x.wrapping_add(*y);
                return self.int(r);
            }
            (ExprNode::FloatConst(x), ExprNode::FloatConst(y)) => {
                let r = *x + *y;
                return self.float(r);
            }
            (ExprNode::IntConst(0), _) => return b,
            (_, ExprNode::IntConst(0)) => return a,
            (ExprNode::FloatConst(x), _) if *x == 0.0 => return b,
            (_, ExprNode::FloatConst(y)) if *y == 0.0 => return a,
            _ => {}
        }
        self.push(ExprNode::Add(a, b))
    }

    /// Subtraction with constant folding.
    pub fn sub(&mut self, a: ExprId, b: ExprId) -> ExprId {
        match (self.node(a), self.node(b)) {
            (ExprNode::IntConst(x), ExprNode::IntConst(y)) => {
                let r = x.wrapping_sub(*y);
                return self.int(r);
            }
            (ExprNode::FloatConst(x), ExprNode::FloatConst(y)) => {
                let r = *x - *y;
                return self.float(r);
            }
            (_, ExprNode::IntConst(0)) => return a,
            (_, ExprNode::FloatConst(y)) if *y == 0.0 => return a,
            _ => {}
        }
        self.push(ExprNode::Sub(a, b))
    }

    /// Multiplication with constant folding.
    ///
    /// `0.0 * x = 0` and `1.0 * x = x` hold over the reals (we do not model
    /// IEEE `0 * inf`); the zero annihilation is what keeps `-INFINITY`
    /// running-max seeds out of live expressions (`d * e^{m_0 - m_1}` with
    /// `d = 0.0` on the first FlashAttention iteration).
    pub fn mul(&mut self, a: ExprId, b: ExprId) -> ExprId {
        match (self.node(a), self.node(b)) {
            (ExprNode::IntConst(x), ExprNode::IntConst(y)) => {
                let r = x.wrapping_mul(*y);
                return self.int(r);
            }
            (ExprNode::FloatConst(x), ExprNode::FloatConst(y)) => {
                let r = *x * *y;
                return self.float(r);
            }
            (ExprNode::IntConst(0), _) | (_, ExprNode::IntConst(0)) => return self.int(0),
            (ExprNode::FloatConst(x), _) if *x == 0.0 => return self.float(0.0),
            (_, ExprNode::FloatConst(y)) if *y == 0.0 => return self.float(0.0),
            (ExprNode::IntConst(1), _) => return b,
            (_, ExprNode::IntConst(1)) => return a,
            (ExprNode::FloatConst(x), _) if *x == 1.0 => return b,
            (_, ExprNode::FloatConst(y)) if *y == 1.0 => return a,
            _ => {}
        }
        self.push(ExprNode::Mul(a, b))
    }

    /// Division with constant folding.
    pub fn div(&mut self, a: ExprId, b: ExprId) -> ExprId {
        match (self.node(a), self.node(b)) {
            (ExprNode::IntConst(x), ExprNode::IntConst(y)) if *y != 0 => {
                let r = x.wrapping_div(*y);
                return self.int(r);
            }
            (ExprNode::FloatConst(x), ExprNode::FloatConst(y)) => {
                let r = *x / *y;
                return self.float(r);
            }
            (ExprNode::IntConst(0), _) => return self.int(0),
            (_, ExprNode::IntConst(1)) => return a,
            _ => {}
        }
        self.push(ExprNode::Div(a, b))
    }

    /// Remainder with constant folding (i64 semantics when concrete).
    pub fn rem(&mut self, a: ExprId, b: ExprId) -> ExprId {
        match (self.node(a), self.node(b)) {
            (ExprNode::IntConst(x), ExprNode::IntConst(y)) if *y != 0 => {
                let r = x.wrapping_rem(*y);
                return self.int(r);
            }
            _ => {}
        }
        self.push(ExprNode::Rem(a, b))
    }

    /// Negation with constant folding.
    pub fn neg(&mut self, a: ExprId) -> ExprId {
        match *self.node(a) {
            ExprNode::IntConst(x) => return self.int(-x),
            ExprNode::FloatConst(x) => return self.float(-x),
            _ => {}
        }
        self.push(ExprNode::Neg(a))
    }

    /// Exponential: e^a
    pub fn exp(&mut self, a: ExprId) -> ExprId {
        self.push(ExprNode::Exp(a))
    }

    /// Natural logarithm: ln(a)
    pub fn log(&mut self, a: ExprId) -> ExprId {
        self.push(ExprNode::Log(a))
    }

    /// Square root: sqrt(a)
    pub fn sqrt(&mut self, a: ExprId) -> ExprId {
        self.push(ExprNode::Sqrt(a))
    }

    /// Reciprocal: 1/a
    pub fn rcp(&mut self, a: ExprId) -> ExprId {
        self.push(ExprNode::Rcp(a))
    }

    /// Maximum with constant folding.
    /// `max(-inf, x) = x`: running-max chains start at -INFINITY.
    pub fn max(&mut self, a: ExprId, b: ExprId) -> ExprId {
        match (self.node(a), self.node(b)) {
            (ExprNode::IntConst(x), ExprNode::IntConst(y)) => {
                let r = *x.max(y);
                return self.int(r);
            }
            (ExprNode::FloatConst(x), ExprNode::FloatConst(y)) => {
                let r = x.max(*y);
                return self.float(r);
            }
            (ExprNode::FloatConst(x), _) if *x == f64::NEG_INFINITY => return b,
            (_, ExprNode::FloatConst(y)) if *y == f64::NEG_INFINITY => return a,
            _ => {}
        }
        if a == b {
            return a;
        }
        self.push(ExprNode::Max(a, b))
    }

    /// Minimum with constant folding.
    /// `min(inf, x) = x`: running-min chains start at INFINITY.
    pub fn min(&mut self, a: ExprId, b: ExprId) -> ExprId {
        match (self.node(a), self.node(b)) {
            (ExprNode::IntConst(x), ExprNode::IntConst(y)) => {
                let r = *x.min(y);
                return self.int(r);
            }
            (ExprNode::FloatConst(x), ExprNode::FloatConst(y)) => {
                let r = x.min(*y);
                return self.float(r);
            }
            (ExprNode::FloatConst(x), _) if *x == f64::INFINITY => return b,
            (_, ExprNode::FloatConst(y)) if *y == f64::INFINITY => return a,
            _ => {}
        }
        if a == b {
            return a;
        }
        self.push(ExprNode::Min(a, b))
    }

    /// Absolute value.
    pub fn abs(&mut self, a: ExprId) -> ExprId {
        self.push(ExprNode::Abs(a))
    }

    /// Fused multiply-add with constant folding: a * b + c
    /// (`fma(0, b, c) = c` over the reals; see `mul` for why.)
    pub fn fma(&mut self, a: ExprId, b: ExprId, c: ExprId) -> ExprId {
        if let (ExprNode::FloatConst(x), ExprNode::FloatConst(y), ExprNode::FloatConst(z)) =
            (self.node(a), self.node(b), self.node(c))
        {
            let r = x.mul_add(*y, *z);
            return self.float(r);
        }
        let zero_a = matches!(self.node(a), ExprNode::IntConst(0))
            || matches!(self.node(a), ExprNode::FloatConst(x) if *x == 0.0);
        let zero_b = matches!(self.node(b), ExprNode::IntConst(0))
            || matches!(self.node(b), ExprNode::FloatConst(y) if *y == 0.0);
        if zero_a || zero_b {
            return c;
        }
        self.push(ExprNode::Fma(a, b, c))
    }

    /// Create a symbolic read from launch-config input array `array_name`:
    /// `array_name[index]`.
    ///
    /// When `index` is concrete, immediately resolves to the typed
    /// `InputElement` identity - the same one lazy input materialization
    /// produces. Otherwise stores a `SymbolicRead` node that resolves upon
    /// TID substitution. (A negative concrete index wraps into `u64`;
    /// every resolver uses the same wrapping, and no materialized element
    /// can collide with it since element indices are bounded by the
    /// array's length.)
    pub fn symbolic_read(&mut self, array_name: &str, index: ExprId) -> ExprId {
        let sid = self.strings.push(array_name.to_string());
        // Eagerly resolve if index is concrete
        if let Some(i) = self.as_i64(index) {
            return self.input_element(sid, i as u64);
        }
        self.push(ExprNode::SymbolicRead { array: sid, index })
    }

    // =================================================================
    // Bitwise builders (with eager constant folding)
    // =================================================================

    /// Bitwise AND with constant folding.
    pub fn bit_and(&mut self, a: ExprId, b: ExprId) -> ExprId {
        if let (ExprNode::IntConst(x), ExprNode::IntConst(y)) = (self.node(a), self.node(b)) {
            let r = *x & *y;
            return self.int(r);
        }
        self.push(ExprNode::BitAnd(a, b))
    }

    /// Bitwise OR with constant folding.
    pub fn bit_or(&mut self, a: ExprId, b: ExprId) -> ExprId {
        if let (ExprNode::IntConst(x), ExprNode::IntConst(y)) = (self.node(a), self.node(b)) {
            let r = *x | *y;
            return self.int(r);
        }
        self.push(ExprNode::BitOr(a, b))
    }

    /// Bitwise XOR with constant folding.
    pub fn bit_xor(&mut self, a: ExprId, b: ExprId) -> ExprId {
        if let (ExprNode::IntConst(x), ExprNode::IntConst(y)) = (self.node(a), self.node(b)) {
            let r = *x ^ *y;
            return self.int(r);
        }
        self.push(ExprNode::BitXor(a, b))
    }

    /// Bitwise NOT with constant folding.
    pub fn bit_not(&mut self, a: ExprId) -> ExprId {
        if let ExprNode::IntConst(x) = *self.node(a) {
            return self.int(!x);
        }
        self.push(ExprNode::BitNot(a))
    }

    /// Left shift with constant folding.
    pub fn shl(&mut self, a: ExprId, b: ExprId) -> ExprId {
        match (self.node(a), self.node(b)) {
            (ExprNode::IntConst(x), ExprNode::IntConst(y)) => {
                let r = x.wrapping_shl(*y as u32);
                return self.int(r);
            }
            (_, ExprNode::IntConst(0)) => return a,
            _ => {}
        }
        self.push(ExprNode::Shl(a, b))
    }

    /// Arithmetic right shift with constant folding.
    pub fn shr(&mut self, a: ExprId, b: ExprId) -> ExprId {
        match (self.node(a), self.node(b)) {
            (ExprNode::IntConst(x), ExprNode::IntConst(y)) => {
                let r = x.wrapping_shr(*y as u32);
                return self.int(r);
            }
            (_, ExprNode::IntConst(0)) => return a,
            _ => {}
        }
        self.push(ExprNode::Shr(a, b))
    }

    /// Logical right shift with constant folding.
    pub fn lshr(&mut self, a: ExprId, b: ExprId) -> ExprId {
        match (self.node(a), self.node(b)) {
            (ExprNode::IntConst(x), ExprNode::IntConst(y)) => {
                let r = (*x as u64).wrapping_shr(*y as u32) as i64;
                return self.int(r);
            }
            (_, ExprNode::IntConst(0)) => return a,
            (ExprNode::IntConst(0), _) => return self.int(0),
            _ => {}
        }
        self.push(ExprNode::LShr(a, b))
    }

    // =================================================================
    // Comparison builders (with eager constant folding)
    // =================================================================

    /// Equal with constant folding.
    pub fn eq(&mut self, a: ExprId, b: ExprId) -> ExprId {
        if let (ExprNode::IntConst(x), ExprNode::IntConst(y)) = (self.node(a), self.node(b)) {
            let r = *x == *y;
            return self.bool_val(r);
        }
        self.push(ExprNode::Eq(a, b))
    }

    /// Not-equal with constant folding.
    pub fn ne(&mut self, a: ExprId, b: ExprId) -> ExprId {
        if let (ExprNode::IntConst(x), ExprNode::IntConst(y)) = (self.node(a), self.node(b)) {
            let r = *x != *y;
            return self.bool_val(r);
        }
        self.push(ExprNode::Ne(a, b))
    }

    /// Less-than with constant folding.
    pub fn lt(&mut self, a: ExprId, b: ExprId) -> ExprId {
        match (self.node(a), self.node(b)) {
            (ExprNode::IntConst(x), ExprNode::IntConst(y)) => {
                let r = *x < *y;
                return self.bool_val(r);
            }
            (ExprNode::FloatConst(x), ExprNode::FloatConst(y)) => {
                let r = *x < *y;
                return self.bool_val(r);
            }
            _ => {}
        }
        self.push(ExprNode::Lt(a, b))
    }

    /// Less-or-equal with constant folding.
    pub fn le(&mut self, a: ExprId, b: ExprId) -> ExprId {
        match (self.node(a), self.node(b)) {
            (ExprNode::IntConst(x), ExprNode::IntConst(y)) => {
                let r = *x <= *y;
                return self.bool_val(r);
            }
            (ExprNode::FloatConst(x), ExprNode::FloatConst(y)) => {
                let r = *x <= *y;
                return self.bool_val(r);
            }
            _ => {}
        }
        self.push(ExprNode::Le(a, b))
    }

    /// Greater-than with constant folding.
    pub fn gt(&mut self, a: ExprId, b: ExprId) -> ExprId {
        match (self.node(a), self.node(b)) {
            (ExprNode::IntConst(x), ExprNode::IntConst(y)) => {
                let r = *x > *y;
                return self.bool_val(r);
            }
            (ExprNode::FloatConst(x), ExprNode::FloatConst(y)) => {
                let r = *x > *y;
                return self.bool_val(r);
            }
            _ => {}
        }
        self.push(ExprNode::Gt(a, b))
    }

    /// Greater-or-equal with constant folding.
    pub fn ge(&mut self, a: ExprId, b: ExprId) -> ExprId {
        match (self.node(a), self.node(b)) {
            (ExprNode::IntConst(x), ExprNode::IntConst(y)) => {
                let r = *x >= *y;
                return self.bool_val(r);
            }
            (ExprNode::FloatConst(x), ExprNode::FloatConst(y)) => {
                let r = *x >= *y;
                return self.bool_val(r);
            }
            _ => {}
        }
        self.push(ExprNode::Ge(a, b))
    }

    // =================================================================
    // Boolean builders
    // =================================================================

    /// Logical AND.
    pub fn and(&mut self, a: ExprId, b: ExprId) -> ExprId {
        self.push(ExprNode::And(a, b))
    }

    /// Logical OR.
    pub fn or(&mut self, a: ExprId, b: ExprId) -> ExprId {
        self.push(ExprNode::Or(a, b))
    }

    /// Logical NOT.
    pub fn not(&mut self, a: ExprId) -> ExprId {
        self.push(ExprNode::Not(a))
    }

    // =================================================================
    // Conditional (with eager folding on concrete condition)
    // =================================================================

    /// Select: cond ? then_val : else_val
    pub fn select(&mut self, cond: ExprId, then_val: ExprId, else_val: ExprId) -> ExprId {
        match self.as_bool(cond) {
            Some(true) => return then_val,
            Some(false) => return else_val,
            None => {}
        }
        self.push(ExprNode::Select(cond, then_val, else_val))
    }

    // =================================================================
    // Conversions
    // =================================================================

    /// Convert to float (from int), with constant folding.
    pub fn to_float(&mut self, a: ExprId) -> ExprId {
        if let ExprNode::IntConst(v) = *self.node(a) {
            return self.float(v as f64);
        }
        self.push(ExprNode::ToFloat(a))
    }

    /// Convert to int (from float, truncating), with constant folding.
    pub fn to_int(&mut self, a: ExprId) -> ExprId {
        if let ExprNode::FloatConst(v) = *self.node(a) {
            return self.int(v as i64);
        }
        self.push(ExprNode::ToInt(a))
    }

    /// Sign extend from `from_bits` to `to_bits`, with constant folding.
    pub fn sign_extend(&mut self, a: ExprId, from_bits: u32, to_bits: u32) -> ExprId {
        if let ExprNode::IntConst(v) = *self.node(a) {
            let shift = 64 - from_bits;
            let extended = (v << shift) >> shift;
            return self.int(extended);
        }
        self.push(ExprNode::SignExtend {
            value: a,
            from_bits,
            to_bits,
        })
    }

    /// Zero extend from `from_bits` to `to_bits`, with constant folding.
    pub fn zero_extend(&mut self, a: ExprId, from_bits: u32, to_bits: u32) -> ExprId {
        if let ExprNode::IntConst(v) = *self.node(a) {
            let mask = if from_bits >= 64 {
                u64::MAX
            } else {
                (1u64 << from_bits) - 1
            };
            return self.int((v as u64 & mask) as i64);
        }
        self.push(ExprNode::ZeroExtend {
            value: a,
            from_bits,
            to_bits,
        })
    }

    /// Truncate to `to_bits`, with constant folding.
    pub fn truncate(&mut self, a: ExprId, to_bits: u32) -> ExprId {
        if let ExprNode::IntConst(v) = *self.node(a) {
            let mask = if to_bits >= 64 {
                u64::MAX
            } else {
                (1u64 << to_bits) - 1
            };
            return self.int((v as u64 & mask) as i64);
        }
        self.push(ExprNode::Truncate { value: a, to_bits })
    }

    // =================================================================
    // Query methods
    // =================================================================

    /// Try to evaluate as a concrete i64.
    pub fn as_i64(&self, id: ExprId) -> Option<i64> {
        match self.node(id) {
            ExprNode::IntConst(v) => Some(*v),
            ExprNode::FloatConst(v) => Some(*v as i64),
            ExprNode::BoolConst(b) => Some(if *b { 1 } else { 0 }),
            _ => None,
        }
    }

    /// Try to evaluate as a concrete u64.
    pub fn as_u64(&self, id: ExprId) -> Option<u64> {
        match self.node(id) {
            ExprNode::IntConst(v) => Some(*v as u64),
            ExprNode::FloatConst(v) => Some(*v as u64),
            ExprNode::BoolConst(b) => Some(if *b { 1 } else { 0 }),
            _ => None,
        }
    }

    /// Try to evaluate as a concrete f64.
    pub fn as_f64(&self, id: ExprId) -> Option<f64> {
        match self.node(id) {
            ExprNode::IntConst(v) => Some(*v as f64),
            ExprNode::FloatConst(v) => Some(*v),
            _ => None,
        }
    }

    /// Try to evaluate as a concrete bool.
    pub fn as_bool(&self, id: ExprId) -> Option<bool> {
        match self.node(id) {
            ExprNode::BoolConst(b) => Some(*b),
            ExprNode::IntConst(v) => Some(*v != 0),
            _ => None,
        }
    }

    /// Check if this is a concrete value (int, float, or bool constant).
    pub fn is_concrete(&self, id: ExprId) -> bool {
        matches!(
            self.node(id),
            ExprNode::IntConst(_) | ExprNode::FloatConst(_) | ExprNode::BoolConst(_)
        )
    }

    /// Check if this is an undefined value.
    pub fn is_undefined(&self, id: ExprId) -> bool {
        matches!(self.node(id), ExprNode::Undefined)
    }

    /// Check if this is a discarded value (from re-aggregation).
    pub fn is_discarded(&self, id: ExprId) -> bool {
        matches!(self.node(id), ExprNode::Discarded)
    }

    /// Create a discarded-value node.
    pub fn discarded(&mut self) -> ExprId {
        self.push(ExprNode::Discarded)
    }

    // =================================================================
    // Display
    // =================================================================

    /// Format an expression to the given formatter, using a stacker guard
    /// to handle deep recursion.
    pub fn fmt_expr(&self, id: ExprId, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        stacker::maybe_grow(64 * 1024, 8 * 1024 * 1024, || self.fmt_expr_inner(id, f))
    }

    fn fmt_expr_inner(&self, id: ExprId, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.node(id) {
            ExprNode::IntConst(v) => write!(f, "{}", v),
            ExprNode::FloatConst(v) => write!(f, "{}", v),
            ExprNode::BoolConst(b) => write!(f, "{}", b),
            ExprNode::Symbol(sid) => write!(f, "{}", sid),
            ExprNode::ParamSymbol(sid) => write!(f, "{}", self.string(*sid)),
            ExprNode::InputElement { array, index } => {
                write!(f, "{}[{}]", self.string(*array), index)
            }
            ExprNode::Add(a, b) => {
                write!(f, "(")?;
                self.fmt_expr(*a, f)?;
                write!(f, " + ")?;
                self.fmt_expr(*b, f)?;
                write!(f, ")")
            }
            ExprNode::Sub(a, b) => {
                write!(f, "(")?;
                self.fmt_expr(*a, f)?;
                write!(f, " - ")?;
                self.fmt_expr(*b, f)?;
                write!(f, ")")
            }
            ExprNode::Mul(a, b) => {
                write!(f, "(")?;
                self.fmt_expr(*a, f)?;
                write!(f, " * ")?;
                self.fmt_expr(*b, f)?;
                write!(f, ")")
            }
            ExprNode::Div(a, b) => {
                write!(f, "(")?;
                self.fmt_expr(*a, f)?;
                write!(f, " / ")?;
                self.fmt_expr(*b, f)?;
                write!(f, ")")
            }
            ExprNode::Rem(a, b) => {
                write!(f, "(")?;
                self.fmt_expr(*a, f)?;
                write!(f, " % ")?;
                self.fmt_expr(*b, f)?;
                write!(f, ")")
            }
            ExprNode::Neg(a) => {
                write!(f, "(-")?;
                self.fmt_expr(*a, f)?;
                write!(f, ")")
            }
            ExprNode::Exp(a) => {
                write!(f, "exp(")?;
                self.fmt_expr(*a, f)?;
                write!(f, ")")
            }
            ExprNode::Log(a) => {
                write!(f, "log(")?;
                self.fmt_expr(*a, f)?;
                write!(f, ")")
            }
            ExprNode::Sqrt(a) => {
                write!(f, "sqrt(")?;
                self.fmt_expr(*a, f)?;
                write!(f, ")")
            }
            ExprNode::Rcp(a) => {
                write!(f, "rcp(")?;
                self.fmt_expr(*a, f)?;
                write!(f, ")")
            }
            ExprNode::Max(a, b) => {
                write!(f, "max(")?;
                self.fmt_expr(*a, f)?;
                write!(f, ", ")?;
                self.fmt_expr(*b, f)?;
                write!(f, ")")
            }
            ExprNode::Min(a, b) => {
                write!(f, "min(")?;
                self.fmt_expr(*a, f)?;
                write!(f, ", ")?;
                self.fmt_expr(*b, f)?;
                write!(f, ")")
            }
            ExprNode::Abs(a) => {
                write!(f, "abs(")?;
                self.fmt_expr(*a, f)?;
                write!(f, ")")
            }
            ExprNode::BitAnd(a, b) => {
                write!(f, "(")?;
                self.fmt_expr(*a, f)?;
                write!(f, " & ")?;
                self.fmt_expr(*b, f)?;
                write!(f, ")")
            }
            ExprNode::BitOr(a, b) => {
                write!(f, "(")?;
                self.fmt_expr(*a, f)?;
                write!(f, " | ")?;
                self.fmt_expr(*b, f)?;
                write!(f, ")")
            }
            ExprNode::BitXor(a, b) => {
                write!(f, "(")?;
                self.fmt_expr(*a, f)?;
                write!(f, " ^ ")?;
                self.fmt_expr(*b, f)?;
                write!(f, ")")
            }
            ExprNode::BitNot(a) => {
                write!(f, "(~")?;
                self.fmt_expr(*a, f)?;
                write!(f, ")")
            }
            ExprNode::Shl(a, b) => {
                write!(f, "(")?;
                self.fmt_expr(*a, f)?;
                write!(f, " << ")?;
                self.fmt_expr(*b, f)?;
                write!(f, ")")
            }
            ExprNode::Shr(a, b) => {
                write!(f, "(")?;
                self.fmt_expr(*a, f)?;
                write!(f, " >> ")?;
                self.fmt_expr(*b, f)?;
                write!(f, ")")
            }
            ExprNode::LShr(a, b) => {
                write!(f, "(")?;
                self.fmt_expr(*a, f)?;
                write!(f, " >>> ")?;
                self.fmt_expr(*b, f)?;
                write!(f, ")")
            }
            ExprNode::Eq(a, b) => {
                write!(f, "(")?;
                self.fmt_expr(*a, f)?;
                write!(f, " == ")?;
                self.fmt_expr(*b, f)?;
                write!(f, ")")
            }
            ExprNode::Ne(a, b) => {
                write!(f, "(")?;
                self.fmt_expr(*a, f)?;
                write!(f, " != ")?;
                self.fmt_expr(*b, f)?;
                write!(f, ")")
            }
            ExprNode::Lt(a, b) => {
                write!(f, "(")?;
                self.fmt_expr(*a, f)?;
                write!(f, " < ")?;
                self.fmt_expr(*b, f)?;
                write!(f, ")")
            }
            ExprNode::Le(a, b) => {
                write!(f, "(")?;
                self.fmt_expr(*a, f)?;
                write!(f, " <= ")?;
                self.fmt_expr(*b, f)?;
                write!(f, ")")
            }
            ExprNode::Gt(a, b) => {
                write!(f, "(")?;
                self.fmt_expr(*a, f)?;
                write!(f, " > ")?;
                self.fmt_expr(*b, f)?;
                write!(f, ")")
            }
            ExprNode::Ge(a, b) => {
                write!(f, "(")?;
                self.fmt_expr(*a, f)?;
                write!(f, " >= ")?;
                self.fmt_expr(*b, f)?;
                write!(f, ")")
            }
            ExprNode::And(a, b) => {
                write!(f, "(")?;
                self.fmt_expr(*a, f)?;
                write!(f, " && ")?;
                self.fmt_expr(*b, f)?;
                write!(f, ")")
            }
            ExprNode::Or(a, b) => {
                write!(f, "(")?;
                self.fmt_expr(*a, f)?;
                write!(f, " || ")?;
                self.fmt_expr(*b, f)?;
                write!(f, ")")
            }
            ExprNode::Not(a) => {
                write!(f, "(!")?;
                self.fmt_expr(*a, f)?;
                write!(f, ")")
            }
            ExprNode::Select(c, t, e) => {
                write!(f, "(")?;
                self.fmt_expr(*c, f)?;
                write!(f, " ? ")?;
                self.fmt_expr(*t, f)?;
                write!(f, " : ")?;
                self.fmt_expr(*e, f)?;
                write!(f, ")")
            }
            ExprNode::ToFloat(a) => {
                write!(f, "float(")?;
                self.fmt_expr(*a, f)?;
                write!(f, ")")
            }
            ExprNode::ToInt(a) => {
                write!(f, "int(")?;
                self.fmt_expr(*a, f)?;
                write!(f, ")")
            }
            ExprNode::SignExtend {
                value,
                from_bits,
                to_bits,
            } => {
                write!(f, "sext{}to{}(", from_bits, to_bits)?;
                self.fmt_expr(*value, f)?;
                write!(f, ")")
            }
            ExprNode::ZeroExtend {
                value,
                from_bits,
                to_bits,
            } => {
                write!(f, "zext{}to{}(", from_bits, to_bits)?;
                self.fmt_expr(*value, f)?;
                write!(f, ")")
            }
            ExprNode::Truncate { value, to_bits } => {
                write!(f, "trunc{}(", to_bits)?;
                self.fmt_expr(*value, f)?;
                write!(f, ")")
            }
            ExprNode::Fma(a, b, c) => {
                write!(f, "fma(")?;
                self.fmt_expr(*a, f)?;
                write!(f, ", ")?;
                self.fmt_expr(*b, f)?;
                write!(f, ", ")?;
                self.fmt_expr(*c, f)?;
                write!(f, ")")
            }
            ExprNode::SymbolicRead { array, index } => {
                write!(f, "{}[", self.string(*array))?;
                self.fmt_expr(*index, f)?;
                write!(f, "]")
            }
            ExprNode::Discarded => write!(f, "discarded"),
            ExprNode::Undefined => write!(f, "undefined"),
        }
    }

    /// Convenience method: format an expression to a `String`.
    pub fn display_expr(&self, id: ExprId) -> String {
        use std::fmt::Write;
        let mut buf = String::new();
        // We use a wrapper that implements Display so we can use write!
        struct ExprDisplay<'a> {
            arena: &'a ExprArena,
            id: ExprId,
        }
        impl<'a> fmt::Display for ExprDisplay<'a> {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                self.arena.fmt_expr(self.id, f)
            }
        }
        write!(buf, "{}", ExprDisplay { arena: self, id })
            .expect("formatting an ExprId to String should not fail");
        buf
    }
}

impl Default for ExprArena {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for ExprArena {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ExprArena")
            .field("num_nodes", &self.nodes.count().to_value())
            .field("num_strings", &self.strings.count().to_value())
            .finish()
    }
}

impl Clone for ExprArena {
    fn clone(&self) -> Self {
        Self {
            nodes: self.nodes.clone(),
            strings: self.strings.clone(),
        }
    }
}

/// Check structural equality of two expressions across different arenas.
///
/// Walks both expression trees in lockstep, comparing node variants and
/// leaf values. This is the cross-arena equivalent of the old `Expr: PartialEq`.
pub fn structurally_equal(a_arena: &ExprArena, a: ExprId, b_arena: &ExprArena, b: ExprId) -> bool {
    stacker::maybe_grow(64 * 1024, 8 * 1024 * 1024, || {
        structurally_equal_inner(a_arena, a, b_arena, b)
    })
}

fn structurally_equal_inner(
    a_arena: &ExprArena,
    a: ExprId,
    b_arena: &ExprArena,
    b: ExprId,
) -> bool {
    use ExprNode::*;
    match (a_arena.node(a), b_arena.node(b)) {
        (IntConst(x), IntConst(y)) => x == y,
        (FloatConst(x), FloatConst(y)) => x == y,
        (BoolConst(x), BoolConst(y)) => x == y,
        (Symbol(x), Symbol(y)) => x == y,
        (ParamSymbol(x), ParamSymbol(y)) => a_arena.string(*x) == b_arena.string(*y),
        (
            InputElement {
                array: xa,
                index: xi,
            },
            InputElement {
                array: ya,
                index: yi,
            },
        ) => xi == yi && a_arena.string(*xa) == b_arena.string(*ya),
        (Undefined, Undefined) => true,

        // Binary ops
        (Add(a1, a2), Add(b1, b2))
        | (Sub(a1, a2), Sub(b1, b2))
        | (Mul(a1, a2), Mul(b1, b2))
        | (Div(a1, a2), Div(b1, b2))
        | (Rem(a1, a2), Rem(b1, b2))
        | (Max(a1, a2), Max(b1, b2))
        | (Min(a1, a2), Min(b1, b2))
        | (BitAnd(a1, a2), BitAnd(b1, b2))
        | (BitOr(a1, a2), BitOr(b1, b2))
        | (BitXor(a1, a2), BitXor(b1, b2))
        | (Shl(a1, a2), Shl(b1, b2))
        | (Shr(a1, a2), Shr(b1, b2))
        | (LShr(a1, a2), LShr(b1, b2))
        | (Eq(a1, a2), Eq(b1, b2))
        | (Ne(a1, a2), Ne(b1, b2))
        | (Lt(a1, a2), Lt(b1, b2))
        | (Le(a1, a2), Le(b1, b2))
        | (Gt(a1, a2), Gt(b1, b2))
        | (Ge(a1, a2), Ge(b1, b2))
        | (And(a1, a2), And(b1, b2))
        | (Or(a1, a2), Or(b1, b2)) => {
            structurally_equal(a_arena, *a1, b_arena, *b1)
                && structurally_equal(a_arena, *a2, b_arena, *b2)
        }

        // Unary ops
        (Neg(a1), Neg(b1))
        | (Exp(a1), Exp(b1))
        | (Log(a1), Log(b1))
        | (Sqrt(a1), Sqrt(b1))
        | (Rcp(a1), Rcp(b1))
        | (Abs(a1), Abs(b1))
        | (BitNot(a1), BitNot(b1))
        | (Not(a1), Not(b1))
        | (ToFloat(a1), ToFloat(b1))
        | (ToInt(a1), ToInt(b1)) => structurally_equal(a_arena, *a1, b_arena, *b1),

        // Ternary ops
        (Fma(a1, a2, a3), Fma(b1, b2, b3)) | (Select(a1, a2, a3), Select(b1, b2, b3)) => {
            structurally_equal(a_arena, *a1, b_arena, *b1)
                && structurally_equal(a_arena, *a2, b_arena, *b2)
                && structurally_equal(a_arena, *a3, b_arena, *b3)
        }

        // Conversions with metadata
        (
            SignExtend {
                value: v1,
                from_bits: f1,
                to_bits: t1,
            },
            SignExtend {
                value: v2,
                from_bits: f2,
                to_bits: t2,
            },
        )
        | (
            ZeroExtend {
                value: v1,
                from_bits: f1,
                to_bits: t1,
            },
            ZeroExtend {
                value: v2,
                from_bits: f2,
                to_bits: t2,
            },
        ) => f1 == f2 && t1 == t2 && structurally_equal(a_arena, *v1, b_arena, *v2),

        (
            Truncate {
                value: v1,
                to_bits: t1,
            },
            Truncate {
                value: v2,
                to_bits: t2,
            },
        ) => t1 == t2 && structurally_equal(a_arena, *v1, b_arena, *v2),

        // Discarded values are structurally equal to each other
        (Discarded, Discarded) => true,

        // Symbolic array read
        (
            SymbolicRead {
                array: a1,
                index: i1,
            },
            SymbolicRead {
                array: a2,
                index: i2,
            },
        ) => {
            a_arena.string(*a1) == b_arena.string(*a2)
                && structurally_equal(a_arena, *i1, b_arena, *i2)
        }

        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constant_folding() {
        let mut arena = ExprArena::new();

        // Integer arithmetic
        let a = arena.int(3);
        let b = arena.int(4);
        let e = arena.add(a, b);
        assert_eq!(arena.as_i64(e), Some(7));

        let a = arena.int(10);
        let b = arena.int(3);
        let e = arena.sub(a, b);
        assert_eq!(arena.as_i64(e), Some(7));

        let a = arena.int(3);
        let b = arena.int(4);
        let e = arena.mul(a, b);
        assert_eq!(arena.as_i64(e), Some(12));

        // Shifts
        let a = arena.int(1);
        let b = arena.int(4);
        let e = arena.shl(a, b);
        assert_eq!(arena.as_i64(e), Some(16));

        // Bitwise
        let a = arena.int(0xFF);
        let b = arena.int(0x0F);
        let e = arena.bit_and(a, b);
        assert_eq!(arena.as_i64(e), Some(0x0F));
    }

    #[test]
    fn test_identity_simplification() {
        let mut arena = ExprArena::new();

        // x + 0 = x
        let x = arena.symbol();
        let zero = arena.int(0);
        let e = arena.add(x, zero);
        assert_eq!(e, x);

        // 0 + x = x
        let x = arena.symbol();
        let zero = arena.int(0);
        let e = arena.add(zero, x);
        assert_eq!(e, x);

        // x * 1 = x
        let x = arena.symbol();
        let one = arena.int(1);
        let e = arena.mul(x, one);
        assert_eq!(e, x);

        // 1 * x = x
        let x = arena.symbol();
        let one = arena.int(1);
        let e = arena.mul(one, x);
        assert_eq!(e, x);

        // x - 0 = x
        let x = arena.symbol();
        let zero = arena.int(0);
        let e = arena.sub(x, zero);
        assert_eq!(e, x);

        // x / 1 = x
        let x = arena.symbol();
        let one = arena.int(1);
        let e = arena.div(x, one);
        assert_eq!(e, x);
    }

    #[test]
    fn test_logical_shift_right() {
        let mut arena = ExprArena::new();

        // 128 >>> 1 = 64
        let a = arena.int(128);
        let b = arena.int(1);
        let e = arena.lshr(a, b);
        assert_eq!(arena.as_i64(e), Some(64));

        // -1 >>> 1 should be a large positive number (logical shift)
        let a = arena.int(-1);
        let b = arena.int(1);
        let e = arena.lshr(a, b);
        assert_eq!(arena.as_i64(e), Some(i64::MAX));

        // Nested: (128 >>> 1) << 2 = 256
        let a = arena.int(128);
        let b = arena.int(1);
        let c = arena.int(2);
        let step1 = arena.lshr(a, b);
        let e = arena.shl(step1, c);
        assert_eq!(arena.as_i64(e), Some(256));
    }

    #[test]
    fn test_sign_extend() {
        let mut arena = ExprArena::new();

        // Sign extend 8-bit -1 (0xFF) to 64 bits
        let v = arena.int(0xFF);
        let e = arena.sign_extend(v, 8, 64);
        assert_eq!(arena.as_i64(e), Some(-1));

        // Sign extend 8-bit 127 (0x7F) to 64 bits - stays positive
        let v = arena.int(0x7F);
        let e = arena.sign_extend(v, 8, 64);
        assert_eq!(arena.as_i64(e), Some(127));

        // Sign extend 32-bit 0 to 64 bits
        let v = arena.int(0);
        let e = arena.sign_extend(v, 32, 64);
        assert_eq!(arena.as_i64(e), Some(0));
    }

    #[test]
    fn test_zero_extend() {
        let mut arena = ExprArena::new();

        // Zero extend 8-bit 0xFF to 64 bits - stays 255
        let v = arena.int(0xFF);
        let e = arena.zero_extend(v, 8, 64);
        assert_eq!(arena.as_i64(e), Some(255));
    }

    #[test]
    fn test_truncate() {
        let mut arena = ExprArena::new();

        // Truncate 0x1234 to 8 bits = 0x34
        let v = arena.int(0x1234);
        let e = arena.truncate(v, 8);
        assert_eq!(arena.as_i64(e), Some(0x34));
    }

    #[test]
    fn test_div() {
        let mut arena = ExprArena::new();

        // 10 / 3 = 3 (eager folding in constructor)
        let a = arena.int(10);
        let b = arena.int(3);
        let e = arena.div(a, b);
        assert_eq!(arena.as_i64(e), Some(3));

        // 10.0 / 2.0 = 5.0
        let a = arena.float(10.0);
        let b = arena.float(2.0);
        let e = arena.div(a, b);
        assert_eq!(arena.as_f64(e), Some(5.0));
    }

    #[test]
    fn test_node_lookup() {
        let mut arena = ExprArena::new();

        let a = arena.int(42);
        assert_eq!(arena.node(a), &ExprNode::IntConst(42));

        let b = arena.float(2.5);
        assert!(matches!(arena.node(b), ExprNode::FloatConst(v) if (*v - 2.5).abs() < 1e-10));

        let c = arena.bool_val(true);
        assert_eq!(arena.node(c), &ExprNode::BoolConst(true));

        let d = arena.param_symbol("alpha");
        if let ExprNode::ParamSymbol(sid) = arena.node(d) {
            assert_eq!(arena.string(*sid), "alpha");
        } else {
            panic!("expected ParamSymbol");
        }

        let sid = arena.intern_string("input");
        let e = arena.input_element(sid, 0);
        assert!(matches!(
            arena.node(e),
            ExprNode::InputElement { array, index: 0 } if arena.string(*array) == "input"
        ));

        let u = arena.undefined();
        assert_eq!(arena.node(u), &ExprNode::Undefined);
    }

    #[test]
    fn test_query_methods() {
        let mut arena = ExprArena::new();

        let i = arena.int(10);
        assert_eq!(arena.as_i64(i), Some(10));
        assert_eq!(arena.as_u64(i), Some(10));
        assert_eq!(arena.as_f64(i), Some(10.0));
        assert_eq!(arena.as_bool(i), Some(true));
        assert!(arena.is_concrete(i));
        assert!(!arena.is_undefined(i));

        let f = arena.float(2.5);
        assert_eq!(arena.as_f64(f), Some(2.5));
        assert!(arena.is_concrete(f));

        let b = arena.bool_val(false);
        assert_eq!(arena.as_bool(b), Some(false));
        assert!(arena.is_concrete(b));

        let s = arena.symbol();
        assert!(!arena.is_concrete(s));
        assert!(arena.as_i64(s).is_none());

        let u = arena.undefined();
        assert!(arena.is_undefined(u));
        assert!(!arena.is_concrete(u));
    }

    #[test]
    fn test_select_folding() {
        let mut arena = ExprArena::new();

        let t = arena.bool_val(true);
        let a = arena.int(1);
        let b = arena.int(2);
        let e = arena.select(t, a, b);
        assert_eq!(e, a);

        let f = arena.bool_val(false);
        let c = arena.int(3);
        let d = arena.int(4);
        let e = arena.select(f, c, d);
        assert_eq!(e, d);
    }

    #[test]
    fn test_display() {
        let mut arena = ExprArena::new();

        let a = arena.int(3);
        let b = arena.int(4);
        let e = arena.add(a, b);
        // Constant folding means this is just "7"
        assert_eq!(arena.display_expr(e), "7");

        // Build a symbolic expression: (x + 1)
        let x = arena.symbol();
        let one = arena.int(1);
        let e = arena.add(x, one);
        let s = arena.display_expr(e);
        // Should contain " + 1" and parentheses
        assert!(s.contains(" + "));
        assert!(s.contains("1"));
    }

    /// `validate` accepts every arena the constructors can build...
    #[test]
    fn validate_accepts_constructed_arenas() {
        let mut ar = ExprArena::new();
        let x = ar.param_symbol("x");
        let y = ar.symbol();
        let s = ar.add(x, y);
        let m = ar.max(s, x);
        let _ = ar.select(m, s, x);
        assert_eq!(ar.validate(), Ok(()));
    }

    /// ...and rejects forward/self references and out-of-range string
    /// ids, which only a corrupt or crafted dump can contain. A self
    /// reference like `Add(0, 0)` at node 0 previously passed the
    /// bounds-only check and sent consumers into unbounded recursion.
    #[test]
    fn validate_rejects_malformed_arenas() {
        use id_collections::Id;

        let cyclic = ExprArena::from_raw_parts_for_test(
            vec![ExprNode::Add(Id::from_index(0), Id::from_index(0))],
            vec![],
        );
        assert!(cyclic.validate().is_err());

        let forward = ExprArena::from_raw_parts_for_test(
            vec![ExprNode::Neg(Id::from_index(1)), ExprNode::IntConst(1)],
            vec![],
        );
        assert!(forward.validate().is_err());

        let oob_string = ExprArena::from_raw_parts_for_test(
            vec![ExprNode::ParamSymbol(Id::from_index(3))],
            vec!["x".to_string()],
        );
        assert!(oob_string.validate().is_err());
    }
}

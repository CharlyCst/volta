//! High-level analysis driver: parse tree in, per-output-element symbolic
//! expressions (or a race/deadlock/structured-CTA error) out.

use std::fmt;
use std::time::{Duration, Instant};

use volta_frontend::ast::{Function, Module, TopLevelItem, VarDecl};

use crate::equiv::{DEFAULT_RECYCLE_TERMS, EquivError, EquivSession};
use crate::eval::{AnalysisConfig, AnalysisOutput, EvalError, Interpreter, Stats};
use crate::logging::info;
use crate::lower_error::LowerError;
use crate::lowering::lower_function;
use crate::numeric;
use crate::symbolic::{ExprArena, ExprId};

/// Errors from the end-to-end analysis of one kernel.
#[derive(Debug)]
pub enum AnalysisError {
    KernelNotFound { name: Option<String> },
    Lower(LowerError),
    Eval(EvalError),
}

impl fmt::Display for AnalysisError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::KernelNotFound { name: Some(name) } => {
                write!(f, "no kernel named '{}' in module", name)
            }
            Self::KernelNotFound { name: None } => write!(f, "no kernel entry in module"),
            Self::Lower(e) => write!(f, "lowering failed: {}", e),
            Self::Eval(e) => write!(f, "{}", e),
        }
    }
}

impl std::error::Error for AnalysisError {}

impl From<LowerError> for AnalysisError {
    fn from(e: LowerError) -> Self {
        Self::Lower(e)
    }
}

impl From<EvalError> for AnalysisError {
    fn from(e: EvalError) -> Self {
        Self::Eval(e)
    }
}

/// Find a kernel entry point by name, or the unique entry if `name` is None.
pub fn find_kernel<'m>(
    module: &'m Module,
    name: Option<&str>,
) -> Result<&'m Function, AnalysisError> {
    let mut entries = module.items.iter().filter_map(|item| match item {
        TopLevelItem::Entry(f) => Some(f),
        _ => None,
    });
    match name {
        Some(name) => {
            entries
                .find(|f| f.name.to_string() == name)
                .ok_or(AnalysisError::KernelNotFound {
                    name: Some(name.to_string()),
                })
        }
        None => entries
            .next()
            .ok_or(AnalysisError::KernelNotFound { name: None }),
    }
}

/// Module-level variable declarations (extern shared memory, module globals).
pub fn module_vars(module: &Module) -> Vec<VarDecl> {
    module
        .items
        .iter()
        .filter_map(|item| match item {
            TopLevelItem::Variable(v) => Some(v.clone()),
            _ => None,
        })
        .collect()
}

/// Analyze one kernel: lower it and symbolically execute all threads of
/// CTA (0,0,0) under the given configuration.
pub fn analyze_kernel(
    module: &Module,
    kernel: Option<&str>,
    config: AnalysisConfig,
) -> Result<AnalysisOutput, AnalysisError> {
    let func = find_kernel(module, kernel)?;
    let vars = module_vars(module);
    let program = lower_function(func, &vars)?;
    info!(
        "analyzing kernel {:?}: block={:?} grid={:?}",
        kernel, config.block_dim, config.grid_dim
    );
    let mut interp = Interpreter::new(&program, config)?;
    interp.run()?;
    Ok(interp.into_output()?)
}

/// A single output element where the two kernels disagree.
#[derive(Debug, Clone)]
pub struct Mismatch {
    pub array: String,
    pub index: u64,
}

/// Result of comparing two analysis outputs.
#[derive(Debug)]
pub enum EquivOutcome {
    Equivalent,
    NotEquivalent { mismatches: Vec<Mismatch> },
}

/// Errors from output comparison.
#[derive(Debug)]
pub enum EquivCheckError {
    /// The two outputs have different arrays or element counts.
    ShapeMismatch { message: String },
    /// The underlying symbolic check failed.
    Equiv(EquivError),
    /// The f64 oracle contradicted (or could not confirm) a verdict.
    Numeric { message: String },
}

impl fmt::Display for EquivCheckError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ShapeMismatch { message } => write!(f, "output shape mismatch: {}", message),
            Self::Equiv(e) => write!(f, "equivalence check failed: {}", e),
            Self::Numeric { message } => write!(f, "numeric oracle: {}", message),
        }
    }
}

impl std::error::Error for EquivCheckError {}

impl From<EquivError> for EquivCheckError {
    fn from(e: EquivError) -> Self {
        Self::Equiv(e)
    }
}

/// Options for [`check_output_equivalence_with`].
#[derive(Debug, Clone)]
pub struct EquivCheckOptions {
    /// Check at most this many common elements per array (0 = all).
    pub sample: u64,
    /// Confirm every verdict with the f64 numeric oracle.
    pub verify_numeric: bool,
    /// Recycle the VC intern tables past this many interned terms
    /// (0 = never); see `EquivSession::with_recycle_terms`.
    pub recycle_terms: usize,
}

impl Default for EquivCheckOptions {
    fn default() -> Self {
        Self {
            sample: 0,
            verify_numeric: false,
            recycle_terms: DEFAULT_RECYCLE_TERMS,
        }
    }
}

/// The outcome of a comparison plus how much of the footprint it covered.
#[derive(Debug)]
pub struct EquivCheckReport {
    pub outcome: EquivOutcome,
    /// Elements actually compared (less than total when sampling).
    pub elements_checked: u64,
    /// Comparable elements in the reference footprints.
    pub elements_total: u64,
    /// Time spent in the decision procedure itself: the summed
    /// `EquivSession::check` calls, and nothing else. VC pairing and the
    /// optional numeric-oracle verification are excluded, so this is the
    /// number to put beside another backend's solver time (the paper's
    /// tables) - it does not move when `verify_numeric` is toggled.
    pub check_time: Duration,
}

/// Pair up the two runs' written elements for each array the caller
/// names: both runs must have written every named array with an
/// identical index set, element for element (arrays the caller does not
/// name are not compared - e.g. auxiliary exports like FlashAttention's
/// softmax `l`/`m` statistics that only the optimized kernel computes).
/// The list must be nonempty: checking nothing is an error, not a
/// vacuous pass. Shared by `check_output_equivalence_with` (the decision
/// procedure) and any other backend (e.g. `volta_z3`) that needs the
/// exact same element correspondence to be a fair comparison.
pub fn paired_elements(
    reference: &AnalysisOutput,
    optimized: &AnalysisOutput,
    arrays: &[String],
) -> Result<Vec<(String, Vec<(u64, ExprId, ExprId)>)>, EquivCheckError> {
    if arrays.is_empty() {
        return Err(EquivCheckError::ShapeMismatch {
            message: "no arrays specified to check".to_string(),
        });
    }
    let mut result = Vec::with_capacity(arrays.len());
    for name in arrays {
        let Some((_, ref_elems)) = reference.outputs.iter().find(|(n, _)| n == name) else {
            return Err(EquivCheckError::ShapeMismatch {
                message: format!("reference run has no output array '{}'", name),
            });
        };
        let Some((_, opt_elems)) = optimized.outputs.iter().find(|(n, _)| n == name) else {
            return Err(EquivCheckError::ShapeMismatch {
                message: format!("optimized run has no output array '{}'", name),
            });
        };

        if ref_elems.len() != opt_elems.len() {
            return Err(EquivCheckError::ShapeMismatch {
                message: format!(
                    "array '{}': {} elements written vs {}",
                    name,
                    ref_elems.len(),
                    opt_elems.len()
                ),
            });
        }
        let mut common = Vec::with_capacity(ref_elems.len());
        for (&(ri, r), &(oi, o)) in ref_elems.iter().zip(opt_elems.iter()) {
            if ri != oi {
                return Err(EquivCheckError::ShapeMismatch {
                    message: format!(
                        "array '{}': written footprints differ (element {} vs {})",
                        name, ri, oi
                    ),
                });
            }
            common.push((ri, r, o));
        }
        result.push((name.clone(), common));
    }
    Ok(result)
}

/// Check two analysis outputs element by element under `options`. One
/// `EquivSession` is shared across all elements: structure shared between
/// elements (and between the two kernels) canonicalizes once.
pub fn check_output_equivalence_with(
    reference: &AnalysisOutput,
    optimized: &AnalysisOutput,
    arrays: &[String],
    options: &EquivCheckOptions,
) -> Result<EquivCheckReport, EquivCheckError> {
    let paired = paired_elements(reference, optimized, arrays)?;

    let mut session =
        EquivSession::with_recycle_terms(&reference.arena, &optimized.arena, options.recycle_terms);
    let mut mismatches = Vec::new();
    let mut elements_checked = 0u64;
    let mut elements_total = 0u64;
    let mut check_time = Duration::ZERO;

    for (name, common) in &paired {
        elements_total += common.len() as u64;
        let limit = match options.sample {
            0 => common.len(),
            n => common.len().min(n as usize),
        };
        for &(index, r, o) in common.iter().take(limit) {
            let check_start = Instant::now();
            let equivalent = session.check(r, o)?;
            check_time += check_start.elapsed();
            if options.verify_numeric {
                numeric::verify_verdict(&reference.arena, r, &optimized.arena, o, equivalent)
                    .map_err(|message| EquivCheckError::Numeric {
                        message: format!("array '{}' element {}: {}", name, index, message),
                    })?;
            }
            if !equivalent {
                mismatches.push(Mismatch {
                    array: name.clone(),
                    index,
                });
            }
            elements_checked += 1;
        }
    }

    let outcome = if mismatches.is_empty() {
        EquivOutcome::Equivalent
    } else {
        EquivOutcome::NotEquivalent { mismatches }
    };
    Ok(EquivCheckReport {
        outcome,
        elements_checked,
        elements_total,
        check_time,
    })
}

/// A snapshot of one kernel's verification conditions: the expression arena
/// plus the output footprint (index -> root `ExprId`). This is exactly what
/// `check_output_equivalence_with` needs and nothing else - `Stats`/op-counts
/// describe the symbolic-execution run that produced it, not the VCs
/// themselves, so they aren't part of the snapshot.
#[derive(serde::Serialize, serde::Deserialize)]
pub struct VcSnapshot {
    pub arena: ExprArena,
    pub outputs: Vec<(String, Vec<(u64, ExprId)>)>,
}

impl VcSnapshot {
    pub fn from_output(output: AnalysisOutput) -> Self {
        Self {
            arena: output.arena,
            outputs: output.outputs,
        }
    }

    /// Rehydrate into the shape `check_output_equivalence_with` accepts.
    /// `stats`/`op_counts` are empty: they belong to a symbolic-execution
    /// run, and a dump has none to report.
    pub fn into_analysis_output(self) -> AnalysisOutput {
        AnalysisOutput {
            arena: self.arena,
            outputs: self.outputs,
            stats: Stats::default(),
            op_counts: std::collections::BTreeMap::new(),
        }
    }

    /// Check that every id in the snapshot points inside its arena. Run
    /// this on snapshots rebuilt from external data (a dump file): a
    /// corrupt or version-skewed file that still decodes would otherwise
    /// panic with an index-out-of-bounds deep inside the equivalence check.
    pub fn validate(&self) -> Result<(), String> {
        self.arena.validate()?;
        let n_nodes = self.arena.node_count();
        for (name, elems) in &self.outputs {
            for &(index, root) in elems {
                if id_collections::Id::to_index(root) as usize >= n_nodes {
                    return Err(format!(
                        "output '{}' element {} references expression {} but the arena has {} nodes",
                        name,
                        index,
                        id_collections::Id::to_index(root),
                        n_nodes
                    ));
                }
            }
        }
        Ok(())
    }
}

/// The reference and optimized kernels' verification conditions, as
/// persisted by `volta compare --dump-vcs` and reloaded by `--from-dump`.
#[derive(serde::Serialize, serde::Deserialize)]
pub struct VcDump {
    pub reference: VcSnapshot,
    pub optimized: VcSnapshot,
}

impl VcDump {
    /// Validate both snapshots (see `VcSnapshot::validate`).
    pub fn validate(&self) -> Result<(), String> {
        self.reference
            .validate()
            .map_err(|e| format!("reference: {}", e))?;
        self.optimized
            .validate()
            .map_err(|e| format!("optimized: {}", e))
    }
}

/// Write a per-instruction-kind execution profile, most-executed first.
/// The one formatter for `AnalysisOutput::op_counts`, shared by `volta`
/// and `volta-bench` so their profile tables cannot drift.
pub fn write_op_counts(
    out: &mut dyn std::io::Write,
    label: &str,
    counts: &std::collections::BTreeMap<&'static str, u64>,
) -> std::io::Result<()> {
    if counts.is_empty() {
        return Ok(());
    }
    let total: u64 = counts.values().sum();
    let mut entries: Vec<_> = counts.iter().collect();
    entries.sort_by(|a, b| b.1.cmp(a.1).then(a.0.cmp(b.0)));
    writeln!(out, "{} profile:", label)?;
    for (kind, count) in entries {
        let pct = 100.0 * *count as f64 / total as f64;
        writeln!(out, "  {:<16} {:>10}  ({:>5.1}%)", kind, count, pct)?;
    }
    Ok(())
}

/// Check that two analysis outputs agree on every element of every named
/// array under the default options: all elements checked, no numeric
/// oracle.
pub fn check_output_equivalence(
    reference: &AnalysisOutput,
    optimized: &AnalysisOutput,
    arrays: &[String],
) -> Result<EquivOutcome, EquivCheckError> {
    check_output_equivalence_with(reference, optimized, arrays, &EquivCheckOptions::default())
        .map(|report| report.outcome)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval::Stats;

    fn output_with(arrays: &[(&str, &[u64])]) -> AnalysisOutput {
        let mut arena = ExprArena::new();
        let outputs = arrays
            .iter()
            .map(|(name, indices)| {
                let elems = indices
                    .iter()
                    .map(|&i| {
                        let sid = arena.intern_string(*name);
                        (i, arena.input_element(sid, i))
                    })
                    .collect();
                (name.to_string(), elems)
            })
            .collect();
        AnalysisOutput {
            arena,
            outputs,
            stats: Stats::default(),
            op_counts: std::collections::BTreeMap::new(),
        }
    }

    fn names(names: &[&str]) -> Vec<String> {
        names.iter().map(|n| n.to_string()).collect()
    }

    /// The caller's array list is the spec: named arrays must exist on
    /// BOTH sides, unnamed arrays on either side are ignored, and naming
    /// nothing is an error rather than a vacuous pass.
    #[test]
    fn paired_elements_follows_the_callers_list() {
        let reference = output_with(&[("out", &[0, 1])]);
        let optimized = output_with(&[("out", &[0, 1]), ("aux", &[0])]);

        // Unnamed optimized-only "aux" is ignored.
        let paired = paired_elements(&reference, &optimized, &names(&["out"])).unwrap();
        assert_eq!(paired.len(), 1);
        assert_eq!(paired[0].1.len(), 2);

        // Naming an array absent from either side is an error.
        assert!(paired_elements(&reference, &optimized, &names(&["aux"])).is_err());
        assert!(paired_elements(&optimized, &reference, &names(&["out", "aux"])).is_err());
        assert!(paired_elements(&reference, &optimized, &names(&["missing"])).is_err());

        // An empty list is an error, not a vacuous pass.
        assert!(paired_elements(&reference, &optimized, &[]).is_err());

        // Differing footprints for a named array are an error.
        let narrower = output_with(&[("out", &[0])]);
        assert!(paired_elements(&reference, &narrower, &names(&["out"])).is_err());
    }
}

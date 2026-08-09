//! Analysis configuration: launch dimensions, parameter values, and the
//! input/output arrays that define the kernel's memory interface.

/// Value bound to a kernel parameter, positional by `ParamId` order.
#[derive(Debug, Clone)]
pub enum ParamValue {
    /// Concrete integer (also used for raw pointer values)
    Int(i64),
    /// Concrete float
    Float(f64),
    /// Symbolic float input with the given name (e.g. "alpha")
    SymFloat(String),
    /// Pointer to a named array from `AnalysisConfig::arrays`
    ArrayPtr(String),
}

/// Whether an array is a kernel input (pre-initialized with fresh symbols),
/// an output (extracted after execution), or both.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArrayKind {
    Input,
    Output,
    InputOutput,
    /// An input of concrete integers where element i holds the value i
    /// (identity mapping). For index arrays that feed addressing, e.g.
    /// OpenMM's `posq[particles[i]]`: symbolic elements would make the
    /// derived addresses symbolic and violate structured-CTA.
    IndexInput,
}

impl ArrayKind {
    pub fn is_input(self) -> bool {
        matches!(self, Self::Input | Self::InputOutput | Self::IndexInput)
    }

    pub fn is_output(self) -> bool {
        matches!(self, Self::Output | Self::InputOutput)
    }
}

/// A global-memory array visible to the kernel.
///
/// Input arrays are pre-populated with named symbols `name[0]`, `name[1]`,
/// ... at granule width `elem_width`. Bases must not fall in the reserved
/// module-global region (see `symbols::MODULE_GLOBAL_BASE`).
#[derive(Debug, Clone)]
pub struct ArrayDef {
    pub name: String,
    pub base: u64,
    /// Element width in bytes (2 for f16, 4 for f32/int, 8 for f64)
    pub elem_width: u64,
    /// Number of elements
    pub len: u64,
    pub kind: ArrayKind,
}

impl ArrayDef {
    pub fn size_bytes(&self) -> u64 {
        self.elem_width * self.len
    }
}

/// Full configuration for analyzing one kernel.
#[derive(Debug, Clone)]
pub struct AnalysisConfig {
    /// Threads per block; the CTA under analysis is block (0,0,0)
    pub block_dim: (u32, u32, u32),
    /// Grid dimensions (only used for `%nctaid`)
    pub grid_dim: (u32, u32, u32),
    /// Parameter values in `ParamId` (declaration) order
    pub params: Vec<ParamValue>,
    /// Global-memory arrays
    pub arrays: Vec<ArrayDef>,
    /// Concrete values for module-scope `.global` variables, by PTX name
    pub global_values: Vec<(String, i64)>,
    /// Size of dynamic (extern) shared memory in bytes
    pub dynamic_shared_bytes: u64,
    /// Abort analysis after this many executed instructions
    pub max_instructions: u64,
}

impl AnalysisConfig {
    pub fn new(block_dim: (u32, u32, u32)) -> Self {
        Self {
            block_dim,
            grid_dim: (1, 1, 1),
            params: Vec::new(),
            arrays: Vec::new(),
            global_values: Vec::new(),
            dynamic_shared_bytes: 0,
            max_instructions: 2_000_000_000,
        }
    }

    pub fn num_threads(&self) -> u32 {
        self.block_dim.0 * self.block_dim.1 * self.block_dim.2
    }

    pub fn array(&self, name: &str) -> Option<&ArrayDef> {
        self.arrays.iter().find(|a| a.name == name)
    }

    /// Names of the declared output arrays, in declaration order - the
    /// natural "arrays to check" list for an equivalence comparison over
    /// this config.
    pub fn output_array_names(&self) -> Vec<String> {
        self.arrays
            .iter()
            .filter(|a| a.kind.is_output())
            .map(|a| a.name.clone())
            .collect()
    }

    /// Reject configs whose identities are ambiguous: array names and
    /// `sym:` parameter names define which values correlate (see
    /// `symbolic::SymbolRef`), so duplicate names or overlapping ranges
    /// would silently conflate distinct values.
    pub fn validate(&self) -> Result<(), String> {
        for (i, a) in self.arrays.iter().enumerate() {
            if a.elem_width == 0 || a.len == 0 {
                return Err(format!(
                    "array '{}' has zero element width or length",
                    a.name
                ));
            }
            for b in &self.arrays[i + 1..] {
                if a.name == b.name {
                    return Err(format!(
                        "two arrays share the name '{}'; array names are identities \
                         and must be unique",
                        a.name
                    ));
                }
                let (a_end, b_end) = (a.base + a.size_bytes(), b.base + b.size_bytes());
                if a.base < b_end && b.base < a_end {
                    return Err(format!(
                        "arrays '{}' and '{}' overlap in memory \
                         ([{:#x}, {:#x}) vs [{:#x}, {:#x}))",
                        a.name, b.name, a.base, a_end, b.base, b_end
                    ));
                }
            }
        }
        let mut sym_names = std::collections::BTreeSet::new();
        for value in &self.params {
            if let ParamValue::SymFloat(name) = value {
                if name.is_empty() {
                    return Err("sym: parameter name is empty".to_string());
                }
                if name.contains('[') || name.contains(']') {
                    return Err(format!(
                        "sym: parameter name '{}' contains brackets; bracketed names \
                         are reserved for array elements",
                        name
                    ));
                }
                if !sym_names.insert(name) {
                    return Err(format!(
                        "two sym: parameters share the name '{}'; parameter names are \
                         identities and must be unique (a duplicate would silently \
                         constrain the check to inputs where both parameters are equal)",
                        name
                    ));
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn array(name: &str, base: u64, len: u64) -> ArrayDef {
        ArrayDef {
            name: name.to_string(),
            base,
            elem_width: 4,
            len,
            kind: ArrayKind::Input,
        }
    }

    #[test]
    fn validate_accepts_disjoint_arrays_and_plain_sym_names() {
        let mut config = AnalysisConfig::new((1, 1, 1));
        config.arrays.push(array("in", 0x1000, 16));
        config.arrays.push(array("out", 0x2000, 16));
        config
            .params
            .push(ParamValue::SymFloat("alpha".to_string()));
        assert_eq!(config.validate(), Ok(()));
    }

    /// Identity ambiguities are rejected, not silently conflated: two
    /// arrays sharing a name would make unrelated storage produce the
    /// same `InputElement` symbols, and overlapping ranges would give one
    /// address two identities.
    #[test]
    fn validate_rejects_ambiguous_identities() {
        let mut dup = AnalysisConfig::new((1, 1, 1));
        dup.arrays.push(array("x", 0x1000, 16));
        dup.arrays.push(array("x", 0x2000, 16));
        assert!(dup.validate().is_err());

        let mut overlap = AnalysisConfig::new((1, 1, 1));
        overlap.arrays.push(array("a", 0x1000, 16));
        overlap.arrays.push(array("b", 0x1020, 16));
        assert!(overlap.validate().is_err());

        let mut bracketed = AnalysisConfig::new((1, 1, 1));
        bracketed
            .params
            .push(ParamValue::SymFloat("x[0]".to_string()));
        assert!(bracketed.validate().is_err());

        let mut dup_sym = AnalysisConfig::new((1, 1, 1));
        dup_sym
            .params
            .push(ParamValue::SymFloat("alpha".to_string()));
        dup_sym
            .params
            .push(ParamValue::SymFloat("alpha".to_string()));
        assert!(dup_sym.validate().is_err());
    }
}

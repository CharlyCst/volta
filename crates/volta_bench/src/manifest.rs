//! The vcs-directory manifest (`<out-dir>/vcs/manifest.json`): `generate`'s
//! record of what each dump contains, and `solve`'s staleness guard.
//!
//! Dumps are keyed by benchmark-name slug and overwritten in place, so a
//! vcs directory assembled by different volta versions (or different
//! benchmark definitions) can silently mix stale and fresh dumps. The
//! manifest pins each slug to the benchmark name and per-array footprint
//! element counts `generate` produced; `solve` compares the manifest
//! against every dump it loads and hard-errors on disagreement - the
//! stale/mixed-directory failure mode this file exists to catch. A
//! *missing* manifest (or a missing entry) is only a warning: hand-copied
//! dump directories stay usable.
//!
//! `generate` updates the manifest incrementally (read-modify-write per
//! written dump), so `generate single`/`generate category` refresh their
//! own entries without discarding the rest.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use volta_analysis::symbolic::ExprId;

use crate::results::sanitize_name;

/// Manifest filename inside the vcs directory.
pub const MANIFEST_FILE: &str = "manifest.json";

/// Manifest schema version; bump on breaking changes so an old manifest
/// fails loudly instead of misguiding the staleness check.
pub const MANIFEST_FORMAT: u32 = 1;

/// The whole manifest: dump slug -> what `generate` wrote there.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VcsManifest {
    pub format: u32,
    pub entries: BTreeMap<String, ManifestEntry>,
}

impl VcsManifest {
    pub fn new() -> Self {
        Self {
            format: MANIFEST_FORMAT,
            entries: BTreeMap::new(),
        }
    }
}

impl Default for VcsManifest {
    fn default() -> Self {
        Self::new()
    }
}

/// One dump's identity: which benchmark it belongs to, when it was
/// generated, and both snapshots' per-array footprint element counts.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ManifestEntry {
    /// The benchmark's display name (slugs are many-to-one, so the full
    /// name pins the slug to one benchmark).
    pub benchmark: String,
    /// When `generate` wrote the dump (unix seconds).
    pub timestamp_unix: u64,
    /// Reference snapshot: output array name -> written element count.
    pub reference_elements: BTreeMap<String, u64>,
    /// Optimized snapshot: output array name -> written element count.
    pub optimized_elements: BTreeMap<String, u64>,
}

/// Where the manifest lives inside a vcs directory.
pub fn manifest_path(vcs_dir: &Path) -> PathBuf {
    vcs_dir.join(MANIFEST_FILE)
}

/// Per-array element counts of one snapshot's output footprint (the
/// `outputs` shape shared by `AnalysisOutput` and `VcSnapshot`).
pub fn element_counts(outputs: &[(String, Vec<(u64, ExprId)>)]) -> BTreeMap<String, u64> {
    outputs
        .iter()
        .map(|(name, elems)| (name.clone(), elems.len() as u64))
        .collect()
}

/// Read the manifest. `Ok(None)` when the file does not exist (a
/// hand-assembled vcs directory - the caller warns and skips the guard);
/// a present-but-unreadable manifest is a hard error, because a guard
/// that silently degrades is worse than none.
pub fn read_manifest(vcs_dir: &Path) -> Result<Option<VcsManifest>> {
    let path = manifest_path(vcs_dir);
    let text = match std::fs::read_to_string(&path) {
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        other => other.with_context(|| format!("reading {}", path.display()))?,
    };
    let manifest: VcsManifest = serde_json::from_str(&text).with_context(|| {
        format!(
            "{} is not a valid vcs manifest; delete it and re-run \
             `volta-bench generate` to rebuild it",
            path.display()
        )
    })?;
    if manifest.format != MANIFEST_FORMAT {
        bail!(
            "{} has manifest format {} but this build writes format {}; \
             delete it (and regenerate the dumps) with `volta-bench generate`",
            path.display(),
            manifest.format,
            MANIFEST_FORMAT
        );
    }
    Ok(Some(manifest))
}

/// Read the manifest, starting a fresh one when none exists (the
/// `generate` read-modify-write entry point). Errors are `read_manifest`'s.
pub fn read_or_new(vcs_dir: &Path) -> Result<VcsManifest> {
    Ok(read_manifest(vcs_dir)?.unwrap_or_default())
}

/// Write the manifest to `<vcs_dir>/manifest.json` (the directory must
/// already exist - `generate` creates it when it writes the dump).
pub fn write_manifest(vcs_dir: &Path, manifest: &VcsManifest) -> Result<()> {
    let path = manifest_path(vcs_dir);
    let text = serde_json::to_string_pretty(manifest).context("serializing the vcs manifest")?;
    std::fs::write(&path, text).with_context(|| format!("writing {}", path.display()))
}

/// Record one just-written dump in the manifest (in memory; the caller
/// writes the file): the slugged key, the benchmark name, a now
/// timestamp, and both footprints' per-array element counts.
pub fn record_dump(
    manifest: &mut VcsManifest,
    benchmark: &str,
    reference_outputs: &[(String, Vec<(u64, ExprId)>)],
    optimized_outputs: &[(String, Vec<(u64, ExprId)>)],
) {
    let timestamp_unix = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    manifest.entries.insert(
        sanitize_name(benchmark),
        ManifestEntry {
            benchmark: benchmark.to_string(),
            timestamp_unix,
            reference_elements: element_counts(reference_outputs),
            optimized_elements: element_counts(optimized_outputs),
        },
    );
}

/// What checking one loaded dump against the manifest concluded (the
/// error case - disagreement - is the `Err` of [`check_dump`]).
#[derive(Debug, PartialEq, Eq)]
pub enum ManifestCheck {
    /// The manifest has this benchmark's entry and it agrees with the
    /// loaded dump.
    Verified,
    /// The manifest has no entry for this benchmark (e.g. a hand-copied
    /// dump); the caller warns and carries on.
    NoEntry,
}

/// The staleness guard: compare a loaded dump's footprints (and the
/// benchmark's name) against the manifest. Any disagreement means the
/// vcs directory does not hold what `generate` last recorded for this
/// benchmark - stale, mixed, or clobbered - and is a hard error telling
/// the user to regenerate.
pub fn check_dump(
    manifest: &VcsManifest,
    benchmark: &str,
    reference_outputs: &[(String, Vec<(u64, ExprId)>)],
    optimized_outputs: &[(String, Vec<(u64, ExprId)>)],
) -> Result<ManifestCheck> {
    let slug = sanitize_name(benchmark);
    let Some(entry) = manifest.entries.get(&slug) else {
        return Ok(ManifestCheck::NoEntry);
    };
    if entry.benchmark != benchmark {
        bail!(
            "the vcs manifest says dump '{}' belongs to benchmark '{}', not '{}'; \
             the vcs directory is stale or mixed - re-run `volta-bench generate`",
            slug,
            entry.benchmark,
            benchmark
        );
    }
    for (side, expected, found) in [
        (
            "reference",
            &entry.reference_elements,
            element_counts(reference_outputs),
        ),
        (
            "optimized",
            &entry.optimized_elements,
            element_counts(optimized_outputs),
        ),
    ] {
        if let Some(diff) = first_count_diff(expected, &found) {
            bail!(
                "dump '{}' disagrees with the vcs manifest for '{}': {} snapshot {}; \
                 the vcs directory is stale or mixed - re-run `volta-bench generate`",
                slug,
                benchmark,
                side,
                diff
            );
        }
    }
    Ok(ManifestCheck::Verified)
}

/// First difference between the manifest's counts and the dump's, as a
/// message fragment; `None` when identical.
fn first_count_diff(
    expected: &BTreeMap<String, u64>,
    found: &BTreeMap<String, u64>,
) -> Option<String> {
    for (name, want) in expected {
        match found.get(name) {
            None => return Some(format!("is missing output array '{}'", name)),
            Some(got) if got != want => {
                return Some(format!(
                    "array '{}' has {} elements, manifest recorded {}",
                    name, got, want
                ));
            }
            Some(_) => {}
        }
    }
    found
        .keys()
        .find(|name| !expected.contains_key(*name))
        .map(|name| format!("has an output array '{}' the manifest did not record", name))
}

#[cfg(test)]
mod tests {
    use id_collections::Id;

    use super::*;

    fn outputs(arrays: &[(&str, u64)]) -> Vec<(String, Vec<(u64, ExprId)>)> {
        arrays
            .iter()
            .map(|&(name, len)| {
                let elems = (0..len)
                    .map(|i| (i, ExprId::from_index(i as u32)))
                    .collect();
                (name.to_string(), elems)
            })
            .collect()
    }

    /// A manifest written by `write_manifest` reads back identical, and a
    /// missing file is `Ok(None)`, not an error.
    #[test]
    fn manifest_round_trips_and_missing_is_none() {
        let dir = std::env::temp_dir().join(format!("volta_manifest_rt_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();

        assert_eq!(read_manifest(&dir).unwrap(), None);

        let mut manifest = VcsManifest::new();
        record_dump(
            &mut manifest,
            "(Red-1, Red-2)",
            &outputs(&[("out", 1)]),
            &outputs(&[("out", 1), ("aux", 4)]),
        );
        record_dump(
            &mut manifest,
            "(Attention, FA1)",
            &outputs(&[("o", 64)]),
            &outputs(&[("o", 64)]),
        );
        write_manifest(&dir, &manifest).unwrap();
        let loaded = read_manifest(&dir).unwrap().expect("manifest exists");
        assert_eq!(loaded, manifest);

        let entry = &loaded.entries["red-1-red-2"];
        assert_eq!(entry.benchmark, "(Red-1, Red-2)");
        assert_eq!(entry.reference_elements["out"], 1);
        assert_eq!(entry.optimized_elements["aux"], 4);

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A corrupt or wrong-format manifest is a hard error naming the
    /// file, never `Ok` - a guard that silently degrades is worse than
    /// none.
    #[test]
    fn corrupt_and_wrong_format_manifests_are_rejected() {
        let dir = std::env::temp_dir().join(format!("volta_manifest_bad_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();

        std::fs::write(manifest_path(&dir), "{ not json").unwrap();
        let err = format!("{:#}", read_manifest(&dir).unwrap_err());
        assert!(err.contains("manifest"), "{err}");
        assert!(err.contains("generate"), "{err}");

        let future = serde_json::json!({ "format": MANIFEST_FORMAT + 1, "entries": {} });
        std::fs::write(manifest_path(&dir), future.to_string()).unwrap();
        let err = format!("{:#}", read_manifest(&dir).unwrap_err());
        assert!(err.contains("format"), "{err}");

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The staleness guard: agreement verifies, a missing entry is only
    /// `NoEntry`, and every kind of disagreement (count drift, missing or
    /// extra arrays, a slug claimed by another benchmark) is an error
    /// pointing at `generate`.
    #[test]
    fn check_dump_flags_disagreements() {
        let reference = outputs(&[("out", 2)]);
        let optimized = outputs(&[("out", 2), ("aux", 3)]);
        let mut manifest = VcsManifest::new();
        record_dump(&mut manifest, "(Red-1, Red-2)", &reference, &optimized);

        assert_eq!(
            check_dump(&manifest, "(Red-1, Red-2)", &reference, &optimized).unwrap(),
            ManifestCheck::Verified
        );
        assert_eq!(
            check_dump(&manifest, "(Red-1, Red-3)", &reference, &optimized).unwrap(),
            ManifestCheck::NoEntry
        );

        // Element-count drift on either side.
        let err = check_dump(
            &manifest,
            "(Red-1, Red-2)",
            &outputs(&[("out", 5)]),
            &optimized,
        )
        .unwrap_err();
        let msg = format!("{:#}", err);
        assert!(
            msg.contains("'out' has 5 elements, manifest recorded 2"),
            "{msg}"
        );
        assert!(msg.contains("generate"), "{msg}");

        // A whole array missing from the dump, and an extra one.
        let err = check_dump(
            &manifest,
            "(Red-1, Red-2)",
            &reference,
            &outputs(&[("out", 2)]),
        )
        .unwrap_err();
        assert!(format!("{:#}", err).contains("missing output array 'aux'"));
        let err = check_dump(
            &manifest,
            "(Red-1, Red-2)",
            &outputs(&[("out", 2), ("extra", 1)]),
            &optimized,
        )
        .unwrap_err();
        assert!(format!("{:#}", err).contains("'extra'"));

        // Slugs are many-to-one: the same slug claimed under a different
        // display name is a mixed directory, not a match.
        let err = check_dump(&manifest, "Red 1 Red: 2", &reference, &optimized).unwrap_err();
        let msg = format!("{:#}", err);
        assert!(msg.contains("(Red-1, Red-2)"), "{msg}");
        assert!(msg.contains("Red 1 Red: 2"), "{msg}");
    }
}

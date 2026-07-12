//! Enforces the workspace's dependency-direction rule against the *resolved*
//! package graph, not the folder layout.
//!
//! Every crate declares its architectural layer locally, in its manifest:
//!
//! ```toml
//! [package.metadata.pipecrab]
//! layer = "runtime"
//! ```
//!
//! The rule: a package may depend only on **strictly lower** layers (normal
//! and build deps; dev-deps are exempt, since a test may legitimately reach
//! for anything). The order, lowest first:
//!
//! ```text
//! core  <  runtime  <  { trait, facade }  <  adapter  <  app
//! ```
//!
//! `trait` and `facade` share a rank: they never depend on each other, and a
//! strictly-lower rule forbids same-rank edges, so if that ever changes the
//! gate will flag it — the signal to split a layer, not to loosen the rule.
//!
//! `support` is exempt from the ordering entirely (dev-only tooling), but it
//! is still a *declared* layer: the gate **fails closed**, so any workspace
//! member with no `layer` at all is an error. A new crate therefore can't slip
//! through unlabeled — it forces a one-line decision at creation time.

use std::collections::{HashMap, HashSet};

use cargo_metadata::{DependencyKind, MetadataCommand, Package};

/// Ordinal rank of a production layer. `None` means "not part of the ordering"
/// — either the exempt `support` layer or an unknown/misspelled name, which
/// the caller distinguishes and reports.
fn rank(layer: &str) -> Option<i32> {
    match layer {
        "core" => Some(0),
        "runtime" => Some(1),
        "trait" | "facade" => Some(2),
        "adapter" => Some(3),
        "app" => Some(4),
        _ => None,
    }
}

/// The set of layer names the gate understands. `support` is valid but
/// deliberately unranked.
fn is_known_layer(layer: &str) -> bool {
    layer == "support" || rank(layer).is_some()
}

fn declared_layer(pkg: &Package) -> Option<&str> {
    pkg.metadata
        .get("pipecrab")
        .and_then(|m| m.get("layer"))
        .and_then(|l| l.as_str())
}

#[test]
fn layering() {
    let metadata = MetadataCommand::new()
        .exec()
        .expect("run `cargo metadata`");

    let workspace: HashSet<_> = metadata.workspace_members.iter().cloned().collect();
    let members: Vec<&Package> = metadata
        .packages
        .iter()
        .filter(|p| workspace.contains(&p.id))
        .collect();

    let mut errors: Vec<String> = Vec::new();
    let mut layer_of: HashMap<&str, &str> = HashMap::new();

    // Pass 1 — fail closed: every workspace member must declare a known layer.
    for pkg in &members {
        match declared_layer(pkg) {
            Some(layer) if is_known_layer(layer) => {
                layer_of.insert(pkg.name.as_str(), layer);
            }
            Some(layer) => errors.push(format!(
                "{}: unknown layer \"{}\" — expected one of \
                 core, runtime, trait, facade, adapter, app, support",
                pkg.name, layer
            )),
            None => errors.push(format!(
                "{}: no [package.metadata.pipecrab] layer declared \
                 (use \"support\" for dev-only crates)",
                pkg.name
            )),
        }
    }

    // Pass 2 — downward-only: no dependency may point to an equal-or-higher
    // layer. Only workspace members are ranked; external crates and the exempt
    // `support` layer are skipped as both source and target.
    let by_name: HashMap<&str, &Package> =
        members.iter().map(|p| (p.name.as_str(), *p)).collect();

    for pkg in &members {
        let Some(from) = layer_of.get(pkg.name.as_str()).and_then(|l| rank(l)) else {
            continue; // unlabeled (already reported) or exempt support crate
        };
        for dep in &pkg.dependencies {
            if dep.kind == DependencyKind::Development {
                continue;
            }
            let Some(target) = by_name.get(dep.name.as_str()) else {
                continue; // external crate — not ours to rank
            };
            let Some(to) = layer_of.get(target.name.as_str()).and_then(|l| rank(l)) else {
                continue; // dep on an exempt support crate
            };
            if to >= from {
                errors.push(format!(
                    "{} ({}) may not depend on {} ({}) — dependencies must \
                     point to a strictly lower layer",
                    pkg.name,
                    layer_of[pkg.name.as_str()],
                    target.name,
                    layer_of[target.name.as_str()],
                ));
            }
        }
    }

    assert!(
        errors.is_empty(),
        "workspace layering violations:\n  - {}",
        errors.join("\n  - "),
    );
}

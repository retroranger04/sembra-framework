# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0] — 2026-07-12

Second major release of the SEMBRA framework. This release introduces a
richer variational ansatz, a three-branch equilibrium classification
scheme, and a redesigned inference pipeline. It is not backward-compatible
with v1.

### Added

- Four-parameter Ritz ansatz for the equilibrium shape, adding two
  enrichment coordinates (`a`, `b`) on top of the base parameters
  (`λ₀`, `η₀`) used in v1.
- Three-branch equilibrium router that classifies each query point as
  `stable`, `near_fold`, or `no_equilibrium`, replacing v1's two-way
  stable / past-limit split.
- Wrinkling-envelope Gaussian process used only for design-space
  visualization in the shipped Jupyter notebooks. Not part of the
  inference pipeline.
- User-facing Jupyter notebooks for both packages, with stress-field
  visualizations, design-space plots showing stable, unstable, and
  no-equilibrium regions, and failure-mode diagnostics.

### Changed

- **Inference architecture**: v2 uses a symbolic core for the energy
  functional and Hessian, a neural surrogate for the four Ritz
  parameters, and Gaussian process envelopes for the wrinkling and
  critical-pressure boundaries. The equilibrium condition is now
  enforced as a post-inference step rather than being embedded in the
  training loss.
- **Neural surrogate output**: predicts `(p, a, η₀, b)` instead of v1's
  `(p, η₀)`, reflecting the richer ansatz.
- **Loss function**: standard-deviation-normalized MSE replaces v1's
  weighted MSE combined with a physics-informed penalty term.
- **Package folder names**: `SEMBRA` and `SEMBRA_CC` from v1 are
  renamed to `sembra_vc` and `sembra_cc` in v2 for consistency with
  the internal Python package names and imports.

### Breaking changes

- **Inference wrapper API**: the `predict(...)` function now accepts
  `Phi` (the nondimensional electrostatic parameter) in place of v1's
  `alpha_p`. Any code written against v1's signature will need to be
  updated.
- **Package imports**: v1's `SEMBRA` and `SEMBRA_CC` become `sembra_vc`
  and `sembra_cc` respectively.
- **Trained model artifacts**: v1 model weights are not compatible
  with the v2 wrapper. v2 ships its own trained weights.

### Removed

- v1's physics-informed penalty term in the training loss (superseded
  by the post-inference equilibrium enforcement described above).

### Preserved

- v1.0.0 remains available under the [`v1.0.0` tag](https://github.com/retroranger04/sembra-framework/releases/tag/v1.0.0)
  and at its [archived Zenodo version](https://doi.org/10.5281/zenodo.20723833).

## [1.0.0] — 2026-06-16

Initial release accompanying the paper by Kumar, Mathur, and DasGupta.

- Two Python surrogate packages: `SEMBRA` (voltage-controlled) and
  `SEMBRA_CC` (charge-controlled).
- Trained neural surrogate, Gaussian process envelope, analytical
  symbolic core, and inference wrapper for each package.
- Archived on Zenodo (DOI [10.5281/zenodo.20723832](https://doi.org/10.5281/zenodo.20723832)).

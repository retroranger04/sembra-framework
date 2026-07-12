# SEMBRA Framework

**Surrogate for Elastomer Membrane Behavior under Radial loading and Actuation** — a surrogate framework for dielectric elastomer membrane instability.

This repository accompanies the paper *On the Instability of Inflated
Dielectric Elastomer Membranes: Energy-Based Analysis and a Surrogate
Modeling Framework* by Kumar, Mathur, and DasGupta. It provides two
neural-surrogate packages that predict the equilibrium state and
stability of an inflated circular dielectric elastomer membrane under
combined mechanical pressure and electrical actuation.

## Packages

- **[sembra_vc](./sembra_vc)** — voltage-controlled actuation. Use this when
  the membrane is connected to a constant voltage source.
- **[sembra_cc](./sembra_cc)** — charge-controlled actuation. Use this
  when the membrane is initially charged and then disconnected from
  the power supply.

Both packages share an identical architecture and methodology, differing
only in the underlying electrostatic energy formulation. Each is
self-contained and can be installed and used independently.

## Quick start

Choose the package corresponding to your loading scenario:

- For voltage-controlled analysis, see [sembra_vc's README](./sembra_vc/README.md).
- For charge-controlled analysis, see [sembra_cc's README](./sembra_cc/README.md).

Each package provides a `predict(alpha, Phi, p)` function and a
Jupyter notebook for interactive use, along with full inference and
validation pipelines.

## What's new in v2

See [CHANGELOG.md](./CHANGELOG.md) for the full list of changes from v1 to v2,
including the four-parameter Ritz ansatz, the three-branch equilibrium
router, and the updated inference API.

## Requirements

- Python 3.11 or higher

See the individual package READMEs for dependency details.

## Citation

If you use this software, please cite the archived release:

> Kumar, N., Mathur, A., & DasGupta, A. (2026). *SEMBRA Framework: Surrogate models for instability of dielectric elastomer membranes*. Zenodo. <https://doi.org/10.5281/zenodo.20723832>

A machine-readable citation is available via the "Cite this repository"
button on the GitHub page, backed by [CITATION.cff](./CITATION.cff).

The accompanying paper is in preparation; its citation will be added here upon publication.

## License

MIT — see [sembra_vc/LICENSE](./sembra_vc/LICENSE) and [sembra_cc/LICENSE](./sembra_cc/LICENSE).

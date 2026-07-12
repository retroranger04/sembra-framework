# SEMBRA v2 — Voltage-Controlled (`sembra_vc`)

`sembra_vc` is a fast surrogate for the equilibrium state and stability of an
inflated dielectric elastomer membrane under combined mechanical and electrical
loading, in the **voltage-controlled** scenario. Given the three loading
parameters `(alpha, Phi, p)`, it returns the equilibrium Ritz state, a stability
classification, and failure-mode diagnostics in milliseconds — without requiring
the user to run expensive numerical continuation.

## Installation

From inside this directory:

```bash
pip install -e .
```

> **GPU training note.** Training the surrogate requires a CUDA-enabled build of
> PyTorch. A plain `pip install` may resolve the CPU-only wheel. For GPU
> training, install PyTorch from the CUDA index URL appropriate to your system
> (see https://pytorch.org/get-started/locally/) before or after installing this
> package. Inference runs on either GPU or CPU.

## Usage

The package is queried through the Jupyter notebook in `notebooks/`, or
programmatically:

```python
from sembra_vc.wrapper import predict

result = predict(alpha=1.5, Phi=0.2, p=0.3)
print(result)
```

See the companion package `sembra_cc` for the charge-controlled scenario.

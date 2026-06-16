"""Numerical constants referenced by the surrogate wrapper.

Values here MUST mirror the active DEC entries in ``logs/decisions.md``.
Do not edit a constant without first logging a corresponding DEC entry.
"""

from __future__ import annotations

P_ZERO_THRESHOLD: float = 1e-6
GREY_ZONE_HALF_WIDTH: float = 0.005

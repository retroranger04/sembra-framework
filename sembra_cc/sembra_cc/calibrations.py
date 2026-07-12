"""Numerical constants for SEMBRA v2 (charge-controlled scenario).

This module is the single source of truth for every tunable numerical constant
used by the package. No other module defines these values as magic numbers; they
import from here so the user can tune them in exactly one place.
"""

# Radial integration grid for Π evaluation.
TRAPEZOIDAL_GRID_SIZE = 2001

# Projection layer convergence parameters.
PROJECTION_TOLERANCE = 1e-6
PROJECTION_MAX_ITERATIONS = 30
PROJECTION_REGULARIZATION = 1e-6

# Three-branch router threshold.
GREY_ZONE_THRESHOLD = 0.005

# Input validation ranges.
ALPHA_MIN = 1.2
ALPHA_MAX = 2.0
PHI_MIN = 0.0
PHI_MAX = 0.40
P_MIN = 0.0

# Radial grid for user-facing stress field display.
STRESS_DISPLAY_GRID_SIZE = 401

# Numerical zero threshold for the trivial p ≈ 0 shortcut.
TRIVIAL_P_THRESHOLD = 1e-6

# Stable-branch 1D inversion (p -> lambda_0) bracket for Brent's method.
# The bracket spans the whole stable range: from the undeformed lower bound up to
# just below the limit-point stretch (lambda_limit - margin). See DEV-003.
INVERSION_LAMBDA_MIN = 1.0
INVERSION_LIMIT_MARGIN = 1e-3

# Warm-start for the 4D equilibrium Newton solve on the grey-zone branch
# (a, eta_0, b). lambda_0 is warm-started from the GP's lambda_limit.
PAST_LIMIT_A_INIT = 0.05
PAST_LIMIT_ETA0_INIT = 0.3
PAST_LIMIT_B_INIT = -0.05

# Adaptive retreat margins for the 4D Newton solve near the fold (p_crit), where
# the Jacobian is singular (det H = 0) and a solve exactly at the fold frequently
# fails to converge. The near_fold / no_equilibrium handlers retreat to
# p*(1 - margin), increasing the margin from MIN to MAX until the solve converges,
# and report the margin actually used. See DEV-018.
PAST_LIMIT_MARGIN_MIN = 0.005
PAST_LIMIT_MARGIN_MAX = 0.05

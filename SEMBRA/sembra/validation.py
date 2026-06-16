"""Full validation suite for SEMBRA (PRD Section 6.6)."""

from __future__ import annotations

import json
import math
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path  # noqa: F401  (kept for return-type usage in helpers below)
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from sembra import LOGS_DIR, PLOTS_DIR
from sembra.data import load_full_dataset, split_stable_unstable
from sembra.envelope import load_envelope_models, predict_p_crit
from sembra.wrapper import PredictionResult, predict

_LOGS_DIR = LOGS_DIR
_PLOTS_DIR = PLOTS_DIR

PASS_CRITERIA = {
    "heldout_mean_rel_lambda_0": 0.02,
    "heldout_mean_rel_eta_0": 0.02,
    "heldout_p95_rel_lambda_0": 0.05,
    "heldout_p95_rel_eta_0": 0.05,
    "near_limit_mean_rel_lambda_0": 0.05,
    "near_limit_mean_rel_eta_0": 0.05,
    "heldout_max_rel_lambda_0": 0.15,
    "heldout_max_rel_eta_0": 0.15,
    "limit_point_all_unstable": True,
    "latency_p95_ms": 50.0,
}

HOLDOUT_SLICES = [(1.4, 0.06), (1.6, 0.16), (1.8, 0.10), (1.8, 0.24)]


@dataclass
class ValidationReport:
    statistics: dict[str, Any]
    pass_criteria: dict[str, bool]
    failed: list[str]

    @property
    def all_passed(self) -> bool:
        return not self.failed


def _slice_mask(df: pd.DataFrame, slices: Iterable[tuple[float, float]]) -> pd.Series:
    mask = pd.Series(False, index=df.index)
    for alpha_val, alpha_p_val in slices:
        mask |= (
            np.isclose(df["alpha"], alpha_val, atol=1e-6)
            & np.isclose(df["alpha_p"], alpha_p_val, atol=1e-6)
        )
    return mask


def _relative_errors(predictions: np.ndarray, truths: np.ndarray) -> np.ndarray:
    return np.abs(predictions - truths) / np.maximum(np.abs(truths), 1e-12)


def _summarize(rel_errs: np.ndarray) -> dict[str, float]:
    if rel_errs.size == 0:
        return {"n": 0, "mean": float("nan"), "median": float("nan"),
                "p95": float("nan"), "max": float("nan")}
    return {
        "n": int(rel_errs.size),
        "mean": float(rel_errs.mean()),
        "median": float(np.median(rel_errs)),
        "p95": float(np.percentile(rel_errs, 95)),
        "max": float(rel_errs.max()),
    }


def _evaluate_on_rows(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[bool]]:
    """Evaluate ``predict`` on every row of ``df``; per DEC-012 state is always populated."""
    predicted = np.empty((len(df), 3), dtype=float)
    truths = df[["lambda_0", "eta_0", "det_H"]].to_numpy(dtype=float)
    stables: list[bool] = []
    for i, (_, row) in enumerate(df.iterrows()):
        result = predict(row["alpha"], row["alpha_p"], row["p"])
        stables.append(result.stable)
        predicted[i] = (result.lambda_0, result.eta_0, result.det_H)
    return predicted, truths, stables


def _run_ground_truth_test(stable_df: pd.DataFrame, limit_df: pd.DataFrame) -> dict[str, Any]:
    predicted, truths, stables = _evaluate_on_rows(stable_df)
    rel_err = _relative_errors(predicted, truths)
    overall = {
        "lambda_0": _summarize(rel_err[:, 0]),
        "eta_0": _summarize(rel_err[:, 1]),
        "det_H": _summarize(rel_err[:, 2]),
        "n_total_stable": int(len(stable_df)),
    }

    held = _slice_mask(stable_df, HOLDOUT_SLICES)
    held_idx = np.where(held.to_numpy())[0]
    rel_held = _relative_errors(predicted[held_idx], truths[held_idx])
    heldout = {
        "lambda_0": _summarize(rel_held[:, 0]),
        "eta_0": _summarize(rel_held[:, 1]),
        "det_H": _summarize(rel_held[:, 2]),
    }

    near_idx_list: list[int] = []
    for (alpha_val, alpha_p_val), grp in stable_df.groupby(["alpha", "alpha_p"], sort=False):
        sorted_by_lambda = grp.sort_values("lambda_0").tail(5)
        positions = np.where(stable_df.index.isin(sorted_by_lambda.index))[0]
        near_idx_list.extend(positions.tolist())
    near_idx = np.array(sorted(set(near_idx_list)), dtype=int)
    rel_near = _relative_errors(predicted[near_idx], truths[near_idx])
    near_limit = {
        "lambda_0": _summarize(rel_near[:, 0]),
        "eta_0": _summarize(rel_near[:, 1]),
        "det_H": _summarize(rel_near[:, 2]),
    }

    limit_stables = []
    for _, row in limit_df.iterrows():
        result = predict(row["alpha"], row["alpha_p"], row["p"])
        limit_stables.append(result.stable)
    limit_check = {
        "n_total": int(len(limit_df)),
        "n_unstable_reported": int(sum(1 for s in limit_stables if not s)),
        "all_unstable": all(not s for s in limit_stables),
    }

    return {
        "overall": overall,
        "heldout": heldout,
        "near_limit": near_limit,
        "limit_point_check": limit_check,
    }


def _plot_heldout_slices(stable_df: pd.DataFrame) -> list[Path]:
    saved: list[Path] = []
    for alpha_val, alpha_p_val in HOLDOUT_SLICES:
        sub = stable_df[
            np.isclose(stable_df["alpha"], alpha_val, atol=1e-6)
            & np.isclose(stable_df["alpha_p"], alpha_p_val, atol=1e-6)
        ].sort_values("lambda_0")
        if sub.empty:
            continue
        preds = []
        for _, row in sub.iterrows():
            r = predict(row["alpha"], row["alpha_p"], row["p"])
            preds.append((r.lambda_0, r.eta_0, r.det_H))
        preds_arr = np.array(preds, dtype=float)

        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        axes[0].plot(sub["p"], sub["lambda_0"], "k-", label="true")
        axes[0].plot(sub["p"], preds_arr[:, 0], "r--", label="predicted")
        axes[0].set_xlabel("p")
        axes[0].set_ylabel("lambda_0")
        axes[0].set_title(f"lambda_0 vs p  (alpha={alpha_val}, alpha_p={alpha_p_val})")
        axes[0].legend()
        axes[1].plot(sub["p"], sub["det_H"], "k-", label="true")
        axes[1].plot(sub["p"], preds_arr[:, 2], "r--", label="predicted")
        axes[1].set_xlabel("p")
        axes[1].set_ylabel("det_H")
        axes[1].set_title(f"det_H vs p  (alpha={alpha_val}, alpha_p={alpha_p_val})")
        axes[1].legend()
        fig.tight_layout()
        out_path = _PLOTS_DIR / f"heldout_slice_{alpha_val}_{alpha_p_val}.png"
        fig.savefig(out_path, dpi=120)
        plt.close(fig)
        saved.append(out_path)
    return saved


def _plot_pcrit_contour(limit_df: pd.DataFrame) -> Path:
    p_model, _ = load_envelope_models()
    alphas = np.linspace(1.2, 2.0, 80)
    alpha_ps = np.linspace(0.0, 0.72, 80)
    aa, ap = np.meshgrid(alphas, alpha_ps, indexing="xy")
    grid = predict_p_crit(p_model, aa.ravel(), ap.ravel()).reshape(aa.shape)
    fig, ax = plt.subplots(figsize=(7, 5))
    cs = ax.contourf(aa, ap, grid, levels=24)
    ax.scatter(
        limit_df["alpha"], limit_df["alpha_p"],
        c="white", s=14, edgecolors="k", label="limit-point training points",
    )
    ax.set_xlabel("alpha")
    ax.set_ylabel("alpha_p")
    ax.set_title("p_crit envelope (GP)")
    ax.legend(loc="upper left")
    fig.colorbar(cs, ax=ax, label="p_crit")
    out_path = _PLOTS_DIR / "pcrit_envelope.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def _benchmark_latency(rng: np.random.Generator, n: int = 1000) -> dict[str, float]:
    inputs = np.column_stack(
        [
            rng.uniform(1.2, 2.0, n),
            rng.uniform(0.0, 0.72, n),
            rng.uniform(0.001, 2.5, n),
        ]
    )
    times_ms = np.empty(n)
    for i, (a, ap, p) in enumerate(inputs):
        t0 = time.perf_counter()
        predict(float(a), float(ap), float(p))
        times_ms[i] = (time.perf_counter() - t0) * 1000.0
    return {
        "n": int(n),
        "mean_ms": float(times_ms.mean()),
        "median_ms": float(np.median(times_ms)),
        "p95_ms": float(np.percentile(times_ms, 95)),
        "max_ms": float(times_ms.max()),
    }


def _run_edge_cases() -> dict[str, Any]:
    p_model, _ = load_envelope_models()
    pc_16_10 = float(predict_p_crit(p_model, 1.6, 0.10))
    cases = [
        ("p_eq_zero", (1.5, 0.2, 0.0)),
        ("p_eq_1e-10", (1.5, 0.2, 1e-10)),
        ("first_stable_row", None),
        ("p_below_pcrit", (1.6, 0.10, max(pc_16_10 - 1e-3, 1e-3))),
        ("p_at_pcrit", (1.6, 0.10, pc_16_10)),
        ("p_1_05_pcrit", (1.6, 0.10, 1.05 * pc_16_10)),
        ("p_3_pcrit", (1.6, 0.10, 3.0 * pc_16_10)),
        ("boundary_alpha_1_2", (1.2, 0.2, 0.5)),
        ("boundary_alpha_2_0", (2.0, 0.5, 0.8)),
    ]
    stable_df, _ = split_stable_unstable(load_full_dataset())
    first_row = stable_df.iloc[0]
    cases[2] = ("first_stable_row", (float(first_row["alpha"]), float(first_row["alpha_p"]), float(first_row["p"])))

    results: dict[str, Any] = {}
    for name, payload in cases:
        a, ap, p = payload
        try:
            r = predict(a, ap, p)
            results[name] = {
                "input": [a, ap, p],
                "raised": None,
                "stable": r.stable,
                "lambda_0": r.lambda_0 if not math.isnan(r.lambda_0) else None,
                "eta_0": r.eta_0 if not math.isnan(r.eta_0) else None,
                "det_H": r.det_H if not math.isnan(r.det_H) else None,
                "message": r.message,
            }
        except Exception as exc:  # noqa: BLE001
            results[name] = {"input": [a, ap, p], "raised": f"{type(exc).__name__}: {exc}"}

    error_cases = [
        ("alpha_lt_1_2", (0.5, 0.0, 0.5)),
        ("alpha_gt_2_0", (3.0, 0.0, 0.5)),
        ("negative_alpha_p", (1.5, -0.1, 0.5)),
        ("negative_p", (1.5, 0.2, -1.0)),
        ("nan_alpha", (float("nan"), 0.2, 0.5)),
    ]
    error_results = {}
    for name, payload in error_cases:
        a, ap, p = payload
        try:
            predict(a, ap, p)
            error_results[name] = {"raised": None, "expected_value_error": True, "passed": False}
        except ValueError as exc:
            error_results[name] = {"raised": str(exc), "expected_value_error": True, "passed": True}
        except Exception as exc:  # noqa: BLE001
            error_results[name] = {"raised": str(exc), "expected_value_error": True, "passed": False}
    return {"valid_inputs": results, "value_error_inputs": error_results}


def run_full_validation(seed: int = 42) -> ValidationReport:
    """Execute the full validation suite (Section 6.6) and persist artifacts."""
    _PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    df = load_full_dataset()
    stable_df, limit_df = split_stable_unstable(df)

    log_path = _LOGS_DIR / "validation.log"
    log_lines: list[str] = []

    log_lines.append("=== SEMBRA validation (PRD Section 6.6) ===")
    log_lines.append(f"date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log_lines.append(f"n_stable={len(stable_df)} n_limit={len(limit_df)}")

    log_lines.append("\n--- End-to-end ground-truth test ---")
    gt = _run_ground_truth_test(stable_df, limit_df)
    log_lines.append(json.dumps(gt, indent=2))

    log_lines.append("\n--- Plot artifacts ---")
    slice_plots = _plot_heldout_slices(stable_df)
    pcrit_plot = _plot_pcrit_contour(limit_df)
    log_lines.append(f"slice plots: {[p.name for p in slice_plots]}")
    log_lines.append(f"pcrit plot: {pcrit_plot.name}")

    log_lines.append("\n--- Latency benchmark ---")
    rng = np.random.default_rng(seed)
    latency = _benchmark_latency(rng)
    log_lines.append(json.dumps(latency, indent=2))

    log_lines.append("\n--- Edge-case suite ---")
    edge = _run_edge_cases()
    edge_path = _LOGS_DIR / "edge_case_results.json"
    with open(edge_path, "w", encoding="utf-8") as fh:
        json.dump(edge, fh, indent=2, default=str)
    log_lines.append(f"saved to {edge_path.name}")

    statistics = {
        "ground_truth": gt,
        "latency": latency,
        "edge_cases": edge,
        "slice_plots": [str(p) for p in slice_plots],
        "pcrit_plot": str(pcrit_plot),
    }

    pass_results = {
        "heldout_mean_rel_lambda_0": gt["heldout"]["lambda_0"]["mean"] <= PASS_CRITERIA["heldout_mean_rel_lambda_0"],
        "heldout_mean_rel_eta_0": gt["heldout"]["eta_0"]["mean"] <= PASS_CRITERIA["heldout_mean_rel_eta_0"],
        "heldout_p95_rel_lambda_0": gt["heldout"]["lambda_0"]["p95"] <= PASS_CRITERIA["heldout_p95_rel_lambda_0"],
        "heldout_p95_rel_eta_0": gt["heldout"]["eta_0"]["p95"] <= PASS_CRITERIA["heldout_p95_rel_eta_0"],
        "near_limit_mean_rel_lambda_0": gt["near_limit"]["lambda_0"]["mean"] <= PASS_CRITERIA["near_limit_mean_rel_lambda_0"],
        "near_limit_mean_rel_eta_0": gt["near_limit"]["eta_0"]["mean"] <= PASS_CRITERIA["near_limit_mean_rel_eta_0"],
        "heldout_max_rel_lambda_0": gt["heldout"]["lambda_0"]["max"] <= PASS_CRITERIA["heldout_max_rel_lambda_0"],
        "heldout_max_rel_eta_0": gt["heldout"]["eta_0"]["max"] <= PASS_CRITERIA["heldout_max_rel_eta_0"],
        "limit_point_all_unstable": gt["limit_point_check"]["all_unstable"],
        "latency_p95_ms": latency["p95_ms"] <= PASS_CRITERIA["latency_p95_ms"],
        "value_error_cases_all_passed": all(v["passed"] for v in edge["value_error_inputs"].values()),
        "edge_cases_no_unhandled": all(
            v.get("raised") is None for v in edge["valid_inputs"].values()
        ),
    }
    failed = [k for k, v in pass_results.items() if not v]
    log_lines.append("\n--- Pass criteria ---")
    log_lines.append(json.dumps(pass_results, indent=2))
    log_lines.append(f"failed: {failed}")

    with open(log_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(log_lines) + "\n")

    if failed:
        failure_path = _LOGS_DIR / "validation_failure.md"
        lines = ["# Validation failure report", ""]
        lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")
        lines.append("## Failing criteria")
        for crit in failed:
            actual: Any
            if crit.startswith("heldout_") and "lambda_0" in crit:
                actual = gt["heldout"]["lambda_0"]
            elif crit.startswith("heldout_") and "eta_0" in crit:
                actual = gt["heldout"]["eta_0"]
            elif crit.startswith("near_limit_") and "lambda_0" in crit:
                actual = gt["near_limit"]["lambda_0"]
            elif crit.startswith("near_limit_") and "eta_0" in crit:
                actual = gt["near_limit"]["eta_0"]
            elif crit == "limit_point_all_unstable":
                actual = gt["limit_point_check"]
            elif crit == "latency_p95_ms":
                actual = latency
            else:
                actual = edge
            lines.append(f"- **{crit}**: {json.dumps(actual, indent=2)}")
        with open(failure_path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines))

    return ValidationReport(statistics=statistics, pass_criteria=pass_results, failed=failed)

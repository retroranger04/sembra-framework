"""Validation suite (PRD §6.6). Produces the cached summary used by Hard Halt 3."""

from __future__ import annotations

import json
import math
import time
import traceback
from typing import Any

import numpy as np
import pandas as pd

from sembra_cc import LOGS_DIR
from sembra_cc.data import (
    HELDOUT_SLICES,
    load_full_dataset,
    make_train_val_split,
    split_stable_unstable,
)
from sembra_cc.wrapper import predict


_VALIDATION_LOG = LOGS_DIR / "validation_log.md"
_CACHED_SUMMARY = LOGS_DIR / "cached_validation_summary.json"


def _relative_errors(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    return np.abs(pred - true) / np.maximum(np.abs(true), 1e-12)


def _measure_heldout(heldout_df: pd.DataFrame) -> dict[str, Any]:
    rel_lambda = []
    rel_eta = []
    per_slice = {}
    for slice_alpha, slice_alpha_p in HELDOUT_SLICES:
        mask = (
            (heldout_df["alpha"].sub(slice_alpha).abs() < 1e-9)
            & (heldout_df["alpha_p"].sub(slice_alpha_p).abs() < 1e-9)
        )
        sub = heldout_df[mask]
        if len(sub) == 0:
            continue
        l_true = sub["lambda_0"].to_numpy()
        e_true = sub["eta_0"].to_numpy()
        l_pred = []
        e_pred = []
        for _, row in sub.iterrows():
            res = predict(float(row["alpha"]), float(row["alpha_p"]), float(row["p"]))
            l_pred.append(res.lambda_0)
            e_pred.append(res.eta_0)
        l_pred = np.array(l_pred)
        e_pred = np.array(e_pred)
        rel_l = _relative_errors(l_pred, l_true)
        rel_e = _relative_errors(e_pred, e_true)
        rel_lambda.append(rel_l)
        rel_eta.append(rel_e)
        per_slice[f"{slice_alpha}_{slice_alpha_p}"] = {
            "n": int(len(sub)),
            "mean_rel_lambda_0": float(np.mean(rel_l)),
            "mean_rel_eta_0": float(np.mean(rel_e)),
            "max_rel_lambda_0": float(np.max(rel_l)),
            "max_rel_eta_0": float(np.max(rel_e)),
        }
    rel_lambda = np.concatenate(rel_lambda) if rel_lambda else np.array([])
    rel_eta = np.concatenate(rel_eta) if rel_eta else np.array([])
    return {
        "heldout_mean_rel_lambda_0": float(np.mean(rel_lambda)) if rel_lambda.size else float("nan"),
        "heldout_mean_rel_eta_0": float(np.mean(rel_eta)) if rel_eta.size else float("nan"),
        "heldout_p95_rel_lambda_0": float(np.percentile(rel_lambda, 95)) if rel_lambda.size else float("nan"),
        "heldout_p95_rel_eta_0": float(np.percentile(rel_eta, 95)) if rel_eta.size else float("nan"),
        "heldout_max_rel_lambda_0": float(np.max(rel_lambda)) if rel_lambda.size else float("nan"),
        "heldout_max_rel_eta_0": float(np.max(rel_eta)) if rel_eta.size else float("nan"),
        "per_slice": per_slice,
    }


def _measure_near_limit(heldout_df: pd.DataFrame) -> dict[str, Any]:
    rel_lambda = []
    rel_eta = []
    for slice_alpha, slice_alpha_p in HELDOUT_SLICES:
        mask = (
            (heldout_df["alpha"].sub(slice_alpha).abs() < 1e-9)
            & (heldout_df["alpha_p"].sub(slice_alpha_p).abs() < 1e-9)
        )
        sub = heldout_df[mask].nlargest(5, "lambda_0")
        if len(sub) == 0:
            continue
        for _, row in sub.iterrows():
            res = predict(float(row["alpha"]), float(row["alpha_p"]), float(row["p"]))
            l_true = float(row["lambda_0"])
            e_true = float(row["eta_0"])
            if math.isfinite(res.lambda_0):
                rel_lambda.append(abs(res.lambda_0 - l_true) / max(abs(l_true), 1e-12))
            if math.isfinite(res.eta_0):
                rel_eta.append(abs(res.eta_0 - e_true) / max(abs(e_true), 1e-12))
    return {
        "near_limit_mean_rel_lambda_0": float(np.mean(rel_lambda)) if rel_lambda else float("nan"),
        "near_limit_mean_rel_eta_0": float(np.mean(rel_eta)) if rel_eta else float("nan"),
        "near_limit_n_points": len(rel_lambda),
    }


def _measure_limit_point_classification(limit_df: pd.DataFrame) -> dict[str, Any]:
    total = len(limit_df)
    n_unstable = 0
    for _, row in limit_df.iterrows():
        res = predict(float(row["alpha"]), float(row["alpha_p"]), float(row["p"]))
        if not res.stable:
            n_unstable += 1
    return {
        "limit_point_total": total,
        "limit_point_unstable_count": n_unstable,
        "limit_point_all_unstable": n_unstable == total,
    }


def _measure_latency(seed: int = 7) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    n = 200
    samples = []
    for _ in range(n):
        a = float(rng.uniform(1.2, 2.0))
        ap = float(rng.uniform(0.0, 0.4))
        p = float(rng.uniform(0.05, 1.0))
        t0 = time.perf_counter()
        predict(a, ap, p)
        samples.append((time.perf_counter() - t0) * 1000.0)
    samples = np.array(samples)
    return {
        "latency_mean_ms": float(np.mean(samples)),
        "latency_p95_ms": float(np.percentile(samples, 95)),
        "latency_max_ms": float(np.max(samples)),
        "latency_n_samples": int(n),
    }


def _measure_value_error_coverage() -> dict[str, Any]:
    cases = [
        ("nan_alpha", lambda: predict(float("nan"), 0.1, 0.5)),
        ("inf_alpha_p", lambda: predict(1.5, float("inf"), 0.5)),
        ("string_p", lambda: predict(1.5, 0.1, "0.5")),
        ("alpha_too_low", lambda: predict(1.0, 0.1, 0.5)),
        ("alpha_too_high", lambda: predict(2.5, 0.1, 0.5)),
        ("negative_alpha_p", lambda: predict(1.5, -0.1, 0.5)),
        ("negative_p", lambda: predict(1.5, 0.1, -0.5)),
    ]
    passed = []
    failed = []
    for name, fn in cases:
        try:
            fn()
        except ValueError:
            passed.append(name)
        except Exception as e:  # noqa
            failed.append((name, f"raised {type(e).__name__}: {e}"))
        else:
            failed.append((name, "did not raise"))
    return {
        "value_error_cases_total": len(cases),
        "value_error_cases_passed": len(passed),
        "value_error_cases_all_passed": len(failed) == 0,
        "value_error_failed_cases": failed,
    }


def _measure_edge_cases(limit_df: pd.DataFrame) -> dict[str, Any]:
    failures = []
    p_crit_sample = float(limit_df["p"].iloc[0])
    a = float(limit_df["alpha"].iloc[0])
    ap = float(limit_df["alpha_p"].iloc[0])
    cases = [
        (a, ap, 0.0),
        (a, ap, 1e-9),
        (a, ap, max(p_crit_sample * 0.9, 0.01)),
        (a, ap, p_crit_sample * 1.005),
        (a, ap, p_crit_sample * 1.5),
    ]
    for c in cases:
        try:
            r = predict(*c)
            if not hasattr(r, "lambda_0"):
                failures.append((c, "no lambda_0 attribute"))
        except Exception as e:  # noqa
            failures.append((c, f"{type(e).__name__}: {e}\n{traceback.format_exc()}"))
    return {
        "edge_cases_total": len(cases),
        "edge_cases_no_unhandled": len(failures) == 0,
        "edge_case_failures": failures,
    }


def _pass_criteria(summary: dict) -> dict[str, Any]:
    thresholds = {
        "heldout_mean_rel_lambda_0": 0.02,
        "heldout_mean_rel_eta_0": 0.02,
        "heldout_p95_rel_lambda_0": 0.05,
        "heldout_p95_rel_eta_0": 0.05,
        "heldout_max_rel_lambda_0": 0.15,
        "heldout_max_rel_eta_0": 0.15,
        "near_limit_mean_rel_lambda_0": 0.05,
        "near_limit_mean_rel_eta_0": 0.05,
        "latency_p95_ms": 50.0,
    }
    results = {}
    for key, threshold in thresholds.items():
        val = summary.get(key, float("nan"))
        results[key] = {
            "value": val,
            "threshold": threshold,
            "pass": bool(np.isfinite(val) and val <= threshold),
        }
    results["limit_point_all_unstable"] = {
        "value": summary.get("limit_point_all_unstable", False),
        "threshold": "all 105",
        "pass": bool(summary.get("limit_point_all_unstable", False)),
    }
    results["value_error_cases_all_passed"] = {
        "value": summary.get("value_error_cases_all_passed", False),
        "threshold": True,
        "pass": bool(summary.get("value_error_cases_all_passed", False)),
    }
    results["edge_cases_no_unhandled"] = {
        "value": summary.get("edge_cases_no_unhandled", False),
        "threshold": True,
        "pass": bool(summary.get("edge_cases_no_unhandled", False)),
    }
    all_pass = all(item["pass"] for item in results.values())
    return {"criteria": results, "all_pass": all_pass}


def _write_validation_log(summary: dict, criteria_report: dict) -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    md = []
    md.append("\n---\n")
    md.append("## Hard Halt 3 — full validation (PRD §6.6)\n")
    md.append(f"\nRun timestamp: {time.strftime('%Y-%m-%dT%H:%M:%S')}\n")
    md.append("\n### Pass criteria\n")
    md.append("| Criterion | Value | Threshold | Status |")
    md.append("|---|---|---|---|")
    for k, v in criteria_report["criteria"].items():
        val_str = (
            f"{v['value']:.6f}"
            if isinstance(v["value"], (int, float)) and not isinstance(v["value"], bool)
            else str(v["value"])
        )
        md.append(
            f"| {k} | {val_str} | {v['threshold']} | {'PASS' if v['pass'] else 'FAIL'} |"
        )
    md.append("")
    md.append(f"**Overall:** {'PASS' if criteria_report['all_pass'] else 'FAIL'}")
    md.append("")
    md.append("### Detailed summary")
    md.append("```json")
    md.append(json.dumps(summary, indent=2, default=str))
    md.append("```")
    with open(_VALIDATION_LOG, "a") as fh:
        fh.write("\n".join(md))


def run_full_validation() -> dict:
    df = load_full_dataset()
    stable, limit = split_stable_unstable(df)
    _, _, heldout = make_train_val_split(stable, seed=42)

    summary: dict = {}
    summary.update(_measure_heldout(heldout))
    summary.update(_measure_near_limit(heldout))
    summary.update(_measure_limit_point_classification(limit))
    summary.update(_measure_latency())
    summary.update(_measure_value_error_coverage())
    summary.update(_measure_edge_cases(limit))

    criteria_report = _pass_criteria(summary)
    summary["pass_criteria"] = criteria_report

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    with open(_CACHED_SUMMARY, "w") as fh:
        json.dump(summary, fh, indent=2, default=str)
    _write_validation_log(summary, criteria_report)

    return summary

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Exp2 -> Exp3 predictor

Uses Exp2 Pass@1 to predict Exp3 (until-correct up to k) success rate and
expected attempts under the baseline truncated-geometric model:
  - q = 1 - (1 - p)^k
  - A = [1 - (1 - p)^k] / p

By default, the script:
  - Treats the Exp2 Pass@1 per (model, task_type) as p̂ and plugs it directly
    into the formulas above (no empirical-Bayes shrinkage or delta-method
    corrections).
  - Optionally applies a finite-population (hypergeometric) correction when
    --pool-size is provided, to better approximate sampling without replacement.
  - Optionally fits a simple logit-linear calibration on available Exp3
    observations when --calibrate is set, mainly for diagnostic purposes.

Inputs are loaded from the structured results directory via
visualize_results.CAPTCHAVisualizer, so you can point to any results tree.

Outputs a CSV with per-(provider,model,task_type) predictions and, when
available, Exp3 observations for comparison.
"""

from __future__ import annotations

import argparse
from math import comb
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from .visualize_results import CAPTCHAVisualizer


def g_q(p: float, k: int) -> float:
    return 1.0 - (1.0 - p) ** max(k, 0)


def g2_q(p: float, k: int) -> float:
    if k < 2:
        return 0.0
    return -k * (k - 1) * (1.0 - p) ** (k - 2)


def shrink_p_beta(x: int, n: int, alpha0: float = 1.0, beta0: float = 1.0) -> float:
    return (alpha0 + x) / (alpha0 + beta0 + max(n, 0))


def predict_exp3_delta(p_hat: float, n: int, k: int) -> float:
    """Delta-method correction for q = 1-(1-p)^k with Var(p̂)=p(1-p)/n."""
    p = float(np.clip(p_hat, 1e-9, 1 - 1e-9))
    var = p * (1.0 - p) / max(n, 1)
    q0 = g_q(p, k)
    q_corr = q0 - 0.5 * g2_q(p, k) * var
    return float(np.clip(q_corr, 0.0, 1.0))


def predict_exp3_hypergeom(p_eff: float, k: int, N: int) -> float:
    """Finite-population correction: probability of ≥1 success in k draws without replacement.
    p_eff is the effective single-draw success proportion; N is pool size.
    """
    N = int(max(0, N))
    if N == 0 or k <= 0:
        return 0.0
    k = min(k, N)
    M = int(round(float(np.clip(p_eff, 0.0, 1.0)) * N))
    M = max(0, min(M, N))
    if M == 0:
        return 0.0
    if M == N:
        return 1.0
    if k > (N - M):
        p_none = 0.0
    else:
        p_none = comb(N - M, k) / comb(N, k)
    return 1.0 - p_none


def expected_attempts_hypergeom(p_eff: float, k: int, N: int) -> float:
    """Expected attempts for finite pool (approx): E[min(G,k)] with without-replacement draws.
    Uses identity E[min(G,k)] = sum_{r=0}^{k-1} P(G ≥ r+1) and hypergeometric no-success terms.
    """
    N = int(max(0, N))
    if N == 0 or k <= 0:
        return 0.0
    k = min(k, N)
    M = int(round(float(np.clip(p_eff, 0.0, 1.0)) * N))
    M = max(0, min(M, N))
    tot = 0.0
    for r in range(0, k):
        if r > (N - M):
            p_none = 0.0
        else:
            denom = comb(N, r)
            p_none = 0.0 if denom == 0 else comb(N - M, r) / denom
        tot += 1.0 - p_none
    return tot


def predict_q_from_exp2(
    p_hat: float,
    n: int,
    k: int,
    N_pool: Optional[int] = None,
    alpha0: float = 1.0,
    beta0: float = 1.0,
) -> float:
    """
    Baseline mapping without statistical corrections:
      q = 1 - (1 - p_hat)^k

    We keep the finite-pool option (hypergeometric) as an optional override
    when N_pool is provided, but disable:
      - Beta shrinkage (EB prior)
      - Delta-method (Jensen) correction
    """
    # Clip p_hat to avoid numerical edge cases at exactly 0/1
    p = float(np.clip(p_hat, 1e-9, 1.0 - 1e-9))

    # Pure i.i.d. Bernoulli baseline
    q = g_q(p, k)

    # Optional finite-population correction when pool size is explicitly given
    if N_pool is not None:
        return predict_exp3_hypergeom(p, k, N_pool)
    return q


def predict_A_from_exp2(
    p_hat: float,
    n: int,
    k: int,
    N_pool: Optional[int] = None,
    alpha0: float = 1.0,
    beta0: float = 1.0,
) -> float:
    """
    Baseline expected attempts (truncated geometric) without corrections:
      A = [1 - (1 - p_hat)^k] / p_hat

    As above, we keep the optional finite-pool variant when N_pool is set.
    """
    p = float(np.clip(p_hat, 1e-9, 1.0 - 1e-9))

    if N_pool is not None:
        return expected_attempts_hypergeom(p, k, N_pool)

    if p <= 0:
        return float(k)
    return float((1.0 - (1.0 - p) ** k) / p)


def _logit(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 1e-6, 1 - 1e-6)
    return np.log(x / (1 - x))


def calibrate_logit_linear(q_pred: np.ndarray, q_obs: np.ndarray) -> Tuple[float, float]:
    """Fit logit(q_obs) ≈ a + b * logit(q_pred) via least squares.
    Returns (a,b).
    """
    X = _logit(q_pred)
    y = _logit(q_obs)
    A = np.vstack([np.ones_like(X), X]).T
    # least squares
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    a, b = coef[0], coef[1]
    return float(a), float(b)


def apply_calibration(q_pred: np.ndarray, a: float, b: float) -> np.ndarray:
    z = a + b * _logit(q_pred)
    q = 1.0 / (1.0 + np.exp(-z))
    return np.clip(q, 0.0, 1.0)


def main():
    ap = argparse.ArgumentParser(description="Predict Exp3 from Exp2 with bias corrections")
    ap.add_argument("--results-dir", default="./results", help="Root results directory")
    ap.add_argument("--output", default="./exp2_to_exp3_predictions.csv", help="Output CSV path")
    ap.add_argument("--k", type=int, default=10, help="Max attempts per type in Exp3")
    ap.add_argument("--pool-size", type=int, default=None, help="Pool size (N) per type in Exp3; if omitted, no finite-pop correction")
    ap.add_argument("--alpha0", type=float, default=1.0, help="Beta prior alpha for EB shrinkage")
    ap.add_argument("--beta0", type=float, default=1.0, help="Beta prior beta for EB shrinkage")
    ap.add_argument(
        "--provider",
        dest="providers",
        action="append",
        default=None,
        help="Optional provider filter(s); repeat or use comma-separated values (e.g., --provider openai,gemini)",
    )
    ap.add_argument(
        "--model",
        dest="models",
        action="append",
        default=None,
        help="Optional model filter(s) in provider/model form; repeat or comma-separate to include multiple models in one run",
    )
    ap.add_argument("--calibrate", action="store_true", help="Fit simple logit-linear calibration on available Exp3")
    args = ap.parse_args()

    viz = CAPTCHAVisualizer(results_dir=args.results_dir)
    if viz.data.empty:
        raise SystemExit("No results found under --results-dir")

    df = viz.data.copy()

    def _normalize_multi(values):
        """Expand repeated/CSV CLI inputs into a unique list."""
        if values is None:
            return None
        if isinstance(values, str):
            values = [values]
        items = []
        for v in values:
            if v is None:
                continue
            items.extend([s.strip() for s in str(v).split(",") if s.strip()])
        return sorted(set(items)) or None

    providers_filter = _normalize_multi(args.providers)
    models_filter = _normalize_multi(args.models)

    if providers_filter:
        df = df[df['provider'].isin(providers_filter)]
    if models_filter:
        df = df[df['provider_model'].isin(models_filter)]

    # Exp2 summary (per provider/model/task_type): n and pass (Pass@1)
    exp2 = df[df['experiment'] == 'exp2'].copy()
    if exp2.empty:
        raise SystemExit("Exp2 results not found; cannot predict")

    # keep columns if present; some results already aggregated per type
    cols_keep = ['provider','model','provider_model','task_type','n','pass']
    for c in cols_keep:
        if c not in exp2.columns:
            # some columns may be missing; fill defaults
            if c == 'n':
                exp2['n'] = 1
            else:
                pass
    exp2_grp = exp2.groupby(['provider','model','provider_model','task_type'], as_index=False).agg(
        n=('n','max'),  # already aggregated per type; use max as safe default
        p_hat=('pass','mean')  # Pass@1
    )

    # Exp3 observations for validation (success within k, avg attempts)
    exp3 = df[df['experiment'] == 'exp3'].copy()
    exp3_obs = None
    if not exp3.empty:
        # Already converted in loader: 'pass' (success prob), 'avg_attempts'
        cols = ['provider','model','provider_model','task_type','pass','avg_attempts']
        for c in cols:
            if c not in exp3.columns:
                # tolerate absence
                exp3[c] = np.nan
        exp3_obs = exp3.groupby(['provider','model','provider_model','task_type'], as_index=False).agg(
            q_obs=('pass','mean'),
            A_obs=('avg_attempts','mean')
        )

    # Predict
    rows = []
    for _, r in exp2_grp.iterrows():
        prov = r['provider']
        model = r['model']
        pm = r['provider_model']
        t = r['task_type']
        n = int(r.get('n', 1) or 1)
        p_hat = float(r['p_hat'])

        # Final q_pred / A_pred (may include finite-pool correction if --pool-size is set)
        q_pred = predict_q_from_exp2(p_hat, n, args.k, args.pool_size, args.alpha0, args.beta0)
        A_pred = predict_A_from_exp2(p_hat, n, args.k, args.pool_size, args.alpha0, args.beta0)

        rows.append([prov, model, pm, t, n, p_hat, q_pred, A_pred])

    pred_df = pd.DataFrame(
        rows,
        columns=[
            'provider',
            'model',
            'provider_model',
            'task_type',
            'n',
            'p_hat',
            'q_pred',
            'A_pred',
        ],
    )

    # Optional calibration on observed Exp3
    if args.calibrate and exp3_obs is not None and not exp3_obs.empty:
        merged = pred_df.merge(exp3_obs, on=['provider','model','provider_model','task_type'], how='inner')
        valid = merged.dropna(subset=['q_pred','q_obs'])
        if len(valid) >= 3:
            a, b = calibrate_logit_linear(valid['q_pred'].to_numpy(), valid['q_obs'].to_numpy())
            pred_df['q_pred_cal'] = apply_calibration(pred_df['q_pred'].to_numpy(), a, b)
            pred_df['calibration_a'] = a
            pred_df['calibration_b'] = b
        else:
            print("[INFO] Not enough pairs to calibrate; skipping")

    # Attach observations for reference
    if exp3_obs is not None and not exp3_obs.empty:
        pred_df = pred_df.merge(exp3_obs, on=['provider','model','provider_model','task_type'], how='left')
        # basic error metrics if obs available
        if 'q_obs' in pred_df.columns:
            pred_df['q_abs_err'] = (pred_df['q_pred'] - pred_df['q_obs']).abs()
        if 'A_obs' in pred_df.columns:
            pred_df['A_abs_err'] = (pred_df['A_pred'] - pred_df['A_obs']).abs()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(out_path, index=False)
    print(f"[SAVED] Predictions -> {out_path}")


if __name__ == "__main__":
    main()

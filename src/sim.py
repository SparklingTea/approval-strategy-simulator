"""Unit economics engine: policy x assumptions -> conversion, losses, profit and CLV, with Monte Carlo bands.

Decisions use the tier's *predicted* PD; outcomes use each loan's *realised* default and loss from the
out-of-time vintage. A tier's accuracy therefore shows up in money, not just in AUC.
"""
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PD_WINDOW_Y = 7.0
BAL_FACTOR = 0.55  # average outstanding balance / original amount for an amortising loan


@dataclass(frozen=True)
class Policy:
    tier: str = "standard"           # instant | standard | full | routed
    route_pd: float = 0.03           # routed: ask for documents when instant PD >= this ...
    route_amount: float = 250_000    # ... or the requested amount >= this
    pd_cutoff: float = 0.04          # approve if predicted PD over the product term <= cutoff
    pricing: str = "risk_based"      # "flat" or "risk_based"
    flat_apr: float = 0.12
    margin: float = 0.06             # risk-based: APR = cost of funds + margin + loss_load * annualised EL
    loss_load: float = 1.0
    max_apr: float = 0.25
    amount_mult: float = 1.0         # offer = requested x multiplier
    term_y: float = 2.0


@dataclass(frozen=True)
class Assumptions:
    cof: float = 0.05
    opex_per_loan: float = 400.0
    uw_cost: dict = field(default_factory=lambda: {"instant": 5.0, "standard": 25.0, "full": 120.0})
    completion: dict = field(default_factory=lambda: {"instant": 0.90, "standard": 0.80, "full": 0.60})
    base_takeup: float = 0.65        # at ref_apr, for a full-amount offer
    ref_apr: float = 0.12
    beta: float = -0.25              # take-up logit change per +1pp APR (assumed: not identified in the data)
    beta_sd: float = 0.08
    adverse_sel: float = 0.03        # extra take-up logit per +1pp APR for applicants who will default
    amount_elast: float = 1.0        # take-up logit change per unit log(amount multiplier)
    pd_amount_elast: float = 0.3     # PD scales with multiplier ** this
    repeat_prob: float = 0.30
    discount: float = 0.12
    default_stress: float = 1.0
    stress_sd: float = 0.15


def load_population(n: int | None = 20_000, seed: int = 0) -> pd.DataFrame:
    df = pd.read_parquet(ROOT / "data" / "processed" / "scored_oot.parquet")
    df = df[df["funded"] == 1]
    if n and n < len(df):
        df = df.sample(n, random_state=seed)
    return df.reset_index(drop=True)


def horizon_pd(p7, term_y):
    return 1 - (1 - p7) ** (term_y / PD_WINDOW_Y)


def _logit(p):
    return np.log(p / (1 - p))


def _expit(x):
    return 1 / (1 + np.exp(-x))


def simulate(pop: pd.DataFrame, pol: Policy, a: Assumptions, n_draws: int = 0, seed: int = 0) -> dict:
    """Return point estimates (and P10/P50/P90 bands if n_draws > 0), all per 1,000 applications."""
    T, m = pol.term_y, pol.amount_mult
    if pol.tier == "routed":
        pd_inst = horizon_pd(pop["pd_instant"].to_numpy(), T)
        routed = (pd_inst >= pol.route_pd) | (pop["amount"].to_numpy() >= pol.route_amount)
        pd_dec = np.where(routed, horizon_pd(pop["pd_full"].to_numpy(), T), pd_inst)
        completion = np.where(routed, a.completion["full"], a.completion["instant"])
        uw = np.where(routed, a.uw_cost["full"], a.uw_cost["instant"])
    else:
        routed = np.zeros(len(pop), bool)
        pd_dec = horizon_pd(pop[f"pd_{pol.tier}"].to_numpy(), T)
        completion = np.full(len(pop), a.completion[pol.tier])
        uw = np.full(len(pop), a.uw_cost[pol.tier])
    lgd_bar = 0.77
    approve = pd_dec <= pol.pd_cutoff
    if pol.pricing == "flat":
        apr = np.full(len(pop), pol.flat_apr)
    else:
        apr = np.minimum(a.cof + pol.margin + pol.loss_load * pd_dec * lgd_bar / (BAL_FACTOR * T), pol.max_apr)

    # Realised 7-year default, scaled to the product horizon with a constant-hazard ratio.
    p_full = pop["pd_full"].to_numpy()
    d = pop["default"].to_numpy() * horizon_pd(p_full, T) / p_full * m ** a.pd_amount_elast
    lgd = pop["lgd"].to_numpy()
    amt = pop["amount"].to_numpy() * m
    gap_pp = (apr - a.ref_apr) * 100

    def run(beta, t0, stress, w_boot):
        D = np.clip(d * stress, 0, 1)
        logit_t = _logit(t0) + beta * gap_pp + a.amount_elast * np.log(m)
        # Adverse selection: at higher prices, the applicants who will default are more willing to accept.
        t_good, t_bad = _expit(logit_t), _expit(logit_t + a.adverse_sel * gap_pp)
        take = (1 - D) * t_good + D * t_bad
        w = completion * approve * take * w_boot
        revenue = apr * amt * BAL_FACTOR * T * (1 - 0.5 * D)
        loss = D * lgd * amt
        profit = revenue - a.cof * amt * BAL_FACTOR * T - loss - a.opex_per_loan
        exp_profit = (apr * (1 - 0.5 * pd_dec) - a.cof) * amt * BAL_FACTOR * T - pd_dec * lgd_bar * amt - a.opex_per_loan
        q = np.clip(a.repeat_prob * t_good / t0, 0, 0.95) * (1 + a.discount) ** -T
        clv = profit + (1 - D) * q / (1 - q) * exp_profit
        n_app = w_boot.sum()
        uw_total = (uw * completion * w_boot).sum()
        vol = (w * amt).sum()
        k = 1000 / n_app
        return {
            "approval_rate": (approve * w_boot).sum() / n_app,
            "conversion": w.sum() / n_app,
            "funded_volume": vol * k,
            "avg_apr": (w * apr).sum() / max(w.sum(), 1e-9),
            "loss_rate": (w * loss).sum() / max(vol, 1e-9),
            "default_rate": (w * D).sum() / max(w.sum(), 1e-9),
            "profit": ((w * profit).sum() - uw_total) * k,
            "clv": ((w * clv).sum() - uw_total) * k,
            "clv_per_customer": ((w * clv).sum() - uw_total) / max(w.sum(), 1e-9),
            "doc_request_rate": (routed * w_boot).sum() / n_app,
            "roa": ((w * profit).sum() - uw_total) / max(vol * BAL_FACTOR * T, 1e-9),
        }

    point = run(a.beta, a.base_takeup, a.default_stress, np.ones(len(pop)))
    if not n_draws:
        return {"point": point}
    rng = np.random.default_rng(seed)
    draws = [
        run(rng.normal(a.beta, a.beta_sd), float(np.clip(rng.normal(a.base_takeup, 0.05), 0.05, 0.98)),
            a.default_stress * rng.lognormal(0, a.stress_sd), rng.poisson(1.0, len(pop)).astype(float))
        for _ in range(n_draws)
    ]
    dd = pd.DataFrame(draws)
    return {"point": point, "p10": dd.quantile(0.1).to_dict(), "p50": dd.quantile(0.5).to_dict(),
            "p90": dd.quantile(0.9).to_dict(), "draws": dd}


def frontier_grid(pop, a: Assumptions, base: Policy, cutoffs=None, margins=None, tiers=("instant", "standard", "full", "routed")):
    cutoffs = cutoffs if cutoffs is not None else [0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.06, 0.08, 1.0]
    margins = margins if margins is not None else np.round(np.arange(0.0, 0.141, 0.01), 3)
    rows = []
    for tier in tiers:
        for c in cutoffs:
            for mg in margins:
                pol = replace(base, tier=tier, pd_cutoff=float(c), margin=float(mg), pricing="risk_based")
                rows.append({"tier": tier, "pd_cutoff": c, "margin": mg, **simulate(pop, pol, a)["point"]})
    g = pd.DataFrame(rows)
    g["pareto"] = pareto_mask(g["conversion"].to_numpy(), g["clv"].to_numpy())
    return g


def pareto_mask(x, y):
    """True where no other point has both higher x and higher y."""
    order = np.argsort(-x, kind="stable")
    mask = np.zeros(len(x), bool)
    best = -np.inf
    for i in order:
        if y[i] > best:
            mask[i], best = True, y[i]
    return mask


def power_two_prop(p1: float, p2: float, alpha=0.05, power=0.8) -> int:
    from statistics import NormalDist
    z = NormalDist().inv_cdf
    pbar = (p1 + p2) / 2
    num = z(1 - alpha / 2) * np.sqrt(2 * pbar * (1 - pbar)) + z(power) * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))
    return int(np.ceil(num ** 2 / (p1 - p2) ** 2))

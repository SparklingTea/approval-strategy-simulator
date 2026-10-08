"""Build the efficient frontier, compare policies with uncertainty bands, and save charts + numbers for the memo."""
import json
import sys
from dataclasses import replace
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from sim import Assumptions, Policy, frontier_grid, load_population, simulate  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
REP = ROOT / "reports"
TIER_COLORS = {"instant": "#2a78d6", "standard": "#eb6834", "full": "#1baf7a", "routed": "#eda100"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

BASELINE = Policy(tier="standard", pricing="flat", flat_apr=0.12, pd_cutoff=0.03)


def best_by_tier(grid):
    return grid.loc[grid.groupby("tier")["clv"].idxmax()].set_index("tier")


def breakeven_completion(pop, a, pol_docs, target_clv, tier_key="full"):
    """Lowest docs-step completion at which `pol_docs` matches `target_clv` (bisection)."""
    lo, hi = 0.05, 1.0
    for _ in range(30):
        mid = (lo + hi) / 2
        aa = replace(a, completion={**a.completion, tier_key: mid})
        if simulate(pop, pol_docs, aa)["point"]["clv"] >= target_clv:
            hi = mid
        else:
            lo = mid
    return hi


def plot_frontier(grid, rec, base_pt, path):
    fig, ax = plt.subplots(figsize=(8.5, 5.2), dpi=160)
    ax.scatter(grid["conversion"] * 100, grid["clv"] / 1e6, s=9, color="#c3c2b7", zorder=1, label="All grid policies")
    for tier, col in TIER_COLORS.items():
        g = grid[grid.tier == tier]
        f = g[g.pareto_tier].sort_values("conversion")
        ax.plot(f["conversion"] * 100, f["clv"] / 1e6, color=col, lw=2, zorder=3, label=f"{tier} frontier")
    ax.scatter([base_pt["conversion"] * 100], [base_pt["clv"] / 1e6], s=70, marker="s", color=INK, zorder=5)
    ax.annotate("Baseline (flat 12% APR, standard docs)", (base_pt["conversion"] * 100, base_pt["clv"] / 1e6),
                xytext=(8, -14), textcoords="offset points", fontsize=8, color=INK)
    ax.scatter([rec["conversion"] * 100], [rec["clv"] / 1e6], s=110, marker="*", color=INK, zorder=5)
    ax.annotate("Recommended", (rec["conversion"] * 100, rec["clv"] / 1e6), xytext=(8, 6),
                textcoords="offset points", fontsize=8, color=INK, weight="bold")
    ax.set_xlabel("Conversion: funded loans per application (%)", color=MUTED)
    ax.set_ylabel("Portfolio CLV per 1,000 applications ($m)", color=MUTED)
    ax.set_title("Efficient frontier: growth vs customer lifetime value", loc="left", color=INK, fontsize=11)
    ax.grid(color=GRID, lw=0.6)
    ax.set_ylim(bottom=max(grid["clv"].min() / 1e6, -2))
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_calibration(metrics, path):
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.4), dpi=160, sharey=True)
    for ax, (tier, m) in zip(axes, metrics["tiers"].items()):
        c = m["calibration"]
        pred, obs = [b["pred"] * 100 for b in c], [b["obs"] * 100 for b in c]
        lim = max(max(pred), max(obs)) * 1.1
        ax.plot([0, lim], [0, lim], color=MUTED, lw=1, ls="--")
        ax.plot(pred, obs, color=TIER_COLORS[tier], lw=2, marker="o", ms=4)
        ax.set_title(f"{tier}: AUC {m['auc']:.3f}, A/E {m['ae_ratio']:.2f}", fontsize=9, loc="left")
        ax.set_xlabel("Predicted 7y PD (%)", color=MUTED, fontsize=8)
        ax.grid(color=GRID, lw=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("Observed default rate (%)", color=MUTED, fontsize=8)
    fig.suptitle("Out-of-time calibration (FY2017-18 vintages, models trained on FY2010-16)", x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    REP.mkdir(exist_ok=True)
    metrics = json.loads((ROOT / "artifacts" / "metrics.json").read_text())
    pop = load_population()
    a = Assumptions()
    base = Policy()

    grid = frontier_grid(pop, a, base)
    grid["pareto_tier"] = False
    from sim import pareto_mask
    for t, g in grid.groupby("tier"):
        grid.loc[g.index, "pareto_tier"] = pareto_mask(g["conversion"].to_numpy(), g["clv"].to_numpy())
    best = best_by_tier(grid)

    rt = best.loc["routed"]
    route_scan = []
    for rp in (0.015, 0.02, 0.025, 0.03, 0.04):
        pol = replace(base, tier="routed", pd_cutoff=float(rt.pd_cutoff), margin=float(rt.margin), route_pd=rp, route_amount=1e9)
        route_scan.append({"route_pd": rp, **simulate(pop, pol, a)["point"]})

    top = best["clv"].idxmax()
    rec_pol = replace(base, tier=top, pd_cutoff=float(best.loc[top, "pd_cutoff"]), margin=float(best.loc[top, "margin"]))
    rec = simulate(pop, rec_pol, a, n_draws=300)
    bl = simulate(pop, BASELINE, a, n_draws=300)
    diff = rec["draws"]["clv"] - bl["draws"]["clv"]

    inst = best.loc["instant"]
    full_pol = replace(base, tier="full", pd_cutoff=float(best.loc["full", "pd_cutoff"]), margin=float(best.loc["full", "margin"]))
    be = breakeven_completion(pop, a, full_pol, float(inst["clv"]))
    no_friction = replace(a, completion={k: a.completion["instant"] for k in a.completion},
                          uw_cost={k: a.uw_cost["instant"] for k in a.uw_cost})
    voi = {t: simulate(pop, replace(rec_pol, tier=t), no_friction)["point"]["clv"] for t in ("instant", "standard", "full")}

    stress = {}
    for s in (1.0, 1.5, 2.0, 3.0):
        aa = replace(a, default_stress=s)
        g = frontier_grid(pop, aa, base, tiers=(top,))
        b = g.loc[g["clv"].idxmax()]
        stress[s] = {"pd_cutoff": float(b.pd_cutoff), "margin": float(b.margin), "clv": float(b.clv),
                     "conversion": float(b.conversion), "loss_rate": float(b.loss_rate),
                     "approval_rate": float(b.approval_rate)}

    beta_sens = {}
    for beta in (-0.1, -0.25, -0.4, -0.6):
        g = frontier_grid(pop, replace(a, beta=beta), base, tiers=(top,))
        b = g.loc[g["clv"].idxmax()]
        beta_sens[beta] = {"margin": float(b.margin), "avg_apr": float(b.avg_apr), "conversion": float(b.conversion), "clv": float(b.clv)}

    plot_frontier(grid, rec["point"], bl["point"], REP / "frontier.png")
    plot_calibration(metrics, REP / "calibration.png")
    grid.to_csv(REP / "frontier_grid.csv", index=False)

    def band(r):
        return {k: {"p10": r["p10"][k], "p50": r["p50"][k], "p90": r["p90"][k], "point": r["point"][k]} for k in r["point"]}

    out = {
        "best_by_tier": best.drop(columns=["pareto", "pareto_tier"]).to_dict(orient="index"),
        "recommended_policy": rec_pol.__dict__, "recommended": band(rec), "baseline": band(bl),
        "clv_uplift": {"p10": float(diff.quantile(0.1)), "p50": float(diff.quantile(0.5)), "p90": float(diff.quantile(0.9)),
                       "prob_positive": float((diff > 0).mean())},
        "full_docs_breakeven_completion": be, "value_of_information_no_friction": voi,
        "route_scan": route_scan, "stress": stress, "beta_sensitivity": beta_sens,
    }
    (REP / "results.json").write_text(json.dumps(out, indent=2, default=float))
    print(json.dumps({k: out[k] for k in ("best_by_tier", "recommended_policy", "clv_uplift", "full_docs_breakeven_completion",
                                          "value_of_information_no_friction", "stress", "beta_sensitivity")}, indent=1, default=float))
    print("rec", {k: (round(v["p10"], 3), round(v["point"], 3), round(v["p90"], 3)) for k, v in out["recommended"].items()})
    print("base", {k: (round(v["p10"], 3), round(v["point"], 3), round(v["p90"], 3)) for k, v in out["baseline"].items()})
    print("routes", [(r["route_pd"], round(r["clv"] / 1e6, 3), round(r["doc_request_rate"], 2), round(r["loss_rate"], 4)) for r in route_scan])


if __name__ == "__main__":
    main()

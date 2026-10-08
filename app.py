"""Approval & pricing strategy simulator -- streamlit run app.py"""
import json
import re
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))
from sim import Assumptions, Policy, frontier_grid, horizon_pd, load_population, pareto_mask, power_two_prop, simulate  # noqa: E402

ROOT = Path(__file__).parent
TIER_COLORS = {"instant": "#2a78d6", "standard": "#eb6834", "full": "#1baf7a", "routed": "#eda100"}
TIER_LABELS = {
    "instant": "Instant: application form only",
    "standard": "Standard: + company checks (age, franchise, headcount)",
    "full": "Full docs: + underwriting (collateral, lender risk spread, guarantee)",
    "routed": "Routed: instant, but ask for docs when instant PD is high",
}

st.set_page_config(page_title="Approval Strategy Simulator", layout="wide")


@st.cache_data
def population(n):
    return load_population(n)


@st.cache_data
def metrics():
    return json.loads((ROOT / "artifacts" / "metrics.json").read_text())


@st.cache_data
def grid(_pop, _a, _base, key):
    g = frontier_grid(_pop, _a, _base)
    g["pareto_tier"] = False
    for _, sub in g.groupby("tier"):
        g.loc[sub.index, "pareto_tier"] = pareto_mask(sub["conversion"].to_numpy(), sub["clv"].to_numpy())
    return g


def fmt_money(x, md=True):
    # Streamlit markdown treats paired "$" as LaTeX delimiters.
    d = r"\$" if md else "$"
    return f"{d}{x / 1e6:,.2f}m" if abs(x) >= 1e6 else f"{d}{x / 1e3:,.0f}k"


def band_metric(col, label, res, key, kind):
    f = {"pct": lambda v: f"{v * 100:.1f}%", "money": fmt_money, "pct2": lambda v: f"{v * 100:.2f}%"}[kind]
    col.metric(label, f(res["point"][key]).replace(r"\$", "$"))
    if "p10" in res:
        col.caption(f"P10–P90: {f(res['p10'][key])} – {f(res['p90'][key])}")


m = metrics()
pop = population(20_000)

# ---------------- Sidebar: policy + assumptions ----------------
st.sidebar.header("Policy levers")
tier = st.sidebar.radio("Information collected before deciding", list(TIER_LABELS), index=0,
                        format_func=lambda t: TIER_LABELS[t])
route_pd = 0.03
if tier == "routed":
    route_pd = st.sidebar.slider("Ask for documents if instant PD ≥ (%)", 0.5, 6.0, 3.0, 0.5) / 100
term_y = st.sidebar.slider("Loan term (years)", 0.5, 5.0, 2.0, 0.5)
pd_cut = st.sidebar.slider("Approval cutoff: max PD over loan term (%)", 0.5, 15.0, 15.0, 0.5,
                           help="Max is effectively 'no cutoff' for this population") / 100
pricing = st.sidebar.radio("Pricing policy", ["risk_based", "flat"],
                           format_func=lambda p: {"risk_based": "Risk-based", "flat": "Flat APR"}[p], horizontal=True)
if pricing == "flat":
    flat_apr = st.sidebar.slider("Flat APR (%)", 6.0, 30.0, 12.0, 0.5) / 100
    margin, max_apr = 0.08, 0.25
else:
    flat_apr = 0.12
    margin = st.sidebar.slider("Margin over cost of funds (pp)", 0.0, 15.0, 8.0, 0.5) / 100
    max_apr = st.sidebar.slider("APR cap (%)", 10.0, 40.0, 25.0, 1.0) / 100
amount_mult = st.sidebar.slider("Offer amount (× requested)", 0.5, 1.25, 1.0, 0.05)

with st.sidebar.expander("Assumptions (not identified in the data)"):
    base_takeup = st.slider("Take-up at 12% APR", 0.2, 0.95, 0.65, 0.05)
    beta = st.slider("Price sensitivity: take-up logit per +1pp APR", -1.0, 0.0, -0.25, 0.05)
    beta_sd = st.slider("…uncertainty (sd)", 0.0, 0.3, 0.08, 0.01)
    adverse = st.slider("Adverse selection: extra logit per +1pp for future defaulters", 0.0, 0.15, 0.03, 0.01)
    c_inst = st.slider("Completion: instant", 0.3, 1.0, 0.90, 0.01)
    c_std = st.slider("Completion: standard", 0.3, 1.0, 0.80, 0.01)
    c_full = st.slider("Completion: full docs", 0.3, 1.0, 0.60, 0.01)
    cof = st.slider("Cost of funds (%)", 1.0, 10.0, 5.0, 0.5) / 100
    repeat = st.slider("Repeat-loan probability (good payers)", 0.0, 0.8, 0.30, 0.05,
                       help=f"SBA data shows {m['repeat_3y']:.0%} within 3 years for bank loans; online SME lenders see far more.")
    stress = st.slider("Default stress (× observed)", 0.5, 4.0, 1.0, 0.25,
                       help="SBA borrowers were already screened by a bank; an online through-the-door population is riskier.")
    n_draws = st.slider("Monte Carlo draws", 50, 500, 200, 50)

a = Assumptions(cof=cof, base_takeup=base_takeup, beta=beta, beta_sd=beta_sd, adverse_sel=adverse,
                completion={"instant": c_inst, "standard": c_std, "full": c_full}, repeat_prob=repeat,
                default_stress=stress)
pol = Policy(tier=tier, pd_cutoff=pd_cut, pricing=pricing, flat_apr=flat_apr, margin=margin, max_apr=max_apr,
             amount_mult=amount_mult, term_y=term_y, route_pd=route_pd, route_amount=1e12)
baseline = Policy(tier="standard", pricing="flat", flat_apr=0.12, pd_cutoff=0.03, term_y=term_y)

st.title("Approval & pricing strategy simulator")
st.caption("New-customer SME lending · Decisions use each tier's predicted PD; outcomes use **realised** defaults and "
           "losses on out-of-time SBA 7(a) loans (FY2017–18). All figures per 1,000 applications, USD.")

tab_sim, tab_front, tab_model, tab_exp, tab_memo = st.tabs(
    ["Simulator", "Efficient frontier", "PD model & calibration", "Experiment design", "Memo"])

# ---------------- Simulator ----------------
with tab_sim:
    res = simulate(pop, pol, a, n_draws=n_draws)
    bl = simulate(pop, baseline, a, n_draws=n_draws, seed=0)
    c = st.columns(5)
    band_metric(c[0], "Conversion", res, "conversion", "pct")
    band_metric(c[1], "Loss rate (of funded)", res, "loss_rate", "pct2")
    band_metric(c[2], "Profit / 1k apps", res, "profit", "money")
    band_metric(c[3], "CLV / 1k apps", res, "clv", "money")
    band_metric(c[4], "Avg APR", res, "avg_apr", "pct")

    diff = res["draws"]["clv"] - bl["draws"]["clv"]
    st.markdown(
        f"**vs baseline** (standard docs, flat 12% APR, 3% cutoff): CLV "
        f"{'+' if diff.median() >= 0 else ''}{fmt_money(diff.median())} per 1k applications "
        f"(P10 {fmt_money(diff.quantile(.1))}, P90 {fmt_money(diff.quantile(.9))}); "
        f"P(better) = **{(diff > 0).mean():.0%}**.")

    left, right = st.columns([1, 1])
    with left:
        p = res["point"]
        completion = (a.completion["instant"] * (1 - p["doc_request_rate"]) + a.completion["full"] * p["doc_request_rate"]
                      if tier == "routed" else a.completion[tier])
        stages = ["Applications", "Completed", "Approved", "Funded"]
        vals = [1000, 1000 * completion, 1000 * completion * p["approval_rate"], 1000 * p["conversion"]]
        fig = go.Figure(go.Funnel(y=stages, x=vals, textinfo="value+percent initial", marker_color="#2a78d6"))
        fig.update_layout(title="Funnel per 1,000 applications", height=330, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)
    with right:
        fig = go.Figure()
        fig.add_histogram(x=bl["draws"]["clv"] / 1e6, name="Baseline", marker_color="#c3c2b7", opacity=0.8, nbinsx=30)
        fig.add_histogram(x=res["draws"]["clv"] / 1e6, name="This policy", marker_color="#2a78d6", opacity=0.75, nbinsx=30)
        fig.update_layout(barmode="overlay", title="CLV uncertainty (Monte Carlo)", xaxis_title="CLV per 1k apps ($m)",
                          height=330, margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("How the numbers are built"):
        st.markdown(r"""
- **Decision PD**: tier model's 7-year PD, converted to the loan term at constant hazard, $1-(1-p)^{T/7}$.
- **APR** (risk-based): cost of funds + margin + annualised expected loss $p_T \cdot \overline{LGD} / (0.55\,T)$, capped.
- **Take-up**: $\text{logit}\,t = \text{logit}\,t_0 + \beta\,\Delta APR_{pp} + \ln(\text{amount mult})$; future defaulters get an extra $\kappa\,\Delta APR_{pp}$ (adverse selection).
- **Outcome**: each loan's *realised* default and loss-given-default, scaled to the term; revenue on 55% average balance.
- **CLV**: first-loan profit + expected discounted repeat-loan profit for non-defaulters; repeat propensity falls with price.
- **Bands**: bootstrap over applicants × draws of price sensitivity, base take-up and a default-stress factor.
""")

# ---------------- Frontier ----------------
with tab_front:
    st.markdown("Every combination of **information tier × approval cutoff × margin** (risk-based pricing). "
                "Lines are each tier's Pareto frontier: no other policy in that tier has both higher conversion and higher CLV.")
    base_pol = replace(pol, pricing="risk_based")
    g = grid(pop, a, base_pol, repr((a, base_pol)))
    fig = go.Figure()
    fig.add_scatter(x=g.conversion * 100, y=g.clv / 1e6, mode="markers", marker=dict(color="#c3c2b7", size=5),
                    name="All policies", customdata=g[["tier", "pd_cutoff", "margin", "loss_rate"]].to_numpy(),
                    hovertemplate="%{customdata[0]} · cutoff %{customdata[1]:.1%} · margin %{customdata[2]:.0%}"
                                  "<br>conversion %{x:.1f}% · CLV $%{y:.2f}m · loss %{customdata[3]:.2%}<extra></extra>")
    for t, col in TIER_COLORS.items():
        f = g[(g.tier == t) & g.pareto_tier].sort_values("conversion")
        fig.add_scatter(x=f.conversion * 100, y=f.clv / 1e6, mode="lines+markers", line=dict(color=col, width=2),
                        marker=dict(size=6), name=t, customdata=f[["pd_cutoff", "margin", "loss_rate"]].to_numpy(),
                        hovertemplate=f"{t} · cutoff %{{customdata[0]:.1%}} · margin %{{customdata[1]:.0%}}"
                                      "<br>conversion %{x:.1f}% · CLV $%{y:.2f}m · loss %{customdata[2]:.2%}<extra></extra>")
    cur = res["point"]
    fig.add_scatter(x=[cur["conversion"] * 100], y=[cur["clv"] / 1e6], mode="markers", name="Your policy",
                    marker=dict(symbol="star", size=16, color="#0b0b0b"))
    fig.add_scatter(x=[bl["point"]["conversion"] * 100], y=[bl["point"]["clv"] / 1e6], mode="markers", name="Baseline",
                    marker=dict(symbol="square", size=11, color="#52514e"))
    fig.update_layout(xaxis_title="Conversion: funded per application (%)", yaxis_title="CLV per 1k applications ($m)",
                      height=520, margin=dict(l=10, r=10, t=10, b=10), yaxis=dict(range=[max(g.clv.min() / 1e6, -2), None]))
    st.plotly_chart(fig, use_container_width=True)
    best = g.loc[g.groupby("tier")["clv"].idxmax()][["tier", "pd_cutoff", "margin", "conversion", "approval_rate",
                                                       "loss_rate", "avg_apr", "clv"]]
    st.markdown("**CLV-maximising policy per tier**")
    st.dataframe(best.style.format({"pd_cutoff": "{:.1%}", "margin": "{:.0%}", "conversion": "{:.1%}",
                                    "approval_rate": "{:.1%}", "loss_rate": "{:.2%}", "avg_apr": "{:.1%}",
                                    "clv": lambda v: fmt_money(v, md=False)}), hide_index=True, use_container_width=True)

# ---------------- Model ----------------
with tab_model:
    st.markdown(f"Gradient-boosted PD models with isotonic calibration, trained on **{m['n_train']:,}** funded loans "
                f"(FY2010–16) and tested out-of-time on **{m['n_test']:,}** (FY2017–18). Target: charged off within 84 months.")
    rows = [{"tier": t, "AUC": v["auc"], "AUC 95% CI": f"{v['auc_ci'][0]:.3f}–{v['auc_ci'][1]:.3f}", "Gini": v["gini"],
             "Brier": v["brier"], "Actual/Expected": v["ae_ratio"], "features": ", ".join(v["features"])}
            for t, v in m["tiers"].items()]
    st.dataframe(pd.DataFrame(rows).style.format({"AUC": "{:.3f}", "Gini": "{:.3f}", "Brier": "{:.4f}", "Actual/Expected": "{:.2f}"}),
                 hide_index=True, use_container_width=True)
    fig = go.Figure()
    mx = 0
    for t, v in m["tiers"].items():
        x = [b["pred"] * 100 for b in v["calibration"]]
        y = [b["obs"] * 100 for b in v["calibration"]]
        mx = max(mx, *x, *y)
        fig.add_scatter(x=x, y=y, mode="lines+markers", name=t, line=dict(color=TIER_COLORS[t], width=2))
    fig.add_scatter(x=[0, mx], y=[0, mx], mode="lines", line=dict(color="#52514e", dash="dash", width=1), name="perfect")
    fig.update_layout(title="Out-of-time reliability (deciles)", xaxis_title="Predicted 7y PD (%)",
                      yaxis_title="Observed default rate (%)", height=420, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(f"""
**What this says**
- **Drift:** all tiers under-predict defaults on the later vintages by ~{(m['tiers']['instant']['ae_ratio'] - 1):.0%}, most at the risky end. A production PD needs a vintage-level recalibration loop, and pricing needs a buffer until it's in place.
- **Information value is uneven:** company checks add almost nothing over the form (AUC {m['tiers']['instant']['auc']:.3f} → {m['tiers']['standard']['auc']:.3f}); underwriting signals add more ({m['tiers']['full']['auc']:.3f}).
- **Leakage caught:** `TermInMonths` is rewritten when loans are restructured (91% of charge-offs have non-standard terms). Using it gave a spurious AUC of 0.95. It's excluded.
- **Price elasticity is not identifiable here:** a take-up regression on the observed price spread gives a *positive* coefficient ({m['takeup']['beta_spread']:+.3f} ± {m['takeup']['beta_se']:.3f}), because lenders price on risk and context. Price sensitivity is therefore an explicit assumption until it's measured with a randomised price test.
- **LGD** ≈ {m['lgd']['mean']:.0%} of the original amount ({m['lgd']['by_collateral']['1']:.0%} with collateral, {m['lgd']['by_collateral']['0']:.0%} without).
""")

# ---------------- Experiment design ----------------
with tab_exp:
    st.markdown("#### Sizing a test on a journey change")
    e1, e2, e3, e4 = st.columns(4)
    p0 = e1.number_input("Baseline conversion", 0.01, 0.99, round(float(res["point"]["conversion"]), 3), 0.01)
    mde = e2.number_input("Minimum detectable effect (pp)", 0.1, 20.0, 2.0, 0.1) / 100
    weekly = e3.number_input("Applications per week", 100, 100_000, 2_000, 100)
    split = e4.slider("Share of traffic in test", 0.1, 1.0, 1.0, 0.1)
    n = power_two_prop(p0, p0 + mde)
    st.metric("Sample per arm (α=5%, power=80%)", f"{n:,}", f"{2 * n / (weekly * split):.1f} weeks to run")

    st.markdown("#### Why loss outcomes need a different readout")
    ct = m["chargeoff_timing"]
    fig = go.Figure(go.Scatter(x=[int(k[:-1]) for k in ct], y=[v * 100 for v in ct.values()], mode="lines+markers",
                               line=dict(color="#2a78d6", width=2)))
    fig.update_layout(xaxis_title="Months since approval", yaxis_title="% of 7-year charge-offs already visible",
                      height=320, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(f"""
Only **{ct['12m']:.0%}** of eventual charge-offs are visible at 12 months and **{ct['24m']:.0%}** at 24 months. A test that changes *who* gets funded can't wait for realised losses:
1. **Primary metric:** conversion and funded volume, read out in weeks.
2. **Risk guardrail:** the model-predicted expected loss of the funded mix (scored with the *current* PD model, so both arms are measured the same way), plus early-arrears (30+ DPD) as the first observed signal.
3. **Price tests:** randomise APR in ±1–2pp bands *within* a risk grade. This identifies the take-up curve (β) that the observational data can't, and it's the single biggest driver of the recommended margin.
4. **Holdout:** keep a small (~2–5%) random approve/price holdout so future PD models can be trained without selection bias (reject inference).
""")

# ---------------- Memo ----------------
with tab_memo:
    memo = ROOT / "reports" / "memo.md"
    if memo.exists():
        buf = []
        for line in memo.read_text(encoding="utf-8").splitlines():
            img = re.fullmatch(r"!\[(.*)\]\((.+)\)", line.strip())
            if img:
                st.markdown("\n".join(buf))
                st.image(str(ROOT / "reports" / img.group(2)), caption=img.group(1))
                buf = []
            else:
                buf.append(line)
        st.markdown("\n".join(buf))
    else:
        st.info("Run `python src/report.py` and see reports/memo.md")

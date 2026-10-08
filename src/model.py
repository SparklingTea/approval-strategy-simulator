"""Fit calibrated PD models at three information tiers, estimate take-up and LGD, and score the out-of-time set."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
ART = ROOT / "artifacts"

# Each tier adds what a lender would learn by asking the applicant for more.
TIERS = {
    "instant": ["log_amount", "naics2", "state", "biz_type", "revolver", "variable_rate"],
    "standard": ["biz_age", "franchise", "log_jobs"],
    # Collateral, the lender's own risk spread and the guarantee level encode what the bank learned
    # from full underwriting (financials, statements) -- the closest public proxy for "documents".
    "full": ["collateral", "spread", "guar_pct", "express"],
}
TRAIN_FY, TEST_FY = range(2010, 2017), range(2017, 2019)
RNG = np.random.default_rng(7)


def tier_features() -> dict[str, list[str]]:
    feats, acc = {}, []
    for name, cols in TIERS.items():
        acc = acc + cols
        feats[name] = acc
    return feats


def calibration_bins(y, p, n=10):
    q = pd.qcut(p, n, labels=False, duplicates="drop")
    g = pd.DataFrame({"y": y, "p": p, "q": q}).groupby("q")
    return [{"pred": float(r.p), "obs": float(r.y), "n": int(r.n)}
            for r in g.agg(p=("p", "mean"), y=("y", "mean"), n=("y", "size")).itertuples()]


def boot_auc(y, p, n=200):
    idx = RNG.integers(0, len(y), (n, len(y)))
    vals = [roc_auc_score(y[i], p[i]) for i in idx]
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def fit_takeup(df: pd.DataFrame) -> dict:
    """Logit of take-up (not cancelled) on price spread, controlling for risk, size and vintage."""
    X = pd.get_dummies(df[["spread", "pd_instant", "log_amount", "revolver", "fy"]].astype({"fy": str}),
                       columns=["fy"], drop_first=True).astype(float)
    X["pd_instant"] = np.log(X["pd_instant"])
    y = df["funded"].to_numpy()
    lr = LogisticRegression(C=np.inf, max_iter=2000).fit(X, y)
    Xd = np.column_stack([np.ones(len(X)), X.to_numpy()])
    p = lr.predict_proba(X)[:, 1]
    cov = np.linalg.inv(Xd.T @ (Xd * (p * (1 - p))[:, None]))
    i = list(X.columns).index("spread") + 1
    return {"beta_spread": float(lr.coef_[0][i - 1]), "beta_se": float(np.sqrt(cov[i, i])),
            "base_takeup": float(y.mean()), "n": int(len(y))}


def main() -> None:
    df = pd.read_parquet(PROC / "loans.parquet")
    funded = df[df["funded"] == 1]
    tr, te = funded[funded.fy.isin(TRAIN_FY)], funded[funded.fy.isin(TEST_FY)]
    metrics = {"tiers": {}, "n_train": len(tr), "n_test": len(te)}
    scored = df[df.fy.isin(TEST_FY)].copy()

    for name, cols in tier_features().items():
        base = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, max_leaf_nodes=31,
                                              min_samples_leaf=200, l2_regularization=1.0,
                                              categorical_features="from_dtype", random_state=0)
        model = CalibratedClassifierCV(base, method="isotonic", cv=3).fit(tr[cols], tr["default"])
        p = np.clip(model.predict_proba(te[cols])[:, 1], 1e-4, 0.999)
        y = te["default"].to_numpy()
        auc = roc_auc_score(y, p)
        metrics["tiers"][name] = {
            "features": cols, "auc": auc, "auc_ci": boot_auc(y, p), "gini": 2 * auc - 1,
            "brier": brier_score_loss(y, p), "logloss": log_loss(y, p),
            "ae_ratio": float(y.mean() / p.mean()), "calibration": calibration_bins(y, p),
        }
        scored[f"pd_{name}"] = np.clip(model.predict_proba(scored[cols])[:, 1], 1e-4, 0.999)
        print(f"{name:9s} AUC {auc:.3f}  Brier {metrics['tiers'][name]['brier']:.4f}  A/E {metrics['tiers'][name]['ae_ratio']:.2f}")

    metrics["takeup"] = fit_takeup(scored)
    co = funded[funded["default"] == 1]
    lgd_boot = [co["lgd"].sample(frac=1, replace=True, random_state=s).mean() for s in range(200)]
    metrics["lgd"] = {"mean": float(co["lgd"].mean()), "se": float(np.std(lgd_boot)),
                      "by_collateral": co.groupby("collateral")["lgd"].mean().to_dict()}
    good = funded[(funded["default"] == 0) & funded.fy.between(2010, 2016)]
    metrics["repeat_3y"] = float(good["repeat_3y"].mean())
    t = co["months_to_co"].dropna()
    metrics["chargeoff_timing"] = {f"{m}m": float((t <= m).mean()) for m in (6, 12, 18, 24, 36, 48, 60)}
    metrics["default_rate_by_fy"] = funded.groupby("fy")["default"].mean().to_dict()

    ART.mkdir(exist_ok=True)
    (ART / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    scored.to_parquet(PROC / "scored_oot.parquet")
    print(json.dumps({k: metrics[k] for k in ("takeup", "lgd", "repeat_3y", "chargeoff_timing")}, indent=1, default=float))


if __name__ == "__main__":
    main()

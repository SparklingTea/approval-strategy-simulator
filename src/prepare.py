"""Clean the SBA 7(a) FOIA file into a modelling table for the approval simulator."""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "FOIA_7a_FY2010_FY2019_asof_260630.csv"
OUT = ROOT / "data" / "processed" / "loans.parquet"

# WSJ prime rate change dates; used to turn the quoted rate into a spread over base.
PRIME = pd.Series(
    [3.25, 3.50, 3.75, 4.00, 4.25, 4.50, 4.75, 5.00, 5.25, 5.50, 5.25, 5.00, 4.75],
    index=pd.to_datetime([
        "2008-12-16", "2015-12-17", "2016-12-15", "2017-03-16", "2017-06-15", "2017-12-14",
        "2018-03-22", "2018-06-14", "2018-09-27", "2018-12-20", "2019-08-01", "2019-09-19",
        "2019-10-31",
    ]),
)

AGE_MAP = {
    "Startup, Loan Funds will Open Business": "startup",
    "New, Less than 1 Year old": "<2y",
    "New Business or 2 years or less": "<2y",
    "Less than 3 years old but at least 2": "2-5y",
    "Less than 4 years old but at least 3": "2-5y",
    "Less than 5 years old but at least 4": "2-5y",
    "Existing or more than 2 years old": "2y+",
    "Existing, 5 or more years": "5y+",
    "Change of Ownership": "change_owner",
}

COLS = [
    "BorrName", "BorrZip", "BorrState", "LocationID", "GrossApproval", "SBAGuaranteedApproval",
    "ApprovalDate", "ApprovalFY", "FirstDisbursementDate", "ProcessingMethod", "InitialInterestRate",
    "FixedorVariableInterestInd", "TermInMonths", "NaicsCode", "FranchiseCode", "BusinessType",
    "BusinessAge", "LoanStatus", "ChargeOffDate", "GrossChargeOffAmount", "RevolverStatus",
    "JobsSupported", "CollateralInd",
]


def repeat_flag(df: pd.DataFrame, window_days: int = 3 * 365) -> pd.Series:
    """1 if the same borrower (name + zip) was approved for another 7(a) loan within the window."""
    key = df["BorrName"].str.lower().str.replace(r"[^a-z0-9]", "", regex=True) + "|" + df["BorrZip"].astype(str).str[:5]
    tmp = pd.DataFrame({"key": key, "d": df["ApprovalDate"]}).sort_values(["key", "d"])
    nxt = tmp.groupby("key")["d"].shift(-1)
    gap = (nxt - tmp["d"]).dt.days
    return ((gap > 30) & (gap <= window_days)).astype(int).reindex(df.index)


def main() -> None:
    df = pd.read_csv(RAW, usecols=COLS, low_memory=False,
                     parse_dates=["ApprovalDate", "FirstDisbursementDate", "ChargeOffDate"])
    df["LoanStatus"] = df["LoanStatus"].str.replace(" ", "")
    df["repeat_3y"] = repeat_flag(df)

    # TermInMonths is rewritten when a loan is restructured (91% of charge-offs carry non-standard terms),
    # so it is neither filtered on nor used as a feature. The target is instead charge-off within a fixed
    # 84-month window from approval, which every FY2010-2018 vintage has fully observed by the 2026-06 snapshot.
    m = (df["ApprovalFY"].between(2010, 2018) & (df["GrossApproval"] <= 500_000)
         & df["LoanStatus"].isin(["PIF", "CHGOFF", "EXEMPT", "CANCLD"]))
    df = df[m].copy()

    months_to_co = (df["ChargeOffDate"] - df["ApprovalDate"]).dt.days / 30.44
    defaulted = (df["LoanStatus"] == "CHGOFF") & (months_to_co <= 84)
    prime = PRIME.reindex(df["ApprovalDate"], method="ffill").to_numpy()
    out = pd.DataFrame({
        "fy": df["ApprovalFY"].astype(int),
        "amount": df["GrossApproval"],
        "log_amount": np.log(df["GrossApproval"]),
        "naics2": df["NaicsCode"].astype("Int64").astype(str).str[:2].replace({"<N": "NA"}),
        "state": df["BorrState"].fillna("NA"),
        "biz_type": df["BusinessType"].fillna("NA"),
        "revolver": (df["RevolverStatus"] == "Y").astype(int),
        "variable_rate": (df["FixedorVariableInterestInd"] == "V").astype(int),
        "biz_age": df["BusinessAge"].map(AGE_MAP).fillna("unknown"),
        "franchise": df["FranchiseCode"].notna().astype(int),
        "log_jobs": np.log1p(df["JobsSupported"].fillna(0)),
        "collateral": (df["CollateralInd"] == "Y").astype(int),
        "express": df["ProcessingMethod"].str.contains("Express", na=False).astype(int),
        "spread": df["InitialInterestRate"] - prime,
        "guar_pct": df["SBAGuaranteedApproval"] / df["GrossApproval"],
        "funded": (df["LoanStatus"] != "CANCLD").astype(int),
        "default": defaulted.astype(int),
        "lgd": (df["GrossChargeOffAmount"] / df["GrossApproval"]).clip(0, 1).where(defaulted, 0.0),
        "months_to_co": months_to_co.where(defaulted),
        "repeat_3y": df["repeat_3y"],
    })
    for c in ["naics2", "state", "biz_type", "biz_age"]:
        out[c] = out[c].astype("category")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.reset_index(drop=True).to_parquet(OUT)
    print(f"{len(out):,} loans -> {OUT}")
    print(out.groupby("fy")[["funded", "default"]].mean().round(3))


if __name__ == "__main__":
    main()

# Approval & pricing strategy simulator

Set the information you collect before deciding, the approval cutoff, the loan amount and the pricing policy. See expected conversion, losses and customer lifetime value with uncertainty bands, and where each policy sits on the efficient frontier.

Built on 398k public US SBA 7(a) small-business loans. Decisions use each information tier's *predicted* PD; outcomes use each loan's *realised* default and loss out of time. Model accuracy therefore shows up in money, not just in AUC.

- **Memo:** [reports/memo.md](reports/memo.md)
- **Frontier:** [reports/frontier.png](reports/frontier.png)
- **Calibration:** [reports/calibration.png](reports/calibration.png)

## Run

```bash
pip install -r requirements.txt
# data: SBA FOIA 7(a) FY2010-FY2019 file (public domain), ~255 MB
curl -L -o data/raw/FOIA_7a_FY2010_FY2019_asof_260630.csv https://data.sba.gov/sites/default/files/uploaded_resources/FOIA_7a_FY2010_FY2019_asof_260630.csv
python src/prepare.py   # clean -> data/processed/loans.parquet
python src/model.py     # PD models x3 tiers, take-up, LGD -> artifacts/metrics.json
python src/report.py    # frontier, policy comparison, charts -> reports/
streamlit run app.py
```

## Structure

| File | What it does |
|---|---|
| `src/prepare.py` | Filters to ≤$500k loans FY2010–18. Target: charge-off within 84 months. Builds features and the prime-rate spread. |
| `src/model.py` | Gradient boosting + isotonic calibration at three information tiers. Out-of-time AUC/Brier/A-E, take-up regression, LGD, repeat rate, charge-off timing. |
| `src/sim.py` | Unit-economics engine: take-up with price sensitivity and adverse selection, document-step friction, routed doc requests, CLV, Monte Carlo bands, frontier grid, power calculation. |
| `src/report.py` | Builds the frontier, recommended vs baseline policy, break-even completion, stress and price-sensitivity tests. |
| `app.py` | Streamlit UI with five tabs: Simulator, Frontier, PD model, Experiment design, Memo. |

## Modelling notes

- **Leakage:** `TermInMonths` is rewritten on restructure (91% of charge-offs have non-standard terms), so it is excluded. With it, AUC was a fake 0.95.
- **Fixed default window:** the target is charge-off within 84 months of approval. Every FY2010–18 vintage has had at least that long to play out by the June 2026 snapshot, so the target is not censored.
- **Not identifiable from observational data:** price sensitivity (the observed coefficient has the wrong sign) and document-step completion. Both are explicit, adjustable assumptions, and the memo proposes the experiments that would measure them.
- **Limitations:** US, bank-approved borrowers only (no rejects). USD. FY2017–18 vintages include COVID-era SBA payment relief.

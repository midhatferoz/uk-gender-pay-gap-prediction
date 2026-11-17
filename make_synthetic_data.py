"""Generate FAKE data with the same schema as the UK gender pay gap files.

Purpose: smoke-test the pipeline (and let you develop offline). The numbers are
invented, so NEVER report results obtained from this data in the paper.

    python make_synthetic_data.py            # writes data/synthetic/gpg_<year>.csv
    python run_pipeline.py --raw-dir data/synthetic --tag smoke_test
"""
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent / "data" / "synthetic"
YEARS = range(2017, 2026)
N_EMPLOYERS = 4000
SIZE_LABELS = ["Less than 250", "250 to 499", "500 to 999", "1000 to 4999",
               "5000 to 19,999", "20,000 or more"]
SIC = {
    "fashion": ["47710", "47721", "14110", "14131", "46420", "47710, 47910"],
    "retail_other": ["47110", "47190", "47730", "47910", "47520"],
    "other": ["62012", "86101", "85200", "41100", "70229", "84110", "62012\n70229"],
}
MU = {"fashion": 8.0, "retail_other": 9.0, "other": 13.0}


def make():
    rng = np.random.default_rng(0)
    OUT.mkdir(parents=True, exist_ok=True)
    sector = rng.choice(["fashion", "retail_other", "other"], N_EMPLOYERS, p=[0.05, 0.13, 0.82])
    emp_id = 10_000 + np.arange(N_EMPLOYERS)
    size = rng.choice(6, N_EMPLOYERS, p=[0.03, 0.45, 0.25, 0.20, 0.05, 0.02])
    start = np.where(rng.random(N_EMPLOYERS) < 0.85, 2017, rng.integers(2017, 2023, N_EMPLOYERS))
    drop = np.where(rng.random(N_EMPLOYERS) < 0.2, rng.integers(2020, 2026, N_EMPLOYERS), 3000)
    sic = [rng.choice(SIC[s]) for s in sector]
    mu = np.array([MU[s] for s in sector])

    gap = mu + rng.normal(0, 9, N_EMPLOYERS)
    ftop = np.clip(35 - 0.5 * (gap - mu) + rng.normal(0, 8, N_EMPLOYERS), 5, 95)
    sig = np.where(size == 1, 3.5, 2.5)

    for year in YEARS:
        if year > 2017:
            change = (-0.25 * (gap - mu) + 0.08 * (ftop - 35)
                      + rng.normal(0, 1, N_EMPLOYERS) * sig)
            gap = gap + change
            ftop = np.clip(ftop - 0.4 * change + rng.normal(0, 3, N_EMPLOYERS), 5, 95)
        reports = (start <= year) & (drop > year)
        if year == 2019:                       # reporting was suspended: lower coverage
            reports &= rng.random(N_EMPLOYERS) < 0.55
        n = int(reports.sum())
        g = gap[reports]
        fl = np.clip(62 - 0.2 * g + rng.normal(0, 5, n), 5, 95)
        flm = np.clip(55 + rng.normal(0, 6, n), 5, 95)
        fum = np.clip(45 - 0.1 * g + rng.normal(0, 6, n), 5, 95)
        ft = ftop[reports]
        has_bonus = rng.random(n) > 0.25
        bonus_mean = np.where(has_bonus, 0.8 * g + rng.normal(0, 15, n), np.nan)
        bonus_med = np.where(has_bonus, 0.7 * g + rng.normal(0, 15, n), np.nan)
        due = datetime(year + 1, 4, 4)
        days = np.where(rng.random(n) < 0.1, rng.integers(1, 200, n), -rng.integers(0, 30, n))
        submitted = [due + timedelta(days=int(d), hours=int(rng.integers(0, 23))) for d in days]
        df = pd.DataFrame({
            "EmployerName": [f"Employer {i}" for i in emp_id[reports]],
            "EmployerId" if year != 2018 else "Employer ID": emp_id[reports],
            "Address": "1 Example Street", "PostCode": "AB1 2CD",
            "CompanyNumber": [f"{i:08d}" for i in emp_id[reports]],
            "SicCodes": np.array(sic, dtype=object)[reports],
            "DiffMeanHourlyPercent": np.round(g + rng.normal(0, 1.5, n), 1),
            "DiffMedianHourlyPercent": np.round(g, 1),
            "DiffMeanBonusPercent": np.round(bonus_mean, 1),
            "DiffMedianBonusPercent": np.round(bonus_med, 1),
            "MaleBonusPercent": np.where(has_bonus, np.round(rng.uniform(5, 60, n), 1), 0.0),
            "FemaleBonusPercent": np.where(has_bonus, np.round(rng.uniform(5, 60, n), 1), 0.0),
            "MaleLowerQuartile": np.round(100 - fl, 1), "FemaleLowerQuartile": np.round(fl, 1),
            "MaleLowerMiddleQuartile": np.round(100 - flm, 1),
            "FemaleLowerMiddleQuartile": np.round(flm, 1),
            "MaleUpperMiddleQuartile": np.round(100 - fum, 1),
            "FemaleUpperMiddleQuartile": np.round(fum, 1),
            "MaleTopQuartile": np.round(100 - ft, 1), "FemaleTopQuartile": np.round(ft, 1),
            "CompanyLinkToGPGInfo": "", "ResponsiblePerson": "A Person",
            "EmployerSize": np.array(SIZE_LABELS, dtype=object)[size[reports]],
            "CurrentName": [f"Employer {i}" for i in emp_id[reports]],
            "SubmittedAfterTheDeadline": np.where(days > 0, "TRUE", "FALSE"),
            "DueDate": due.strftime("%d/%m/%Y %H:%M"),
            "DateSubmitted": [s.strftime("%d/%m/%Y %H:%M") for s in submitted],
        })
        # inject messiness the cleaner must handle
        dup = df.sample(frac=0.01, random_state=year).copy()
        dup["DateSubmitted"] = (due - timedelta(days=40)).strftime("%d/%m/%Y %H:%M")
        bad = df.sample(frac=0.003, random_state=year + 1).index
        df.loc[bad, "MaleTopQuartile"] = 10.0                     # quartile does not sum to 100
        df.loc[df.sample(frac=0.02, random_state=year + 2).index, "EmployerSize"] = "Not Provided"
        pd.concat([df, dup]).to_csv(OUT / f"gpg_{year}.csv", index=False)
    print(f"Wrote synthetic files to {OUT}")


if __name__ == "__main__":
    make()

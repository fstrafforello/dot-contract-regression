"""Turn raw DOT contract transactions into the two analysis tables.

awards.parquet   one row per definitive contract / order / purchase order. Features describe the award as
                 signed (the base transaction, modification 0); outcomes summarise everything that happened
                 to it afterwards (modifications, ceiling growth, terminations).
monthly.parquet  DOT obligations and new awards per calendar month, for forecasting.

Usage:
    python -m src.build
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
OUT_DIR = ROOT / "data" / "processed"

# Last day covered by the archive; awards are followed up to here.
DATA_CUTOFF = pd.Timestamp("2026-09-30")
# Awards signed in these fiscal years have at least three years of follow-up.
FIRST_BASE_FY, LAST_BASE_FY = 2013, 2023
# Ignore awards whose initial ceiling is below the micro-purchase range; their ratios are mostly noise.
MIN_INITIAL_CEILING = 25_000
# A modification counts as a ceiling increase only if it adds at least this share of the initial ceiling.
MIN_INCREASE_SHARE = 0.01

SCOPE_ACTIONS = {
    "CHANGE ORDER",
    "SUPPLEMENTAL AGREEMENT FOR WORK WITHIN SCOPE",
    "ADDITIONAL WORK (NEW AGREEMENT, JUSTIFICATION REQUIRED)",
    "DEFINITIZE CHANGE ORDER",
}
NUMERIC = [
    "federal_action_obligation", "base_and_exercised_options_value", "base_and_all_options_value",
    "number_of_offers_received",
]
DATES = [
    "action_date", "solicitation_date", "period_of_performance_start_date",
    "period_of_performance_current_end_date", "period_of_performance_potential_end_date",
]


def load_transactions() -> pd.DataFrame:
    files = sorted(RAW_DIR.glob("dot_contracts_fy*.parquet"))
    if not files:
        raise FileNotFoundError("No raw files found. Run `python -m src.download` first.")
    df = pd.concat((pd.read_parquet(f) for f in files), ignore_index=True)
    df = df.drop_duplicates("contract_transaction_unique_key")
    for col in NUMERIC:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in DATES:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    df["action_date_fiscal_year"] = pd.to_numeric(df["action_date_fiscal_year"], errors="coerce")
    return df


def fiscal_year(dates: pd.Series) -> pd.Series:
    return dates.dt.year + (dates.dt.month >= 10).astype(int)


def build_monthly(tx: pd.DataFrame) -> pd.DataFrame:
    tx = tx[tx["action_date"].notna()]
    month = tx["action_date"].dt.to_period("M").dt.to_timestamp()
    is_new = (tx["modification_number"] == "0") & (tx["award_or_idv_flag"] == "AWARD")
    monthly = pd.DataFrame({
        "obligations": tx.groupby(month)["federal_action_obligation"].sum(),
        "new_awards": is_new.groupby(month).sum(),
        "transactions": tx.groupby(month).size(),
    })
    monthly.index.name = "month"
    # Keep whole fiscal years only.
    monthly = monthly.loc["2011-10-01":DATA_CUTOFF]
    monthly["fiscal_year"] = fiscal_year(monthly.index.to_series())
    monthly["fiscal_month"] = (monthly.index.month - 10) % 12 + 1
    return monthly.reset_index()


def build_awards(tx: pd.DataFrame) -> pd.DataFrame:
    tx = tx[tx["award_or_idv_flag"] == "AWARD"].sort_values(["contract_award_unique_key", "action_date"])

    # Base transaction: the earliest modification 0 for each award.
    base = tx[tx["modification_number"] == "0"].drop_duplicates("contract_award_unique_key")
    base = base[base["action_date_fiscal_year"].between(FIRST_BASE_FY, LAST_BASE_FY)]
    base = base[base["base_and_all_options_value"] >= MIN_INITIAL_CEILING]

    keys = set(base["contract_award_unique_key"])
    tx = tx[tx["contract_award_unique_key"].isin(keys)]
    mods = tx[tx["modification_number"] != "0"]

    award = base.set_index("contract_award_unique_key")
    initial = award["base_and_all_options_value"]
    start = award["period_of_performance_start_date"].fillna(award["action_date"])
    planned_end = award["period_of_performance_potential_end_date"].fillna(
        award["period_of_performance_current_end_date"])

    out = pd.DataFrame(index=award.index)
    out["award_id_piid"] = award["award_id_piid"]
    out["signed_date"] = award["action_date"]
    out["start_date"] = start
    out["planned_end_date"] = planned_end
    out["planned_days"] = (planned_end - start).dt.days
    out["base_fiscal_year"] = award["action_date_fiscal_year"].astype(int)
    out["fiscal_month"] = (award["action_date"].dt.month - 10) % 12 + 1
    out["days_to_fy_end"] = (
        pd.to_datetime(award["action_date_fiscal_year"].astype(int).astype(str) + "-09-30")
        - award["action_date"]).dt.days
    out["solicitation_days"] = (award["action_date"] - award["solicitation_date"]).dt.days

    out["initial_ceiling"] = initial
    out["initial_obligation"] = award["federal_action_obligation"]
    for col in [
        "awarding_sub_agency_name", "awarding_office_code", "awarding_office_name", "recipient_uei",
        "recipient_parent_uei", "recipient_name", "award_type", "type_of_contract_pricing",
        "product_or_service_code", "naics_code", "extent_competed", "solicitation_procedures",
        "type_of_set_aside", "number_of_offers_received",
        "contracting_officers_determination_of_business_size", "commercial_item_acquisition_procedures",
        "performance_based_service_acquisition", "undefinitized_action", "multi_year_contract",
        "contract_financing", "primary_place_of_performance_state_code",
    ]:
        out[col] = award[col]

    # Outcomes from the modification history.
    g = mods.groupby("contract_award_unique_key")
    out["n_mods"] = g.size().reindex(out.index, fill_value=0)
    out["n_scope_mods"] = (
        mods["action_type"].isin(SCOPE_ACTIONS).groupby(mods["contract_award_unique_key"]).sum()
        .reindex(out.index, fill_value=0))
    out["final_ceiling"] = initial + g["base_and_all_options_value"].sum().reindex(out.index, fill_value=0)
    out["final_obligation"] = (
        tx.groupby("contract_award_unique_key")["federal_action_obligation"].sum().reindex(out.index))
    out["growth_ratio"] = out["final_ceiling"] / initial

    final_end = tx.groupby("contract_award_unique_key")["period_of_performance_current_end_date"].max()
    out["final_end_date"] = final_end.reindex(out.index)
    out["schedule_slip_days"] = (out["final_end_date"] - planned_end).dt.days

    increases = mods[mods["base_and_all_options_value"] >= MIN_INCREASE_SHARE *
                     mods["contract_award_unique_key"].map(initial)]
    out["first_increase_date"] = increases.groupby("contract_award_unique_key")["action_date"].min()

    is_term = mods["action_type"].fillna("").str.contains("TERMINATE|CANCELLATION")
    out["terminated"] = is_term.groupby(mods["contract_award_unique_key"]).any().reindex(
        out.index, fill_value=False)
    out["termination_date"] = mods[is_term].groupby("contract_award_unique_key")["action_date"].min()

    # Observation window for time-to-event analyses: from start until the work ends or the data stops.
    window_end = out["final_end_date"].fillna(planned_end).clip(upper=DATA_CUTOFF)
    out["window_end_date"] = window_end
    out["observed_days"] = (window_end - start).dt.days

    out = out[(out["planned_days"] > 0) & (out["growth_ratio"] > 0)]
    return out.reset_index()


def main():
    tx = load_transactions()
    print(f"{len(tx):,} transactions loaded")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    monthly = build_monthly(tx)
    monthly.to_parquet(OUT_DIR / "monthly.parquet", index=False)
    print(f"monthly: {len(monthly)} months")

    awards = build_awards(tx)
    awards.to_parquet(OUT_DIR / "awards.parquet", index=False)
    print(f"awards: {len(awards):,} rows")


if __name__ == "__main__":
    main()

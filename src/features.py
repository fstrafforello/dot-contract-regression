"""Shared feature engineering so every notebook models the same award-level variables."""
from pathlib import Path

import numpy as np
import pandas as pd

PROCESSED = Path(__file__).resolve().parents[1] / "data" / "processed"


# Right-hand side shared by the award-level models (patsy / statsmodels formula syntax).
# Reference levels are the most common category, so each coefficient reads as "relative to a typical award".
RHS = (
    "log_ceiling + log_planned_years"
    " + C(pricing, Treatment('firm_fixed'))"
    " + C(competition, Treatment('full_open')) + single_offer + log_offers + offers_missing"
    " + small_business + commercial_item + performance_based"
    " + C(award_kind, Treatment('delivery_order'))"
    " + C(psc_group, Treatment('products'))"
    " + C(sub_agency, Treatment('FAA'))"
    " + log_solicitation_days + solicitation_missing"
    " + september + final_two_weeks"
    " + C(base_fiscal_year)"
)


def load_awards() -> pd.DataFrame:
    return pd.read_parquet(PROCESSED / "awards.parquet")


def load_monthly() -> pd.DataFrame:
    return pd.read_parquet(PROCESSED / "monthly.parquet")


def _pricing_group(s: pd.Series) -> pd.Series:
    s = s.fillna("")
    return np.select(
        [s.eq("FIRM FIXED PRICE"),
         s.isin(["TIME AND MATERIALS", "LABOR HOURS"]),
         s.str.startswith("COST"),
         s.str.startswith("FIXED PRICE")],
        ["firm_fixed", "time_materials", "cost_reimbursement", "fixed_other"],
        default="other",
    )


def _competition_group(s: pd.Series) -> pd.Series:
    s = s.fillna("")
    return np.select(
        [s.eq("FULL AND OPEN COMPETITION"),
         s.eq("FULL AND OPEN COMPETITION AFTER EXCLUSION OF SOURCES"),
         s.eq("COMPETED UNDER SAP"),
         s.isin(["NOT COMPETED", "NOT COMPETED UNDER SAP", "NOT AVAILABLE FOR COMPETITION",
                 "NON-COMPETITIVE DELIVERY ORDER"])],
        ["full_open", "set_aside", "simplified", "not_competed"],
        default="other",
    )


PSC_GROUPS = {
    "R": "professional_services", "D": "it_services", "A": "research", "Y": "construction",
    "Z": "facility_maintenance", "J": "equipment_maintenance", "C": "architect_engineering",
    "S": "utilities_housekeeping", "U": "education_training", "B": "studies",
}


def _psc_group(s: pd.Series) -> pd.Series:
    first = s.fillna("?").str[0]
    return np.where(first.str.isdigit(), "products", first.map(PSC_GROUPS).fillna("other_services"))


def _sub_agency(s: pd.Series) -> pd.Series:
    short = {
        "Federal Aviation Administration": "FAA",
        "Federal Highway Administration": "FHWA",
        "Maritime Administration": "MARAD",
        "Immediate Office of the Secretary of Transportation": "OST",
        "Federal Transit Administration": "FTA",
        "National Highway Traffic Safety Administration": "NHTSA",
    }
    return s.map(short).fillna("other")


def model_frame(awards: pd.DataFrame) -> pd.DataFrame:
    """Award-level modelling table: outcomes plus cleaned, low-cardinality predictors."""
    df = awards.copy()
    offers = pd.to_numeric(df["number_of_offers_received"], errors="coerce")

    df["log_ceiling"] = np.log(df["initial_ceiling"])
    df["log_planned_years"] = np.log(df["planned_days"] / 365.25)
    df["pricing"] = _pricing_group(df["type_of_contract_pricing"])
    df["competition"] = _competition_group(df["extent_competed"])
    df["offers"] = offers.clip(upper=50)
    df["single_offer"] = (offers == 1).astype(int)
    df["offers_missing"] = offers.isna().astype(int)
    df["log_offers"] = np.log1p(df["offers"].fillna(df["offers"].median()))
    df["small_business"] = (df["contracting_officers_determination_of_business_size"] == "SMALL BUSINESS").astype(int)
    # FY2013-FY2015 records use short codes for the same award types.
    kind = df["award_type"].replace({"DO": "DELIVERY ORDER", "PO": "PURCHASE ORDER",
                                     "DCA": "DEFINITIVE CONTRACT", "BPA": "BPA CALL"})
    df["award_kind"] = kind.str.lower().str.replace(" ", "_").fillna("unknown")
    df["psc_group"] = _psc_group(df["product_or_service_code"])
    df["naics2"] = df["naics_code"].fillna("00").str[:2]
    df["sub_agency"] = _sub_agency(df["awarding_sub_agency_name"])
    df["commercial_item"] = df["commercial_item_acquisition_procedures"].fillna("").str.startswith("COMMERCIAL").astype(int)
    df["performance_based"] = df["performance_based_service_acquisition"].fillna("").str.startswith("YES").astype(int)
    df["log_solicitation_days"] = np.log1p(df["solicitation_days"].clip(lower=0))
    df["solicitation_missing"] = df["solicitation_days"].isna().astype(int)
    df["log_solicitation_days"] = df["log_solicitation_days"].fillna(0)

    # Year-end rush: signed in September, and in the last two weeks of the fiscal year.
    df["september"] = (df["fiscal_month"] == 12).astype(int)
    df["final_two_weeks"] = (df["days_to_fy_end"] <= 14).astype(int)

    df["log_growth"] = np.log(df["growth_ratio"])
    df["any_growth"] = (df["growth_ratio"] > 1 + 1e-6).astype(int)
    df["office"] = df["awarding_office_code"].fillna("unknown")

    # Time to first ceiling increase, censored at the end of the observation window.
    days_to_increase = (df["first_increase_date"] - df["start_date"]).dt.days.clip(lower=0)
    df["increase_event"] = (days_to_increase <= df["observed_days"]).astype(int)
    df["increase_days"] = np.where(df["increase_event"] == 1, days_to_increase, df["observed_days"])
    return df

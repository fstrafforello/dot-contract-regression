"""Download Department of Transportation contract transactions from the USAspending award data archive.

The archive publishes one zip per agency per fiscal year (agency 069 = DOT, which includes the FAA).
Each zip holds ~300 columns; only the ones this project uses are kept, saved as one parquet per fiscal year.

Usage:
    python -m src.download                 # FY2012-FY2026
    python -m src.download --first 2015 --last 2020
"""
import argparse
import io
import re
import zipfile
from pathlib import Path

import pandas as pd
import requests

ARCHIVE = "https://files.usaspending.gov/award_data_archive/"
AGENCY = "069"
RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"

COLUMNS = [
    "contract_transaction_unique_key", "contract_award_unique_key", "award_id_piid",
    "modification_number", "transaction_number",
    "federal_action_obligation", "base_and_exercised_options_value", "base_and_all_options_value",
    "current_total_value_of_award", "potential_total_value_of_award",
    "action_date", "action_date_fiscal_year", "solicitation_date",
    "period_of_performance_start_date", "period_of_performance_current_end_date",
    "period_of_performance_potential_end_date",
    "awarding_sub_agency_name", "awarding_office_code", "awarding_office_name",
    "recipient_uei", "recipient_parent_uei", "recipient_name",
    "award_or_idv_flag", "award_type", "type_of_contract_pricing", "action_type_code", "action_type",
    "product_or_service_code", "naics_code", "extent_competed", "solicitation_procedures",
    "type_of_set_aside", "number_of_offers_received",
    "contracting_officers_determination_of_business_size",
    "commercial_item_acquisition_procedures", "performance_based_service_acquisition",
    "undefinitized_action", "multi_year_contract", "contract_financing",
    "primary_place_of_performance_state_code",
]


def latest_archive_key(fiscal_year: int) -> str:
    """Return the newest 'Full' contracts zip for one fiscal year (file names carry a publish date)."""
    prefix = f"FY{fiscal_year}_{AGENCY}_Contracts_Full_"
    listing = requests.get(ARCHIVE, params={"prefix": prefix}, timeout=60).text
    keys = re.findall(r"<Key>([^<]+\.zip)</Key>", listing)
    if not keys:
        raise FileNotFoundError(f"No archive found for FY{fiscal_year}")
    return sorted(keys)[-1]


def download_year(fiscal_year: int, overwrite: bool = False) -> Path:
    out = RAW_DIR / f"dot_contracts_fy{fiscal_year}.parquet"
    if out.exists() and not overwrite:
        print(f"FY{fiscal_year}: already downloaded")
        return out

    key = latest_archive_key(fiscal_year)
    print(f"FY{fiscal_year}: downloading {key}")
    payload = requests.get(ARCHIVE + key, timeout=600).content

    frames = []
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        for name in sorted(n for n in zf.namelist() if n.endswith(".csv")):
            with zf.open(name) as fh:
                frames.append(pd.read_csv(fh, usecols=lambda c: c in COLUMNS, dtype=str))
    df = pd.concat(frames, ignore_index=True)

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"FY{fiscal_year}: {len(df):,} transactions -> {out.name}")
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--first", type=int, default=2012)
    parser.add_argument("--last", type=int, default=2026)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    for fy in range(args.first, args.last + 1):
        download_year(fy, overwrite=args.overwrite)


if __name__ == "__main__":
    main()

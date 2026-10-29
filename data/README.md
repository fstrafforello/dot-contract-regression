# Data

Nothing in this folder is committed. Recreate it with:

```bash
python -m src.download   # ~3 MB parquet per fiscal year into data/raw/ (FY2012-FY2026, ~15 min)
python -m src.build      # data/processed/awards.parquet and monthly.parquet
```

**Source:** [USAspending.gov award data archive](https://files.usaspending.gov/award_data_archive/), "Contracts Full"
files for agency `069` (Department of Transportation). One file per fiscal year, one row per contract action
(the base award and every modification), as reported to FPDS. 40 of the ~300 published columns are kept.

## awards.parquet: one row per award

Definitive contracts, delivery orders, purchase orders and BPA calls with a base transaction (modification `0`) in
FY2013-FY2023 and an initial ceiling of at least $25,000. Indefinite-delivery vehicles (IDVs) are excluded; their
task orders are included as delivery orders. Follow-up runs to 30 September 2026.

| Column | Meaning |
|---|---|
| `initial_ceiling` | Base and all options value on the base transaction |
| `final_ceiling` | Initial ceiling plus the change in base-and-all-options value from every later modification |
| `growth_ratio` | `final_ceiling / initial_ceiling` |
| `n_mods`, `n_scope_mods` | Number of modifications; of which change orders, in-scope supplemental agreements, additional work, definitizations |
| `planned_days` | Start to potential end date (with options) at signing |
| `first_increase_date` | First modification adding at least 1% of the initial ceiling |
| `terminated`, `termination_date` | Any termination for convenience, default or cause, or legal cancellation |
| `observed_days` | Start to the earlier of the final end date and the data cutoff (the survival window) |

Award-level totals published on individual transaction rows are snapshots and are not used. Ceilings are rebuilt
from the transaction history across all fiscal-year files.

## monthly.parquet

Obligations (`federal_action_obligation` summed), new awards and transaction counts per calendar month,
October 2011 to September 2026.

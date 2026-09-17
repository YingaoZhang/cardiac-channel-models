from __future__ import annotations

import json
import math
import time
from collections import defaultdict
from pathlib import Path

import httpx
import pandas as pd
from rdkit import Chem
from rdkit.Chem import inchi


API_ROOT = "https://www.ebi.ac.uk/chembl/api/data"
QUERY = (
    f"{API_ROOT}/activity.json?target_chembl_id=CHEMBL240"
    "&standard_type=IC50&standard_relation=%3D&standard_units=nM&limit=1000"
)


def fetch_json(client: httpx.Client, url: str) -> dict:
    for attempt in range(5):
        try:
            response = client.get(url)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError:
            if attempt == 4:
                raise
            time.sleep(2**attempt)
    raise RuntimeError("unreachable")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    rows_by_key: dict[str, list[dict]] = defaultdict(list)
    raw_activity_count = 0
    url: str | None = QUERY

    with httpx.Client(timeout=60, follow_redirects=True) as client:
        status = fetch_json(client, f"{API_ROOT}/status.json")
        while url:
            payload = fetch_json(client, url)
            for activity in payload["activities"]:
                raw_activity_count += 1
                if activity.get("data_validity_comment") not in (None, "Manually validated"):
                    continue
                smiles = activity.get("canonical_smiles")
                try:
                    value_nm = float(activity["standard_value"])
                except (TypeError, ValueError, KeyError):
                    continue
                if not smiles or not math.isfinite(value_nm) or value_nm <= 0:
                    continue
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    continue
                canonical = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
                key = inchi.MolToInchiKey(mol)
                rows_by_key[key].append(
                    {
                        "smiles": canonical,
                        "value_nm": value_nm,
                        "molecule_chembl_id": activity.get("molecule_chembl_id"),
                        "document_chembl_id": activity.get("document_chembl_id"),
                    }
                )
            next_path = payload["page_meta"].get("next")
            url = f"https://www.ebi.ac.uk{next_path}" if next_path else None
            print(f"downloaded {raw_activity_count} activities", flush=True)

    curated = []
    for key, measurements in rows_by_key.items():
        values = pd.Series([item["value_nm"] for item in measurements], dtype=float)
        median_nm = float(values.median())
        pic50 = 9.0 - math.log10(median_nm)
        first = measurements[0]
        curated.append(
            {
                "pIC50": pic50,
                "Source": f"{status['chembl_db_version']} CHEMBL240",
                "Activity": "Blocker" if pic50 >= 5.0 else "Non-blocker",
                "Activity_Label": "Blocker" if pic50 >= 5.0 else "Non-blocker",
                "Compound_ID": first["molecule_chembl_id"],
                "SMILES": first["smiles"],
                "InChIKey": key,
                "IC50_nM_median": median_nm,
                "Measurement_Count": len(measurements),
            }
        )

    frame = pd.DataFrame(curated).sort_values("pIC50", ascending=False)
    destination = root / "data" / "dev" / "herg_dev.csv"
    frame.to_csv(destination, index=False)
    provenance = {
        "source": "ChEMBL REST API",
        "chembl_version": status["chembl_db_version"],
        "release_date": status["chembl_release_date"],
        "target": "CHEMBL240 (KCNH2/hERG)",
        "filters": {
            "standard_type": "IC50",
            "standard_relation": "=",
            "standard_units": "nM",
            "invalid_measurements_excluded": True,
        },
        "aggregation": "canonical InChIKey; median IC50 for repeated measurements",
        "raw_activity_count": raw_activity_count,
        "unique_compound_count": len(frame),
        "download_query": QUERY,
    }
    (root / "data" / "herg_chembl_source.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"saved {len(frame)} unique hERG compounds to {destination}")


if __name__ == "__main__":
    main()

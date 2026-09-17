from pathlib import Path

import pandas as pd


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    source = root / "data" / "dev_upstream" / "herg_dev.csv"
    destination = root / "data" / "dev" / "herg_dev.csv"
    frame = pd.read_csv(source)
    frame = frame.dropna(subset=["SMILES", "pIC50"]).copy()
    frame["pIC50"] = pd.to_numeric(frame["pIC50"], errors="raise")
    frame["Activity_Label"] = frame["pIC50"].ge(5.0).map(
        {True: "Blocker", False: "Non-blocker"}
    )
    frame["Activity"] = frame["Activity_Label"]
    frame["Compound_ID"] = frame["ChEMBL ID"].fillna(frame["PubChem CID"])
    frame["InChIKey"] = frame["InChl Key"]
    columns = [
        "pIC50", "Source", "Activity", "Activity_Label",
        "Compound_ID", "SMILES", "InChIKey",
    ]
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame[columns].to_csv(destination, index=False)
    print(f"{destination}: {len(frame)} valid compounds")


if __name__ == "__main__":
    main()

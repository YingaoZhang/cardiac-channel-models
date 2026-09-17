import os
import sys

from rdkit import Chem
from rdkit.Chem import RDConfig


def _load_sascorer():
    if not hasattr(RDConfig, "RDContribDir"):
        raise AttributeError(
            "Current RDKit installation does not expose RDConfig.RDContribDir; "
            "cannot locate Contrib/SA_Score/sascorer.py."
        )

    sa_score_path = os.path.join(RDConfig.RDContribDir, "SA_Score")

    if not os.path.exists(sa_score_path):
        raise FileNotFoundError(f"RDKit SA_Score directory not found: {sa_score_path}")

    if sa_score_path not in sys.path:
        sys.path.append(sa_score_path)

    import sascorer

    return sascorer


sascorer = _load_sascorer()


def calculate_sa_score(smiles: str) -> float:
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return 99.9

    return round(float(sascorer.calculateScore(mol)), 2)


def describe_sa_score(sa_score: float) -> str:
    if sa_score < 4:
        return "容易合成"

    if sa_score < 7:
        return "中等合成难度"

    return "合成难度较高"

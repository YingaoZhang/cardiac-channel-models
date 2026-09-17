from __future__ import annotations

from rdkit import Chem, DataStructs
from rdkit.Chem import Crippen, Descriptors, Lipinski, QED, rdFingerprintGenerator


SIMILARITY_THRESHOLD = 0.40
QED_THRESHOLD = 0.40
MAX_LIPINSKI_VIOLATIONS = 1

_MORGAN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def _molecule(smiles: str):
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    return molecule


def _drug_likeness(molecule) -> dict:
    properties = {
        "molecular_weight": float(Descriptors.MolWt(molecule)),
        "logp": float(Crippen.MolLogP(molecule)),
        "h_bond_donors": int(Lipinski.NumHDonors(molecule)),
        "h_bond_acceptors": int(Lipinski.NumHAcceptors(molecule)),
        "rotatable_bonds": int(Lipinski.NumRotatableBonds(molecule)),
        "tpsa": float(Descriptors.TPSA(molecule)),
        "qed": float(QED.qed(molecule)),
    }
    violations = {
        "molecular_weight_gt_500": properties["molecular_weight"] > 500,
        "logp_gt_5": properties["logp"] > 5,
        "h_bond_donors_gt_5": properties["h_bond_donors"] > 5,
        "h_bond_acceptors_gt_10": properties["h_bond_acceptors"] > 10,
    }
    violation_count = sum(violations.values())
    return {
        **properties,
        "lipinski_violations": violations,
        "lipinski_violation_count": violation_count,
        "passed": properties["qed"] >= QED_THRESHOLD
        and violation_count <= MAX_LIPINSKI_VIOLATIONS,
        "criteria": {
            "qed_min": QED_THRESHOLD,
            "max_lipinski_violations": MAX_LIPINSKI_VIOLATIONS,
        },
    }


def review_candidate(original_smiles: str, candidate_smiles: str) -> dict:
    original = _molecule(original_smiles)
    candidate = _molecule(candidate_smiles)
    similarity = float(
        DataStructs.TanimotoSimilarity(
            _MORGAN.GetFingerprint(original),
            _MORGAN.GetFingerprint(candidate),
        )
    )
    similarity_passed = similarity >= SIMILARITY_THRESHOLD
    return {
        "structure_similarity": {
            "method": "Morgan radius=2, 2048-bit Tanimoto",
            "score": similarity,
            "threshold": SIMILARITY_THRESHOLD,
            "passed": similarity_passed,
        },
        "activity_retention_proxy": {
            "method": "structural-similarity proxy",
            "score": similarity,
            "threshold": SIMILARITY_THRESHOLD,
            "passed": similarity_passed,
            "limitation": (
                "No target-activity model is configured; this is not a direct prediction "
                "of pharmacological activity."
            ),
        },
        "drug_likeness": _drug_likeness(candidate),
    }

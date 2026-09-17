from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import Crippen, Descriptors, Lipinski, MACCSkeys, QED, rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold


DESCRIPTOR_NAMES = (
    "MolWt", "MolLogP", "MolMR", "TPSA", "HBD", "HBA",
    "RotatableBonds", "RingCount", "AromaticRings", "AliphaticRings",
    "FractionCSP3", "HeavyAtomCount", "HeteroAtomCount", "FormalCharge",
    "NHOHCount", "NOCount", "LabuteASA", "QED",
)

MORGAN_RADIUS = 2
MORGAN_BITS = 2048
MACCS_BITS = 167
FEATURE_DIM = MORGAN_BITS + MACCS_BITS + len(DESCRIPTOR_NAMES)


def physicochemical_descriptors(mol: Chem.Mol) -> list[float]:
    return [
        Descriptors.MolWt(mol), Crippen.MolLogP(mol), Crippen.MolMR(mol),
        Descriptors.TPSA(mol), Lipinski.NumHDonors(mol),
        Lipinski.NumHAcceptors(mol), Lipinski.NumRotatableBonds(mol),
        Lipinski.RingCount(mol), Lipinski.NumAromaticRings(mol),
        Lipinski.NumAliphaticRings(mol), Lipinski.FractionCSP3(mol),
        Lipinski.HeavyAtomCount(mol), Lipinski.NumHeteroatoms(mol),
        Chem.rdmolops.GetFormalCharge(mol), Lipinski.NHOHCount(mol),
        Lipinski.NOCount(mol), Descriptors.LabuteASA(mol), QED.qed(mol),
    ]


def bit_vector_to_array(vector: DataStructs.ExplicitBitVect) -> np.ndarray:
    array = np.zeros(vector.GetNumBits(), dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(vector, array)
    return array


def scaffold_smiles(mol: Chem.Mol) -> str:
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    return scaffold or Chem.MolToSmiles(mol, canonical=True)


def featurize_mol(mol: Chem.Mol) -> np.ndarray:
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=MORGAN_RADIUS, fpSize=MORGAN_BITS
    )
    return np.concatenate(
        [
            bit_vector_to_array(generator.GetFingerprint(mol)),
            bit_vector_to_array(MACCSkeys.GenMACCSKeys(mol)),
            np.asarray(physicochemical_descriptors(mol), dtype=np.float32),
        ]
    ).astype(np.float32)


def featurize_smiles(smiles: str) -> tuple[str, np.ndarray]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    canonical = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    return canonical, featurize_mol(mol)


def build_feature_file(csv_path: Path, output_path: Path) -> None:
    frame = pd.read_csv(csv_path)
    required = {"SMILES", "pIC50", "Activity_Label"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing columns in {csv_path.name}: {sorted(missing)}")

    frame = frame.dropna(subset=["SMILES", "pIC50"]).reset_index(drop=True)
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=MORGAN_RADIUS, fpSize=MORGAN_BITS)
    morgan, maccs, descriptors, scaffolds, canonical = [], [], [], [], []
    for row_number, smiles in enumerate(frame["SMILES"], start=2):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES in {csv_path.name}, CSV row {row_number}")
        canonical.append(Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True))
        morgan.append(bit_vector_to_array(generator.GetFingerprint(mol)))
        maccs.append(bit_vector_to_array(MACCSkeys.GenMACCSKeys(mol)))
        descriptors.append(physicochemical_descriptors(mol))
        scaffolds.append(scaffold_smiles(mol))

    labels = frame["Activity_Label"].map({"Non-blocker": 0, "Blocker": 1})
    if labels.isna().any():
        unknown = sorted(frame.loc[labels.isna(), "Activity_Label"].unique())
        raise ValueError(f"Unknown activity labels: {unknown}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        morgan=np.asarray(morgan, dtype=np.uint8),
        maccs=np.asarray(maccs, dtype=np.uint8),
        physicochemical=np.asarray(descriptors, dtype=np.float32),
        descriptor_names=np.asarray(DESCRIPTOR_NAMES),
        y_class=labels.to_numpy(dtype=np.uint8),
        y_reg=frame["pIC50"].to_numpy(dtype=np.float32),
        smiles=np.asarray(canonical),
        scaffolds=np.asarray(scaffolds),
    )

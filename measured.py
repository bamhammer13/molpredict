import numpy as np
import pandas as pd
from rdkit import Chem

DATA_PATH = "data/esol.csv"
SMILES_COL = "smiles"
TARGET_COL = "measured log solubility in mols per litre"

# Builds a lookup of molecules appearing in the data, converting to and using canonical form, so that it can match varied ways of writing molecules
# Can't just use fingerprints as they are lossy, and could lead to collisions of different molecules with the same fingerprint
def build_measured_lookup():
    df = pd.read_csv(DATA_PATH)
    # Builds the list of valid molecules in the same order as load_dataset
    canonical_list = []
    measured_list = []
    for smiles, target in zip(df[SMILES_COL], df[TARGET_COL]):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue # Skips invalid SMILES
        try:
            measured = float(target)
        except ValueError:
            continue
        canonical_list.append(Chem.MolToSmiles(mol))
        measured_list.append(measured)

    
    n = len(canonical_list)
    idx = np.random.RandomState(0).permutation(n) # Uses same shuffle used for training
    split = int(0.8 * n) # Same split as well
    val_positions = set(idx[split:].tolist())

    lookup = {}
    # Fills up the lookup, noting if it was used in training or validation 
    for pos, (canonical, measured) in enumerate(zip(canonical_list, measured_list)):
        lookup[canonical] = {
            "measured": measured,
            "in_training": pos not in val_positions,
        }
    return lookup

_lookup = build_measured_lookup() # Runs once on module importation

# Gets the molecule's actual water solubility, if it is in the data, using the canonical form
def get_measured(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    canonical = Chem.MolToSmiles(mol)
    return _lookup.get(canonical) # Returns None if it isn't in the dataset
import numpy as np
import pandas as pd
from features import smiles_to_fingerprint, FP_SIZE

DATA_PATH = "data/esol.csv"
SMILES_COL = "smiles"
TARGET_COL = "measured log solubility in mols per litre"

# Function that loads in the molecules strings from the csv and converts them into fingerprints 
def load_dataset():
    df = pd.read_csv(DATA_PATH)
    X, y = [], []
    for smiles, target in zip(df[SMILES_COL], df[TARGET_COL]):
        try:
            X.append(smiles_to_fingerprint(smiles))
            y.append(float(target))
        except ValueError:
            continue # Skips the molecules RDKit can't parse
    X = np.stack(X)
    y = np.array(y, dtype=np.float32).reshape(-1, 1)
    return X, y

if __name__=="__main__":
    X, y = load_dataset()
    print("Loaded", len(X), "molecules")
    print("X shape:", X.shape, "| y shape:", y.shape)

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.DataStructs import ConvertToNumpyArray

FP_SIZE = 2048
_generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=FP_SIZE)

# Turns a Simplified Molecular input Line Entry System(SMILES) string and return its fingerprint as a NumPy array if it is valid
# Fingerprints are the numeric form of the molecule, which we need to compare with other molecules
def smiles_to_fingerprint(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles!r}") #Raises a ValueError that will be picked up in train.py, so it is skipped
    else:
        fp = _generator.GetFingerprint(mol)
        arr = np.zeros((FP_SIZE,), dtype=np.float32)
        ConvertToNumpyArray(fp, arr)
        return arr

# Quick test of the function by testing to see if it converts the SMILES string for ethanol into its fingerprint
# Should return ethanol fingerprint: shape (2048,) bits set 6
if __name__ == "__main__":
    fp = smiles_to_fingerprint("CCO")
    print("ethanol fingerprint: shape", fp.shape, "bits set", int(fp.sum()))
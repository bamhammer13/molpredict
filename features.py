import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.DataStructs import ConvertToNumpyArray

FP_SIZE = 2048
_generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=FP_SIZE)

def smiles_to_fingerprint(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles!r}")
    else:
        fp = _generator.GetFingerprint(mol)
        arr = np.zeros((FP_SIZE,), dtype=np.float32)
        ConvertToNumpyArray(fp, arr)
        return arr

if __name__ == "__main__":
    fp = smiles_to_fingerprint("CCO")
    print("ethanol fingerprint: shape", fp.shape, "bits set", int(fp.sum()))
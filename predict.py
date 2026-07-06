import sys
import torch
from features import smiles_to_fingerprint
from train import SolubilityNet

# Loads in model once, so it doesn't have to reload it every function call
_model = SolubilityNet()
_model.load_state_dict(torch.load("model.pt"))
_model.eval()

def predict(smiles):
    fingerprint = smiles_to_fingerprint(smiles) # Converts the SMILES input into a fingerprint
    x = torch.tensor(fingerprint).unsqueeze(0) # Turns the fingerprint into a batch, which the model expects
    with torch.no_grad(): # We're just looking so gradient tracking is disabled
        return _model(x).item() # runs fingerprint through the model, calculating prediction and returning it

if __name__ == "__main__":
    # Sets the SMILES value to the argument given in the terminal, defaults to ethanol if none is given
    smiles = sys.argv[1] if len(sys.argv) > 1 else "CCO" 
    logS = predict(smiles) # Runs predict on the given molecule, storing result in logS
    print(f"SMILES: {smiles}") # Prints out the SMILES string that was predicted for verification
    # Prints out the predicted water solubility of input, should be between -12 and 2
    # Due to the dataset it was trained on, it will probably flub non-organic molecules
    print(f"Predicted logS: {logS:.3f} (higher means more soluble)") 
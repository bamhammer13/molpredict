from predict import predict
from measured import get_measured

# Function to make prediction for the molecules in a list as a batch
def predict_batch(smiles_list):
    results = []
    for smiles in smiles_list:
        try:
            logS = predict(smiles)
            measured = get_measured(smiles)
            results.append({"smiles": smiles, "logS": logS, "measured": measured})
        except ValueError:
            results.append({"smiles": smiles, "error": "Invalid SMILES"})
    return results
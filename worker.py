from predict import predict

# Function to make prediction for the molecules in a list as a batch
def predict_batch(smiles_list):
    results = []
    for smiles in smiles_list:
        try:
            logS = predict(smiles)
            results.append({"smiles": smiles, "logS": logS})
        except ValueError:
            results.append({"smiles": smiles, "error": "Invalid SMILES"})
    return results
from predict import predict
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI() # Creates the FastAPI instance that will be the basis for the code

# Defines the request shape
class MoleculeRequest(BaseModel):
    smiles: str # The JSON given a field name of smiles, it will take in a molecule value

# Defines the /predict endpoint, that runs the predict function and returns the result
@app.post("/predict") 
def predict_endpoint(request: MoleculeRequest):
    logS = predict(request.smiles)
    return {"smiles": request.smiles, "logS": logS}

# Simple health checking endpoint, just to check if the server is actually alive
@app.get("/healthz")
def health():
    return {"status": "ok"} # If asked for status will say ok, if it doesn't answer that's when you know you got a problem


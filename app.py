from predict import predict
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.responses import FileResponse


app = FastAPI() # Creates the FastAPI instance that will be the basis for the code

# Defines the request shape
class MoleculeRequest(BaseModel):
    smiles: str # The JSON given a field name of smiles, it will take in a molecule value

# Defines the /predict endpoint, that runs the predict function and returns the result
@app.post("/predict") 
def predict_endpoint(request: MoleculeRequest):
    # Tries to use the request's SMILES string, if invalid sends a 400, Bad Request, error with a message
    try:
        logS = predict(request.smiles)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid SMILES: {request.smiles}")
    return {"smiles": request.smiles, "logS": logS}

# Simple health checking endpoint, just to check if the server is actually alive
@app.get("/healthz")
def health():
    return {"status": "ok"} # If asked for status will say ok, if it doesn't answer that's when you know you got a problem

app.mount("/static", StaticFiles(directory="static"), name="static") # Makes the contents of the static/ folder reachable

# The main web page at root URL
@app.get("/")
def index():
    return FileResponse("static/index.html")


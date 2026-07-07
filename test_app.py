from features import smiles_to_fingerprint, FP_SIZE
from predict import predict
from fastapi.testclient import TestClient
from app import app
import pytest

client = TestClient(app)

def test_fingerprint_shape():
    fp = smiles_to_fingerprint("CCO") # Gets the fingerprint for the SMILES string for Ethanol
    assert fp.shape == (FP_SIZE,) # Must be a 2048 long vector

def test_invalid_smiles_raises():
    with pytest.raises(ValueError): # Reverses it so that having a ValueError is what it wants, failing if it doesn't get it
        smiles_to_fingerprint("xyz") # Gives an invalid SMILES to the function

def test_predict_returns_number():
    logS = predict("CCO") # Gets the predicted solubility of Ethanol
    assert isinstance(logS, float) # The returned prediction must be a float
    assert -15 < logS < 5 # The returned prediction must be in a reasonable range

def test_healthz():
    response = client.get("/healthz")
    assert response.status_code == 200 # Status code must be 200(ok)
    assert response.json() == {"status": "ok"} # The response must be status is ok response

def test_predict_endpoint():
    response = client.post("/predict", json={"smiles": "CCO"})
    assert response.status_code == 200 
    assert "logS" in response.json() # logS must be included in the response

def test_predict_invalid_returns_400():
    response = client.post("/predict", json={"smiles": "xy"})
    assert response.status_code == 400 # The response must come back with a 400, bad request, error
from predict import predict
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from redis import Redis
from rq import Queue
from rq.job import Job
from worker import predict_batch
import os


app = FastAPI() # Creates the FastAPI instance that will be the basis for the code

redis_host = os.environ.get("REDIS_HOST", "localhost") # Gets the enivronment's REDIS host
redis_conn = Redis(host=redis_host, port=6379) # Creates the Remote Dictionary Server(REDIS) connection
queue = Queue(connection=redis_conn) # Creates the job queue

# Defines the request shape for requesting a batch of predictions
class BatchRequest(BaseModel):
    smiles_list: list[str]

# Defines the request shape for singular molecules
class MoleculeRequest(BaseModel):
    smiles: str # The JSON given a field name of smiles, it will take in a molecule value

# Defines the endpoint for adding batch jobs
@app.post("/predict/batch")
def predict_batch_endpoint(request: BatchRequest):
    job = queue.enqueue(predict_batch, request.smiles_list, result_ttl=86400) # Creates the job of doing the batch of prediction and adds it to the queue, results last 24hrs
    return {"job_id": job.id, "status": job.get_status()} # Returns the job's id and status for reference

# Defines the endpoint for getting the status of a job
@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    job = Job.fetch(job_id, connection=redis_conn)
    return{"job_id": job.id, "status": job.get_status(), "result": job.result}

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


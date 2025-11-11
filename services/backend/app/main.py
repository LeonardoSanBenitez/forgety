from typing import List, Tuple, Optional
import shutil
import os
import subprocess
import time
from fastapi import FastAPI, Form, File, UploadFile, HTTPException
from libs import Request, RequestInferred, RequestLaunched, RequestCompleted, DatabaseLocalJson, infer_request


app = FastAPI(debug=True)
database = DatabaseLocalJson(filepath="app/database.json")

@app.post("/v1/public-api-create-request")
async def create_request(
    customer_id: str = Form(...),
    experiment_name: str = Form(...),
    model_base_name: str = Form(...),
    concept_forget: Optional[str] = Form(None),
    concept_overwrite: Optional[str] = Form(None),
    concept_retain: Optional[str] = Form(None),
    unlearning_algorithm: str = Form(...),
    model_output_hf_id: str = Form(...),
    dataset: Optional[UploadFile] = File(None),
) -> str:
    """
    Accepts experiment request + dataset zip upload.
    Returns (status_code, message_or_id).
    """

    print(
        f"Received request:\n"
        f"  customer_id={customer_id}\n"
        f"  experiment_name={experiment_name}\n"
        f"  model_base_name={model_base_name}\n"
        f"  concept_forget={concept_forget}\n"
        f"  concept_overwrite={concept_overwrite}\n"
        f"  concept_retain={concept_retain}\n"
        f"  unlearning_algorithm={unlearning_algorithm}\n"
        f"  model_output_hf_id={model_output_hf_id}\n",
        flush=True,
    )
    request = Request(
        customer_id=customer_id,
        experiment_name=experiment_name,
        model_base_name=model_base_name,
        concept_forget=concept_forget,
        concept_overwrite=concept_overwrite,
        concept_retain=concept_retain,
        unlearning_algorithm=unlearning_algorithm,
        model_output_hf_id=model_output_hf_id,
    )
    identifier = request.uuid
    

    ###############################################
    # Handle dataset upload
    ###############################################
    num_forget_images = 0
    num_retain_images = 0
    if dataset is not None:
        print("Starting data extraction...", flush=True)
        
        # Check if the uploaded file is a .zip file
        if not dataset.filename.lower().endswith('.zip'):
            raise HTTPException(status_code=400, detail="Uploaded file must be a .zip archive.")

        # Create folder structure for the request
        data_path = os.path.join("/requests", str(identifier), "data")
        zip_location = os.path.join(data_path, dataset.filename)
        forget_path = os.path.join(data_path, "forget")
        retain_path = os.path.join(data_path, "retain")
        os.makedirs(data_path, exist_ok=True)
        
        # Save
        with open(zip_location, "wb") as buffer:
            shutil.copyfileobj(dataset.file, buffer)
        print(f"file '{dataset.filename}' saved at '{zip_location}'", flush=True)

        # Unzip
        shutil.unpack_archive(zip_location, data_path)
        if not os.path.exists(forget_path):
            raise HTTPException(status_code=400, detail="The ZIP file must contain a 'forget' folder.")
        if not os.path.exists(retain_path):
            raise HTTPException(status_code=400, detail="The ZIP file must contain a 'retain' folder.")

        num_forget_images = len([f for f in os.listdir(forget_path) if f.lower().endswith('.png') and os.path.isfile(os.path.join(forget_path, f))])
        num_retain_images = len([f for f in os.listdir(retain_path) if f.lower().endswith('.png') and os.path.isfile(os.path.join(retain_path, f))])
        if num_forget_images == 0:
            raise HTTPException(status_code=400, detail="The 'forget' folder must contain at least one PNG image.")
        if num_retain_images == 0:
            raise HTTPException(status_code=400, detail="The 'retain' folder must contain at least one PNG image.")
        print(f"Dataset extracted to {data_path}", flush=True)

    ###############################################
    # Infer missing parameters and validate request
    ###############################################
    request_inferred: RequestInferred = infer_request(request, num_forget_images, num_retain_images)
    print(f"Inferred request: {request_inferred}", flush=True)
    
    ###############################################
    # Launch
    ###############################################
    try:
        # TODO
        request_launched = request_inferred
        database.insert_request(request_launched)
    except Exception as e:
        print(f"Error launching job: {e}", flush=True)

    return identifier



@app.post("/v1/test-cluster-communication")
async def test_cluster_communication() -> str:
    import subprocess
    import time

    print(">>>>>>>>>>>>>>> LAUNCH JOB")
    result = subprocess.run(["cd app; make run-batch-cluster"], capture_output=True, text=True, shell=True)
    print(result.stdout)
    print(result.stderr)
    time.sleep(1)

    print(">>>>>>>>>>>>>>> GET INFO")
    result = subprocess.run(["cd app; make debug-last-cluster"], capture_output=True, text=True, shell=True)
    print(result.stdout)
    print(result.stderr)

    print(">>>>>>>>>>>>>>> CANCEL JOB")
    result = subprocess.run(["cd app; make stop-cluster"], capture_output=True, text=True, shell=True)
    print(result.stdout)
    print(result.stderr)

    return "tudo certin, trem baom!"

@app.get("/v1/public-api-read-requests")
async def read_incidents(
    customer_id: str
) -> List[RequestLaunched | RequestCompleted]:
    ###############################################
    # Get info about unfinished requests
    ###############################################
    # TODO

    ###############################################
    # Get latest
    ###############################################
    results = database.get_requests()
    results = sorted(results, key=lambda r: getattr(r, "timestamp_started", 0), reverse=True)
    return results

@app.post("/v1/availability-test")
async def availability_test() -> int:
    return 200

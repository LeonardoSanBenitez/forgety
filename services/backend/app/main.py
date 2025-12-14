from typing import List, Tuple, Optional
import shutil
import os
import subprocess
import time
from fastapi import FastAPI, Form, File, UploadFile, HTTPException
from libs import Request, RequestInferred, RequestLaunched, RequestCompleted, DatabaseLocalJson, InfraSlurm, infer_request


app = FastAPI(debug=True)
database = DatabaseLocalJson(filepath="app/database.json")
infra = InfraSlurm()

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

        num_forget_images = len([f for f in os.listdir(forget_path) if (f.lower().endswith('.png') or f.lower().endswith('.jpg')) and os.path.isfile(os.path.join(forget_path, f))])
        num_retain_images = len([f for f in os.listdir(retain_path) if (f.lower().endswith('.png') or f.lower().endswith('.jpg')) and os.path.isfile(os.path.join(retain_path, f))])
        if num_forget_images == 0:
            raise HTTPException(status_code=400, detail="The 'forget' folder must contain at least one PNG or JPG image.")
        if num_retain_images == 0:
            raise HTTPException(status_code=400, detail="The 'retain' folder must contain at least one PNG or JPG image.")
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
        request_launched = infra.launch(request_inferred)
        database.insert_request(request_launched)
    except Exception as e:
        print(f"Error launching job: {e}", flush=True)

    return identifier



@app.post("/v1/test-launch")
async def test_launch() -> str:
    request_inferred: RequestInferred = infer_request(Request(
        customer_id="demo-customer",
        experiment_name="Test experiment",
        model_base_name="stable-diffusion-v1-5/stable-diffusion-v1-5",
        concept_forget="cat",
        concept_overwrite=None,
        concept_retain="lion; tiger; leopard",
        unlearning_algorithm="Automatic (recommended)",
        model_output_hf_id="LeonardoBenitez/demo-forgety"
    ), num_forget_images=0, num_retain_images=0)

    request_launched = infra.launch(request_inferred)
    database.insert_request(request_launched)
    updated_request = infra.status(request_launched)
    result = f"Request {request_launched.uuid} updated to {updated_request} (type {type(updated_request)})"
    print(result, flush=True)
    return result

@app.get("/v1/test-read")
async def test_read(uuid: str) -> str:
    # Find request by UUID in the local database (uuid provided as a query parameter)
    request = database.get_request(uuid=uuid)
    if request is None:
        raise HTTPException(status_code=404, detail=f"Request {uuid} not found")

    updated_request = infra.status(request)

    return f"Request {request.uuid} updated to {updated_request} (type {type(updated_request)})"


@app.get("/v1/public-api-read-requests")
async def read_incidents(
    customer_id: str
) -> List[RequestLaunched | RequestCompleted]:
    ###############################################
    # Get info about unfinished requests
    ###############################################
    requests_to_update = database.get_requests_status_none()
    print(f"Found {len(requests_to_update)} requests to update", flush=True)
    for request in requests_to_update:
        try:
            updated_request = infra.status(request)
            if isinstance(updated_request, RequestCompleted):
                database.update_request(request.uuid, updated_request.model_dump())
                print(f"Request {request.uuid} updated to {updated_request}", flush=True)
        except Exception as e:
            print(f"Error updating request {request.uuid}: {e}", flush=True)

    ###############################################
    # Get latest
    ###############################################
    results = database.get_requests()
    results = sorted(results, key=lambda r: getattr(r, "timestamp_started", 0), reverse=True)
    return results
    #     # TODO: get images from server too
    results = [
        RequestLaunched(
            customer_id='demo-customer',
            experiment_name='Modern art generation, remove John Snow works',
            model_base_name='stable-diffusion-v1-5/stable-diffusion-v1-5',
            concept_forget='John Snow',
            concept_overwrite='Vanilla modern art',
            concept_retain=None,
            unlearning_algorithm = 'FADE',
            model_output_hf_id = 'demo-customer/modern-art-no-john-snow',
            hyperparameters = {
                'epochs': 10
            },
            timestamp_started = 1234567890,
            slurm_job_id = '123456'
        ),
        RequestCompleted(
            customer_id='demo-customer',
            experiment_name='Prehistoric painting generator, forget cellphones',
            model_base_name='stable-diffusion-v1-5/stable-diffusion-v1-5',
            concept_forget='Someone holding a cellphone',
            concept_overwrite='Someone holding a rock',
            concept_retain=None,
            unlearning_algorithm = 'FADE',
            model_output_hf_id = 'demo-customer/prehistoric-art-no-cellphones',
            hyperparameters = {
                'epochs': 10
            },
            timestamp_started = 1234567890,
            slurm_job_id = '123456',
            status = 'SUCCEEDED',
            metrics = [{'name': 'accuracy', 'value': 0.95}],
            credits_consumed = 12.5,
        ),
        RequestCompleted(
            customer_id='demo-customer',
            experiment_name='Prehistoric painting generator, forget cellphones',
            model_base_name='stable-diffusion-v1-5/stable-diffusion-v1-5',
            concept_forget=None,
            concept_overwrite=None,
            concept_retain=None,
            unlearning_algorithm = 'Munba',
            model_output_hf_id = 'demo-customer/prehistoric-art-no-cellphones',
            hyperparameters = {
                'epochs': 10
            },
            timestamp_started = 1234567890,
            slurm_job_id = '123456',
            status = 'FAILED',
            metrics = [{}],
            credits_consumed = 0,
        ),
    ]
    return results


@app.post("/v1/availability-test")
async def availability_test() -> int:
    return 200

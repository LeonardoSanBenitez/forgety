from typing import List, Dict, Optional, Any
import shutil
import os
from fastapi import FastAPI, Form, File, UploadFile, HTTPException, Body, Depends
from libs import Request, RequestInferred, RequestLaunched, RequestCompleted, DatabaseLocalJson, InfraSlurm, infer_request
from libs.database.base import Database
from libs.infra.base import Infra
from libs.vision_unlearning_benchmarks_I_care_TEMP import rt_name_to_class, convert_params_from_gui_to_backend, type_task, InterferencePerEntity

app = FastAPI(debug=True)

_database: Database = DatabaseLocalJson(filepath="app/database.json")
_infra: Infra = InfraSlurm()


def get_database() -> Database:
    return _database


def get_infra() -> Infra:
    return _infra


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
    database: Database = Depends(get_database),
    infra: Infra = Depends(get_infra),
) -> str:
    """
    Accepts experiment request + dataset zip upload.
    Returns request UUID.
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
        if not dataset.filename or not dataset.filename.lower().endswith('.zip'):
            raise HTTPException(status_code=400, detail="Uploaded file must be a .zip archive.")
        dataset_filename: str = dataset.filename  # narrowed from Optional[str]

        # Create folder structure for the request
        data_path = os.path.join("/requests", str(identifier), "data")
        zip_location = os.path.join(data_path, dataset_filename)
        forget_path = os.path.join(data_path, "forget")
        retain_path = os.path.join(data_path, "retain")
        os.makedirs(data_path, exist_ok=True)

        # Save
        with open(zip_location, "wb") as buffer:
            shutil.copyfileobj(dataset.file, buffer)
        print(f"file '{dataset_filename}' saved at '{zip_location}'", flush=True)

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


@app.post("/v1/public-api-compute-rt")
async def compute_rt(template: str = Body(...), params: dict = Body(...)) -> dict:
    """
    Compute and return resulting data (as returned by the compute method of the
    corresponding ResultTemplate subclass) as a JSON response.
    """
    rt = rt_name_to_class[template](**convert_params_from_gui_to_backend(params))
    data = rt.compute()
    return data


@app.get("/v1/public-api-read-interference-per-entity-all")
async def read_results() -> dict:
    data: Dict[type_task, List[Dict[str, Any]]] = {}
    for task in list(type_task.__args__):  # type: ignore[attr-defined]
        data[task] = InterferencePerEntity(task=task).compute()
    return data


@app.post("/v1/test-launch")
async def test_launch(
    database: Database = Depends(get_database),
    infra: Infra = Depends(get_infra),
) -> str:
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
async def test_read(
    uuid: str,
    database: Database = Depends(get_database),
    infra: Infra = Depends(get_infra),
) -> str:
    # Find request by UUID in the local database (uuid provided as a query parameter)
    request = database.get_request(uuid=uuid)
    if request is None:
        raise HTTPException(status_code=404, detail=f"Request {uuid} not found")

    updated_request = infra.status(request)  # type: ignore[arg-type]

    return f"Request {request.uuid} updated to {updated_request} (type {type(updated_request)})"


@app.get("/v1/public-api-read-requests")
async def read_incidents(
    customer_id: str,
    database: Database = Depends(get_database),
    infra: Infra = Depends(get_infra),
) -> List[RequestLaunched | RequestCompleted]:
    ###############################################
    # Get info about unfinished requests
    ###############################################
    requests_to_update = database.get_requests_status_none()
    print(f"Found {len(requests_to_update)} requests to update", flush=True)
    for request in requests_to_update:
        try:
            updated_request = infra.status(request)  # type: ignore[arg-type]
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


@app.post("/v1/availability-test")
async def availability_test() -> int:
    return 200

import uuid
from typing import Literal, List, Optional
from pydantic import BaseModel, Field


class Request(BaseModel):
    uuid: str = Field(default_factory=lambda: str(uuid.uuid4()))
    customer_id: str
    experiment_name: str
    model_base_name: str
    concept_forget: Optional[str] = None
    concept_overwrite: Optional[str] = None
    concept_retain: Optional[str] = None
    unlearning_algorithm: str
    model_output_hf_id: str


class RequestInferred(Request):
    hyperparameters: Optional[dict] = None  # Specific to unlearning_algorithm... very hard to type, but would be good
    num_forget_images: Optional[int] = None
    num_retain_images: Optional[int] = None


class RequestLaunched(RequestInferred):
    timestamp_started: Optional[int] = None
    slurm_job_id: Optional[str] = None


class RequestCompleted(RequestLaunched):
    status: Optional[Literal['SUCCEEDED', 'FAILED']] = None
    metrics: Optional[List[dict]] = None
    credits_consumed: Optional[float] = None

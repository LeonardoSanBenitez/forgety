from typing import Literal, List, Optional
from pydantic import BaseModel


class Request(BaseModel):
    customer_id: str
    experiment_name: str
    model_base_name: str
    concept_forget: Optional[str]
    concept_overwrite: Optional[str]
    unlearning_algorithm: str
    model_output_hf_id: str


class RequestInferred(Request):
    epochs: int


class RequestLaunched(RequestInferred):
    timestamp_started: int
    slurm_job_id: str


class RequestCompleted(RequestLaunched):
    status: Literal['SUCCEEDED', 'FAILED']
    metrics: List[dict]
    credits_consumed: float

import os
from typing import Optional
from fastapi import HTTPException
from libs.schemas.request import Request, RequestInferred
import math


def infer_request(request: Request, num_forget_images: int, num_retain_images: int) -> RequestInferred:
    # Choose algorithm
    if request.unlearning_algorithm == "Automatic (recommended)":
        if num_forget_images == 0:
            if request.concept_overwrite or request.concept_retain:
                request.unlearning_algorithm = "UCE"
            else:
                raise HTTPException(status_code=400, detail="Cannot automatically choose unlearning algorithm: either pass image data or concepts to overwrite/retain.")
        else:  # we have images to forget
            if request.concept_overwrite:
                request.unlearning_algorithm = "FADE"
            else:
                request.unlearning_algorithm = "Munba"

    # Check if hyperpameters match algorithm
    if request.unlearning_algorithm == "FADE":
        if not request.concept_forget:
            raise HTTPException(status_code=400, detail="FADE requires a concept to forget.")
        if not request.concept_overwrite:
            raise HTTPException(status_code=400, detail="FADE requires a concept to overwrite.")
        if num_forget_images <= 0 or num_retain_images <= 0:
            raise HTTPException(status_code=400, detail="FADE requires image data")
    elif request.unlearning_algorithm == "Munba":
        if num_forget_images <= 0 or num_retain_images <= 0:
            raise HTTPException(status_code=400, detail="Munba requires image data")
    elif request.unlearning_algorithm == "UCE":
        if not request.concept_forget:
            raise HTTPException(status_code=400, detail="UCE requires a concept to forget.")
        if not request.concept_retain:
            raise HTTPException(status_code=400, detail="UCE requires a concept to retain.")
    
    # Choose hyperparameters
    hyperparameters: dict
    if request.unlearning_algorithm == "FADE" or request.unlearning_algorithm == "Munba":
        hyperparameters = {
            "model_name_or_path": request.model_base_name,
            "dataset_forget_name": os.path.join("data", "forget"),
            "dataset_retain_name": os.path.join("data", "retain"),
            "output_dir": 'model',
            "hub_model_id": request.model_output_hf_id,

            "dataloader_num_workers": 2,
            "resolution": 512,
            "num_validation_images": 1,

            "mixed_precision": "no",
            "learning_rate": 1e-4,
            "max_grad_norm": 1.0,
            
            "checkpointing_steps": 10000,
            "lr_scheduler_type": "constant",
            "lr_warmup_steps": 0,
            "save_strategy": "epoch",
            "save_total_limit": 2,
            "random_flip": True,
            
            "lora_r": 4,
            "target_modules": ["q_proj", "k_proj", "v_proj"],
            "lora_alpha": 4,
            "lora_dropout": 0.1,
            
            "seed": 42,

            "per_device_train_batch_size": 2,
            "gradient_accumulation_steps": 2,
        }
        if request.unlearning_algorithm == "FADE":
            hyperparameters.update({
                'overwritting_concept': request.concept_overwrite,
                "validation_prompt": f"An image of {request.concept_forget}",
                "final_eval_prompts_forget": [
                    f'An image of {request.concept_forget}',
                    f'Photograph of {request.concept_forget}; high definition',
                    f'An picture of {request.concept_forget} in the rain',
                ],
                "final_eval_prompts_retain": [
                    f'An image of {request.concept_overwrite}',
                    f'Photograph of {request.concept_overwrite}; high definition',
                    f'An picture of {request.concept_overwrite} in the rain',
                ],
                "validation_epochs": 1,
                "num_train_epochs": math.ceil(1600 / min(num_forget_images, num_retain_images)),
            })
        elif request.unlearning_algorithm == "Munba":
            hyperparameters.update({
                "validation_prompt": None,
                "final_eval_prompts_forget": [],
                "final_eval_prompts_retain": [],
                "validation_epochs": 400,
                "num_train_epochs": math.ceil(400 / min(num_forget_images, num_retain_images)),
            })
    elif request.unlearning_algorithm == "UCE":
        hyperparameters = {
            "model_name_or_path": request.model_base_name,
            "output_dir": 'model',
            "edit_concepts": request.concept_forget,
            "guide_concepts": "object",
            "concept_type": "object",
            "preserve_concepts": request.concept_retain,
            "hub_model_id": request.model_output_hf_id,
        }

    return RequestInferred(**request.model_dump(), hyperparameters=hyperparameters, num_forget_images=num_forget_images, num_retain_images=num_retain_images)

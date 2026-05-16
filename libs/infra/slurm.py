import os
import subprocess
import time
import re
from typing import Tuple, List, Dict
from io import StringIO

from libs.infra.base import Infra
from libs.schemas.request import RequestInferred, RequestLaunched, RequestCompleted


_TEMPLATE_MAIN_PY_TEST = """
import logging
logging.info("Starting unlearning experiment with {unlearning_algorithm}")
hyperparameters = {hyperparameters}
logging.info(f"Hyperparameters: " + str(hyperparameters))
logging.error("o oh...")
"""

_TEMPLATE_MAIN_PY_FADE_UCE_MUNBA = """
import os
import sys
import dotenv
import pandas as pd
import torch
import os


sys.path = ['../../../TRDP-unlearning'] + sys.path
sys.path = ['../../../vision-unlearning'] + sys.path
dotenv.load_dotenv()
os.environ['WANDB_DISABLED'] = "true"
assert len(os.getenv('HF_TOKEN'))>0

from unlearner_lora_distillation import UnlearnerLoraDistillation  # FADE
from vision_unlearning.unlearner import UnlearnerLoraDirect  # Munba
from vision_unlearning.unlearner import UCE
from vision_unlearning.utils.logger import get_logger, setup_loggers
from vision_unlearning.utils.gradient_weighting import GradientWeightingMethodSimple, GradientWeightingMethodMunba
from vision_unlearning.utils.parameter_attribution import ParameterAttributionMethod


logger = get_logger('forgety')
setup_loggers(modules_info=['unlearning'])
device = 'cuda' if torch.cuda.is_available() else 'cpu'
hyperparameters = {hyperparameters}
hyperparameters['device'] = device

logger.info("Starting unlearning with {unlearning_algorithm}")
if '{unlearning_algorithm}' == 'FADE':
    hyperparameters['gradient_weighting_method'] = GradientWeightingMethodSimple(forget_weight=0.3, retain_weight=1.0)
    unlearner = UnlearnerLoraDistillation(**hyperparameters)
elif '{unlearning_algorithm}' == 'Munba':
    hyperparameters['gradient_weighting_method'] = GradientWeightingMethodMunba()
    unlearner=UnlearnerLoraDirect(**hyperparameters)
elif '{unlearning_algorithm}' == 'UCE':
    unlearner = UCE(**hyperparameters)
else:
    raise NotImplementedError()

logger.info(hyperparameters)
eval_results = unlearner.train()
logger.info('''Evaluation metrics:\\n''' + pd.DataFrame([dict(Name=r.metric_name, Value=r.metric_value) for r in eval_results]).to_markdown())
"""

_TEMPLATE_MAIN_PY_MACE = """
TODO
"""

_TEMPLATE_LAUNCHER_SH = """#!/bin/sh
#SBATCH --job-name=unlearn_forgety_{identifier}	# Name of the job
#SBATCH --partition=gpu				# Use the GPU partition
#SBATCH --time=12:00:00				# Set maximum run time for the job (hh:mm:ss)
#SBATCH --output=outputs/%j-train-out		# Redirect stdout
#SBATCH --error=outputs/%j-train-err		# Redirect stderr
#SBATCH --gres=gpu:v100:1\t\t\t\t# Select machine config (gpu:v100:1 = 1 V100 16GB GPU, gpu:a100:1 = 1 A100 40GB GPU)
srun apptainer exec --nv ../../infra/run_cluster.sif accelerate launch --num_processes 2 --num_machines 1 --dynamo_backend no {request_file}
"""

CREDITS_PER_HOUR = 5.0

JOB_DETAILS_SEP = r'-------+\nJob details using sacct\n-------+\n'
JOB_ERROR_SEP = r'-------+\nJob error\n-------+\n'
JOB_OUTPUT_SEP = r'-------+\nJob output\n-------+\n'


def execute_command(command: str, max_retries: int = 5, wait_seconds: float = 0., timeout_one_seconds: float = 60.) -> Tuple[str, str]:
    '''
    If can't, raises exception
    '''
    last_exc: Exception = RuntimeError("max_retries must be > 0")
    for attempt in range(max_retries):
        try:
            result = subprocess.run([command], capture_output=True, text=True, shell=True, check=True, timeout=timeout_one_seconds)
            return result.stdout, result.stderr
        except Exception as e:
            last_exc = e
            print(f"Warning: Command '{command}' failed on attempt {attempt + 1}/{max_retries} with error: {e}", flush=True)
            if attempt < max_retries - 1:
                time.sleep(wait_seconds)
    raise last_exc


def parse_markdown_table(stdout: str, start_anchor: str = 'Evaluation metrics:\n') -> List[Dict]:
    """
    Extracts a markdown table starting after a specified anchor string from a
    larger text block and returns it as a list of dictionaries.

    The metric names will include the trailing performance indicators (e.g., (~↑)).
    """
    results = []

    # 1. Locate the start of the table
    try:
        start_index = stdout.index(start_anchor) + len(start_anchor)
    except ValueError:
        return []

    sub_stdout = stdout[start_index:].splitlines()

    # 2. Extract consecutive table lines
    table_lines = []
    for line in sub_stdout:
        stripped_line = line.strip()
        if stripped_line.startswith('|'):
            table_lines.append(stripped_line)
        elif table_lines:
            # Stop when the table structure is broken
            break

    # We need at least 3 lines: header, separator, and one data row
    if len(table_lines) < 3:
        return []

    # 3. Process data rows (skip header and separator lines)
    data_rows = table_lines[2:]

    for row_line in data_rows:
        # Split the line by '|', removing the empty strings that result from
        # the leading and trailing pipe characters.
        parts = [part.strip() for part in row_line.split('|') if part.strip()]

        # We expect three parts: [Index, Name, Value]
        if len(parts) != 3:
            continue

        # Parts: [0] = Index, [1] = Name, [2] = Value
        raw_name = parts[1]
        raw_value = parts[2]

        try:
            # Name: The raw name already contains the indicator and needs no further
            # cleaning besides stripping surrounding whitespace.
            clean_name = raw_name

            # Value: Convert to float (handles standard and scientific notation)
            value = float(raw_value)

            results.append({
                'name': clean_name,
                'value': value
            })
        except ValueError:
            # Skip row if the value cannot be converted to float
            continue

    return results


class InfraSlurm(Infra):
    def launch(self, request: RequestInferred) -> RequestLaunched:
        identifier = request.uuid
        request_file = "main.py"
        bash_script = "run_batch_cluster.sh"
        files_path = os.path.join("/requests", str(identifier))
        os.makedirs(files_path, exist_ok=True)
        print(f"ID: {identifier}")

        with open(os.path.join(files_path, request_file), "w") as f:
            f.write(_TEMPLATE_MAIN_PY_FADE_UCE_MUNBA.format(
                unlearning_algorithm=request.unlearning_algorithm,
                hyperparameters=request.hyperparameters,
            ))
        print(f"Unlearning script '{request_file}' created!", flush=True)

        with open(os.path.join(files_path, bash_script), "w") as f:
            f.write(_TEMPLATE_LAUNCHER_SH.format(identifier=identifier, request_file=request_file))
        os.chmod(os.path.join(files_path, bash_script), 0o755)
        print(f"Bash script '{bash_script}' created!", flush=True)

        # Just the exception being raised is not guaranteed that the command failed;
        # we either check in the cluster or blindly proceed.
        stdout, stderr = execute_command(f"cd app; make copy-folder UUID={identifier}", timeout_one_seconds=30.)

        print(f">>>>>>>>>>>>>>> LAUNCH JOB: make run-batch-cluster UUID={identifier}", flush=True)
        stdout, stderr = execute_command(f"cd app; make run-batch-cluster UUID={identifier}", timeout_one_seconds=10.)
        job_id_match = re.search(r'Submitted batch job (\d+)\n', stdout)
        if job_id_match is None:
            raise ValueError(f"Could not parse Slurm job ID from output: {stdout!r}")
        slurm_job_id: str = job_id_match.group(1)

        result: dict = request.model_dump()
        result.update({'timestamp_started': int(time.time()), 'slurm_job_id': slurm_job_id})
        return RequestLaunched(**result)

    def stop(self, request: RequestLaunched) -> None:
        return None

    def status(self, request: RequestLaunched) -> RequestLaunched | RequestCompleted:
        # Check job status in SLURM
        print(f">>>>>>>>>>>>>>> CHECK JOB STATUS: make debug-one-cluster job_id={request.slurm_job_id} UUID={request.uuid}", flush=True)
        stdout, stderr = execute_command(
            f"cd app; make debug-one-cluster job_id={request.slurm_job_id} UUID={request.uuid}",
            timeout_one_seconds=10.,
        )
        print('DEBUG stdout', stdout, flush=True)
        print('DEBUG stderr', stderr, flush=True)

        # Get job outputs
        sections = re.split(f'({JOB_DETAILS_SEP}|{JOB_ERROR_SEP}|{JOB_OUTPUT_SEP})', stdout)
        if len(sections) != 7:
            raise ValueError(
                f"Unexpected format after splitting sections: expected 7 parts, got {len(sections)}. Sections: {sections}"
            )
        job_details = ""
        job_error = ""
        job_output = ""
        try:
            details_match = re.search(JOB_DETAILS_SEP, stdout)
            error_match = re.search(JOB_ERROR_SEP, stdout)
            output_match = re.search(JOB_OUTPUT_SEP, stdout)
            if details_match is None or error_match is None or output_match is None:
                print("Could not find all section separators in the string.", flush=True)
            else:
                details_index = sections.index(details_match.group(0))
                error_index = sections.index(error_match.group(0))
                output_index = sections.index(output_match.group(0))
                job_details += sections[details_index + 1].strip()
                job_error += sections[error_index + 1].strip()
                job_output += sections[output_index + 1].strip()
        except ValueError:
            print("Error processing sections after splitting.", flush=True)

        # If completed, update request
        status_line = job_details.split('\n')[2]  # third line
        if ('COMPLETED' in status_line) or ('FAILED' in status_line) or ('CANCELLED' in status_line):
            # Get status
            if 'COMPLETED' in status_line:
                status = 'SUCCEEDED'
            else:
                status = 'FAILED'

            # Get metrics
            metrics = parse_markdown_table(job_output)
            if len(metrics) == 0:
                print(f"Warning: No metrics found in job output for request {request.uuid}", flush=True)
            runtime: float = 0
            runtime_entries = [d for d in metrics if d['name'] == 'Runtime data loading seconds (~↓)']
            if runtime_entries:
                runtime = runtime_entries[0]['value']

            # Update
            result = request.model_dump()
            result.update({
                'status': status,
                'metrics': metrics,
                'job_details': job_details,
                'stdout': job_output,
                'stderr': job_error + '\n' + stderr,
                'credits_consumed': CREDITS_PER_HOUR * (runtime / 3600.0),
            })
            return RequestCompleted(**result)
        else:
            return request


class InfraSlurmMock(Infra):
    def launch(self, request: RequestInferred) -> RequestLaunched:
        result: dict = request.model_dump()
        result.update({'timestamp_started': 1234567890, 'slurm_job_id': "123456"})
        return RequestLaunched(**result)

    def stop(self, request: RequestLaunched) -> None:
        return None

    def status(self, request: RequestLaunched) -> RequestLaunched | RequestCompleted:
        result = request.model_dump()
        result.update({'status': "SUCCEEDED", 'metrics': [{"accuracy": 0.95}], 'credits_consumed': 10.0})
        return RequestCompleted(**result)

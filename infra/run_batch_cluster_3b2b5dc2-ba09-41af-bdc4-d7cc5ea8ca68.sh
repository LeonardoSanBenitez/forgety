#!/bin/sh
#SBATCH --job-name=unlearn_forgety                # Name of the job
#SBATCH --partition=gpu                             # Use the GPU partition
#SBATCH --gres=gpu:a100:1	                        # Select machine config (gpu:v100:1 = 1 V100 16GB GPU, gpu:a100:1 = 1 A100 40GB GPU)
#SBATCH --time=12:00:00                              # Set maximum run time for the job (hh:mm:ss)
#SBATCH --output=assets/cluster_jobs/%j-train-out   # Redirect stdout
#SBATCH --error=assets/cluster_jobs/%j-train-err    # Redirect stderr

if [ "$(basename "$PWD")" != "infra" ]; then
    echo "Error: This script must be run from the 'infra' directory."
    exit 1
fi

srun apptainer exec --nv run_cluster.sif accelerate launch --num_processes 2 --num_machines 1 --dynamo_backend no request_3b2b5dc2-ba09-41af-bdc4-d7cc5ea8ca68.py
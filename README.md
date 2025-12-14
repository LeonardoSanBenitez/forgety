# Forgetsy

A web platform to democratize Machine Unlearning. Quick demo of the Forgetsy web UI in action:

![Forgetsy demo](./images/demo.gif)

It enables experimenting with unlearning in a no-code manner, enabling a broader public (such as policy-makers, philosofers, among others) to exlore the benefits and shortcomings of Machine Unlearning.

It is built fully on top of the [Vision Unlearning](https://github.com/LeonardoSanBenitez/vision-unlearning) library.


# Getting started
Run in the terminal, from the root folder:

```bash
docker compose up
```

then go to `http://localhost:8501`

See also the backend documentation at at `http://localhost:8001/docs`. 

## Secret management
Credentials are stored in a git-ignored .env file. The same file should be present in different locations:
* .env
* infra/.env: Needed because the HF credential is exposed to the code running in the cluster, allowing upload.
* services/backend/app/.env: Needed for backend to connect to slurm

It should be included in both in your local folder and in server's clone.

See `.env.template` for their format

## Server setup
Currently, only a Slurm cluster is supported to execute the jobs.

Our setup was tested in the PPKE's ITK HPC cluster (Esztergom) in December of 2025, with the following configurations:
* Tesla V100-PCIE-16GB with CUDA Version: 12.6
* Slurm 23.11.4
* Rhel centos fedora 8.10 (Green Obsidian)
* Vision-unlearning 0.1.6 with python 3.10

## Tech stack
* **Automation and devops**: Docker, makefile, mypi, PEP8, pytest, pypi, readthedocs, git, github (issues and task board)
* **Frontend**: Streamlit, python
* **Backend**: FastAPI, pydantic, jsonschema, python
* **Infrastructure and cluster integration**: SSH, bash, Slurm commands, apptainer
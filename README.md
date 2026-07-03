# Forgety

A web platform to democratize Machine Unlearning. Quick demo of the Forgetsy web UI in action (respectively exploring the results of a benchmark and running a new unlearnign session):

![Forgetsy demo](./images/demo_rt.gif)

![Forgetsy demo](./images/demo_unlearn.gif)

It enables experimenting with unlearning in a no-code manner, enabling a broader public (such as policy-makers, philosofers, among others) to exlore the benefits and shortcomings of Machine Unlearning.

It is built fully on top of the [Vision Unlearning](https://github.com/LeonardoSanBenitez/vision-unlearning) library, and the benchmark exploration feature is compatible with any HuggingFace repository whose structure follows Appendix 2 of I-CARE paper.


# Getting started
Run in the terminal, from the root folder:

```bash
docker compose up
```

then go to `http://localhost:8501`

See also the backend documentation at at `http://localhost:8001/docs`. 

## Testing
Tests run automatically on GitHub Actions for every push and pull request (`.github/workflows/test.yml`): mypy, pycodestyle, and the offline pytest suite (backend API tests, I-CARE route tests, and Streamlit UI tests via `streamlit.testing.v1.AppTest` — no browser or running backend needed).

To run them locally:

```bash
make test          # inside Docker (mypy + pycodestyle + offline pytest)
```

or on the host, with the dependencies from `services/backend/requirements.txt`, `services/frontend/requirements.txt`, `libs/requirements.txt` and `libs/requirements.dev.txt` installed, and a checkout of [vision-unlearning](https://github.com/LeonardoSanBenitez/vision-unlearning) as a sibling directory of this repository:

```bash
python -m pytest tests -m "not gpu and not integration"
```

Tests marked `integration` download real I-CARE data from HuggingFace; they are excluded from all default runs and executed weekly (or on demand) by `.github/workflows/integration.yml`.

## Secret management
Credentials are stored in a git-ignored .env file. The same file should be present in different locations:
* .env
* infra/.env: Needed because the HF credential is exposed to the code running in the cluster, allowing upload.
* services/backend/app/.env: Needed for backend to connect to slurm

The slurm-related secrets are only needed if you want to run unlearning sessions. They should be included in both in your local folder and in server's clone. See `.env.template` for their format.


## Server setup
For executing new unlearning sessions, it is needed to configure a server (that will perform the actual job erxecution). Currently, only a Slurm cluster is supported.

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

# Architecture

![sequence.png](./images/UML-diagrams-sequence.png)

![component-backend.png](./images/UML-diagrams-component-backend.png)

![component-frontend.png](./images/UML-diagrams-component-frontend.png)

![class-database.png](./images/UML-diagrams-class-database.png)

![class-infra.png](./images/UML-diagrams-class-infra.png)

![class-request.png](./images/UML-diagrams-class-request.png)

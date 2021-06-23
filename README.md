
# Bayesian workflow

Examples for the Bayesian workflow.

ref)

* A. Gelman et al., ["Bayesian Workflow"](https://arxiv.org/abs/2011.01808)
* J. Garby et al., ["Visualization in Bayesian workflow"](https://arxiv.org/abs/1709.01449)

# Setup

## Local machine

Install [poetry](https://python-poetry.org/) with the following commands.

```bash
curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python
```

Install the package. The following command will install the Python package in editable mode.

```bash
poetry install
```

## Docker

The provided Dockerfile can run a Docker container for service.

```bash
docker build -t myapp .
docker run -it myapp bash
```

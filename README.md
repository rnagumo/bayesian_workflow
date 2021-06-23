
# myapp

Template for Python package with Poetry.

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

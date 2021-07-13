
# Bayesian workflow

Examples for the Bayesian workflow.

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

# References

## Bayesian workflow

* A. Gelman et al., ["Bayesian Workflow"](https://arxiv.org/abs/2011.01808)
* J. Garby et al., ["Visualization in Bayesian workflow"](https://arxiv.org/abs/1709.01449)

## Lasso

* [縮小事前分布によるベイズ的変数選択1: Bayesian Lasso](https://qiita.com/ssugasawa/items/b0abce4681f1fcb3216e)
* [The Bayesian Lasso](http://hedibert.org/wp-content/uploads/2018/05/park-casella-2008.pdf)

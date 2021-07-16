
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

# spike-and-slap

* [Example for spike-and-slab regression](https://github.com/pyro-ppl/numpyro/issues/769)
* [Spike and slab: Bayesian linear regression with variable selection](http://www.batisengul.co.uk/post/spike-and-slab-bayesian-linear-regression-with-variable-selection/)
* [Spike and Slab model](https://github.com/AsaCooperStickland/Spike_And_Slab)


* [Comparing Spike and Slab Priors for Bayesian Variable Selection](https://arxiv.org/pdf/1812.07259.pdf)
* [Generalized Spike-and-Slab Priors for Bayesian Group Feature Selection Using Expectation Propagation](https://www.jmlr.org/papers/volume14/hernandez-lobato13a/hernandez-lobato13a.pdf)
* [Expectation propagation in linear regression models with spike-and-slab priors](https://link.springer.com/content/pdf/10.1007/s10994-014-5475-7.pdf)

# Tips

## Sampling from discrete latent variables.

* [invalid prng key error when converting to arviz](https://github.com/pyro-ppl/numpyro/issues/857)
* [HMC with Gibbs sampling](https://num.pyro.ai/en/stable/_modules/numpyro/infer/hmc_gibbs.html)
* [Mixed Hamiltonian Monte Carlo for Mixed Discrete and Continuous Variables](https://arxiv.org/abs/1909.04852)
* [Discontinuous Hamiltonian Monte Carlo for discrete parameters and discontinuous likelihoods](https://arxiv.org/abs/1705.08510)

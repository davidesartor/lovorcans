# localvorcans

Batch Bayesian optimization in JAX for many evaluations. Each round:

1. A scrambled Sobol sample of candidates is pushed, by bisection along a direction, onto the boundaries
   of the Voronoi cells of the points evaluated so far.
2. Every candidate gets its own local GP, fitted on its neighbors only
   ([lagpax](https://github.com/davidesartor/lagpax), a JAX port of laGP). Thousands of small
   GPs run in one XLA call.
3. The `batch_size` candidates with the highest log expected improvement
   ([Ament et al., 2023](https://arxiv.org/abs/2310.20708)) are evaluated next.

Test functions and the L-BFGS-B solver behind the GP fits come from
[vlse](https://github.com/davidesartor/vlse) (`jaxvlse` on PyPI).

## Install

```bash
uv sync                    # add --extra tests for pytest, --extra gpu for CUDA 12
```

lagpax is vendored in `vendor/lagpax` as a uv workspace member until it is on PyPI.

## Use

```python
import jax
jax.config.update("jax_enable_x64", True)   # GP math needs float64

import vlse
from localvorcans import minimize

result = minimize(vlse.Ackley(d=20, normalized=True), d=20, n_batches=100)
result["y"].min()
```

`minimize` works on the unit cube. `normalized=True` maps a vlse function onto it and shifts its
minimum to 0.

## Layout

```
src/localvorcans/
  candidates.py    nearest centers, Voronoi-boundary candidates
  acquisition.py   log EI, top-q selection
  bo.py            the optimization loop
experiments/       entry scripts and their sbatch files; results go to outputs/
tests/             R-free unit tests
vendor/lagpax/     vendored lagpax
```

## Run

```bash
uv run pytest                           # on a compute node, not a login node
sbatch experiments/ackley.sbatch --seed 0
```

"""Batch Bayesian optimization: Voronoi candidates, a local GP per candidate, log EI."""

import time
from typing import Callable

import jax.numpy as jnp
import jax.random as jr
import lagpax
import numpy as np
from jaxtyping import Array, Float

from localvorcans.acquisition import log_expected_improvement, top_q
from localvorcans.candidates import latin_hypercube, voronoi_candidates


def minimize(
    f: Callable[[Float[Array, "n d"]], Float[Array, "n"]],
    d: int,
    seed: int = 0,
    n_init: int = 16,
    batch_size: int = 16,
    n_batches: int = 100,
    raw_samples: int = 1024,
    neighbors: int = 50,
    method: str = "alc",
) -> dict:
    """Minimize f over the unit cube, `batch_size` evaluations per round."""
    rng = np.random.default_rng(seed)
    x = latin_hypercube(n_init, d, rng)
    y = f(x)
    acquisition_time = 0.0

    for i, key in enumerate(jr.split(jr.key(seed), n_batches)):
        timer = time.perf_counter()
        candidates_key, gp_key = jr.split(key)

        # candidates on the boundaries of the Voronoi cells of the design
        candidates = latin_hypercube(raw_samples, d, rng)
        candidates = voronoi_candidates(candidates_key, candidates, centers=x)

        # one local GP per candidate; lagpax assumes a zero mean, so center y
        local = lagpax.aGPsep(
            x, y - y.mean(), candidates, end=min(neighbors, len(x)), method=method, key=gp_key
        )
        score = log_expected_improvement(local.mean + y.mean(), local.var, y.min())
        x_new = candidates[top_q(score, batch_size)]
        acquisition_time += time.perf_counter() - timer

        y_new = f(x_new)
        x, y = jnp.concatenate([x, x_new]), jnp.concatenate([y, y_new])
        print(f"round {i + 1}: batch best {y_new.min():.6g}, best {y.min():.6g}", flush=True)

    return dict(x=x, y=y, acquisition_time=acquisition_time)

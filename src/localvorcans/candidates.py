"""Candidate points on the boundaries of the Voronoi cells of the observed design."""

from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Float, Int, Key


def nearest(x: Float[Array, "n d"], centers: Float[Array, "m d"], k: int = 1) -> Int[Array, "n k"]:
    """Indices of the k nearest centers to each row of x."""
    squared_distance = jnp.sum(jnp.square(x[:, None, :] - centers[None, :, :]), axis=-1)
    return jax.lax.top_k(-squared_distance, k)[1]


@eqx.filter_jit
def voronoi_candidates(
    key: Key[Array, ""],
    x0: Float[Array, "n d"],
    centers: Float[Array, "m d"],
    strategy: Literal["uniform", "proj"] = "uniform",
    iterations: int = 10,
) -> Float[Array, "n d"]:
    """Move each x0 along a direction to the boundary of its Voronoi cell, by bisection."""
    n, d = x0.shape
    own_cell = nearest(x0, centers)[:, 0]

    # unit directions scaled by sqrt(d), so a step of 1 always leaves the unit cube
    if strategy == "uniform":
        direction = jr.normal(key, (n, d))
    elif strategy == "proj":
        direction = x0 - centers[own_cell]
    else:
        raise ValueError(f"unknown strategy {strategy!r}")
    direction = direction / jnp.linalg.norm(direction, axis=-1, keepdims=True) * jnp.sqrt(d)

    def step_to(t: Float[Array, "n"]) -> Float[Array, "n d"]:
        return jnp.clip(x0 + direction * t[:, None], 0, 1)

    # bisect on the step: lower stays inside the own cell, upper outside
    def bisect(_, bracket):
        lower, upper = bracket
        mid = (lower + upper) / 2
        inside = nearest(step_to(mid), centers)[:, 0] == own_cell
        return jnp.where(inside, mid, lower), jnp.where(inside, upper, mid)

    lower, upper = jax.lax.fori_loop(0, iterations, bisect, (jnp.zeros(n), jnp.ones(n)))
    return step_to((lower + upper) / 2)

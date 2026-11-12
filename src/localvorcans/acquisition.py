"""Log expected improvement and batch selection."""

import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jaxtyping import Array, Float, Int, Scalar

EPS = float(jnp.sqrt(jnp.finfo(float).eps))


@jax.jit
def log_expected_improvement(
    mean: Float[Array, "n"], var: Float[Array, "n"], y_best: Scalar
) -> Float[Array, "n"]:
    """log EI for minimization, stable far into the tail (arXiv:2310.20708).

    The branches are deliberate: lax.cond keeps NaNs of the unused branch out of the gradient.
    """

    def log_ei(z, sigma):
        branch1 = lambda: jnp.log(z * jsp.stats.norm.cdf(z) + jsp.stats.norm.pdf(z))
        branch2a = lambda: -2 * jnp.log(-z)
        branch2b = lambda: jax.nn.log1mexp(
            -jnp.log(-z) - jsp.stats.norm.logsf(-z) - z**2 / 2 - jnp.log(2 * jnp.pi) / 2
        )
        branch2 = lambda: (
            -(z**2) / 2
            - jnp.log(2 * jnp.pi) / 2
            + jax.lax.cond(z < -1 / jnp.sqrt(EPS), branch2a, branch2b)
        )
        return jnp.log(sigma) + jax.lax.cond(z > -1, branch1, branch2)

    sigma = jnp.sqrt(var)
    return jax.vmap(log_ei)(z=(y_best - mean) / sigma, sigma=sigma)


def top_q(score: Float[Array, "n"], q: int) -> Int[Array, "q"]:
    """Indices of the q highest scores."""
    return jax.lax.top_k(score, q)[1]

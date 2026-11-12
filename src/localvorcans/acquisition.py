from jaxtyping import Array, Bool, Float, Int, Key, Scalar

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import jax.random as jr
import equinox as eqx


@jax.jit
def upper_confidence_bound(
    mu: Float[Array, "..."],
    sigma: Float[Array, "..."],
    beta: Float[Array, "..."],
) -> Float[Array, "..."]:
    return -mu + jnp.sqrt(beta) * sigma

@jax.jit
def log_expected_improvement(
    mu: Float[Array, "..."],
    sigma: Float[Array, "..."],
    y_best: Float[Array, "..."],
) -> Float[Array, "..."]:
    """Stable log EI: log(sigma) + log(pdf(z) + z * cdf(z)) (Ament et al. 2023).
    https://arxiv.org/abs/2310.20708
    """

    # sanitize inputs for the three branches to avoid NaNs and Infs in gradients
    z = (y_best - mu) / sigma
    z_asymptotic = -jnp.finfo(z.dtype).eps ** -0.25
    upper, lower = z > -1, z < z_asymptotic

    # branch1 (z > -1): direct evaluation
    z1 = jnp.where(upper, z, 0.0)
    log_h1 = jnp.log(jsp.stats.norm.pdf(z1) + z1 * jsp.stats.norm.cdf(z1))

    # branch2 (z_asymptotic <= z <= -1): stable log1mexp trick
    z2 = jnp.where(upper | lower, -2.0, z)
    log_h2 = (
        -(z2**2) / 2
        - jnp.log(2 * jnp.pi) / 2
        + jax.nn.log1mexp(
            -jnp.log(-z2)
            - jsp.stats.norm.logsf(-z2)
            - z2**2 / 2
            - jnp.log(2 * jnp.pi) / 2
        )
    )

    # branch3 (z < z_asymptotic): asymptotic expansion
    z3 = jnp.where(lower, z, 2.0 * z_asymptotic)
    log_h3 = -(z3**2) / 2 - jnp.log(2 * jnp.pi) / 2 - 2 * jnp.log(-z3)

    log_h = jnp.where(upper, log_h1, jnp.where(lower, log_h3, log_h2))
    return jnp.log(sigma) + log_h

@jax.jit
def q_log_expected_improvement(
    mu: Float[Array, "... q"],
    cov: Float[Array, "... q q"],
    y_best: Float[Array, "..."],
    eps: Float[Array, "s q"],
    tau_relu: Scalar | float = 1e-6,
    tau_max: Scalar | float = 1e-2,
) -> Float[Array, "..."]:
    """Stable Monte Carlo batch log EI with fat-tailed relu and smooth max (Ament et al. 2023).
    https://arxiv.org/abs/2310.20708
    """
    L = jnp.linalg.cholesky(cov)
    ys = mu[..., None, :] + jnp.einsum("sq,...rq->...sr", eps, L)

    # log softplus(y_best - y) at temperature tau_relu, branches sanitized for finite gradients
    z = (jnp.asarray(y_best)[..., None, None] - ys) / tau_relu
    z_mid, z_high = z.clip(-30.0, 30.0), jnp.where(z > 30.0, z, 1.0)
    log_softplus = jnp.where(
        z > 30.0,
        jnp.log(z_high),
        jnp.where(z < -30.0, z, jnp.log(jax.nn.softplus(z_mid))),
    )

    # cauchy tail keeps gradients alive for samples far worse than the incumbent
    log_cauchy = jnp.log(0.1) - jnp.log1p(z**2)
    log_imp = jnp.log(tau_relu) + jnp.logaddexp(log_softplus, log_cauchy)

    # smooth max over the batch at temperature tau_max, then log mean over the samples
    log_max = tau_max * jax.nn.logsumexp(log_imp / tau_max, axis=-1)
    return jax.nn.logsumexp(log_max, axis=-1) - jnp.log(len(eps))

@eqx.filter_jit
def boltzmann_select(
    key: Key,
    values: Float[Array, "n"],
    n: int,
    eta: float = 2.0,
) -> Int[Array, "n"]:
    """Sample n distinct indices with weights exp(eta * standardized values), keeping the argmax."""
    finite = jnp.isfinite(values)
    v = jnp.where(finite, values, jnp.nan)
    z = (v - jnp.nanmean(v)) / jnp.maximum(jnp.nanstd(v), jnp.finfo(v.dtype).eps)
    logits = jnp.where(finite, eta * z, -jnp.inf).at[jnp.nanargmax(v)].set(jnp.inf)

    # gumbel top-k is sampling without replacement
    _, idx = jax.lax.top_k(logits + jr.gumbel(key, values.shape), n)
    return idx

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np

from localvorcans.acquisition import log_expected_improvement, top_q


def test_log_ei_matches_the_closed_form_where_it_is_stable():
    mean, var = jnp.linspace(-2.0, 2.0, 41), jnp.full(41, 0.5)
    sigma = jnp.sqrt(var)
    z = (0.0 - mean) / sigma
    ei = sigma * (z * jsp.stats.norm.cdf(z) + jsp.stats.norm.pdf(z))
    np.testing.assert_allclose(log_expected_improvement(mean, var, 0.0), jnp.log(ei), rtol=1e-10)


def test_log_ei_is_finite_with_finite_gradients_far_in_the_tail():
    mean, var = jnp.array([1e2, 1e4, 1e8]), jnp.ones(3)
    assert jnp.all(jnp.isfinite(log_expected_improvement(mean, var, 0.0)))
    grad = jax.grad(lambda m: log_expected_improvement(m, var, 0.0).sum())(mean)
    assert jnp.all(jnp.isfinite(grad))


def test_top_q_picks_the_highest_scores():
    assert set(top_q(jnp.array([0.1, 3.0, -1.0, 2.0]), 2).tolist()) == {1, 3}

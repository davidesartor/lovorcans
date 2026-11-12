import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from localvorcans.candidates import latin_hypercube, nearest, voronoi_candidates


@pytest.mark.parametrize("strategy", ["uniform", "proj"])
def test_candidates_sit_on_a_cell_boundary(strategy):
    rng = np.random.default_rng(0)
    centers, x0 = latin_hypercube(20, 3, rng), latin_hypercube(200, 3, rng)
    x = voronoi_candidates(jr.key(0), x0, centers, strategy=strategy, iterations=40)

    # the two nearest centers are equidistant, unless the ray hit the cube first
    two_nearest = nearest(x, centers, k=2)
    gap = jnp.diff(jnp.linalg.norm(x[:, None] - centers[two_nearest], axis=-1), axis=-1)[:, 0]
    on_cube_face = jnp.any((x == 0) | (x == 1), axis=-1)
    assert jnp.all(on_cube_face | (gap < 1e-8))
    assert jnp.all((x >= 0) & (x <= 1))

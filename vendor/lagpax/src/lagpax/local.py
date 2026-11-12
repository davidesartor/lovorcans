"""Local approximate GP prediction: a small GP per predictive location, on a design built for it.

Shapes: X is n by m, Xref has r rows, XX has q rows, a design has `end` rows (e).
"""

from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int, Key, Scalar

from lagpax.design import greedy_design
from lagpax.gp import GP, darg, garg, jmleGP, mleGP, predGP


class MLE(NamedTuple):
    """Lengthscale(s) and nugget a prediction used: fitted where `mle` asked, else the start."""

    d: Float[Array, "..."]
    g: Float[Array, "..."]


class LocalPrediction(NamedTuple):
    """Student-t predictive at the rows of Xref, from the GP on rows `Xi` of X.

    `Sigma` is None when `lite`; `s2` is its diagonal otherwise.
    """

    mean: Float[Array, "r"]
    s2: Float[Array, "r"]
    Sigma: Float[Array, "r r"] | None
    df: Scalar
    llik: Scalar
    Xi: Int[Array, "e"]
    mle: MLE


class ApproxPrediction(NamedTuple):
    """Predictive mean and variance at each row of XX, each from its own local GP."""

    mean: Float[Array, "q"]
    var: Float[Array, "q"]
    llik: Float[Array, "q"]
    Xi: Int[Array, "q e"]
    mle: MLE


@eqx.filter_jit
def laGP(
    Xref: Float[Array, "r m"] | Float[Array, "m"],
    start: int,
    end: int,
    X: Float[Array, "n m"],
    Z: Float[Array, "n"],
    d: dict | Float[Array, "..."] | float | None = None,
    g: dict | Float[Array, ""] | float | None = 1 / 10000,
    method: str = "alc",
    close: int | None = None,
    numstart: int | None = None,
    rect: Float[Array, "2 m"] | None = None,
    lite: bool = True,
    key: Key[Array, ""] | None = None,
) -> LocalPrediction:
    """Prediction at Xref from a GP on `end` rows of X chosen for it, from the `close` nearest.

    The `start` nearest rows open the design; `method` adds the rest: "alc" by scoring every
    candidate, "alcray" and "alcopt" by a search snapped to a candidate, "nn" by distance.
    A length-m `d` start makes the GP separable.
    """
    X, Z, Xref = jnp.asarray(X), jnp.asarray(Z), jnp.atleast_2d(jnp.asarray(Xref))
    n, m = X.shape

    # laGP's default: the searches that do not score every candidate get ten times as many
    if close is None:
        close = min((1000 + end) * (10 if method in ("alcray", "alcopt") else 1), n)

    # settings against each other and against the data
    if method not in ("alc", "alcopt", "alcray", "nn"):
        raise ValueError(f"method must be 'alc', 'alcopt', 'alcray' or 'nn', not {method!r}")
    if Z.shape != (n,) or Xref.ndim != 2 or Xref.shape[1] != m:
        raise ValueError(
            f"need X (n, m), Z (n,) and Xref (r, m) or (m,), not {X.shape}, {Z.shape}, "
            f"{Xref.shape}"
        )
    if not 1 <= start < end <= close <= n:
        raise ValueError(
            f"need 1 <= start < end <= close <= nrow(X), not start={start}, end={end}, "
            f"close={close}, nrow(X)={n}"
        )
    if method == "alcray" and len(Xref) != 1:
        raise ValueError(f"alcray searches along rays from one reference row, not {len(Xref)}")
    if numstart is not None and numstart < 1:
        raise ValueError(f"numstart must be at least 1, not {numstart}")
    if rect is not None and jnp.shape(rect) != (2, m):
        raise ValueError(f"rect must be (2, {m}): lower then upper corner, not {jnp.shape(rect)}")

    # start, range, prior and whether to fit, for the lengthscale(s) and the nugget
    d, g = darg(d, X, key=key), garg(g, Z)
    d_start, g_start = jnp.asarray(d["start"], X.dtype), jnp.asarray(g["start"], X.dtype)
    if d_start.shape not in ((), (m,)) or g_start.shape != ():
        raise ValueError(
            f"d start must be a scalar or one per column of X and g start a scalar, not shapes "
            f"{d_start.shape} and {g_start.shape}"
        )

    # as laGP's C code: candidates that leave fewer than `start` rows out are all the rows
    if close >= n - start:
        close = n

    # rows of X nearest first, by distance to the closest reference row
    squared_distances = jnp.min(
        jnp.sum(jnp.square(X[None, :, :] - Xref[:, None, :]), axis=-1), axis=0
    )
    _, nearest = jax.lax.top_k(-squared_distances, end if method == "nn" else close)

    # the nearest rows, or the greedy design among the candidates at the start values
    if method == "nn":
        Xi = nearest
    else:
        Xi = nearest[
            greedy_design(X[nearest], Xref, d_start, g_start, start, end, method, numstart, rect)
        ]

    # laGP's mle switch: both jointly, either alone, or neither
    gpi = GP(X[Xi], Z[Xi], d_start, g_start)
    if d["mle"] and g["mle"]:
        gpi = jmleGP(gpi, (d["min"], d["max"]), (g["min"], g["max"]), d["ab"], g["ab"])
    elif d["mle"]:
        gpi = mleGP(gpi, "d", d["min"], d["max"], d["ab"])
    elif g["mle"]:
        gpi = mleGP(gpi, "g", g["min"], g["max"], g["ab"])

    predictive = predGP(gpi, Xref, lite=lite)
    return LocalPrediction(
        predictive.mean,
        predictive.s2,
        predictive.Sigma,
        predictive.df,
        predictive.llik,
        Xi,
        MLE(gpi.d, gpi.g),
    )


@eqx.filter_jit
def laGPsep(
    Xref: Float[Array, "r m"] | Float[Array, "m"],
    start: int,
    end: int,
    X: Float[Array, "n m"],
    Z: Float[Array, "n"],
    d: dict | Float[Array, "..."] | float | None = None,
    g: dict | Float[Array, ""] | float | None = 1 / 10000,
    method: str = "alc",
    close: int | None = None,
    numstart: int | None = None,
    rect: Float[Array, "2 m"] | None = None,
    lite: bool = True,
    key: Key[Array, ""] | None = None,
) -> LocalPrediction:
    """`laGP` with one lengthscale per column of X."""
    X = jnp.asarray(X)
    d = darg(d, X, key=key)
    if jnp.ndim(d["start"]) > 1:
        raise ValueError(
            f"d start must be a scalar or one per column of X, not shape {jnp.shape(d['start'])}"
        )

    # one start, lower and upper bound per column
    per_column = {
        name: jnp.broadcast_to(jnp.asarray(d[name], X.dtype), X.shape[1:])
        for name in ("start", "min", "max")
    }
    return laGP(Xref, start, end, X, Z, {**d, **per_column}, g, method, close, numstart, rect, lite)


@eqx.filter_jit
def aGP(
    X: Float[Array, "n m"],
    Z: Float[Array, "n"],
    XX: Float[Array, "q m"],
    start: int = 6,
    end: int = 50,
    d: dict | Float[Array, "..."] | float | None = None,
    g: dict | Float[Array, "..."] | float | None = 1 / 10000,
    method: str = "alc",
    close: int | None = None,
    numrays: int | None = None,
    batch_size: int = 256,
    key: Key[Array, ""] | None = None,
) -> ApproxPrediction:
    """`laGP` at each row of XX on its own, `batch_size` rows at a time.

    The defaults of `d` and `g` are measured once, on all the data. Their starts may be one per
    row of XX, as when a second pass starts from the first's `mle`; a (q, m) `d` start makes the
    GPs separable.
    """
    X, Z, XX = jnp.asarray(X), jnp.asarray(Z), jnp.asarray(XX)
    if X.ndim != 2 or XX.ndim != 2 or XX.shape[1] != X.shape[1]:
        raise ValueError(f"need X (n, m) and XX (q, m), not {X.shape} and {XX.shape}")
    q, m = XX.shape

    # one start per row of XX
    d, g = darg(d, X, key=key), garg(g, Z)
    d_start, g_start = jnp.asarray(d["start"], X.dtype), jnp.asarray(g["start"], X.dtype)
    if d_start.shape not in ((), (q,), (q, m)) or g_start.shape not in ((), (q,)):
        raise ValueError(
            f"d start must be a scalar, one per row of XX or (nrow(XX), ncol(X)) and g start a "
            f"scalar or one per row of XX, not shapes {d_start.shape} and {g_start.shape}"
        )
    d_start = jnp.broadcast_to(d_start, (q, *d_start.shape[1:]))
    g_start = jnp.broadcast_to(g_start, (q,))

    # laGP's aGP casts rays inside the bounding box of all the data, not of the candidates
    rect = jnp.stack([X.min(axis=0), X.max(axis=0)]) if method == "alcray" else None

    def predict_one(location):
        Xref, d_here, g_here = location
        local = laGP(
            Xref,
            start,
            end,
            X,
            Z,
            {**d, "start": d_here},
            {**g, "start": g_here},
            method,
            close,
            numrays,
            rect,
        )

        # variance of the Student-t whose squared scale is s2
        var = local.s2[0] * local.df / (local.df - 2.0)
        return ApproxPrediction(local.mean[0], var, local.llik, local.Xi, local.mle)

    return jax.lax.map(predict_one, (XX, d_start, g_start), batch_size=batch_size)


@eqx.filter_jit
def aGPsep(
    X: Float[Array, "n m"],
    Z: Float[Array, "n"],
    XX: Float[Array, "q m"],
    start: int = 6,
    end: int = 50,
    d: dict | Float[Array, "..."] | float | None = None,
    g: dict | Float[Array, "..."] | float | None = 1 / 10000,
    method: str = "alc",
    close: int | None = None,
    numrays: int | None = None,
    batch_size: int = 256,
    key: Key[Array, ""] | None = None,
) -> ApproxPrediction:
    """`aGP` with one lengthscale per column of X.

    A `d` start is a scalar, one per column of X, or (q, m): one per column for each row of XX.
    """
    X, XX = jnp.asarray(X), jnp.asarray(XX)
    if X.ndim != 2 or XX.ndim != 2 or XX.shape[1] != X.shape[1]:
        raise ValueError(f"need X (n, m) and XX (q, m), not {X.shape} and {XX.shape}")
    d = darg(d, X, key=key)
    if jnp.shape(d["start"]) not in ((), X.shape[1:], XX.shape):
        raise ValueError(
            f"d start must be a scalar, one per column of X or (nrow(XX), ncol(X)), not shape "
            f"{jnp.shape(d['start'])}"
        )

    # one start per row of XX and column of X, one lower and upper bound per column
    per_column = {
        "start": jnp.broadcast_to(jnp.asarray(d["start"], X.dtype), XX.shape),
        "min": jnp.broadcast_to(jnp.asarray(d["min"], X.dtype), X.shape[1:]),
        "max": jnp.broadcast_to(jnp.asarray(d["max"], X.dtype), X.shape[1:]),
    }
    return aGP(X, Z, XX, start, end, {**d, **per_column}, g, method, close, numrays, batch_size)

"""laGP's Gaussian process layer: zero mean, Gaussian correlation, output scale integrated out.

Notation: d lengthscale as in exp(-|x - x'|^2 / d), one for all inputs (isotropic) or one per
input (separable), nugget g, K = C + g I with lower Cholesky factor K_chol, Ki = K^-1,
psi = Z' Ki Z. A correlation between two sets of points carries their letters, as in Kqn: n the
data, q the predictive locations XX.
"""

import math
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import scipy.special
from jax.scipy.linalg import cho_solve, cholesky
from jax.scipy.special import gammaln
from jaxtyping import Array, ArrayLike, Float, Key, Scalar
from vlse.optim import minimise

# double precision whatever the working dtype; R's optim stops at a relative change of 1e7 of it
MACHINE_EPSILON = float(np.finfo(float).eps)
ROOT_MACHINE_EPSILON = math.sqrt(MACHINE_EPSILON)
RELATIVE_TOLERANCE = 1e7 * MACHINE_EPSILON


class GP(NamedTuple):
    """Data and hyperparameters of a GP, isotropic if d is a scalar and separable if a vector."""

    X: Float[Array, "n m"]
    Z: Float[Array, "n"]
    d: Float[Array, ""] | Float[Array, "m"]
    g: Scalar


class Prediction(NamedTuple):
    """Student-t predictive at q locations: scale s2, the diagonal of Sigma (None when lite)."""

    mean: Float[Array, "q"]
    s2: Float[Array, "q"]
    Sigma: Float[Array, "q q"] | None
    df: Scalar
    llik: Scalar


def covar(
    X1: Float[Array, "... a m"],
    X2: Float[Array, "... b m"],
    d: Float[Array, ""] | Float[Array, "m"],
) -> Float[Array, "... a b"]:
    """exp(-sum_k (x1_k - x2_k)^2 / d_k) between every row of X1 and every row of X2."""
    deltas = X1[..., :, None, :] - X2[..., None, :, :]
    return jnp.exp(-jnp.sum(jnp.square(deltas) / d, axis=-1))


def factorize(gpi: GP) -> tuple[Float[Array, "n n"], Float[Array, "n"], Scalar, Scalar]:
    """K_chol, Ki_Z = Ki Z, psi and the log-likelihood with the output scale integrated out."""
    X, Z, d, g = gpi
    K = covar(X, X, d) + g * jnp.eye(len(Z), dtype=X.dtype)
    K_chol = cholesky(K, lower=True)
    Ki_Z = cho_solve((K_chol, True), Z)
    psi = Z @ Ki_Z
    log_det_K = 2.0 * jnp.sum(jnp.log(jnp.diagonal(K_chol)))
    return K_chol, Ki_Z, psi, -0.5 * (len(Z) * jnp.log(0.5 * psi) + log_det_K)


def with_implicit_gradient(
    objective: Callable[[Float[Array, "p"]], Scalar],
    solution: Float[Array, "p"],
    lower: Float[Array, "p"],
    upper: Float[Array, "p"],
) -> Float[Array, "p"]:
    """`solution`, unchanged in value, carrying the derivative of the argmin of `objective`."""
    solution = jax.lax.stop_gradient(solution)
    free = (solution > lower) & (solution < upper)

    # a Newton step from the frozen solution has the argmin's derivative (implicit function
    # theorem); coordinates on a bound do not move, so they get none
    hessian = jax.lax.stop_gradient(jax.hessian(objective)(solution))
    hessian = jnp.where(
        free[:, None] & free[None, :], hessian, jnp.eye(len(free), dtype=hessian.dtype)
    )
    gradient = jnp.where(free, jax.grad(objective)(solution), 0.0)
    newton_step = jnp.linalg.solve(hessian, gradient)
    newton_step = jnp.where(free & jnp.isfinite(newton_step), newton_step, 0.0)

    # subtracting the step's own value leaves the solution bit-identical
    return solution - (newton_step - jax.lax.stop_gradient(newton_step))


def newGP(X: ArrayLike, Z: ArrayLike, d: ArrayLike, g: ArrayLike) -> GP:
    """GP on the data (X, Z) at lengthscale d and nugget g; a length-m d makes it separable."""
    X = jnp.asarray(X)
    X = X if jnp.issubdtype(X.dtype, jnp.floating) else X.astype(float)
    Z, d, g = (jnp.asarray(value, X.dtype) for value in (Z, d, g))
    if X.ndim != 2 or Z.shape != X.shape[:1]:
        raise ValueError(f"X must be n x m and Z of length n, not {X.shape} and {Z.shape}")
    if d.shape not in ((), X.shape[1:]) or g.shape != ():
        raise ValueError(
            f"d must be a scalar or of length {X.shape[1]} and g a scalar, "
            f"not of shapes {d.shape} and {g.shape}"
        )
    return GP(X, Z, d, g)


def newGPsep(X: ArrayLike, Z: ArrayLike, d: ArrayLike, g: ArrayLike) -> GP:
    """Separable GP on the data (X, Z); a scalar d is repeated for every input."""
    return newGP(X, Z, jnp.broadcast_to(jnp.asarray(d), jnp.shape(X)[1:]), g)


def updateGP(gpi: GP, X: Float[Array, "k m"], Z: Float[Array, "k"]) -> GP:
    """GP with the rows (X, Z) added to its data."""
    X, Z = jnp.asarray(X, gpi.X.dtype), jnp.asarray(Z, gpi.Z.dtype)
    if X.ndim != 2 or X.shape[1:] != gpi.X.shape[1:] or Z.shape != X.shape[:1]:
        raise ValueError(
            f"X must be k x {gpi.X.shape[1]} and Z of length k, not {X.shape} and {Z.shape}"
        )
    return gpi._replace(X=jnp.concatenate([gpi.X, X]), Z=jnp.concatenate([gpi.Z, Z]))


def llikGP(gpi: GP, dab: ArrayLike = (0.0, 0.0), gab: ArrayLike = (0.0, 0.0)) -> Scalar:
    """Log-likelihood plus the log Gamma (shape, rate) prior densities of d and of g.

    A prior applies only if its shape and rate are both positive: (0, 0) is none.
    """
    _, _, _, llik = factorize(gpi)
    for theta, ab in ((gpi.d, dab), (gpi.g, gab)):
        shape, rate = jnp.asarray(ab, gpi.X.dtype)

        # density at stand-in values where there is no prior, so that no nan reaches a gradient
        has_prior = (shape > 0) & (rate > 0)
        shape, rate = jnp.where(has_prior, shape, 1.0), jnp.where(has_prior, rate, 1.0)
        log_density = (
            shape * jnp.log(rate) - gammaln(shape) + (shape - 1.0) * jnp.log(theta) - rate * theta
        )
        llik = llik + jnp.where(has_prior, jnp.sum(log_density), 0.0)
    return llik


def predGP(gpi: GP, XX: Float[Array, "q m"], lite: bool = False, nonug: bool = False) -> Prediction:
    """Student-t predictive at the rows of XX, with n degrees of freedom.

    `lite` skips the covariance between the rows; `nonug` predicts the latent function,
    swapping the nugget of the new points for sqrt(machine epsilon) as laGP does.
    """
    X, Z, d, g = gpi
    XX = jnp.asarray(XX, X.dtype)
    K_chol, Ki_Z, psi, llik = factorize(gpi)
    Kqn = covar(XX, X, d)
    Ki_Knq = cho_solve((K_chol, True), Kqn.T)
    g_new = ROOT_MACHINE_EPSILON if nonug else g

    # scale of a Student-t: psi / n * (Kqq + g I - Kqn Ki Knq)
    if lite:
        Sigma = None
        s2 = psi / len(Z) * (1.0 + g_new - jnp.sum(Kqn * Ki_Knq.T, axis=-1))
    else:
        Kqq = covar(XX, XX, d) + g_new * jnp.eye(len(XX), dtype=X.dtype)
        Sigma = psi / len(Z) * (Kqq - Kqn @ Ki_Knq)
        s2 = jnp.diagonal(Sigma)
    return Prediction(Kqn @ Ki_Z, s2, Sigma, jnp.asarray(len(Z), X.dtype), llik)


def jmleGP(
    gpi: GP,
    drange: tuple[ArrayLike, ArrayLike] = (ROOT_MACHINE_EPSILON, 10.0),
    grange: tuple[ArrayLike, ArrayLike] = (ROOT_MACHINE_EPSILON, 1.0),
    dab: ArrayLike = (0.0, 0.0),
    gab: ArrayLike = (0.0, 0.0),
    maxit: int = 100,
) -> GP:
    """GP at the (d, g) maximizing llikGP inside the ranges, by L-BFGS-B from gpi's own (d, g).

    A range whose ends coincide holds its parameter there; a lower end <= 0 is sqrt(machine
    epsilon). One joint solve, where laGP alternates between d and g.
    """
    X, Z, d, g = gpi

    # start and bounds as vectors [d..., g], what the solver moves
    start, lower, upper = (
        jnp.hstack([jnp.broadcast_to(jnp.asarray(d_end), jnp.atleast_1d(d).shape), g_end])
        for d_end, g_end in ((d, g), *zip(drange, grange))
    )
    start, lower, upper = (value.astype(X.dtype) for value in (start, lower, upper))
    lower = jnp.where(lower > 0, lower, ROOT_MACHINE_EPSILON)
    start = jnp.clip(start, lower, upper)

    def negative_log_posterior(parameters, X, Z, dab, gab):
        return -llikGP(GP(X, Z, parameters[:-1], parameters[-1]), dab, gab)

    # the solver is a while loop, which reverse mode cannot cross: it sees constants only
    data = (X, Z, jnp.asarray(dab, X.dtype), jnp.asarray(gab, X.dtype))
    constants = jax.lax.stop_gradient(data)

    # solved in logs, where a lengthscale and a nugget orders of magnitude smaller share a scale
    log_start, log_lower, log_upper = jnp.log(
        jax.lax.stop_gradient(jnp.stack([start, lower, upper]))
    )
    held = log_lower == log_upper

    def solver_objective(log_parameters):
        # a held parameter's gradient would swamp the curvature scale L-BFGS-B estimates
        log_parameters = jnp.where(held, jax.lax.stop_gradient(log_parameters), log_parameters)
        return negative_log_posterior(jnp.exp(log_parameters), *constants)

    log_solution = minimise(
        solver_objective,
        log_start,
        bounds=(log_lower, log_upper),
        tol=0.0,
        ftol=RELATIVE_TOLERANCE,
        max_iterations=maxit,
    ).x
    on_lower, on_upper = log_solution <= log_lower, log_solution >= log_upper

    def pinned(parameters):
        # a value the solver left on a bound is that bound exactly, and moves with it
        return jnp.where(on_lower, lower, jnp.where(on_upper, upper, parameters))

    parameters = pinned(
        with_implicit_gradient(
            lambda parameters: negative_log_posterior(pinned(parameters), *data),
            pinned(jnp.exp(log_solution)),
            lower,
            upper,
        )
    )
    return GP(X, Z, parameters[:-1] if jnp.ndim(d) else parameters[0], parameters[-1])


def mleGP(
    gpi: GP,
    param: str = "d",
    tmin: ArrayLike = ROOT_MACHINE_EPSILON,
    tmax: ArrayLike = -1.0,
    ab: ArrayLike = (0.0, 0.0),
) -> GP:
    """GP at the d or the g (`param`) maximizing llikGP in [tmin, tmax], the other held.

    A tmax <= 0 is m^2 for d and no bound for g, as in laGP.
    """
    if param not in ("d", "g"):
        raise ValueError(f"param must be 'd' or 'g', not {param!r}")
    tmax = jnp.asarray(tmax, gpi.X.dtype)
    if param == "d":
        dmax = jnp.where(tmax > 0, tmax, gpi.X.shape[1] ** 2)
        return jmleGP(gpi, (tmin, dmax), (gpi.g, gpi.g), dab=ab)
    return jmleGP(gpi, (gpi.d, gpi.d), (tmin, jnp.where(tmax > 0, tmax, jnp.inf)), gab=ab)


def mleGPsep(
    gpsepi: GP,
    param: str = "d",
    tmin: ArrayLike = (ROOT_MACHINE_EPSILON, ROOT_MACHINE_EPSILON),
    tmax: ArrayLike = (-1.0, 1.0),
    ab: ArrayLike = (0.0, 0.0, 0.0, 0.0),
    maxit: int = 100,
) -> GP:
    """GP at the d, the g or "both" (`param`) maximizing llikGP in [tmin, tmax], the rest held.

    Bounds are (d, g) or (d_1, ..., d_m, g) for "both". As in laGP, "d" takes one bound or m,
    reading a pair as its first entry unless m = 2, and "g" reads the first entry; a tmax <= 0
    is m^2 for d and no bound for a g fitted alone.
    """
    if param not in ("d", "g", "both"):
        raise ValueError(f"param must be 'd', 'g' or 'both', not {param!r}")
    m = gpsepi.X.shape[1]
    tmin, tmax, ab = (
        jnp.atleast_1d(jnp.asarray(value, gpsepi.X.dtype)) for value in (tmin, tmax, ab)
    )

    # priors: (d shape, d rate, g shape, g rate), or the pair of the one parameter fitted
    if ab.size == 4:
        dab, gab = ab[:2], ab[2:]
    elif ab.size == 2 and param != "both":
        dab = gab = ab
    else:
        raise ValueError(f"ab must hold 4 values, or 2 unless param is 'both', not {ab.size}")

    if param == "g":
        gmax = jnp.where(tmax[0] > 0, tmax[0], jnp.inf)
        return jmleGP(gpsepi, (gpsepi.d, gpsepi.d), (tmin[0], gmax), gab=gab, maxit=maxit)
    if param == "d":
        dmin, dmax = (
            end[0] if end.size == 1 or (end.size == 2 and m != 2) else end for end in (tmin, tmax)
        )
        grange, gab = (gpsepi.g, gpsepi.g), (0.0, 0.0)
    else:
        dmin, dmax, grange = tmin[:-1], tmax[:-1], (tmin[-1], tmax[-1])
    return jmleGP(gpsepi, (dmin, jnp.where(dmax > 0, dmax, m**2)), grange, dab, gab, maxit)


def darg(
    d: ArrayLike | dict | None,
    X: Float[Array, "n m"],
    samp_size: int = 1000,
    key: Key[Array, ""] | None = None,
) -> dict:
    """`d` (None, a start or a partial dict) as a dict of start, min, max, mle and ab.

    Blanks get laGP's defaults, measured on the squared distances between rows of X, or
    between `samp_size` rows drawn with `key` if there are more. ab is the Gamma (shape, rate)
    prior of a fit, its rate may be None. Defaults are constants under differentiation.
    """
    d = {} if d is None else dict(d) if isinstance(d, dict) else {"start": d}
    start, lower, upper, mle = d.get("start"), d.get("min"), d.get("max"), bool(d.get("mle", True))
    shape, rate = (1.5, None) if d.get("ab") is None else d["ab"]

    # squared distance of each pair of distinct rows once, only if a default needs them
    if start is None or (mle and (lower is None or upper is None or rate is None)):
        X = jax.lax.stop_gradient(jnp.asarray(X))
        if len(X) > samp_size:
            if key is None:
                raise ValueError(
                    f"the defaults of d measure distances on {samp_size} of the {len(X)} rows, "
                    "drawn at random: pass key=jax.random.key(...)"
                )
            X = X[jr.choice(key, len(X), (samp_size,), replace=False)]
        D = jnp.triu(jnp.sum(jnp.square(X[:, None, :] - X[None, :, :]), axis=-1), k=1)
        D = jnp.where(D > 0, D, jnp.nan)
        D_start, D_min, D_max = jnp.nanquantile(D, 0.1), jnp.nanmin(D), jnp.nanmax(D)

    # start at the 10% quantile; a fit stays between half the smallest distance and the largest
    if start is None:
        start = D_start
    if upper is None:
        upper = D_max if mle else jnp.max(jnp.asarray(start))
    if lower is None:
        lower = D_min / 2 if mle else jnp.min(jnp.asarray(start))
        lower = jnp.maximum(lower, ROOT_MACHINE_EPSILON)

    # the prior of a fit holds 95% of its mass below the largest distance
    if not mle:
        shape, rate = 0.0, 0.0
    elif rate is None:
        rate = float(scipy.special.gammaincinv(shape, 0.95)) / D_max
    return {"start": start, "min": lower, "max": upper, "mle": mle, "ab": (shape, rate)}


def garg(g: ArrayLike | dict | None, y: Float[Array, "n"]) -> dict:
    """`g` (None, a start or a partial dict) as a dict of start, min, max, mle and ab.

    Blanks get laGP's defaults, measured on the squared deviations of y from its mean. ab is
    the Gamma (shape, rate) prior of a fit, its rate may be None. Defaults are constants under
    differentiation.
    """
    g = {} if g is None else dict(g) if isinstance(g, dict) else {"start": g}
    start, lower, upper, mle = g.get("start"), g.get("min"), g.get("max"), bool(g.get("mle", False))
    shape, rate = (1.5, None) if g.get("ab") is None else g["ab"]

    # squared deviations from the mean, only if a default needs them
    if start is None or (mle and (upper is None or rate is None)):
        y = jax.lax.stop_gradient(jnp.asarray(y))
        r2s = jnp.square(y - jnp.mean(y))

    # start at the 2.5% quantile; a fit stays below the largest deviation
    if start is None:
        start = jnp.quantile(r2s, 0.025)
    if upper is None:
        upper = jnp.max(r2s) if mle else jnp.max(jnp.asarray(start))
    if lower is None:
        lower = ROOT_MACHINE_EPSILON

    # the prior of a fit holds 95% of its mass below the mean deviation
    if not mle:
        shape, rate = 0.0, 0.0
    elif rate is None:
        rate = float(scipy.special.gammaincinv(shape, 0.95)) / jnp.mean(r2s)
    return {"start": start, "min": lower, "max": upper, "mle": mle, "ab": (shape, rate)}


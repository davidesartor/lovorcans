"""Active learning Cohn (ALC) designs: add the point that most cuts the variance at a reference.

Notation: d and g as in `gp`, K = C + g I, Ki = K^-1, psi = Z' Ki Z. A correlation carries the
letters of the two sets it joins, as in Kpn: n the design, p points that may join it, r the
reference Xref and c the candidates Xcand, which arrive sorted nearest the reference first.
"""

import math
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.linalg import cho_solve, cholesky
from jaxtyping import Array, Bool, Float, Int, Scalar
from vlse.optim import minimise

from lagpax.gp import GP, RELATIVE_TOLERANCE, ROOT_MACHINE_EPSILON, covar

GOLDEN_RATIO_CONJUGATE = (5.0**0.5 - 1.0) / 2.0


def variance_reduction(
    Kpn: Float[Array, "... n"],
    Kpr: Float[Array, "... r"],
    Knr: Float[Array, "n r"],
    Ki: Float[Array, "n n"],
    g: Scalar,
) -> Float[Array, "..."]:
    """Drop in variance, averaged over the reference and per unit scale, if a point joined.

    A point the design already determines scores -inf.
    """
    Kpn_Ki = Kpn @ Ki
    variance_p = 1.0 + g - jnp.sum(Kpn_Ki * Kpn, axis=-1)
    covariance_pr = Kpr - Kpn_Ki @ Knr
    reduction = jnp.mean(jnp.square(covariance_pr), axis=-1) / variance_p
    return jnp.where(variance_p > ROOT_MACHINE_EPSILON, reduction, -jnp.inf)


def bordered_inverse(
    Ki: Float[Array, "n n"],
    Knp: Float[Array, "n"],
    g: Scalar,
    slot: Int[Array, ""],
) -> Float[Array, "n n"]:
    """Ki once the point with correlations Knp to the design fills `slot`."""
    Ki_Knp = Ki @ Knp
    variance_p = 1.0 + g - Knp @ Ki_Knp
    Ki = Ki + jnp.outer(Ki_Knp, Ki_Knp) / variance_p
    border = (-Ki_Knp / variance_p).at[slot].set(1.0 / variance_p)
    return Ki.at[slot, :].set(border).at[:, slot].set(border)


def nearest_untaken(
    point: Float[Array, "m"], Xcand: Float[Array, "c m"], taken: Bool[Array, "c"]
) -> Int[Array, ""]:
    """Untaken candidate nearest the point."""
    squared_distances = jnp.sum(jnp.square(Xcand - point), axis=-1)
    return jnp.argmin(jnp.where(taken, jnp.inf, squared_distances))


def untaken_by_rank(
    taken: Bool[Array, "c"], first_rank: Int[Array, ""], count: int
) -> Int[Array, "{count}"]:
    """Untaken candidates ranked `first_rank`, `first_rank + 1`, ... nearest first, wrapping around.

    The running count of untaken candidates is each one's rank.
    """
    untaken_so_far = jnp.cumsum(~taken)
    ranks = jnp.mod(first_rank + jnp.arange(count), jnp.maximum(untaken_so_far[-1], 1))
    return jnp.searchsorted(untaken_so_far, ranks + 1, side="left")


def ray_ends(
    Xref: Float[Array, "m"],
    Xstart: Float[Array, "k m"],
    lower: Float[Array, "m"],
    upper: Float[Array, "m"],
) -> Float[Array, "k m"]:
    """Each ray runs ten times its start's distance from the reference, or to the box if nearer."""
    direction = 10.0 * (Xstart - Xref)
    room = jnp.where(direction > 0, upper - Xstart, lower - Xstart)
    safe_direction = jnp.where(direction != 0.0, direction, 1.0)
    fraction_to_box = jnp.min(jnp.where(direction != 0.0, room / safe_direction, jnp.inf), axis=-1)
    return Xstart + jnp.clip(fraction_to_box, 0.0, 1.0)[:, None] * direction


def golden_section_maximize(
    function: Callable[[Float[Array, "k"]], Float[Array, "k"]],
    lower: Float[Array, "k"],
    upper: Float[Array, "k"],
    steps: int,
) -> tuple[Float[Array, "k"], Float[Array, "k"]]:
    """Maximizers and maxima on k intervals at once, each interval shrinking 38% per step."""

    def narrow(_, bracket):
        lower, upper = bracket
        width = GOLDEN_RATIO_CONJUGATE * (upper - lower)
        probe_low, probe_high = upper - width, lower + width
        keep_left = function(probe_low) > function(probe_high)
        return jnp.where(keep_left, lower, probe_low), jnp.where(keep_left, probe_high, upper)

    lower, upper = jax.lax.fori_loop(0, steps, narrow, (lower, upper))
    maximizers = (lower + upper) / 2
    return maximizers, function(maximizers)


def ray_search(
    score: Callable[[Float[Array, "k m"]], Float[Array, "k"]],
    Xstart: Float[Array, "k m"],
    Xend: Float[Array, "k m"],
    golden_section_steps: int = 30,
) -> tuple[Float[Array, "k"], Float[Array, "k"]]:
    """How far from each start to its end the score peaks, and the score there.

    As laGP, the start itself (fraction 0) wins whenever it scores higher than the peak found.
    """
    fractions, scores = golden_section_maximize(
        lambda fractions: score(Xstart + fractions[:, None] * (Xend - Xstart)),
        jnp.zeros(len(Xstart), Xstart.dtype),
        jnp.ones(len(Xstart), Xstart.dtype),
        golden_section_steps,
    )
    scores_at_start = score(Xstart)
    at_start = (fractions < ROOT_MACHINE_EPSILON) | (scores_at_start > scores)
    return jnp.where(at_start, 0.0, fractions), jnp.where(at_start, scores_at_start, scores)


def continuous_search(
    score: Callable[[Float[Array, "k m"]], Float[Array, "k"]],
    start: Float[Array, "m"],
    lower: Float[Array, "m"],
    upper: Float[Array, "m"],
    maxit: int = 100,
) -> tuple[Float[Array, "m"], Scalar]:
    """Point of the box where L-BFGS-B from `start` stops minimizing -log score, and that value.

    Stops as R's optim does inside laGP: projected gradient below 0.1 or a relative decrease
    below 1e7 machine epsilons. A score of zero or less counts as the smallest positive number.
    """
    smallest_positive = jnp.finfo(start.dtype).tiny
    solution = minimise(
        lambda point: -jnp.log(jnp.maximum(score(point[None])[0], smallest_positive)),
        start,
        bounds=(lower, upper),
        tol=0.1,
        ftol=RELATIVE_TOLERANCE,
        max_iterations=maxit,
        history_length=5,
    )
    return solution.x, solution.f


def alc_scorer(
    gpi: GP, Xref: Float[Array, "r m"]
) -> Callable[[Float[Array, "k m"]], Float[Array, "k"]]:
    """ALC of any points for this GP and reference, in laGP's scale; K is inverted once."""
    n = len(gpi.X)
    K = covar(gpi.X, gpi.X, gpi.d) + gpi.g * jnp.eye(n, dtype=gpi.X.dtype)
    Ki = cho_solve((cholesky(K, lower=True), True), jnp.eye(n, dtype=K.dtype))
    psi = gpi.Z @ Ki @ gpi.Z
    Knr = covar(gpi.X, Xref, gpi.d)

    def score(points: Float[Array, "k m"]) -> Float[Array, "k"]:
        Kpn = covar(points, gpi.X, gpi.d)
        Kpr = covar(points, Xref, gpi.d)
        return psi / (n - 2) * variance_reduction(Kpn, Kpr, Knr, Ki, gpi.g)

    return score


def alcGP(
    gpi: GP, Xcand: Float[Array, "c m"], Xref: Float[Array, "r m"] | None = None
) -> Float[Array, "c"]:
    """Reduction in predictive variance, averaged over Xref, from adding each candidate alone.

    Xref defaults to the candidates themselves.
    """
    Xcand = jnp.asarray(Xcand)
    Xref = Xcand if Xref is None else jnp.atleast_2d(jnp.asarray(Xref))
    return alc_scorer(gpi, Xref)(Xcand)


def alcrayGP(
    gpi: GP,
    Xref: Float[Array, "m"],
    Xstart: Float[Array, "k m"],
    Xend: Float[Array, "k m"],
) -> tuple[Int[Array, ""], Scalar]:
    """Ray with the highest ALC and how far from its start to its end the peak sits: (r, s)."""
    Xref, Xstart, Xend = (jnp.atleast_2d(jnp.asarray(points)) for points in (Xref, Xstart, Xend))
    if len(Xref) != 1:
        raise ValueError(f"a ray search takes one reference location, not {len(Xref)}")
    if Xstart.shape != Xend.shape:
        raise ValueError(f"Xstart {Xstart.shape} and Xend {Xend.shape} must pair up")

    fractions, scores = ray_search(alc_scorer(gpi, Xref), Xstart, Xend)
    r = jnp.argmax(scores)
    return r, fractions[r]


def alcoptGP(
    gpi: GP,
    Xref: Float[Array, "r m"],
    start: Float[Array, "m"],
    lower: Float[Array, "m"] | float,
    upper: Float[Array, "m"] | float,
    maxit: int = 100,
) -> tuple[Float[Array, "m"], Scalar]:
    """Point of the box [lower, upper] maximizing ALC by L-BFGS-B from `start`: (par, value).

    `value` is -log ALC at `par`, the objective laGP reports.
    """
    # the solver is a while loop, which reverse mode cannot cross
    gpi, Xref, start = jax.lax.stop_gradient(
        (gpi, jnp.atleast_2d(jnp.asarray(Xref)), jnp.asarray(start))
    )
    lower, upper = (
        jnp.broadcast_to(jnp.asarray(bound, start.dtype), start.shape) for bound in (lower, upper)
    )
    return continuous_search(alc_scorer(gpi, Xref), start, lower, upper, maxit)


def greedy_design(
    Xcand: Float[Array, "c m"],
    Xref: Float[Array, "r m"],
    d: Float[Array, "m"] | Scalar,
    g: Scalar,
    start: int,
    end: int,
    method: str = "alc",
    numstart: int | None = None,
    rect: Float[Array, "2 m"] | None = None,
) -> Int[Array, "{end}"]:
    """Rows of Xcand, sorted nearest Xref first, forming the design of size `end` for Xref.

    The first `start` rows open the design; each later slot takes the untaken row with the
    largest ALC, found by scoring them all ("alc"), or the row nearest the best point of
    `numstart` rays ("alcray") or L-BFGS-B runs ("alcopt") inside the box `rect`.
    """
    # the design is a discrete choice: nothing to differentiate
    Xcand, Xref, d, g = jax.lax.stop_gradient(
        (jnp.asarray(Xcand), jnp.atleast_2d(jnp.asarray(Xref)), jnp.asarray(d), jnp.asarray(g))
    )

    # settings against each other and against the data
    if method not in ("alc", "alcray", "alcopt"):
        raise ValueError(f"method must be 'alc', 'alcray' or 'alcopt', not {method!r}")
    if not 1 <= start < end <= len(Xcand):
        raise ValueError(f"need 1 <= start < end <= candidates, not {start}, {end}, {len(Xcand)}")
    if method == "alcray" and len(Xref) != 1:
        raise ValueError(f"alcray takes one reference location, not {len(Xref)}")

    # laGP's defaults: a ray per input dimension, one L-BFGS-B run, the box of the candidates
    # that do not open the design
    if numstart is None:
        numstart = Xcand.shape[1] if method == "alcray" else 1
    numstart = min(numstart, len(Xcand) - end + 1)
    if rect is None:
        lower, upper = Xcand[start:].min(axis=0), Xcand[start:].max(axis=0)
    else:
        lower, upper = jax.lax.stop_gradient(jnp.asarray(rect, Xcand.dtype))

    # which untaken row the searches start from at each step: laGP's round robin
    greedy_steps = end - start
    if method == "alcray":
        first_ranks = np.array([step % math.isqrt(step) for step in range(1, greedy_steps + 1)])
    else:
        first_ranks = np.arange(greedy_steps)

    # empty slots are zero rows and columns of Ki, so whatever fills them is ignored
    Kcr = covar(Xcand, Xref, d)
    K_start = covar(Xcand[:start], Xcand[:start], d) + g * jnp.eye(start, dtype=Xcand.dtype)
    Ki = jnp.zeros((end, end), Xcand.dtype).at[:start, :start].set(jnp.linalg.inv(K_start))
    chosen = jnp.arange(end) * (jnp.arange(end) < start)
    taken = jnp.arange(len(Xcand)) < start

    def add_best(state, step):
        chosen, taken, Ki = state
        slot, first_rank = step
        X_design, Knr = Xcand[chosen], Kcr[chosen]

        def score(points: Float[Array, "k m"]) -> Float[Array, "k"]:
            return variance_reduction(
                covar(points, X_design, d), covar(points, Xref, d), Knr, Ki, g
            )

        if method == "alc":
            best = jnp.argmax(jnp.where(taken, -jnp.inf, score(Xcand)))
        elif method == "alcray":
            start_rows = untaken_by_rank(taken, first_rank, numstart)
            Xstart = Xcand[start_rows]
            Xend = ray_ends(Xref[0], Xstart, lower, upper)
            fractions, scores = ray_search(score, Xstart, Xend)
            ray = jnp.argmax(jnp.where(jnp.isfinite(scores), scores, -jnp.inf))
            optimum = Xstart[ray] + fractions[ray] * (Xend[ray] - Xstart[ray])

            # as laGP, a peak past its start snaps to a row ranked after the ray starts (any
            # untaken row if none is left); a peak at the start is that start's own row
            untaken_so_far = jnp.cumsum(~taken)
            past_starts = first_rank + numstart
            past_starts = jnp.where(past_starts < untaken_so_far[-1], past_starts, 0)
            snapped = nearest_untaken(optimum, Xcand, taken | (untaken_so_far <= past_starts))
            best = jnp.where(fractions[ray] > 0.0, snapped, start_rows[ray])
        else:
            # each run ends on the untaken row nearest its optimum; the best such row wins
            start_rows = untaken_by_rank(taken, first_rank, numstart)
            optima = jax.vmap(lambda x0: continuous_search(score, x0, lower, upper)[0])(
                Xcand[start_rows]
            )
            snapped = jax.vmap(lambda optimum: nearest_untaken(optimum, Xcand, taken))(optima)
            best = snapped[jnp.argmax(score(Xcand[snapped]))]

        Knp = covar(X_design, Xcand[best][None], d)[:, 0]
        Knp = jnp.where(jnp.arange(end) < slot, Knp, 0.0)
        return (
            chosen.at[slot].set(best),
            taken.at[best].set(True),
            bordered_inverse(Ki, Knp, g, slot),
        ), None

    (chosen, _, _), _ = jax.lax.scan(
        add_best, (chosen, taken, Ki), (jnp.arange(start, end), jnp.asarray(first_ranks))
    )
    return chosen

"""Local approximate Gaussian processes in JAX, with the public API of the R package laGP."""

import warnings

import jax

if not jax.config.jax_enable_x64:
    warnings.warn(
        "GP math is likely to fail in float32: enable float64 before importing lagpax with "
        'jax.config.update("jax_enable_x64", True)',
        stacklevel=2,
    )

from lagpax.design import alcGP, alcoptGP, alcrayGP
from lagpax.gp import (
    GP,
    Prediction,
    darg,
    garg,
    jmleGP,
    llikGP,
    mleGP,
    mleGPsep,
    newGP,
    newGPsep,
    predGP,
    updateGP,
)
from lagpax.local import MLE, ApproxPrediction, LocalPrediction, aGP, aGPsep, laGP, laGPsep

__all__ = [
    "GP",
    "MLE",
    "ApproxPrediction",
    "LocalPrediction",
    "Prediction",
    "aGP",
    "aGPsep",
    "alcGP",
    "alcoptGP",
    "alcrayGP",
    "darg",
    "garg",
    "jmleGP",
    "laGP",
    "laGPsep",
    "llikGP",
    "mleGP",
    "mleGPsep",
    "newGP",
    "newGPsep",
    "predGP",
    "updateGP",
]

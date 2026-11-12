"""Minimize Ackley in 20 dimensions; saves outputs/ackley-<seed>.npz."""

import argparse
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np  # noqa: E402
import vlse  # noqa: E402

from localvorcans import minimize  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--d", type=int, default=20)
parser.add_argument("--n-batches", type=int, default=100)
args = parser.parse_args()

result = minimize(vlse.Ackley(d=args.d, normalized=True), args.d, args.seed, n_batches=args.n_batches)
Path("outputs").mkdir(exist_ok=True)
np.savez(f"outputs/ackley-{args.seed}.npz", **{k: np.asarray(v) for k, v in result.items()})

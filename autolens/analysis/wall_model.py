"""
The calibrated wall model behind `al.estimate_runtime`: how long a Nautilus fit takes, before it runs.

Ported (constants and functions unchanged) from the autolens_inference repo,
`scripts/misc/wall/model.py` (autolens_inference#69, release pin 2026.10.25.1, fitted 2026-11-16).
Read that file for the derivation; the short version:

For one stage with `d` free parameters and `n_live` live points::

    evals   = EVALS_A * n_live * d ** EVALS_B * EVALS_FACTOR[dataset_type]
    seconds = evals * (FLOOR_S + price)

and a job is `STARTUP_S` plus the sum of its stages. `price` is the per-call the fit pays: the
profiled per-call, divided by `cores * PARALLEL_EFFICIENCY[cores]` for numba (one process per Nautilus
worker); a JAX per-call already prices one evaluation on its device and is used as is.

**Validity.** The model covers Nautilus only (with the phase-4 settings). It is calibrated in-sample:
0.56-1.59x on the phase-4 legs it was fitted to. Held out (a dataset type's own `EVALS_FACTOR` removed),
interferometer read 0.19-0.45x and weak 0.50-0.52x, so those dataset types are labelled uncalibrated. A
dataset type with no phase-4 row takes `EVALS_FACTOR` 1.0, an uncalibrated default.
"""

from typing import Dict, List, Optional, Tuple

#: evals = EVALS_A * n_live * d ** EVALS_B (least squares in log space over 231 completed,
#: non-resumed phase-4 stages outside interferometer; 2026-11-16).
EVALS_A = 4.30
EVALS_B = 1.637

#: The median of measured / law evaluations per dataset type, over its phase-4 stages.
EVALS_FACTOR: Dict[str, float] = {
    "cluster": 0.77,
    "group": 0.79,
    "imaging": 1.13,
    "interferometer": 0.34,
    "multi_dataset": 1.65,
    "multi_galaxy": 1.06,
    "point_source_image": 0.88,
    "point_source_source": 0.81,
    "weak": 0.42,
}

#: Seconds per evaluation the fit spends outside the likelihood (Nautilus bookkeeping plus sample and
#: output writing).
FLOOR_S = 0.014

#: Seconds per job outside every stage: imports, dataset load, the first compile.
STARTUP_S = 30

#: numba's parallel efficiency under Nautilus's multiprocessing, by core count (phase-4 cores sweep,
#: imaging/rectangular/hst, numba_cpu_sparse_fp64). `al.estimate_runtime` prefers the curve in the
#: header of `autolens/config/runtime_estimates.yaml` (`parallel_efficiency.curve`) and falls back to
#: this one.
PARALLEL_EFFICIENCY: Dict[int, float] = {
    1: 1.0,
    2: 0.68,
    4: 0.62,
    8: 0.55,
    16: 0.36,
    32: 0.28,
}

#: A parametric stage under numba costs about this many times the dataset type's JAX CPU MGE per-call.
NUMBA_PARAMETRIC_FACTOR = 3.0

#: Profiling records whose per-call does not price a fit's evaluation, by record id: `(seconds, why)`.
PER_CALL_OVERRIDE: Dict[str, Tuple[float, str]] = {
    "lifecycle/b5dea27fcaf8b14ec9bb": (
        0.110,
        "multi_dataset/delaunay A100: the record's vmap.per_call (0.176 s) timed a batch of 3, "
        "an OOM fallback, slower than the same record's single-jit median of 110 ms "
        "(wiki/project/state.md, 2026-11-05)",
    ),
}

#: Whole per-evaluation costs (floor included) by `(dataset_type, variant, host_class)`:
#: `(seconds, why)`. Fitted to the legs they price (a measured wall in per-evaluation form).
PER_EVAL_OVERRIDE: Dict[Tuple[str, str, str], Tuple[float, str]] = {
    ("cluster", "image_plane", "hpc_a100"): (
        0.060,
        "the image-plane PointSolver fit costs 15 ms per evaluation in the sampler and ~43 ms "
        "per evaluation in stage output on the A100, against a 0.2 ms vmap per-call "
        "(phase-4 413966: total_wall_s / likelihood_evals 0.056-0.060 s)",
    ),
    ("cluster", "image_plane", "hpc_ral_cpu"): (
        0.049,
        "as the A100 entry; RAL CPU 414145: total_wall_s / likelihood_evals 0.035-0.049 s",
    ),
}

#: Dataset types whose single search is the whole fit: no parametric opening stage to reprice.
SINGLE_SEARCH = {"point_source_source", "point_source_image", "cluster", "weak"}


def stage_evals(free_parameters: int, n_live: int, dataset_type: str) -> int:
    """Evaluations one Nautilus stage takes, from its dimension and live points."""
    law = EVALS_A * n_live * free_parameters**EVALS_B
    return round(law * EVALS_FACTOR.get(dataset_type, 1.0))


def parallel_efficiency(cores: int, curve: Optional[Dict[int, float]] = None) -> float:
    """The measured efficiency at `cores`; between sweep points, the next lower one."""
    curve = curve or PARALLEL_EFFICIENCY
    known = [c for c in sorted(curve) if c <= cores]
    return curve[known[-1]] if known else 1.0


def effective_per_call(
    per_call_s: float,
    backend: str,
    cores: int,
    curve: Optional[Dict[int, float]] = None,
) -> float:
    """What one evaluation costs the job: a numba per-call is one thread and Nautilus forks `cores`
    workers; a JAX per-call (vmap) already prices one evaluation on the device.

    `backend` is the setup backend (`"numba"`) or the original model's `"numba_cpu"`."""
    if backend in ("numba", "numba_cpu"):
        return per_call_s / (cores * parallel_efficiency(cores, curve=curve))
    return per_call_s


def stage_seconds(evals: int, price_s: float) -> float:
    """Wall seconds for one stage of `evals` evaluations at `price_s` likelihood cost."""
    return evals * (FLOOR_S + price_s)


def job_seconds(stage_walls: List[float]) -> float:
    """A job's wall: startup plus the stages it fits."""
    return STARTUP_S + sum(stage_walls)

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Install
```bash
pip install -e ".[dev]"
```

### Run Tests
```bash
# All tests
python -m pytest test_autolens/

# Single test file
python -m pytest test_autolens/lens/test_tracer.py

# With output
python -m pytest test_autolens/imaging/test_fit_imaging.py -s
```

### Codex / sandboxed runs

When running Python from Codex or any restricted environment, set writable cache directories so `numba` and `matplotlib` do not fail on unwritable home or source-tree paths:

```bash
NUMBA_CACHE_DIR=/tmp/numba_cache MPLCONFIGDIR=/tmp/matplotlib python -m pytest test_autolens/
```

This workspace is often imported from `/mnt/c/...` and Codex may not be able to write to module `__pycache__` directories or `/home/jammy/.cache`, which can cause import-time `numba` caching failures without this override.

### Formatting
```bash
black autolens/
```

### Plot Output Mode

Set `PYAUTO_OUTPUT_MODE=1` to capture every figure produced by a script into numbered PNG files in `./output_mode/<script_name>/`. This is useful for visually inspecting all plots from an integration test without needing a display.

```bash
PYAUTO_OUTPUT_MODE=1 python scripts/my_script.py
# -> ./output_mode/my_script/0_fit.png, 1_tracer.png, ...
```

When this env var is set, all `save_figure`, `subplot_save`, and `_save_subplot` calls are intercepted — the normal output path is bypassed and figures are written sequentially to the output_mode directory instead.

## Architecture

**PyAutoLens** is the gravitational lensing layer built on top of PyAutoGalaxy. It adds multi-plane ray-tracing, the `Tracer` object, and lensing-specific fit classes. It depends on:
- **`autogalaxy`** — galaxy morphology, mass/light profiles, single-plane fitting
- **`autoarray`** — low-level data structures (grids, masks, arrays, datasets, inversions)
- **`autofit`** — non-linear search / model-fitting framework

### Core Class Hierarchy

```
Tracer (lens/tracer.py)
  └── List[List[Galaxy]] — galaxies grouped by redshift plane
      ├── ray-traces from source to lens to observer
      ├── delegates to autogalaxy Galaxy/Galaxies for per-plane operations
      └── returns lensed images, deflection maps, convergence, magnification
```

### Dataset Types and Fit Classes

| Dataset | Fit class | Analysis class |
|---|---|---|
| `aa.Imaging` | `FitImaging` | `AnalysisImaging` |
| `aa.Interferometer` | `FitInterferometer` | `AnalysisInterferometer` |
| Point source | `FitPointDataset` | `AnalysisPoint` |

All inherit from the corresponding `autogalaxy` base classes (`ag.FitImaging`, etc.) and extend them with multi-plane lensing via the `Tracer`.

### Key Directories

```
autolens/
  lens/            Tracer, ray-tracing, multi-plane deflection logic
  imaging/         FitImaging, AnalysisImaging
  interferometer/  FitInterferometer, AnalysisInterferometer
  point/           Point-source datasets, fits, and analysis
  quantity/        FitQuantity for arbitrary lensing quantities
  analysis/        Shared analysis base classes, adapt images
  aggregator/      Scraping results from autofit output directories
  plot/            Visualisation for all data types
```

## Decorator System (from autoarray)

PyAutoLens inherits the same decorator conventions as PyAutoGalaxy. Mass and light profile methods that take a grid and return an array/grid/vector are decorated with:

| Decorator | `Grid2D` → | `Grid2DIrregular` → |
|---|---|---|
| `@aa.grid_dec.to_array` | `Array2D` | `ArrayIrregular` |
| `@aa.grid_dec.to_grid` | `Grid2D` | `Grid2DIrregular` |
| `@aa.grid_dec.to_vector_yx` | `VectorYX2D` | `VectorYX2DIrregular` |

The `@aa.grid_dec.transform` decorator (always innermost) transforms the grid to the profile's reference frame. Standard stacking:

```python
@aa.grid_dec.to_array
@aa.grid_dec.transform
def convergence_2d_from(self, grid, xp=np, **kwargs):
    y = grid.array[:, 0]   # .array extracts raw numpy/jax array
    x = grid.array[:, 1]
    return ...             # raw array — decorator wraps it
```

The function body must return a **raw array**. Use `grid.array[:, 0]` (not `grid[:, 0]`) to access coordinates safely for both numpy and jax backends.

See PyAutoArray's `CLAUDE.md` for full decorator internals.

## JAX Support

The `xp` parameter pattern controls the backend:
- `xp=np` (default) — pure NumPy, no JAX dependency
- `xp=jnp` — JAX path; `jax`/`jax.numpy` imported locally inside the function only

### JAX and the `jax.jit` boundary

Two patterns coexist for crossing the JIT boundary:

**Pattern 1: `if xp is np:` guard (raw `jax.Array` return).** Functions intended to be called directly inside `jax.jit` as the outermost op — where no wrapper is needed on the JAX path — guard their autoarray wrapping:

```python
def convergence_2d_via_hessian_from(self, grid, xp=np):
    convergence = 0.5 * (hessian_yy + hessian_xx)

    if xp is np:
        return aa.ArrayIrregular(values=convergence)  # numpy: wrapped
    return convergence                                  # jax: raw jax.Array
```

All `LensCalc` hessian-derived methods use this pattern. Intermediate helpers (e.g. `deflections_yx_2d_from`) don't need the guard — they're consumed by downstream Python before the JIT boundary.

**Pattern 2: pytree-registered wrapper return.** Functions that must return a real autoarray wrapper (or a structured object built from them) opt in to JAX pytree registration. `AbstractNDArray` auto-registers its subclass with `jax.tree_util` the first time an instance is built with `xp=jnp` (via `autoarray.abstract_ndarray._register_as_pytree`). Higher-level types (`FitImaging`, `Tracer`, `DatasetModel`) use `autoarray.abstract_ndarray.register_instance_pytree(cls, no_flatten=...)`, which flattens `__dict__` and carries `no_flatten` names through `aux_data` for per-analysis constants (dataset, settings, cosmology). `AnalysisImaging._register_fit_imaging_pytrees` wires these up when `use_jax=True`, so `jax.jit(analysis.fit_from)(instance)` returns a real `FitImaging` with `jax.Array` leaves.

### `LensCalc` (autogalaxy)

The hessian-derived lensing quantities (`convergence_2d_via_hessian_from`, `shear_yx_2d_via_hessian_from`, `magnification_2d_via_hessian_from`, `magnification_2d_from`, `tangential_eigen_value_from`, `radial_eigen_value_from`) all implement the `if xp is np:` guard in `autogalaxy/operate/lens_calc.py` and return raw `jax.Array` on the JAX path, making them safe to call inside `jax.jit`.

## Namespace Conventions

When importing `autolens as al`:
- `al.mp.*` — mass profiles (re-exported from autogalaxy)
- `al.lp.*` — light profiles (re-exported from autogalaxy)
- `al.Galaxy`, `al.Galaxies`
- `al.Tracer`
- `al.FitImaging`, `al.AnalysisImaging`, `al.SimulatorImaging`
- `al.FitInterferometer`, `al.AnalysisInterferometer`
- `al.FitPointDataset`, `al.AnalysisPoint`

## Line Endings — Always Unix (LF)

All files **must use Unix line endings (LF, `\n`)**. Never write `\r\n` line endings.
## Never rewrite history

NEVER perform these operations on any repo with a remote:

- `git init` in a directory already tracked by git
- `rm -rf .git && git init`
- Commit with subject "Initial commit", "Fresh start", "Start fresh", "Reset
  for AI workflow", or any equivalent message on a branch with a remote
- `git push --force` to `main` (or any branch tracked as `origin/HEAD`)
- `git filter-repo` / `git filter-branch` on shared branches
- `git rebase -i` rewriting commits already pushed to a shared branch

If the working tree needs a clean state, the **only** correct sequence is:

    git fetch origin
    git reset --hard origin/main
    git clean -fd

This applies equally to humans, local Claude Code, cloud Claude agents, Codex,
and any other agent. The "Initial commit — fresh start for AI workflow" pattern
that appeared independently on origin and local for three workspace repos is
exactly what this rule prevents — it costs ~40 commits of redundant local work
every time it happens.

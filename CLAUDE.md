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

### Formatting
```bash
black autolens/
```

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
  plot/            Visualisation (Plotter classes for all data types)
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

Autoarray types (`Array2D`, `ArrayIrregular`, `VectorYX2DIrregular`, etc.) are **not registered as JAX pytrees**. They can be constructed inside a JIT trace, but **cannot be returned** as the output of a `jax.jit`-compiled function.

Functions intended to be called directly inside `jax.jit` must guard autoarray wrapping with `if xp is np:`:

```python
def convergence_2d_via_hessian_from(self, grid, xp=np):
    convergence = 0.5 * (hessian_yy + hessian_xx)

    if xp is np:
        return aa.ArrayIrregular(values=convergence)  # numpy: wrapped
    return convergence                                  # jax: raw jax.Array
```

Functions that are only called as intermediate steps (e.g. `deflections_yx_2d_from`) do not need this guard — they are consumed by downstream Python before the JIT boundary.

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

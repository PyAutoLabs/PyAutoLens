# Copilot Coding Agent Instructions

You are working on **PyAutoLens**, a Python library for strong gravitational lens modeling built on PyAutoGalaxy.

## Key Rules

- Run tests after every change: `python -m pytest test_autolens/`
- Format code with `black autolens/`
- All files must use Unix line endings (LF, `\n`)
- Decorated functions (`@to_array`, `@to_grid`, `@to_vector_yx`) must return **raw arrays**, not autoarray wrappers
- The `xp` parameter controls NumPy (`xp=np`) vs JAX (`xp=jnp`) — never import JAX at module level
- Functions called inside `jax.jit` must guard autoarray wrapping with `if xp is np:`
- If changing public API, clearly document what changed in your PR description — downstream workspaces depend on this

## Architecture

- `autolens/lens/` — `Tracer`, multi-plane ray-tracing, deflection logic
- `autolens/imaging/`, `interferometer/`, `point/` — Dataset-specific fit and analysis classes
- `autolens/plot/` — Visualisation for all data types
- `autolens/aggregator/` — Results scraping
- `test_autolens/` — Test suite

## Sandboxed runs

```bash
NUMBA_CACHE_DIR=/tmp/numba_cache MPLCONFIGDIR=/tmp/matplotlib python -m pytest test_autolens/
```

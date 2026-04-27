# PyAutoLens — Agent Instructions

**PyAutoLens** is a Python library for strong gravitational lens modeling, built on PyAutoGalaxy. It adds multi-plane ray-tracing via the `Tracer` object and lensing-specific fit/analysis classes.

## Setup

```bash
pip install -e ".[dev]"
```

## Running Tests

```bash
python -m pytest test_autolens/
python -m pytest test_autolens/lens/test_tracer.py
python -m pytest test_autolens/imaging/test_fit_imaging.py -s
```

### Sandboxed / Codex runs

```bash
NUMBA_CACHE_DIR=/tmp/numba_cache MPLCONFIGDIR=/tmp/matplotlib python -m pytest test_autolens/
```

## Key Architecture

- **Tracer** (`lens/tracer.py`): groups galaxies by redshift plane, performs multi-plane ray-tracing
- **Fit classes**: `FitImaging`, `FitInterferometer`, `FitPointDataset` — extend autogalaxy equivalents with lensing
- **Analysis classes**: `AnalysisImaging`, `AnalysisInterferometer`, `AnalysisPoint`
- **Namespace**: `al.mp.*` (mass), `al.lp.*` (light), `al.Galaxy`, `al.Tracer`

## Dependencies

- `autogalaxy` — galaxy morphology, profiles, single-plane fitting
- `autoarray` — data structures, grids, masks, inversions
- `autofit` — non-linear search and model-fitting framework

## Key Rules

- The `xp` parameter controls NumPy vs JAX: `xp=np` (default) or `xp=jnp`
- Functions inside `jax.jit` must guard autoarray wrapping with `if xp is np:`
- Decorated functions return **raw arrays** — the decorator wraps them
- All files must use Unix line endings (LF)
- Format with `black autolens/`

## Working on Issues

1. Read the issue description and any linked plan.
2. Identify affected files and write your changes.
3. Run the full test suite: `python -m pytest test_autolens/`
4. Ensure all tests pass before opening a PR.
5. If changing public API, note the change in your PR description — downstream workspaces may need updates.
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

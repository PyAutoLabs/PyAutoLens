# PyAutoLens — Agent Instructions

Canonical, agent-agnostic instructions for this repo. `CLAUDE.md` imports this
file; any tool that does not process `@`-imports should read this directly.

## What this repo is

**PyAutoLens** (package `autolens`) is the strong gravitational-lensing layer
built on PyAutoGalaxy. It adds multi-plane ray-tracing via the `Tracer`, and
lensing-specific `Fit*` / `Analysis*` classes for imaging, interferometer, and
point-source datasets.

Dependency direction: autolens sits at the top of the stack and may import all
four layers below it — **autogalaxy**, **autoarray**, **autofit**, and
**autoconf**. Nothing in the ecosystem imports autolens.

## Related repos

- **Source siblings (all upstream):** PyAutoConf, PyAutoArray, PyAutoFit,
  PyAutoGalaxy.
- **autolens_workspace** — runnable tutorials/examples (`../autolens_workspace`).
- **autolens_workspace_test** — integration + JAX/likelihood parity scripts.
- **autolens_profiling** — performance/profiling harness (`../autolens_profiling`).
- **HowToLens** — the lecture-style tutorial series (`../HowToLens`).
- **docs/** — Sphinx source; published to ReadTheDocs.
- **Science context:** the strong-lensing knowledge wiki at
  `autolens_assistant/wiki/literature/` (concepts, entities, sources) — mass
  models, source reconstruction, degeneracies, substructure, surveys.

## Quick commands

```bash
pip install -e ".[dev]"                                   # install with dev/test extras
python -m pytest test_autolens/                           # full test suite
python -m pytest test_autolens/lens/test_tracer.py        # one focused test (add -s for output)
black autolens/                                           # formatter (advisory — not gated)
```

In a sandboxed / restricted environment, point numba and matplotlib at
writable caches:

```bash
NUMBA_CACHE_DIR=/tmp/numba_cache MPLCONFIGDIR=/tmp/matplotlib python -m pytest test_autolens/
```

## CI / definition of green

PRs must pass `pytest --cov` on the CI matrix (Python 3.12 **and** 3.13). There
is no black/ruff/flake8 gate — formatting is advisory. (`requires-python` in
`pyproject.toml` is `>=3.9`.)

## Configuration & defaults

autoconf supplies the packaged defaults under `autolens/config/`. Workspaces
override them via their own `config/` directory; the test suite pushes a local
config dir via `conf.instance.push(...)` in `test_autolens/conftest.py`. When a
change adds a new config key, mirror it into the packaged defaults so
downstream workspaces inherit it.

## JAX & `xp`

NumPy is the default everywhere; JAX is opt-in and never imported at module
level. `xp=np` (default) selects NumPy; `xp=jnp` selects JAX (imported locally).
Thread `xp` through **every** nested call — a missed site silently defaults to
`xp=np` and fails when a tracer hits an `np.*` op. Two patterns cross the
`jax.jit` boundary: the `if xp is np:` **guard** for raw `jax.Array` returns
(the `LensCalc` hessian methods), and **pytree registration** for functions
returning real wrappers/structured objects — `FitImaging`, `Tracer`, and
`DatasetModel` register via `register_instance_pytree`, so
`jax.jit(analysis.fit_from)(instance)` returns a real `FitImaging` with
`jax.Array` leaves.

**Unit tests are NumPy-only.** A JAX/`xp` change is validated only by the
parity scripts in `autolens_workspace_test` (`jax.jit` round-trip +
`fitness._vmap` batch eval) — never by `test_autolens/`.

Full detail lives in PyAutoArray:
**[`PyAutoArray/docs/agents/jax_and_decorators.md`](../PyAutoArray/docs/agents/jax_and_decorators.md)**.

## Public API

The public surface is defined authoritatively in `autolens/__init__.py` — read
it rather than trusting a hand-maintained namespace table. Canonical import:

```python
import autolens as al
```

Profiles re-export from autogalaxy (`al.mp.*`, `al.lp.*`) alongside `al.Galaxy`,
`al.Tracer`, `al.FitImaging`/`al.AnalysisImaging`, and the point-source classes.

## Key rules / footguns

- Import direction: autolens may use all four upstream packages; nothing
  imports autolens.
- Grid-decorated profile methods return a **raw array** (the decorator wraps
  it); write `aa.decorators.*` and read coordinates via `grid.array[:, 0]`.
- All files use Unix line endings (LF, `\n`) — never `\r\n`.

## Working on issues

1. Read the issue description and any linked plan.
2. Identify affected files and make the change.
3. Run the full suite: `python -m pytest test_autolens/`.
4. If you changed public API, say so explicitly — the workspaces and
   downstream pipelines may need updates.
5. Ensure all tests pass before opening a PR.

## Deep dives

- [`PyAutoArray/docs/agents/jax_and_decorators.md`](../PyAutoArray/docs/agents/jax_and_decorators.md)
  — decorator system, `xp` backend pattern, and the `jax.jit` boundary.

## Clean state

Never rewrite history on a repo with a remote (no `git init` over a tracked
tree, no force-push to `main`, no rebasing pushed shared branches). To reset a
dirty tree the only correct sequence is:

```bash
git fetch origin
git reset --hard origin/main
git clean -fd
```

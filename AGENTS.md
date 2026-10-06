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
**autonerves**. Nothing in the ecosystem imports autolens.

## Related repos

- **Source siblings (all upstream):** PyAutoNerves, PyAutoArray, PyAutoFit,
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
`pyproject.toml` is `>=3.12`.)

## Configuration & defaults

autonerves supplies the packaged defaults under `autolens/config/`. Workspaces
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

<!-- repos_sync:history:begin -->
## Never rewrite history

Never rewrite pushed history on any repo with a remote — no `git init` over a
tracked repo, no force-push to `main`, no fresh-start "Initial commit", no
`filter-repo` / `filter-branch` / `rebase -i` on pushed branches. To get a
clean tree: `git fetch origin && git reset --hard origin/main && git clean -fd`.
<!-- repos_sync:history:end -->

<!-- repos_sync:deliverable:begin -->
## Sessions end at their deliverable

A session ends when it reports its deliverable — never arm anything that
outlives the turn to wait for CI, a review or a merge: no `send_later`, no
`subscribe_pr_activity`, no `CronCreate`, no `ScheduleWakeup`, no `/loop`, no
`RemoteTrigger` create/update/run. Judge once, report, stop; the human re-runs
`/prm` (or the batch review) when it is green. Measured: five batch members
armed hourly check-ins on 2026-08-31, and a mobile `/prm` re-armed a 60-minute
`send_later` hourly all night on 2026-09-03 with no task active, draining usage.
<!-- repos_sync:deliverable:end -->

<!-- repos_sync:filing:begin -->
## Where to file

Questions, help with code or an analysis, ideas, bug reports and results from a
user or collaborator — or an agent acting for one — go to
<https://github.com/orgs/PyAutoLabs/discussions> in the matching category
(Help & Questions, Ideas & Proposals, Bugs & Errors, Show and tell;
Announcements is maintainers-only), never to this repo's Issues. An agent never
runs `gh issue create` for such a report: it drafts the title, category and
body and hands them to the human (sessions cannot create Discussions). Only the
development flow — Mind prompt → `/start_dev` → `/create_issue` → one issue per
task → PR — opens issues here. Why: `PyAutoMind/policy/community_surface.md`.
<!-- repos_sync:filing:end -->

<!-- repos_sync:standards:begin -->
## Shared standards

Before changing a shared interface, consult the applicable
[organism standard](https://github.com/PyAutoLabs/PyAutoBrain/blob/main/docs/standards.md)
on demand, identify affected consumers, and validate their adoption. Change
generated guidance at its canonical source and regenerate.
<!-- repos_sync:standards:end -->

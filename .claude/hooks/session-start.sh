#!/usr/bin/env bash
# GENERATED — canonical source: PyAutoMind/policy/session_start_hook.sh
# Installed into every checked-out repo as .claude/hooks/session-start.sh by
# `python3 PyAutoMind/scripts/repos_sync.py --write`, and drift-checked by
# `--check`. Edit the canonical file, never a copy.
#
# ---------------------------------------------------------------------------
# Python 3.12 is the default in Claude Code web/mobile sessions.
#
# The remote container ships /usr/local/bin/python{,3} -> /usr/bin/python3.11,
# a pip whose shebang is #!/usr/bin/python3, and a uv-managed tool set (pytest,
# ruff, black, mypy, pyright, flake8, poetry) every one of which was built on
# 3.11 — one minor version below the floor the organism set for itself in the
# Python 3.12 floor campaign, and below every CI leg it runs. The image's own
# `use-python 3.12` does not fix it: it moves the update-alternatives links
# under /usr/bin, which /usr/local/bin/python{,3} then shadow.
#
# This hook makes a session 3.12 on three surfaces:
#
#   1. a 3.12 virtualenv first on PATH — python, python3, pip, pytest;
#   2. the /usr/local/bin/python{,3} symlinks repointed at 3.12, so anything
#      resolving PATH without this session's env (a subprocess with a scrubbed
#      environment, a `#!/usr/bin/env python3` script) also gets 3.12;
#   3. the uv-managed tools rebuilt on 3.12 — mypy and flake8 read the
#      interpreter's version, so on 3.11 they judged code against 3.11 rules —
#      and then repaired, because (2) is what breaks each tool env's own
#      `bin/python`, and a rebuild cannot fix a link it is handed.
#
# What it deliberately does NOT touch: the update-alternatives links under
# /usr/bin. Scripts with a literal `#!/usr/bin/python3` shebang follow those,
# and some of the image's own tools (conan) are installed for 3.11 only — a
# flip there breaks them for no gain the three surfaces above don't already
# give.
#
# Remote-only (a local checkout keeps whatever the developer's shell provides),
# idempotent (everything is skipped once it already reads 3.12, so the second
# repo's copy in the same session costs ~0.2s), and non-blocking: every step
# degrades to a logged warning rather than failing the session start.
#
# Per-repo dependencies: a repo that needs more than pytest + PyYAML declares it
# in .claude/session-python.txt — one pip argument per line (`-e .`, a package
# spec, `-r requirements.txt`; `#` comments ignored). Installed additively into
# the shared venv, so the file stays out of this generated hook and the hook
# stays byte-identical in every repo.
set -euo pipefail

[ "${CLAUDE_CODE_REMOTE:-}" = "true" ] || exit 0

VENV="${PYAUTO_SESSION_VENV:-$HOME/.pyauto/session-py312}"
# pytest-xdist is a base dep, not a nicety: a remote container has 4 cores and
# the Brain suite takes 96s on one of them and 28s on four. A session that has
# to `pip install` before it can run tests fast will simply not run them fast.
BASE_DEPS=(pytest PyYAML pytest-xdist)

# Which checkout is this copy of the hook installed in?
#
# NOT $CLAUDE_PROJECT_DIR. That is the session's project directory, which equals
# the repo only when the session holds exactly ONE repo. A session scoped to
# several organs clones them side by side under the project directory
# (/home/user/PyAutoMind, /home/user/PyAutoBrain, ...), and then
# $CLAUDE_PROJECT_DIR is their parent — so reading session-python.txt from it
# found nothing, silently, in exactly the sessions that hold the most repos.
#
# Derive it from where this script actually is, in both the installed and the
# canonical location.
HOOK_SELF="$(readlink -f "$0")"
case "$(dirname "$HOOK_SELF")" in
    */.claude/hooks) REPO_DIR="$(cd "$(dirname "$HOOK_SELF")/../.." && pwd)" ;;
    */policy)        REPO_DIR="$(cd "$(dirname "$HOOK_SELF")/.."    && pwd)" ;;
    *)               REPO_DIR="${CLAUDE_PROJECT_DIR:-$PWD}" ;;
esac
EXTRAS_FILE="$REPO_DIR/.claude/session-python.txt"

# stderr, not stdout: a SessionStart hook's stdout is fed to the agent as
# session context.
log() { printf '[session-start] %s\n' "$*" >&2; }

# And where is the WORKSPACE ROOT — the directory holding the checkouts?
#
# NOT `dirname "$REPO_DIR"`, which is what this was. That is the root only in a
# FLAT workspace. Group the checkouts into family directories and the parent of
# a repo is a family directory, which really exists, really is writable and
# really is not a repo — so every guard below passes and all three consumers go
# wrong in silence: $PYAUTO_ROOT is exported to the whole session as the family
# directory, a second `.claude/` root is written INTO it, and the unshallow
# sweep and the generated fan-out reach that one family instead of every
# checkout.
#
# The rule is the one `PyAutoBrain/bin/_pyauto_root.sh` already ships, in the
# same order, so the session and the Brain's resolvers cannot disagree:
#
#   1. an explicit $PYAUTO_ROOT — the operator's word, taken verbatim;
#   2. the nearest ANCESTOR holding a .pyauto-root marker — the only rule that
#      survives the workspace growing subdirectories, because "does this
#      directory hold a checkout?" is answered yes by a family directory the
#      moment the checkouts move into one;
#   3. the parent of this checkout when it holds a sibling organ;
#   4. the parent regardless, reported as unverified.
#
# The resolver itself is used where it is reachable, so there is one
# implementation to correct. But it CANNOT be a dependency: this hook is
# installed into every repo and runs in containers that hold a single checkout,
# no PyAutoBrain and no marker anywhere above them — the session that most
# needs the hook is the one least likely to have a workspace around it. So the
# same rule runs inline wherever the resolver is absent, steps 3 and 4 never
# require a marker, and the resolver is read in a subshell that reports only
# its two answers: a resolver that is broken, or from a version this copy of
# the hook has never seen, can then never fail a session start.
PYAUTO_ROOT_MARKER=".pyauto-root"

# The nearest ancestor of $1 holding the marker. A checkout is never its own
# workspace root, so the walk starts one level up.
marked_root() {
    local d
    d="$(dirname "$1")"
    while [ "$d" != "/" ]; do
        if [ -f "$d/$PYAUTO_ROOT_MARKER" ]; then
            printf '%s' "$d"
            return 0
        fi
        d="$(dirname "$d")"
    done
    if [ -f "/$PYAUTO_ROOT_MARKER" ]; then
        printf '%s' "/"
        return 0
    fi
    return 1
}

# A directory counts as a workspace root if it holds an organ checkout. Any
# organ: a remote session may hold exactly one, and it is still a root.
holds_an_organ() {
    [ -d "$1/PyAutoMind" ] || [ -d "$1/PyAutoBrain" ] || [ -d "$1/PyAutoHeart" ] \
        || [ -d "$1/PyAutoHands" ] || [ -d "$1/PyAutoMemory" ] || [ -d "$1/PyAutoGut" ] \
        || [ -d "$1/PyAutoNerves" ] || [ -d "$1/PyAutoCortex" ]
}

# Assigns WORKSPACE_ROOT and WORKSPACE_ROOT_REASON rather than printing: one
# decision produces two answers, and a command substitution would drop the
# reason — which is the whole lesson of a defect whose every consequence was
# silent.
resolve_workspace_root() {
    local helper answer
    # 1. The operator's word, never second-guessed. The marker only decides
    #    how the reason reads, so a wrong value from a caller stays visible.
    if [ -n "${PYAUTO_ROOT:-}" ]; then
        WORKSPACE_ROOT="$PYAUTO_ROOT"
        if [ -f "$WORKSPACE_ROOT/$PYAUTO_ROOT_MARKER" ]; then
            WORKSPACE_ROOT_REASON="PYAUTO_ROOT"
        else
            WORKSPACE_ROOT_REASON="PYAUTO_ROOT (unverified - no $PYAUTO_ROOT_MARKER marker)"
        fi
        return 0
    fi
    # The shared resolver, if this checkout can see one: either this checkout
    # IS the Brain, or the Brain is beside it.
    #
    # Its answer is ACCEPTED ONLY IF IT CONTAINS THIS CHECKOUT, because the
    # resolver anchors on the BRAIN checkout, not on this one. In a worktree
    # bundle the bundle's PyAutoBrain is a symlink into the canonical
    # workspace, so the resolver follows it and answers with the canonical
    # root — and this session would then export somebody else's workspace to
    # the whole session, write a `.claude/` root into it and unshallow its
    # repos. Measured on this very bundle: the resolver said
    # the canonical workspace for a hook running in a task bundle beside it.
    # `scripts/session_bootstrap.sh` guards the same trap the same way; the
    # test is "is it an ancestor", not "is the checkout directly under it", so
    # a checkout inside a family directory still passes.
    local repo_real root_real
    repo_real="$(readlink -f "$REPO_DIR" 2>/dev/null || printf '%s' "$REPO_DIR")"
    for helper in "$REPO_DIR/bin/_pyauto_root.sh" \
                  "$(dirname "$REPO_DIR")/PyAutoBrain/bin/_pyauto_root.sh"; do
        [ -r "$helper" ] || continue
        # `|` separates the two answers; no reason the resolver states contains
        # one. The subshell drops this script's `set -euo pipefail` so that a
        # resolver which fails, or which one day needs something this container
        # does not have, degrades to the inline rule instead of to a dead
        # session.
        answer="$( (
            set +euo pipefail
            # shellcheck source=/dev/null
            . "$helper" >/dev/null 2>&1
            printf '%s|%s' "${PYAUTO_ROOT:-}" "${PYAUTO_ROOT_REASON:-}"
        ) 2>/dev/null )" || answer=""
        [ -n "${answer%%|*}" ] && [ -d "${answer%%|*}" ] || continue
        root_real="$(readlink -f "${answer%%|*}" 2>/dev/null || printf '%s' "${answer%%|*}")"
        case "$repo_real" in
            "${root_real%/}"/*) ;;
            *) continue ;;
        esac
        WORKSPACE_ROOT="${answer%%|*}"
        WORKSPACE_ROOT_REASON="${answer#*|}"
        return 0
    done
    # 2. A marked ancestor — the root naming itself, which stays right however
    #    deep under it this checkout sits.
    if answer="$(marked_root "$REPO_DIR")"; then
        WORKSPACE_ROOT="$answer"
        WORKSPACE_ROOT_REASON="$PYAUTO_ROOT_MARKER marker"
        return 0
    fi
    # 3. Beside this checkout — nothing above us says anything, so what sits
    #    beside us is all there is: a single-repo remote session, a CI matrix,
    #    a spawned template. 4. Failing that, the parent anyway: the best guess
    #    available, and a real path a diagnostic can name.
    local parent
    parent="$(dirname "$REPO_DIR")"
    WORKSPACE_ROOT="$parent"
    if holds_an_organ "$WORKSPACE_ROOT"; then
        WORKSPACE_ROOT_REASON="beside this checkout"
    else
        WORKSPACE_ROOT_REASON="unverified (no sibling organ beside this checkout)"
    fi
}
resolve_workspace_root

is_py312() {
    [ -x "$1" ] && "$1" -c 'import sys; raise SystemExit(sys.version_info[:2] != (3, 12))' >/dev/null 2>&1
}

find_base_python() {
    local candidate
    for candidate in /usr/bin/python3.12 /usr/local/bin/python3.12 \
                     "$(command -v python3.12 2>/dev/null || true)"; do
        if [ -n "$candidate" ] && is_py312 "$candidate"; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done
    # No system 3.12 (a future base image could drop it) — uv can fetch one.
    if command -v uv >/dev/null 2>&1; then
        log "no system python3.12; asking uv to install one"
        uv python install 3.12 >&2 || return 1
        candidate="$(uv python find 3.12 2>/dev/null || true)"
        if [ -n "$candidate" ] && is_py312 "$candidate"; then
            printf '%s\n' "$candidate"
            return 0
        fi
    fi
    return 1
}

venv_ready() {
    is_py312 "$VENV/bin/python" \
        && [ -x "$VENV/bin/pip" ] \
        && "$VENV/bin/python" -c 'import pytest, yaml, xdist' >/dev/null 2>&1
}

# 1. The interpreter a session types: python, python3, pip, pytest.
ensure_venv() {
    local base_python
    if venv_ready; then
        log "reusing $VENV ($("$VENV/bin/python" -V 2>&1))"
        return 0
    fi
    if ! base_python="$(find_base_python)"; then
        log "WARNING: no Python 3.12 in this container; PATH left unchanged"
        return 1
    fi
    log "building $VENV on $base_python"
    rm -rf "$VENV"
    mkdir -p "$(dirname "$VENV")"
    # --system-site-packages makes the venv a strict SUPERSET of the base
    # interpreter. That is what lets `point_system_default` make this venv the
    # session's `python3` without silently removing anything the image
    # installed: an isolated venv would swap one set of missing modules for
    # another, and the session would find out through a ModuleNotFoundError
    # that reads like broken code.
    if command -v uv >/dev/null 2>&1; then
        # --seed puts pip inside the venv too, so `pip install` targets 3.12
        # rather than falling through to the container's 3.11 /usr/bin/pip.
        uv venv --seed --system-site-packages --python "$base_python" "$VENV" >&2
        uv pip install --python "$VENV/bin/python" --quiet "${BASE_DEPS[@]}" >&2
    else
        "$base_python" -m venv --system-site-packages "$VENV" >&2
        "$VENV/bin/python" -m pip install --quiet --upgrade pip >&2
        "$VENV/bin/python" -m pip install --quiet "${BASE_DEPS[@]}" >&2
    fi
    venv_ready || { log "WARNING: could not build a 3.12 venv at $VENV"; return 1; }
}

# This repo's own dependencies, if it declares any. Additive and marked, so a
# session holding several repos installs each repo's set exactly once.
ensure_repo_extras() {
    [ -r "$EXTRAS_FILE" ] || return 0
    local marker args=()
    marker="$VENV/.extras-$(cksum <"$EXTRAS_FILE" | tr -d ' /')"
    [ -e "$marker" ] && return 0
    while IFS= read -r line; do
        line="${line%%#*}"
        line="$(printf '%s' "$line" | tr -d '\r' | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')"
        [ -n "$line" ] && args+=("$line")
    done <"$EXTRAS_FILE"
    [ ${#args[@]} -gt 0 ] || return 0
    log "installing this repo's declared deps: ${args[*]}"
    if (cd "$(dirname "$(dirname "$EXTRAS_FILE")")" && "$VENV/bin/python" -m pip install --quiet "${args[@]}" >&2); then
        : >"$marker"
    else
        log "WARNING: $EXTRAS_FILE install failed; continuing without it"
    fi
}

# 2. What PATH means without this session's env file.
#
# This leg is the one that matters most, because the env file is exactly what a
# multi-repo session does NOT get: Claude Code registers project hooks from the
# session's project directory, that directory is the repos' parent, and no hook
# runs there (see leg 5). So every Bash call in such a session resolves whatever
# the image put on PATH, and these two names are the fallback that has to be
# right on its own.
#
# It used to point them at the BASE interpreter — `point_system_default
# "$(readlink -f "$VENV/bin/python")"` resolved the venv's python straight
# through to /usr/bin/python3.12. That satisfied the version question and lost
# everything else: `python3 -m pytest` answered "No module named pytest", and
# the session's own venv, sitting one directory away with pytest in it, was
# unreachable from any shell.
#
# Two details make the fix work:
#
#   * A wrapper script, not a symlink. Python resolves a symlinked executable
#     before looking for `pyvenv.cfg`, so a symlink to $VENV/bin/python lands on
#     the base interpreter's prefix and the venv is lost again — silently, and
#     in the same shape as the bug being fixed. `exec`ing it keeps the venv.
#   * --system-site-packages on the venv (see ensure_venv), so pointing the
#     system default here adds the session's packages without removing the
#     image's.
# Bounded, because this predicate is what runs a candidate interpreter — and
# the one failure mode worth surviving is an interpreter that never returns.
venv_backed() {
    [ -x "$1" ] || return 1
    [ "$(timeout 20 "$1" -c 'import sys; print(sys.prefix)' 2>/dev/null)" = "$VENV" ]
}

# Does following $1's symlinks pass through $2? A venv's python legitimately
# RESOLVES to the same base interpreter a system default points at, so comparing
# endpoints refuses safe rewrites. The question that matters is narrower: is the
# path being rewritten a link in the target's own chain — because then the
# wrapper execs itself.
links_through() {
    local node="$1" needle="$2" hops=0
    needle="$(cd "$(dirname "$needle")" 2>/dev/null && printf '%s/%s' "$(pwd -P)" "$(basename "$needle")")"
    while [ -L "$node" ] && [ "$hops" -lt 40 ]; do
        node="$(cd "$(dirname "$node")" && cd "$(dirname "$(readlink "$node")")" 2>/dev/null && printf '%s/%s' "$(pwd -P)" "$(basename "$(readlink "$node")")")"
        [ "$node" = "$needle" ] && return 0
        hops=$((hops + 1))
    done
    return 1
}

# `rm -f` first, and it is not tidiness. /usr/local/bin/python3 is a SYMLINK to
# the real interpreter, and a redirect opens the link's TARGET: `cat >` there
# overwrites /usr/bin/python3.12 itself with this wrapper. The venv's own
# python symlinks to that same file, so the wrapper then execs itself — the
# container loses its interpreter and every `python3` spins at 100% CPU. Writing
# a fresh file at the path replaces the link instead of following it.
#
# `links_through` is the second half of the same guard: a target that reaches
# the destination through its own symlink chain builds the identical loop by the
# other route, so refuse rather than write it.
write_venv_shim() {
    local dest="$1" target="$2"
    [ -w "$(dirname "$dest")" ] || { log "WARNING: cannot rewrite $dest (not writable)"; return 1; }
    [ -x "$target" ] || { log "WARNING: $target is not executable; leaving $dest alone"; return 1; }
    if links_through "$target" "$dest"; then
        log "WARNING: refusing to point $dest at $target — it links back through $dest"
        return 1
    fi
    rm -f "$dest"
    cat >"$dest" <<SHIM
#!/bin/sh
# GENERATED by PyAutoMind's session-start hook. A wrapper, not a symlink:
# Python resolves a symlink before reading pyvenv.cfg, which loses the venv.
exec "$target" "\$@"
SHIM
    chmod 0755 "$dest"
}

point_system_default() {
    local link bin="${PYAUTO_SESSION_SYSTEM_BIN:-/usr/local/bin}"
    for link in "$bin/python" "$bin/python3"; do
        venv_backed "$link" && continue
        write_venv_shim "$link" "$VENV/bin/python" || continue
    done
    venv_backed "$bin/python3" \
        && log "$bin/python{,3} -> $VENV/bin/python (venv, with pytest and yaml)" \
        || log "WARNING: $bin/python3 is still $(timeout 20 "$bin/python3" -V 2>&1)"
}

# 2b. The `pytest` that PATH actually finds.
#
# $HOME/.local/bin precedes /usr/local/bin, and it holds uv's tool shims — so
# fixing python3 alone still leaves bare `pytest` resolving to uv's ISOLATED
# pytest, which by design cannot see PyYAML or a repo's own extras. In this
# workspace that made `pytest` exit on four collection ImportErrors that read
# like broken source, in a session where the suite was in fact green.
#
# A session should have exactly one pytest, and it should be the venv's.
# Confined to uv's own shim directory: a distro-packaged pytest is not ours to
# overwrite. Runs after retool_uv_tools, which rewrites these same shims.
point_pytest_at_venv() {
    local shim_dir="${HOME}/.local/bin" name path
    [ -x "$VENV/bin/pytest" ] || return 0
    for name in pytest py.test; do
        path="$(command -v "$name" 2>/dev/null)" || continue
        [ -n "$path" ] || continue
        case "$path" in
            "$VENV"/*) continue ;;
            "$shim_dir"/*) ;;
            *) log "WARNING: $name resolves to $path, outside uv's shim dir — left alone"; continue ;;
        esac
        write_venv_shim "$path" "$VENV/bin/$name" \
            && log "$path -> $VENV/bin/$name (so \`$name\` sees this repo's deps)"
    done
}

# 2c. Every OTHER console script the venv owns.
#
# `pytest` and `python3` survive without the venv on PATH because 2 and 2b give
# each one its own shim. Nothing else does — and PATH is exactly what a
# multi-repo session never gets, because the env file that exports it is written
# by Claude Code around a hook it never registers (see leg 5). So a dependency
# declared in `.claude/session-python.txt` that ships a CONSOLE SCRIPT rather
# than an importable module was installed into the venv and then invisible.
#
# Measured, not hypothetical: PyAutoHands shells out to `ipynb-py-convert`, and
# with the package installed its suite still failed five tests with
# `FileNotFoundError: ipynb-py-convert` — the binary sitting in $VENV/bin,
# unreachable, while `--check` called the session healthy. A missing binary and
# an unreachable one are the same symptom, and neither says "environment".
#
# The claim policy is 2b's, because the reasoning is 2b's: a name we can reach
# only through uv's shim dir or not at all is ours to provide; a name the image
# owns elsewhere is not ours to overwrite.
point_venv_scripts_at_venv() {
    local shim_dir="${HOME}/.local/bin" entry name path
    [ -d "$VENV/bin" ] || return 0
    mkdir -p "$shim_dir" 2>/dev/null || return 0
    for entry in "$VENV"/bin/*; do
        [ -f "$entry" ] && [ -x "$entry" ] || continue
        name="$(basename "$entry")"
        case "$name" in
            # python/pip are legs 2 and the venv's own plumbing; pytest is 2b.
            # The activate family is meant to be sourced, never executed.
            python*|pip*|pytest|py.test|activate*|deactivate*|*.bat|*.ps1|*.csh|*.fish|*.nu) continue ;;
        esac
        path="$(command -v "$name" 2>/dev/null)" || path=""
        case "$path" in
            "$VENV"/*) continue ;;
            "") ;;
            "$shim_dir"/*) ;;
            *) log "WARNING: $name resolves to $path, outside uv's shim dir — left alone"; continue ;;
        esac
        write_venv_shim "$shim_dir/$name" "$entry" \
            && log "$shim_dir/$name -> $entry (venv console script)"
    done
}

# 3. The uv-managed tools — rebuilt on 3.12 only where they are not already.
#
# Pinned to the version already installed: this hook changes the INTERPRETER a
# tool runs on, and nothing else. Unpinned, `uv tool install --force` fetches
# the latest release, which quietly moved mypy across a major version (1.19 ->
# 2.3) the first time this ran — a lint-result change nobody asked for, riding
# in on a Python upgrade. Falls back to unpinned only if the pin cannot be
# resolved for 3.12.
retool_uv_tools() {
    command -v uv >/dev/null 2>&1 || return 0
    local tools_dir tool name version spec
    tools_dir="$(uv tool dir 2>/dev/null || echo "$HOME/.local/share/uv/tools")"
    [ -d "$tools_dir" ] || return 0
    for tool in "$tools_dir"/*/; do
        name="$(basename "$tool")"
        is_py312 "$tool/bin/python" && continue
        version="$(uv tool list 2>/dev/null | awk -v n="$name" '$1 == n {print substr($2, 2); exit}')"
        spec="$name"
        [ -n "$version" ] && spec="$name==$version"
        if uv tool install --python 3.12 --force "$spec" >/dev/null 2>&1 \
           || uv tool install --python 3.12 --force "$name" >/dev/null 2>&1; then
            log "rebuilt ${spec} on 3.12"
        else
            log "WARNING: could not rebuild $name on 3.12; it stays on $("$tool/bin/python" -V 2>&1)"
        fi
    done
}

# 3b. The link uv's rebuild cannot fix from inside.
#
# uv creates each tool env with `bin/python` as a SYMLINK to whatever `python3`
# was at install time — here `/usr/local/bin/python3`, which leg 2 has already
# replaced with a wrapper that `exec`s the session venv. Every tool env's python
# then resolves its prefix to the VENV: `sys.prefix` is the venv, the tool's own
# site-packages never reaches `sys.path`, and the console script dies with
# `ModuleNotFoundError: No module named 'flake8'` — with flake8 sitting
# installed two directories away.
#
# TWO paths reach that state, which is why this runs after leg 3 rather than
# only on the envs leg 3 rebuilt: leg 3 rebuilds a 3.11 tool and hands the new
# env the hijacked path, and leg 2 separately breaks every PRE-EXISTING tool env
# that already pointed there and that leg 3 skips (`is_py312 … && continue`).
# Running here covers both, because leg 2 runs before leg 3.
#
# Measured 2026-08-27, post-bootstrap: mypy, flake8, black, poetry and pyright
# all dead this way; `ruff` survived (a native binary) and `pytest` survived
# (leg 2b points its shim straight at the venv, which has pytest). The
# bootstrap's `--check` called every one of them "3.12 OK", because the
# interpreter they reach IS 3.12 — it is simply the wrong one. A session then
# lints clean by not linting at all, and CI is the thing that finds out.
#
# The fix is one link: a venv's `bin/python` must resolve to a BASE interpreter,
# never to a path this hook hijacks.
#
# This lived in `scripts/session_bootstrap.sh` for one pass, which is the wrong
# home. The bootstrap is what a MULTI-repo session runs; a SINGLE-repo session
# registers this hook and never calls the bootstrap, so it got leg 2's breakage
# and none of the repair.
repair_uv_tools() {
    local tools_dir base tool link prefix
    base="$("$VENV/bin/python" -c 'import sys, os; print(os.path.join(sys.base_prefix, "bin", "python3.12"))' 2>/dev/null)" || base=""
    [ -x "$base" ] || base="$(command -v python3.12 2>/dev/null)" || base=""
    [ -x "$base" ] || return 0
    tools_dir="${PYAUTO_UV_TOOLS_DIR:-$(uv tool dir 2>/dev/null || echo "$HOME/.local/share/uv/tools")}"
    [ -d "$tools_dir" ] || return 0
    for tool in "$tools_dir"/*/; do
        link="${tool}bin/python"
        [ -L "$link" ] || continue
        # Ask the interpreter where it thinks it lives, rather than tracing the
        # link: `/usr/local/bin/python3` is a WRAPPER SCRIPT (a symlink there
        # would lose the venv — leg 2's whole note), so `readlink -f` stops at
        # the wrapper and reports nothing about the venv behind it. sys.prefix
        # is the outcome; anything else is the mechanism.
        prefix="$("$link" -c 'import sys; print(sys.prefix)' 2>/dev/null)" || continue
        [ -n "$prefix" ] || continue
        [ "$prefix" = "${tool%/}" ] && continue   # resolves to its own env: correct
        # Non-fatal like every other leg: an unwritable tools dir is a warning,
        # never a failed session start.
        ln -sfn "$base" "$link" || {
            log "WARNING: could not repoint $(basename "${tool%/}") at $base"
            continue
        }
        prefix="$("$link" -c 'import sys; print(sys.prefix)' 2>/dev/null)" || prefix=""
        if [ "$prefix" = "${tool%/}" ]; then
            log "repointed $(basename "${tool%/}") at $base (it resolved into $VENV, not its own env)"
        else
            log "WARNING: $(basename "${tool%/}") still resolves to ${prefix:-nothing} — it will not run"
        fi
    done
}

# 4. Honest git history.
#
# A remote session clones shallow. `git merge-base --is-ancestor` then LIES
# across the graft boundary: it reports "not an ancestor" for a commit whose
# ancestry is simply not in the clone, and the ship/close-out skills act on that
# answer. A completion record already logged this the hard way
# (complete/2026/08/status-sh-repos-missing-source.md, "environment note").
#
# These repos are small (single-digit MB of .git), so unshallowing costs a
# couple of seconds once per container and removes a whole class of wrong
# answer. Bounded and non-fatal: a slow or blocked network leaves a shallow
# clone and a warning, never a failed session start.
unshallow_checkout() {
    local repo
    repo="${1%/}/"   # the callers glob, but do not rely on the trailing slash
    [ -e "${repo}.git/shallow" ] || return 0
    log "unshallowing $(basename "${repo%/}") (shallow clones make ancestry checks lie)"
    if timeout 120 git -C "$repo" fetch --unshallow --quiet 2>/dev/null \
       || timeout 120 git -C "$repo" fetch --depth=2147483647 --quiet 2>/dev/null; then
        log "  $(basename "${repo%/}"): full history ($(git -C "$repo" rev-list --count HEAD 2>/dev/null) commits)"
    else
        log "  WARNING: $(basename "${repo%/}") is still shallow — run 'git fetch --unshallow' before trusting any ancestry check"
    fi
}

# Two levels, not one. A workspace is `<root>/<repo>` or, once its checkouts
# are grouped, `<root>/<family>/<repo>` — and a family directory has no `.git`,
# so a one-level walk steps straight over every checkout inside it and leaves
# them shallow while reporting nothing. A directory that IS a checkout is never
# descended into (a repo's own subdirectories are not siblings), and the walk
# stops at two levels because that is the shape of the workspace: an unbounded
# walk over every checked-out repo is not something a session start can afford.
ensure_full_clone() {
    local entry child
    for entry in "$WORKSPACE_ROOT"/*/; do
        [ -d "$entry" ] || continue
        if [ -e "${entry}.git" ]; then
            unshallow_checkout "$entry"
            continue
        fi
        for child in "$entry"*/; do
            [ -d "$child" ] && [ -e "${child}.git" ] || continue
            unshallow_checkout "$child"
        done
    done
}

# 5. Make the NEXT session in this container start correctly.
#
# Claude Code registers project hooks from the session's project directory. In a
# one-repo session that IS the repo, and `<repo>/.claude/settings.json` is found.
# In a session holding several organs the project directory is their parent,
# which is not a repo and has no `.claude/` — so none of the per-repo hooks are
# registered and none of this script runs. That is why a multi-repo session came
# up on the container's Python 3.11: not a broken hook, an unreachable one.
#
# A repo cannot ship a file into its own parent, so install it at run time. The
# settings we write fan out to every sibling repo's own hook, so the layout
# stays "each repo owns its hook" and this file stays generated-from-one-source.
# It takes effect on the next session start in this container.
install_workspace_settings() {
    # Target the WORKSPACE ROOT, derived from this checkout — not
    # $CLAUDE_PROJECT_DIR, and not only when the two differ.
    #
    # The early return this replaces ("only in the multi-repo layout") made the
    # whole leg unreachable. Writing the fan-out requires the hook to be
    # running; the hook runs only in a session whose project dir is a repo —
    # that is, a SINGLE-repo session — and the early return then skipped it as
    # having nothing to add. So the one session type that could seed the
    # container never did, and the multi-repo session that needed it never ran
    # the hook to find out. Observed directly: a container with two single-repo
    # sessions behind it still had no workspace-root settings, and the next
    # session — three organs, project dir /home/user — fired no hook, ran on
    # PATH's pytest, and left both clones shallow for three minutes until an
    # unrelated verb happened to knock on session_bootstrap.sh.
    #
    # A single-repo session has the same sibling layout one directory up, so it
    # can seed the root for free. Skip only if the root is itself a repo (then
    # it owns its own hook) or is not writable.
    local root="$WORKSPACE_ROOT"
    [ -n "$root" ] && [ -d "$root" ] || return 0
    [ -d "$root/.git" ] && return 0

    local settings="$root/.claude/settings.json"
    local fanout="$root/.claude/hooks/session-start.sh"
    [ -w "$root" ] || { log "WARNING: $root not writable ($WORKSPACE_ROOT_REASON); multi-repo sessions will keep skipping the hook"; return 0; }

    mkdir -p "$(dirname "$fanout")"
    rm -f "$fanout"   # never write through a symlink; see write_venv_shim
    cat >"$fanout" <<'FANOUT'
#!/usr/bin/env bash
# GENERATED at session start by a PyAuto repo's own session-start hook.
# Runs every sibling repo's hook, because the workspace root is not a repo and
# therefore has no hook of its own. Each repo's hook is idempotent, so the
# second and later ones cost ~0.2s.
set -u
ROOT="$(cd "$(dirname "$(readlink -f "$0")")/../.." && pwd)"

# Run one checkout's hook. Non-zero only when there is no hook to run, which is
# how a family directory is told apart from a checkout.
run_repo_hook() {
    hook="${1%/}/.claude/hooks/session-start.sh"
    [ -x "$hook" ] || return 1
    CLAUDE_PROJECT_DIR="${1%/}" "$hook" || \
        printf '[session-start] WARNING: %s failed\n' "$hook" >&2
    return 0
}

# Two levels: the checkouts directly under the root, and those one level down
# inside a family directory once the workspace groups them. Anything that
# already carries a hook is run, never walked into.
for entry in "$ROOT"/*/; do
    [ -d "$entry" ] || continue
    run_repo_hook "$entry" && continue
    for child in "$entry"*/; do
        [ -d "$child" ] || continue
        run_repo_hook "$child" || true
    done
done
FANOUT
    chmod 0755 "$fanout"

    if [ ! -f "$settings" ]; then
        cat >"$settings" <<'SETTINGS'
{
  "hooks": {
    "SessionStart": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "$CLAUDE_PROJECT_DIR/.claude/hooks/session-start.sh"
          }
        ]
      }
    ]
  }
}
SETTINGS
        log "installed $settings ($WORKSPACE_ROOT_REASON) — multi-repo sessions in this container now run the hook"
    fi
}

# Git history and hook reachability have nothing to do with the interpreter, so
# they run first and unconditionally: a container where the 3.12 venv cannot be
# built still wants honest ancestry and a hook that fires next session.
# PYAUTO_SESSION_DEFINE_ONLY=1 defines every function and performs no action —
# the seam this hook's own tests use to drive one leg at a time against a
# temporary directory, instead of against the container they run in. Sourcing
# returns; executing exits.
if [ "${PYAUTO_SESSION_DEFINE_ONLY:-}" = "1" ]; then
    return 0 2>/dev/null || exit 0
fi

# Run leg 3b on its own, without a session start. The door
# `scripts/session_bootstrap.sh` knocks on after it has run every repo's hook —
# a subprocess rather than a source, so this script's `set -euo pipefail` never
# leaks into a caller that is contractually "a bootstrap, never a gate".
if [ "${1:-}" = "--repair-uv-tools" ]; then
    repair_uv_tools
    exit 0
fi

ensure_full_clone
install_workspace_settings

# PYAUTO_SESSION_SKIP_PYTHON=1 stops here — the seam the hook's own tests use to
# exercise the legs above without building an interpreter.
if [ "${PYAUTO_SESSION_SKIP_PYTHON:-}" = "1" ]; then
    log "PYAUTO_SESSION_SKIP_PYTHON=1 — leaving the interpreter alone"
    exit 0
fi

if ensure_venv; then
    ensure_repo_extras
    point_system_default
    retool_uv_tools
    repair_uv_tools
    point_pytest_at_venv
    point_venv_scripts_at_venv
    # Every repo in the session registers this hook, so the second copy must not
    # prepend the venv a second time.
    if [ -n "${CLAUDE_ENV_FILE:-}" ] && ! grep -qs 'PYAUTO_SESSION_PY312=' "$CLAUDE_ENV_FILE"; then
        {
            echo "export PYAUTO_SESSION_PY312=\"$VENV\""
            echo "export VIRTUAL_ENV=\"$VENV\""
            echo "export PATH=\"$VENV/bin:\$PATH\""
            # The workspace root, so the Brain's shell/python resolvers agree
            # with the session instead of each re-deriving it — and how it was
            # reached, because an unverified root that a consumer inherits is
            # exactly the answer that used to be wrong without saying so.
            echo "export PYAUTO_ROOT=\"$WORKSPACE_ROOT\""
            echo "export PYAUTO_ROOT_REASON=\"$WORKSPACE_ROOT_REASON\""
        } >>"$CLAUDE_ENV_FILE"
    fi
    log "default python is now $("$VENV/bin/python" -V 2>&1) ($VENV/bin)"
fi

#!/usr/bin/env bash
# GENERATED — canonical source: PyAutoMind/policy/end_at_deliverable_hook.sh
# Installed into every checked-out repo as .claude/hooks/end-at-deliverable.sh by
# `python3 PyAutoMind/scripts/repos_sync.py --write`, and drift-checked by
# `--check`. Edit the canonical file, never a copy.
#
# ---------------------------------------------------------------------------
# Sessions end at their deliverable (PyAutoMind/policy/end_at_deliverable.md).
#
# A PreToolUse guard on the tools that outlive the turn. It exists because the
# prose rule was already written and was still broken twice: five batch members
# armed hourly check-ins on 2026-08-31 (fixed for batch members only), and on
# 2026-09-02/03 a mobile `/prm` re-armed a 60-minute `send_later` hourly from
# 02:39 to 12:11 UTC with no task active, leaving twenty fired one-shots and a
# drained usage window. A rule a session can talk itself past is not a rule, so
# this one is enforced by the harness.
#
# Registered under `hooks.PreToolUse` with the matcher
#   ^(send_later|subscribe_pr_activity|ScheduleWakeup|CronCreate|RemoteTrigger|Monitor|Bash|mcp__.*(send_later|subscribe_pr_activity).*)$
#
# Allowed through:
#   * `RemoteTrigger` with action list / get / list_runs / get_run_log — reading
#     what already exists never outlives the turn;
#   * `Monitor` running exactly the shift cockpit's stream
#     (`<abs>/PyAutoBrain/bin/pyauto-brain batch watch --stream --record
#     <abs>/PyAutoMind/batches/<slot>.md [--from N] [--for M] [--home DIR]`,
#     paths in [A-Za-z0-9._/-] only, so no shell metacharacter rides along),
#     with the CLI and the record inside the Brain and Mind checkouts this
#     workspace's own resolver names (PYAUTO_ROOT, else the `.pyauto-root`
#     above this hook, then PyAutoBrain/agents/_repo_paths.py),
#     only from 30 min before that record's `dispatched:` to 45 min after its
#     `review-at:` (the carve-out in shift_cockpit_watch.md, human ruling
#     2026-11-16, PyAutoBrain#646);
#   * any other `Monitor` that names neither GitHub nor the watcher — a local
#     tail of a pilot run's log stays allowed;
#   * every foreground `Bash` call, and every background one that is not a loop
#     on GitHub — a one-shot `gh pr checks 12` is how a session reads CI;
#   * anything at all when PYAUTO_ALLOW_TIMERS=1 — the human-authorised escape
#     for a routine the human actually asked for.
#
# Refused, beyond the timer tools: a `Monitor` that names GitHub (`gh`, a
# fetching `git`, github.com, `claude agents`, `ci_verdict`) or opens a
# WebSocket; a `Monitor` that names `batch watch` or `orchestrator_watch` in any
# but the canonical form (a piped or wrapped stream is never let through); the
# stream outside its window or on a record it cannot read; and a background
# `Bash` loop (`while`/`until`/`watch`/`sleep`) that names GitHub. Known gaps,
# listed in policy/shift_cockpit_watch.md "Limits": the check is textual, so
# `gh` inside a script file, `python -c` or `git -c ... fetch` passes it (as
# with scheduler_action_guard.py; the prose rule covers those), and a local
# watch that merely mentions github.com is refused.
#
# Everything else exits 2 (the harness treats stderr as the reason and blocks
# the call). An unreadable payload also exits 2: this fails closed, because the
# failure it guards against is silent and costs a night of usage.
set -uo pipefail

# Human-authorised routine: the one way past this guard, and it has to be set
# deliberately in the environment.
[ "${PYAUTO_ALLOW_TIMERS:-}" = "1" ] && exit 0

payload="$(cat)"

# Fast path, before Python starts: a foreground Bash call is always allowed
# (the Python below would exit 0 for it too), and it is most of the calls this
# hook sees now that the matcher includes Bash. Taken only when the payload is
# unambiguous: one JSON object, exactly one "tool_name" key, that key ahead of
# "tool_input" (so it is the top-level one, not a key nested in an input) with
# the value "Bash", and no "run_in_background" key anywhere. Anything else,
# including every timer tool and every Monitor, goes to Python. Inside a JSON
# string a quote is escaped, so `"tool_name"` cannot match string content.
_name_key='"tool_name"[[:space:]]*:'
_bash_head='"tool_name"[[:space:]]*:[[:space:]]*"Bash"[[:space:]]*[,}]'
case "$payload" in
    *'"run_in_background"'*) ;;
    '{'*'}')
        _head="${payload%%\"tool_input\"*}"
        _rest="${payload#*\"tool_name\"}"
        if [[ $_head =~ $_bash_head ]] && ! [[ $_rest =~ $_name_key ]]; then
            exit 0
        fi
        ;;
esac

# python3 rather than jq: python3 is present everywhere this hook is installed
# (it is what the SessionStart hook guarantees), jq is not. The payload goes to
# Python on stdin and the program as `-c`: an environment variable or an
# argument is capped at 128 KB per string, and a Bash call carrying a longer
# command would then fail to start Python at all (exit 126, a non-blocking hook
# error, so the call would go through unchecked).
PYAUTO_HOOK_DIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
export PYAUTO_HOOK_DIR
IFS= read -r -d '' _program <<'PY'
import json
import os
import re
import sys
from datetime import datetime, timedelta, timezone

# Read-only RemoteTrigger actions: they inspect what exists and schedule nothing.
READ_ONLY_REMOTE_TRIGGER = {"list", "get", "list_runs", "get_run_log"}

REASON = (
    "policy end_at_deliverable: sessions end at their deliverable "
    "— {tool} would outlive the turn.\n"
    "Report and stop; the human re-runs /prm. Set PYAUTO_ALLOW_TIMERS=1 only "
    "for a routine the human asked for.\n"
)

# A watch on GitHub (CI, a review, a merge, a PR, an issue) or a remote feed.
WATCH_REASON = (
    "policy end_at_deliverable: sessions end at their deliverable "
    "— {what} would watch GitHub past the turn.\n"
    "Do not wait for CI in the session: read it once in the foreground "
    "(`gh pr checks <n>` or `ci_verdict.py`), report the verdict and hand "
    "back; the human re-runs /prm when it is green.\n"
    "The only watch a session may hold is the shift cockpit's stream, inside "
    "its window. PYAUTO_ALLOW_TIMERS=1 only for a routine the human asked "
    "for.\n"
)

# The cockpit stream asked for in the wrong form, or outside its shift.
STREAM_REASON = (
    "policy end_at_deliverable: sessions end at their deliverable "
    "— {what}.\n"
    "The only watch a session may hold is the shift cockpit's stream, inside "
    "its window: exactly `<abs>/PyAutoBrain/bin/pyauto-brain batch watch "
    "--stream --record <abs>/PyAutoMind/batches/<slot>.md [--from N] "
    "[--for M] [--home DIR]`, from 30 min before the record's dispatched: to "
    "45 min after its review-at:.\n"
    "If the shift is over, report the digest and stop; do not re-arm.\n"
)

# Every character of an allowed stream command: no shell metacharacter, quote,
# glob, `$`, backtick, bracket, redirection or newline can reach a shell.
STREAM_CHARS_RE = re.compile(r"[A-Za-z0-9._/ -]+")
_PATH = r"/[A-Za-z0-9._/-]+"
STREAM_RE = re.compile(
    rf"(?P<cli>{_PATH}/PyAutoBrain/bin/pyauto-brain) batch watch --stream"
    rf" --record (?P<rec>{_PATH}/PyAutoMind/batches/[A-Za-z0-9-]+\.md)"
    rf"(?: --from [0-9]+)?(?: --for [0-9]+)?(?: --home {_PATH})?"
)
# Anything that reads the watcher's state: only the canonical stream may.
WATCHER_RE = re.compile(r"batch\s+watch|orchestrator_watch")
GITHUB_WATCH_RE = re.compile(
    r"(^|[\s;|&(`$\"'/])(gh\s|git(\s+-C\s+\S+)?\s+(fetch|pull|ls-remote|remote\s+update)\b)"
    r"|github\.com|claude\s+agents\b|ci[_-]verdict"
)
LOOP_RE = re.compile(r"\b(while|until|watch|sleep)\b")
KEY_RE = re.compile(
    r"^- (dispatched|review-at):\s*(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(?::\d{2})?Z)",
    re.M,
)
BEFORE = timedelta(minutes=30)   # before dispatched:
AFTER = timedelta(minutes=45)    # after review-at: (the late-sweep window, H2)


def block(tool):
    sys.stderr.write(REASON.format(tool=tool))
    raise SystemExit(2)


def refuse(template, what):
    sys.stderr.write(template.format(what=what))
    raise SystemExit(2)


def parse_utc(stamp):
    fmt = "%Y-%m-%dT%H:%M:%SZ" if stamp.count(":") == 2 else "%Y-%m-%dT%H:%MZ"
    return datetime.strptime(stamp, fmt).replace(tzinfo=timezone.utc)


def shift_window(record):
    """The (dispatched, review-at) pair of a shift record: the first of each key."""
    with open(record, encoding="utf-8") as handle:
        text = handle.read()
    keys = {}
    for key, stamp in KEY_RE.findall(text):
        keys.setdefault(key, stamp)
    start, end = parse_utc(keys["dispatched"]), parse_utc(keys["review-at"])
    if end < start:
        raise ValueError("review-at before dispatched")
    return start, end


def workspace_root():
    """PYAUTO_ROOT, else the nearest ancestor of this hook holding a
    `.pyauto-root` marker, else the nearest one holding a Brain checkout:
    the order of PyAutoBrain/bin/_pyauto_root.sh."""
    explicit = os.environ.get("PYAUTO_ROOT")
    if explicit:
        return explicit
    start = os.environ.get("PYAUTO_HOOK_DIR") or os.getcwd()
    ancestors = []
    path = os.path.realpath(start)
    while True:
        ancestors.append(path)
        parent = os.path.dirname(path)
        if parent == path:
            break
        path = parent
    for path in ancestors:
        if os.path.isfile(os.path.join(path, ".pyauto-root")):
            return path
    for path in ancestors:
        if brain_resolver_file(path):
            return path
    raise LookupError("no workspace root")


def brain_resolver_file(root):
    found = {
        os.path.realpath(candidate)
        for candidate in (
            os.path.join(root, "PyAutoBrain", "agents", "_repo_paths.py"),
            os.path.join(root, "organs", "PyAutoBrain", "agents", "_repo_paths.py"),
        )
        if os.path.isfile(candidate)
    }
    if len(found) > 1:
        raise LookupError("ambiguous PyAutoBrain checkouts")
    return next(iter(found), None)


def resolved_checkouts():
    """(Brain, Mind) as the workspace's own resolver places them."""
    import importlib.util

    root = workspace_root()
    resolver = brain_resolver_file(root)
    if resolver is None:
        raise LookupError("no PyAutoBrain/agents/_repo_paths.py under the root")
    spec = importlib.util.spec_from_file_location("_pyauto_repo_paths", resolver)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return (
        os.path.realpath(module.repo_path(root, "PyAutoBrain", required=True)),
        os.path.realpath(module.repo_path(root, "PyAutoMind", required=True)),
    )


def check_monitor(tool_input):
    if tool_input.get("ws") is not None:
        refuse(WATCH_REASON, "a WebSocket Monitor")
    raw = str(tool_input.get("command") or "")
    cmd = raw.strip()
    # The stream is matched unstripped: no newline anywhere, not even a last one.
    stream = STREAM_RE.fullmatch(raw) if STREAM_CHARS_RE.fullmatch(raw) else None
    if stream:
        try:
            brain, mind = resolved_checkouts()
        except Exception:
            refuse(STREAM_REASON,
                   "a shift stream with no resolvable Brain and Mind checkouts")
        # Pinned to this workspace's own checkouts, not any file that exists.
        if (os.path.realpath(stream["cli"])
                != os.path.join(brain, "bin", "pyauto-brain")
                or not os.path.isfile(stream["cli"])):
            refuse(STREAM_REASON,
                   "a shift stream from a CLI outside this workspace's PyAutoBrain")
        if (os.path.dirname(os.path.realpath(stream["rec"]))
                != os.path.join(mind, "batches")):
            refuse(STREAM_REASON,
                   "a shift stream on a record outside this workspace's PyAutoMind")
        try:
            start, end = shift_window(stream["rec"])
        except Exception:
            # Fail closed: a window the hook cannot read is one it cannot clear.
            refuse(STREAM_REASON, "a shift stream on an unreadable shift record")
        now = datetime.now(timezone.utc)
        if start - BEFORE <= now <= end + AFTER:
            raise SystemExit(0)
        refuse(STREAM_REASON,
               "a shift stream outside its window (the shift has ended or not begun)")
    if WATCHER_RE.search(cmd):
        refuse(STREAM_REASON, "a Monitor on the shift watcher in a non-canonical form")
    if GITHUB_WATCH_RE.search(cmd):
        refuse(WATCH_REASON, "a Monitor on GitHub")
    raise SystemExit(0)  # H3a: a local watch (a pilot run's log) stays allowed


def check_bash(tool_input):
    background = tool_input.get("run_in_background")
    if isinstance(background, str):
        background = background.strip().lower() == "true"
    if background is not True:
        raise SystemExit(0)  # a foreground call ends inside the turn
    cmd = str(tool_input.get("command") or "")
    if GITHUB_WATCH_RE.search(cmd) and LOOP_RE.search(cmd):
        refuse(WATCH_REASON, "a background Bash loop on GitHub")
    raise SystemExit(0)


try:
    event = json.loads(sys.stdin.read())
    if not isinstance(event, dict):
        raise ValueError("PreToolUse payload is not a JSON object")
    tool = event.get("tool_name")
    if not isinstance(tool, str) or not tool:
        raise ValueError("PreToolUse payload carries no tool_name")
    tool_input = event.get("tool_input")
    if not isinstance(tool_input, dict):
        tool_input = {}
except Exception:
    # Fail closed. A payload this hook cannot read is a call it cannot clear.
    block("an unreadable tool call")

if "RemoteTrigger" in tool:
    action = tool_input.get("action")
    if isinstance(action, str) and action in READ_ONLY_REMOTE_TRIGGER:
        raise SystemExit(0)

if tool == "Monitor":
    check_monitor(tool_input)

if tool == "Bash":
    check_bash(tool_input)

block(tool)
PY

printf '%s' "$payload" | python3 -c "$_program"

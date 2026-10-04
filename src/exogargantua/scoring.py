"""Scoring against sealed truth -- the B2 holdout (amendment A4) and the B1 test split (decision F1).

No accuracy or other comparison with sealed truth may be computed until the resolver is frozen and tagged
`resolver-frozen-v1`, and the resolver may not change after the tag (F1): the frozen paths must be identical
at the tag, at HEAD and in the working tree. Every scoring function must call `require_frozen_resolver()`.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

FROZEN_TAG = "resolver-frozen-v1"
FROZEN_PATHS = ("src/exogargantua/resolver", "models/resolver")  # resolver code and trained combiner + thresholds
ROOT = Path(__file__).resolve().parents[2]


class HoldoutSealed(RuntimeError):
    pass


def _git(*args, cwd=ROOT):
    return subprocess.run(["git", *args], capture_output=True, text=True, cwd=cwd)


def require_frozen_resolver(repo: Path = ROOT) -> str:
    """Refuse unless tag `resolver-frozen-v1` exists and is an ancestor of HEAD. Returns the tag's commit."""
    tag = _git("rev-parse", "--verify", "--quiet", f"refs/tags/{FROZEN_TAG}^{{commit}}", cwd=repo)
    if tag.returncode != 0 or not tag.stdout.strip():
        raise HoldoutSealed(f"holdout is sealed (A4): git tag {FROZEN_TAG!r} does not exist")
    anc = _git("merge-base", "--is-ancestor", tag.stdout.strip(), "HEAD", cwd=repo)
    if anc.returncode != 0:
        raise HoldoutSealed(f"holdout is sealed (A4): {FROZEN_TAG!r} is not an ancestor of HEAD")
    changed = _git("diff", "--name-only", tag.stdout.strip(), "--", *FROZEN_PATHS, cwd=repo)  # vs working tree
    if changed.returncode != 0 or changed.stdout.strip():
        raise HoldoutSealed(f"resolver changed after {FROZEN_TAG!r} (F1): {changed.stdout.split()[:5]}")
    return tag.stdout.strip()


_GUARD_OK: dict = {}


def _guard_once() -> None:
    """require_frozen_resolver() once per process (it runs git; per-call checking made scoring O(rows x git))."""
    if ROOT not in _GUARD_OK:
        _GUARD_OK[ROOT] = require_frozen_resolver()


def period_correct(p_found: float, p_true: float, tol: float = 1e-3) -> bool:
    """The pre-registered correctness rule (docs/success_criteria.md). Guarded."""
    _guard_once()
    return abs(p_found / p_true - 1.0) < tol

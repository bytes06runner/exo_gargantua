"""Scoring against holdout (B2) truth -- sealed by amendment A4 (docs/decisions.md).

No accuracy or other comparison with holdout truth may be computed until the resolver is frozen and
tagged `resolver-frozen-v1`. Every scoring function must call `require_frozen_resolver()` first.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

FROZEN_TAG = "resolver-frozen-v1"
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
    return tag.stdout.strip()


def period_correct(p_found: float, p_true: float, tol: float = 1e-3) -> bool:
    """The pre-registered correctness rule (docs/success_criteria.md). Guarded."""
    require_frozen_resolver()
    return abs(p_found / p_true - 1.0) < tol

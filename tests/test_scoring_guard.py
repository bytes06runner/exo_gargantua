"""Amendment A4: holdout scoring refuses to run until `resolver-frozen-v1` is tagged."""
import subprocess

import pytest

from exogargantua import scoring


def _repo(tmp_path):
    g = lambda *a: subprocess.run(["git", *a], cwd=tmp_path, check=True, capture_output=True)  # noqa: E731
    g("init", "-q")
    g("-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "--allow-empty", "-m", "a")
    return g


def test_sealed_without_tag(tmp_path):
    _repo(tmp_path)
    with pytest.raises(scoring.HoldoutSealed):
        scoring.require_frozen_resolver(tmp_path)


def test_opens_with_tag_on_history(tmp_path):
    g = _repo(tmp_path)
    g("tag", scoring.FROZEN_TAG)
    g("-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "--allow-empty", "-m", "b")
    assert scoring.require_frozen_resolver(tmp_path)


def test_this_repository_is_sealed_now():
    """The real repository has no resolver-frozen-v1 tag yet, so scoring must refuse."""
    with pytest.raises(scoring.HoldoutSealed):
        scoring.period_correct(1.0, 1.0)

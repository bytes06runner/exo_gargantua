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


def test_refuses_if_resolver_changed_after_tag(tmp_path):
    g = _repo(tmp_path)
    (tmp_path / "src/exogargantua/resolver").mkdir(parents=True)
    f = tmp_path / "src/exogargantua/resolver/x.py"
    f.write_text("a = 1\n")
    g("add", ".")
    g("-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "-m", "r")
    g("tag", scoring.FROZEN_TAG)
    assert scoring.require_frozen_resolver(tmp_path)
    f.write_text("a = 2\n")  # uncommitted change
    with pytest.raises(scoring.HoldoutSealed):
        scoring.require_frozen_resolver(tmp_path)
    g("-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "-am", "change")  # committed change
    with pytest.raises(scoring.HoldoutSealed):
        scoring.require_frozen_resolver(tmp_path)


def test_this_repository_is_sealed_until_tagged():
    """Until resolver-frozen-v1 exists in this repository, scoring must refuse."""
    tagged = scoring._git("rev-parse", "--verify", "--quiet", f"refs/tags/{scoring.FROZEN_TAG}").returncode == 0
    if tagged:
        pytest.skip("resolver-frozen-v1 exists")
    with pytest.raises(scoring.HoldoutSealed):
        scoring.period_correct(1.0, 1.0)

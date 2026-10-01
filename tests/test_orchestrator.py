"""Orchestrator GPU quota guard."""
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("orchestrate", ROOT / "kaggle" / "orchestrate.py")
orch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(orch)


def test_quota_guard():
    assert orch.quota_ok(3.23, 22.86)
    assert orch.quota_ok(3.23, 4.6)
    assert not orch.quota_ok(3.23, 4.5)
    assert not orch.quota_ok(3.23, None)


def test_quota_parse(monkeypatch):
    rows = [{"resource": "GPU", "used": "7.14h", "remaining": "22.86h", "total": "30.00h", "refreshAt": "x"}]
    monkeypatch.setattr(orch, "kaggle", lambda *a, **k: (0, "warning line\n" + json.dumps(rows)))
    assert orch.gpu_quota_left() == 22.86
    monkeypatch.setattr(orch, "kaggle", lambda *a, **k: (-1, "timeout"))
    assert orch.gpu_quota_left() is None


def test_screen_jobs_declare_estimates():
    plan = json.loads((ROOT / "results" / "p08_screen_plan.json").read_text())
    for k in range(1, plan["n_sessions"] + 1):
        m = json.loads((ROOT / "kaggle" / "jobs" / f"p08_screen_gpu_{k}" / "kernel-metadata.json").read_text())
        assert m["compute"] == "gpu" and m["est_quota_h"] > 0

"""Decision G2: the launcher refuses compute/session mismatches."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "kaggle"))
import push_job as P  # noqa: E402

GPU = {"compute": "gpu", "enable_gpu": True, "machine_shape": "NvidiaTeslaT4"}
CPU = {"compute": "cpu", "enable_gpu": False}


def test_gpu_job_on_t4_ok():
    assert P.check_compute(dict(GPU)) == ["--accelerator", "NvidiaTeslaT4"]


def test_cpu_job_on_cpu_ok():
    assert P.check_compute(dict(CPU)) == []


@pytest.mark.parametrize("meta", [
    {**CPU, "enable_gpu": True},                       # CPU-bound job asking for a GPU session
    {**CPU, "machine_shape": "NvidiaTeslaT4"},
    {**GPU, "enable_gpu": False},                      # GPU job on a CPU session
    {**GPU, "machine_shape": None},
    {"enable_gpu": False},                             # no compute tag
])
def test_mismatches_refused(meta):
    with pytest.raises(SystemExit):
        P.check_compute(meta)

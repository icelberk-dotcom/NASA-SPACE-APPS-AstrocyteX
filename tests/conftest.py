"""Shared fixtures: an isolated service with a tiny untrained model, temp DB and no network."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.ecg_adapter import ECGAdapter  # noqa: E402
from backend.gemini_agent import GeminiAdvisor  # noqa: E402
from backend.service import AstroGuardService  # noqa: E402
from models.astroguard import AstroGuardNet, ModelConfig  # noqa: E402
from utils.config import Settings  # noqa: E402


@pytest.fixture()
def settings(tmp_path):
    return Settings(db_path=tmp_path / "test.db", checkpoint_path=tmp_path / "none.pt", autotrain=False,
                    gemini_api_key=None, ecg_package_path=None, ecg_autostart=False,
                    ecg_service_url="http://127.0.0.1:59999", mc_samples=4)


@pytest.fixture()
def service(settings):
    model = AstroGuardNet(ModelConfig(d_model=32, head_hidden=48, d_state=16)).eval()
    model.meta = {"data": "synthetic (test)", "metrics": {}}  # type: ignore[attr-defined]
    return AstroGuardService(settings=settings, model=model, advisor=GeminiAdvisor(settings), ecg=ECGAdapter(settings))


@pytest.fixture()
def rules_only_service(settings):
    return AstroGuardService(settings=settings, load_model=False, advisor=GeminiAdvisor(settings), ecg=ECGAdapter(settings))

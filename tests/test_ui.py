"""End-to-end UI test with Streamlit's AppTest harness (no browser needed)."""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parent.parent / "ui" / "app.py")


@pytest.fixture()
def app(tmp_path, monkeypatch):
    monkeypatch.setenv("ASTROGUARD_DB", str(tmp_path / "ui.db"))
    monkeypatch.setenv("ASTROGUARD_CHECKPOINT", str(tmp_path / "missing.pt"))
    monkeypatch.setenv("ASTROGUARD_AUTOTRAIN", "0")  # rules-only mode keeps the test fast
    monkeypatch.setenv("GEMINI_API_KEY", "")
    monkeypatch.delenv("ASTROGUARD_API_URL", raising=False)
    monkeypatch.delenv("ECG_PACKAGE_PATH", raising=False)
    import backend.service as svc

    monkeypatch.setattr(svc, "_SERVICE", None)
    return AppTest.from_file(APP, default_timeout=90)


def test_dashboard_renders_and_runs_full_gamified_flow(app):
    at = app.run()
    assert not at.exception, at.exception
    # 43 number inputs - one per monitored parameter (31 physiological + 12 environment) - each with a coloured indicator beside it
    assert len([n for n in at.number_input if n.key and n.key.startswith("val_")]) == 43

    at.selectbox(key="preset_name").select("dehydration")
    at.button(key="apply_preset").click().run()
    assert at.session_state["val_fluid_intake_ml"] < 1500

    at.button(key="run_top").click().run()
    assert not at.exception, at.exception
    A = at.session_state["assessment"]
    assert A["overall"]["level"] in ("yellow", "red", "black") and A["new_tasks"]

    task_boxes = [c for c in at.checkbox if c.key and c.key.startswith("task_")]
    assert len(task_boxes) == len(A["new_tasks"])
    task_boxes[0].check().run()
    assert not at.exception, at.exception
    remaining = [c for c in at.checkbox if c.key and c.key.startswith("task_")]
    assert len(remaining) == len(task_boxes) - 1  # completed task leaves the active list
    html = " ".join(m.value for m in at.markdown)
    assert "ag-xpbar" in html and "First Step" in html  # XP bar + earned badge rendered


def test_turkish_language_switch(app):
    at = app.run()
    at.selectbox(key="lang").select("tr").run()
    assert not at.exception
    assert any("Panel" in t.label for t in at.tabs)

"""
Central runtime configuration for AstroGuard-X.

All settings come from environment variables (optionally loaded from a local
``.env`` file) so the same code runs on a laptop, in Docker or in the cloud.

Open-source attribution: python-dotenv (BSD-3-Clause) reads the ``.env`` file.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

try:  # python-dotenv is optional at import time so unit tests work without it
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None  # type: ignore[assignment]

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if load_dotenv is not None:
    load_dotenv(PROJECT_ROOT / ".env", override=False)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class Settings:
    # ---- storage / model files ----------------------------------------------------
    db_path: Path = field(default_factory=lambda: Path(os.getenv("ASTROGUARD_DB", PROJECT_ROOT / "data" / "astroguard.db")))
    checkpoint_path: Path = field(
        default_factory=lambda: Path(os.getenv("ASTROGUARD_CHECKPOINT", PROJECT_ROOT / "checkpoints" / "astroguard_v1.pt"))
    )
    autotrain: bool = field(default_factory=lambda: _env_bool("ASTROGUARD_AUTOTRAIN", True))
    # How strongly the transparent reference-range score is trusted vs. the neural net.
    rule_blend: float = field(default_factory=lambda: float(os.getenv("ASTROGUARD_RULE_BLEND", "0.65")))
    mc_samples: int = field(default_factory=lambda: int(os.getenv("ASTROGUARD_MC_SAMPLES", "24")))

    # ---- Gemini (Google GenAI SDK) ------------------------------------------------
    gemini_api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or None
    )
    gemini_model: str = field(default_factory=lambda: os.getenv("GEMINI_MODEL", "gemini-3.6-flash"))
    # Tried in order when the primary model is overloaded / retired (comma separated).
    gemini_fallback_models: List[str] = field(
        default_factory=lambda: [m.strip() for m in os.getenv("GEMINI_FALLBACK_MODELS", "gemini-3.1-flash-lite,gemini-3.5-flash").split(",") if m.strip()]
    )
    gemini_timeout_s: float = field(default_factory=lambda: float(os.getenv("GEMINI_TIMEOUT_S", "25")))

    # ---- ECG analyser (Teknofest V23 package) -------------------------------------
    ecg_package_path: Optional[Path] = field(
        default_factory=lambda: Path(p) if (p := os.getenv("ECG_PACKAGE_PATH")) else None
    )
    ecg_service_url: str = field(default_factory=lambda: os.getenv("ECG_SERVICE_URL", "http://127.0.0.1:5057"))
    ecg_autostart: bool = field(default_factory=lambda: _env_bool("ECG_AUTOSTART", True))
    ecg_python: Optional[str] = field(default_factory=lambda: os.getenv("ECG_PYTHON") or None)
    ecg_startup_timeout_s: float = field(default_factory=lambda: float(os.getenv("ECG_STARTUP_TIMEOUT_S", "180")))

    # ---- subjective-log encoder ---------------------------------------------------
    subjective_backend: str = field(default_factory=lambda: os.getenv("SUBJECTIVE_BACKEND", "lexicon"))

    # ---- API / UI wiring ----------------------------------------------------------
    api_url: Optional[str] = field(default_factory=lambda: os.getenv("ASTROGUARD_API_URL") or None)
    cors_origins: List[str] = field(
        default_factory=lambda: [o.strip() for o in os.getenv("ASTROGUARD_CORS", "http://localhost:8501").split(",") if o.strip()]
    )

    @property
    def ecg_port(self) -> int:
        try:
            return int(self.ecg_service_url.rsplit(":", 1)[1].strip("/"))
        except (IndexError, ValueError):
            return 5057


def get_settings() -> Settings:
    """Fresh settings object (cheap; re-reads the environment each call)."""
    return Settings()

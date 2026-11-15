"""
Backend access for the AstrocyteX Streamlit UI: in-process (``LocalClient``) or over HTTP
(``RemoteClient``, used when ``ASTROGUARD_API_URL`` is set, e.g. with docker-compose).

Open-source attribution: *requests* (Apache-2.0) for the REST client.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import requests

from utils.config import get_settings


class LocalClient:
    """Calls the service layer directly (single-process Streamlit deployment)."""

    mode = "in-process"

    def __init__(self):
        from backend.service import get_service

        self.svc = get_service()

    def health(self) -> Dict[str, Any]:
        s = self.svc
        return {
            "model": s.model_info(),
            "gemini": s.advisor.status(),
            "ecg": s.ecg_status(),
            "voice": s.voice_status(),
        }

    def assess(self, aid: str, payload: Dict[str, Any], lang: str) -> Dict[str, Any]:
        from backend.schemas import ReadingIn

        return self.svc.assess(aid, ReadingIn(**payload), lang).model_dump(mode="json")

    def latest(self, aid: str) -> Optional[Dict[str, Any]]:
        return self.svc.latest(aid)

    def history(self, aid: str) -> List[Dict[str, Any]]:
        return self.svc.risk_history(aid)

    def clear_history(self, aid: str) -> Dict[str, Any]:
        return {"cleared": self.svc.clear_history(aid)}

    def tasks(self, aid: str, status: Optional[str] = None) -> List[Dict[str, Any]]:
        return [t.model_dump(mode="json") for t in self.svc.list_tasks(aid, status)]

    def complete(self, aid: str, task_id: int) -> Dict[str, Any]:
        return self.svc.complete_task(aid, task_id).model_dump(mode="json")

    def gamification(self, aid: str) -> Dict[str, Any]:
        return self.svc.gamification_state(aid).model_dump(mode="json")

    def chat(self, aid: str, question: str, lang: str, role: str = "non_doctor") -> Dict[str, str]:
        return self.svc.chat(aid, question, lang, role=role)

    def ecg_status(self) -> Dict[str, Any]:
        return self.svc.ecg_status()

    def ecg_analyze(self, content: bytes, filename: str) -> Dict[str, Any]:
        return self.svc.analyze_ecg(content, filename)

    def ecg_sample(self, code: str) -> Dict[str, Any]:
        return self.svc.ecg_sample(code)

    def voice_status(self) -> Dict[str, Any]:
        return self.svc.voice_status()

    def voice_analyze(self, content: bytes, filename: str, lang: str = "en") -> Dict[str, Any]:
        return self.svc.analyze_voice(content, filename, lang=lang)

    def voice_sample(self, state_name: str, lang: str = "en") -> Dict[str, Any]:
        return self.svc.voice_sample(state_name, lang=lang)


class RemoteClient:
    """Talks to the FastAPI backend."""

    mode = "REST API"

    def __init__(self, base_url: str):
        self.base = base_url.rstrip("/")

    def _req(self, method: str, path: str, **kw) -> Any:
        r = requests.request(method, f"{self.base}{path}", timeout=kw.pop("timeout", 180), **kw)
        r.raise_for_status()
        return r.json()

    def health(self):
        return self._req("GET", "/health")

    def assess(self, aid, payload, lang):
        return self._req("POST", f"/astronauts/{aid}/assess", params={"lang": lang}, json=payload)

    def latest(self, aid):
        return self._req("GET", f"/astronauts/{aid}/latest")

    def history(self, aid):
        return self._req("GET", f"/astronauts/{aid}/history")

    def clear_history(self, aid):
        return self._req("DELETE", f"/astronauts/{aid}/history")

    def tasks(self, aid, status=None):
        return self._req("GET", f"/astronauts/{aid}/tasks", params={"status": status} if status else None)

    def complete(self, aid, task_id):
        return self._req("POST", f"/astronauts/{aid}/tasks/{task_id}/complete")

    def gamification(self, aid):
        return self._req("GET", f"/astronauts/{aid}/gamification")

    def chat(self, aid, question, lang, role="non_doctor"):
        return self._req("POST", f"/astronauts/{aid}/chat", params={"lang": lang}, json={"question": question, "role": role})

    def ecg_status(self):
        return self._req("GET", "/ecg/status")

    def ecg_analyze(self, content, filename):
        return self._req("POST", "/ecg/analyze", files={"file": (filename, content)}, timeout=300)

    def ecg_sample(self, code):
        return self._req("GET", f"/ecg/sample/{code}", timeout=300)

    def voice_status(self):
        return self._req("GET", "/voice/status")

    def voice_analyze(self, content, filename, lang="en"):
        return self._req("POST", "/voice/analyze", params={"lang": lang}, files={"file": (filename, content)}, timeout=180)

    def voice_sample(self, state_name, lang="en"):
        return self._req("GET", f"/voice/sample/{state_name}", params={"lang": lang}, timeout=180)


def make_client():
    url = get_settings().api_url
    return RemoteClient(url) if url else LocalClient()

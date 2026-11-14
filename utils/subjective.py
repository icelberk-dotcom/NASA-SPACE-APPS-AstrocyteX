"""
Encoder for the astronaut's free-text *subjective log* ("I feel dizzy and my back hurts").

The fusion network needs a fixed-size numeric vector, so we map the text onto
``N_SYMPTOMS`` symptom intensities plus a sentiment score and an intensity score.

Two interchangeable back-ends are provided:

* ``lexicon`` (default) - a transparent, offline, bilingual (English + Turkish)
  keyword model with simple negation handling.  Works on a flight laptop with no
  network and no model download.
* ``transformers`` - zero-shot semantic matching with a multilingual sentence
  encoder from Hugging Face *transformers* (Apache-2.0).  The log and each symptom
  description are embedded with mean pooling and compared by cosine similarity.
  Install with ``pip install -r requirements-nlp.txt``; falls back to the lexicon
  automatically if the model cannot be loaded.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional

import numpy as np

# Symptom -> (English keywords, Turkish keywords, descriptive sentence for the NLP back-end)
SYMPTOMS: Dict[str, Dict[str, object]] = {
    "fatigue": {"kw": ["tired", "fatigue", "exhausted", "sluggish", "yorgun", "bitkin", "halsiz", "uykusuz"],
                "desc": "I feel extremely tired and exhausted"},
    "headache": {"kw": ["headache", "head hurts", "migraine", "baş ağrı", "başım ağrı", "migren"],
                 "desc": "I have a headache"},
    "nausea": {"kw": ["nausea", "nauseous", "vomit", "queasy", "mide bulan", "bulantı", "kusma"],
               "desc": "I feel nauseous and want to vomit"},
    "dizziness": {"kw": ["dizzy", "dizziness", "lightheaded", "vertigo", "spinning", "baş dön", "başım dön", "sersem"],
                  "desc": "I feel dizzy and lightheaded"},
    "vision": {"kw": ["blurry", "blurred", "vision", "eyesight", "double vision", "bulanık", "görme", "göz"],
               "desc": "My vision is blurry"},
    "anxiety": {"kw": ["anxious", "anxiety", "panic", "nervous", "worried", "stress", "kaygı", "endişe", "panik", "gergin", "stres"],
                "desc": "I feel anxious, nervous and stressed"},
    "low_mood": {"kw": ["sad", "depressed", "lonely", "down", "hopeless", "üzgün", "depresif", "yalnız", "mutsuz", "umutsuz"],
                 "desc": "I feel sad, lonely and depressed"},
    "msk_pain": {"kw": ["muscle", "joint", "sore", "cramp", "ache", "kas", "eklem", "kramp", "ağrı"],
                 "desc": "My muscles and joints are sore and painful"},
    "back_pain": {"kw": ["back pain", "back hurts", "spine", "lumbar", "sırt ağrı", "bel ağrı", "belim"],
                  "desc": "I have back pain"},
    "congestion": {"kw": ["congestion", "stuffy", "runny nose", "sinus", "puffy", "tıkanık", "burun", "şiş"],
                   "desc": "My nose is stuffy and my face feels puffy"},
    "palpitations": {"kw": ["palpitation", "racing heart", "heart pounding", "skipped beat", "çarpıntı", "kalbim hızlı", "kalp atış"],
                     "desc": "My heart is racing and pounding"},
    "breathless": {"kw": ["breathless", "short of breath", "can't breathe", "shortness", "nefes darl", "nefes al", "soluk"],
                   "desc": "I am short of breath"},
    "thirst": {"kw": ["thirsty", "dry mouth", "dehydrated", "susuz", "ağzım kuru", "susadım"],
               "desc": "I am very thirsty and my mouth is dry"},
    "appetite_loss": {"kw": ["no appetite", "not hungry", "can't eat", "food tastes", "iştah", "yemek istemiyorum"],
                      "desc": "I have lost my appetite"},
    "insomnia": {"kw": ["insomnia", "can't sleep", "couldn't sleep", "woke up", "awake all night", "uyuyamadım", "uykusuzluk", "uyku sorunu"],
                 "desc": "I could not sleep well last night"},
    "irritability": {"kw": ["irritable", "angry", "frustrated", "snappy", "sinirli", "öfke", "huysuz"],
                     "desc": "I feel irritable and frustrated"},
}
SYMPTOM_KEYS: List[str] = list(SYMPTOMS.keys())
N_SYMPTOMS = len(SYMPTOM_KEYS)
SUBJ_DIM = N_SYMPTOMS + 2  # + sentiment + intensity

# English / Turkish negations that come BEFORE a symptom ("no headache", "hiç ağrı yok" -> before) ...
_NEG_BEFORE = {"no", "not", "without", "never", "none", "hiç"}
# ... and Turkish negations that come AFTER it ("baş ağrım yok", "yorgun değilim").
_NEG_AFTER = {"yok", "yoktu", "değil", "değilim"}
_INTENSIFIERS = {"severe", "very", "extremely", "terrible", "awful", "worst", "unbearable", "intense",
                 "çok", "şiddetli", "dayanılmaz", "aşırı", "korkunç"}
_POSITIVE = {"good", "great", "well", "fine", "rested", "energetic", "happy", "calm", "relaxed", "better",
             "iyi", "harika", "dinlenmiş", "enerjik", "mutlu", "sakin", "rahat"}
_NEGATIVE = {"bad", "worse", "terrible", "awful", "pain", "sick", "ill", "kötü", "hasta", "berbat", "acı"}


def _tokens(text: str) -> List[str]:
    # Turkish-safe lowercasing: "I" -> "ı", "İ" -> "i"
    text = text.replace("İ", "i").replace("I", "ı") if re.search(r"[çğıöşüÇĞİÖŞÜ]", text) else text
    return re.findall(r"[\w']+", text.lower())


class SubjectiveEncoder:
    """Turn a free-text log into a fixed ``SUBJ_DIM`` vector with values in [0, 1]."""

    def __init__(self, backend: str = "lexicon"):
        self.backend = backend
        self._tf = None  # lazily created transformers encoder
        if backend == "transformers":
            self._tf = _TransformersSymptomMatcher.try_create()
            if self._tf is None:
                self.backend = "lexicon"  # graceful fallback

    # --------------------------------------------------------------------------
    def encode(self, text: Optional[str]) -> np.ndarray:
        vec = np.zeros(SUBJ_DIM, dtype=np.float32)
        if not text or not text.strip():
            vec[N_SYMPTOMS] = 0.5  # neutral sentiment when nothing was written
            return vec
        lex = self._lexicon_vector(text)
        if self._tf is not None:
            sem = self._tf.score(text)
            lex[:N_SYMPTOMS] = np.maximum(lex[:N_SYMPTOMS], sem)
        return lex

    def describe(self, vec: np.ndarray, top_k: int = 4) -> List[str]:
        """Human readable list of the strongest symptoms found (for prompts / UI)."""
        idx = np.argsort(vec[:N_SYMPTOMS])[::-1][:top_k]
        return [SYMPTOM_KEYS[i] for i in idx if vec[i] > 0.15]

    # --------------------------------------------------------------------------
    def _lexicon_vector(self, text: str) -> np.ndarray:
        toks = _tokens(text)
        joined = " ".join(toks)
        vec = np.zeros(SUBJ_DIM, dtype=np.float32)
        intensity = sum(t in _INTENSIFIERS for t in toks)
        for i, key in enumerate(SYMPTOM_KEYS):
            hits = 0
            for kw in SYMPTOMS[key]["kw"]:  # type: ignore[union-attr]
                for m in re.finditer(re.escape(kw), joined):
                    # Negation: up to 3 words before ("no headache") or, in Turkish, right after ("baş ağrım yok").
                    before = joined[: m.start()].split()[-3:]
                    after = joined[m.end(): m.end() + 25].split()[:2]
                    if any(w in _NEG_BEFORE for w in before) or any(w in _NEG_AFTER for w in after):
                        continue
                    hits += 1
            vec[i] = min(1.0, 0.45 * hits + (0.25 if hits and intensity else 0.0))
        pos = sum(t in _POSITIVE for t in toks)
        neg = sum(t in _NEGATIVE for t in toks) + float(vec[:N_SYMPTOMS].sum())
        vec[N_SYMPTOMS] = float(np.clip(0.5 + 0.25 * (pos - neg), 0.0, 1.0))  # sentiment: 1 = positive
        vec[N_SYMPTOMS + 1] = float(min(1.0, intensity / 3.0))
        return vec


class _TransformersSymptomMatcher:
    """Zero-shot symptom scoring with a multilingual MiniLM encoder (optional)."""

    MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

    def __init__(self, tokenizer, model, torch_module):
        self.tok, self.model, self.torch = tokenizer, model, torch_module
        self.sym_emb = self._embed([str(SYMPTOMS[k]["desc"]) for k in SYMPTOM_KEYS])

    @classmethod
    def try_create(cls):
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer  # Hugging Face transformers

            tok = AutoTokenizer.from_pretrained(cls.MODEL)
            model = AutoModel.from_pretrained(cls.MODEL).eval()
            return cls(tok, model, torch)
        except Exception:  # model not installed / offline -> use lexicon only
            return None

    def _embed(self, sentences: List[str]):
        torch = self.torch
        enc = self.tok(sentences, padding=True, truncation=True, max_length=128, return_tensors="pt")
        with torch.no_grad():
            out = self.model(**enc).last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1).float()
        pooled = (out * mask).sum(1) / mask.sum(1).clamp(min=1e-9)  # mean pooling
        return torch.nn.functional.normalize(pooled, dim=-1)

    def score(self, text: str) -> np.ndarray:
        emb = self._embed([text])
        cos = (emb @ self.sym_emb.T).squeeze(0).numpy()
        # cosine 0.35 -> 0, 0.75 -> 1 (empirical, conservative: only strong matches count)
        return np.clip((cos - 0.35) / 0.40, 0.0, 1.0).astype(np.float32)

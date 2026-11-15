"""
Astronaut Voice & Psychological Mood / Risk Analyzer for AstrocyteX.

Combines acoustic prosodic signal processing (GeMAPS-inspired F0, jitter, shimmer, HNR,
speech cadence, pause dynamics, spectral tilt, MFCCs) with ``AstroVoiceNet v1.0``
(Spectrogram Conv-Attention + Prosody Gated Residual Network + Bayesian MC-Dropout)
to quantify psychological state, mood dimensions, and vocal health risk.
"""
from __future__ import annotations

import base64
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from models.voice_model import (
    VOICE_MODEL_VERSION,
    AstroVoiceNet,
    load_voice_checkpoint,
    rule_voice_assessment,
)
from utils.config import PROJECT_ROOT, Settings, get_settings
from utils.reference_ranges import level_color, risk_level
from utils.voice_synth import (
    VOICE_FEATURE_NAMES,
    VOICE_STATE_META,
    VOICE_STATES,
    decode_audio_bytes,
    encode_wav_bytes,
    extract_voice_features,
    synthesize_astronaut_voice,
)

log = logging.getLogger("astrocytex.voice")

BIOMARKER_LABELS: Dict[str, Dict[str, str]] = {
    "f0_mean_hz": {"tr": "Temel Frekans (F0 Ort.)", "en": "Fundamental Freq (F0 Mean)"},
    "f0_std_hz": {"tr": "Perde Değişkenliği (F0 SS)", "en": "Pitch Std Dev (F0 SD)"},
    "f0_cv": {"tr": "Bağıl Perde Varyasyonu (CV)", "en": "Relative Pitch Variation (CV)"},
    "f0_iqr_hz": {"tr": "Perde Çeyrekler Açıklığı", "en": "Pitch IQR"},
    "f0_slope_hz_s": {"tr": "Prosodik Eğim", "en": "Prosodic Contour Slope"},
    "jitter_local_pct": {"tr": "Lokal Jitter (Perde Pertürbasyonu)", "en": "Local Jitter"},
    "shimmer_local_pct": {"tr": "Lokal Shimmer (Genlik Pertürbasyonu)", "en": "Local Shimmer"},
    "hnr_db": {"tr": "Harmonik-Gürültü Oranı (HNR)", "en": "Harmonics-to-Noise Ratio (HNR)"},
    "rms_mean_db": {"tr": "Ortalama Ses Enerjisi (RMS)", "en": "Mean Vocal Energy (RMS)"},
    "rms_std_db": {"tr": "Enerji Modülasyonu", "en": "Energy Modulation SD"},
    "dynamic_range_db": {"tr": "Dinamik Aralık", "en": "Dynamic Range"},
    "voiced_fraction": {"tr": "Sesli Çerçeve Oranı", "en": "Voiced Frame Fraction"},
    "pause_ratio": {"tr": "Duraksama Oranı", "en": "Speech Pause Ratio"},
    "pause_rate_hz": {"tr": "Duraksama Sıklığı", "en": "Pause Frequency"},
    "mean_pause_sec": {"tr": "Ortalama Duraksama Süresi", "en": "Mean Pause Duration"},
    "speech_rate_syl_s": {"tr": "Konuşma Hızı (Hece/sn)", "en": "Speech Cadence (Syl/s)"},
    "alpha_ratio_db": {"tr": "Spektral Alfa Oranı (1-5 kHz / <1 kHz)", "en": "Alpha Spectral Tilt Ratio"},
    "hammarberg_index_db": {"tr": "Hammarberg İndeksi", "en": "Hammarberg Spectral Index"},
    "spectral_centroid_hz": {"tr": "Spektral Kütle Merkezi", "en": "Spectral Centroid"},
    "spectral_rolloff_hz": {"tr": "Spektral Rolloff (%85)", "en": "Spectral Rolloff (85%)"},
    "spectral_flatness": {"tr": "Spektral Düzlük", "en": "Spectral Flatness"},
    "spectral_flux": {"tr": "Spektral Akı", "en": "Spectral Flux"},
    "zcr_mean": {"tr": "Sıfır Geçiş Oranı (ZCR)", "en": "Zero-Crossing Rate"},
    "zcr_std": {"tr": "ZCR Değişkenliği", "en": "ZCR Variability"},
    "breath_noise_ratio": {"tr": "İnspiratuvar Solunum Türbülansı", "en": "Inspiratory Breath Noise Ratio"},
    "crest_factor": {"tr": "Tepe Faktörü", "en": "Waveform Crest Factor"},
}

STATE_GUIDANCE: Dict[str, Dict[str, str]] = {
    "calm_regulated": {
        "clinical_tr": (
            "Prosodik kontur dengeli, glottal pertürbasyon indeksleri (jitter/shimmer) fizyolojik "
            "sınırlarda ve harmonik-gürültü oranı (HNR) korunmuş. Sempatovagal denge ve bilişsel "
            "uyanıklık açısından vokal risk saptanmadı."
        ),
        "simple_tr": (
            "Ses tonunuz dengeli, sakin ve dinlenmiş görünüyor. Mevcut çalışma ve dinlenme "
            "düzeninizi güvenle sürdürebilirsiniz."
        ),
        "clinical_en": (
            "Balanced prosodic contour with physiological glottal perturbation (jitter/shimmer) and "
            "preserved harmonics-to-noise ratio (HNR). No vocal markers of autonomic dysregulation."
        ),
        "simple_en": (
            "Your voice sounds calm, steady, and well-rested. Keep maintaining your current sleep "
            "and daily mission routine."
        ),
    },
    "cognitive_fatigue": {
        "clinical_tr": (
            "Azalmış artikülasyon/hece hızı, uzamış duraksama oranı ve düşmüş spektral alfa eğimi "
            "nörobilişsel yorgunluk ve sirkadiyen uyku borcu ile uyumludur. Kritik EVA veya hassas "
            "kenetlenme manevraları öncesinde uyku penceresi ve iş yükü revizyonu önerilir."
        ),
        "simple_tr": (
            "Konuşma hızınızdaki yavaşlama ve duraksamalar yorgun düştüğünüzü gösteriyor. Kısa bir "
            "dinlenme molası verin, su için ve bu akşam uyku saatinizi öne çekin."
        ),
        "clinical_en": (
            "Reduced syllabic cadence, prolonged inter-lexical pauses, and attenuated spectral alpha "
            "tilt indicate neurocognitive fatigue and sleep debt. Review crew workload prior to EVA."
        ),
        "simple_en": (
            "Slower speech pace and longer pauses suggest you are tired. Take a short rest break, "
            "hydrate, and protect your sleep schedule tonight."
        ),
    },
    "acute_stress": {
        "clinical_tr": (
            "Artmış F0 varyasyonu, yükselmiş vokal kord gerilimi (pozitif alfa spektral kayması, yüksek "
            "lokal jitter) ve hızlanmış konuşma temposu akut sempatoadrenal aktivasyona işaret etmektedir. "
            "GSR, tükürük kortizolü ve kardiyovasküler parametrelerle korelasyon ve yavaş tempolu "
            "solunum (0.1 Hz rezonans frekansı) protokolü önerilir."
        ),
        "simple_tr": (
            "Sesinizde belirgin bir gerginlik ve hızlı konuşma temposu algılandı. Sessiz bir bölmede "
            "15 dakika yavaş nefes egzersizi yapmanız gerginliğinizi azaltmaya yardımcı olacaktır."
        ),
        "clinical_en": (
            "Elevated relative F0 excursion, increased vocal-fold tension (positive alpha ratio shift, "
            "elevated local jitter), and rapid cadence indicate acute sympathetic arousal. Correlate with "
            "GSR/cortisol and initiate 0.1 Hz paced vagal breathing."
        ),
        "simple_en": (
            "Your voice shows signs of tension and elevated stress. Taking 15 minutes for slow, "
            "steady breathing in a quiet module will help you reset."
        ),
    },
    "low_mood_isolation": {
        "clinical_tr": (
            "Belirgin monopitch (düşük bağıl F0 varyasyon katsayısı), azalmış vokal RMS enerjisi ve "
            "uzamış duraksama latansı; uzun süreli izolasyon/kapalı kalma kaynaklı afektif düzleşme veya "
            "motivasyonel çökkünlükle uyumludur. Psikososyal destek protokolü ve aile/ekip iletişimi önerilir."
        ),
        "simple_tr": (
            "Ses enerjiniz düşük ve tonlamanız normalden daha durgun görünüyor. Ekip arkadaşlarınızla "
            "ortak bir mola planlamak veya yakınlarınızla görüntülü/sesli görüşme yapmak iyi gelecektir."
        ),
        "clinical_en": (
            "Marked monopitch (reduced F0 coefficient of variation), hypophonic RMS energy, and prolonged "
            "pause latency align with isolation-related flat affect. Recommend psychosocial support protocol."
        ),
        "simple_en": (
            "Your vocal energy sounds lower and flatter than usual. Connecting with crewmates or "
            "scheduling a family call can help lift your mood."
        ),
    },
    "dyspnea_hypoxic": {
        "clinical_tr": (
            "Kısa soluk grupları, duraksama aralıklarında yüksek frekanslı inspiratuvar türbülans ve "
            "düşmüş HNR değeri solunumsal zorlanma veya kabin gaz kompozisyonu (yüksek ppCO2 / düşük ppO2) "
            "stresini düşündürmektedir. SpO2, solunum hızı ve ECLSS kabin CO2/O2 sensörleri derhal doğrulanmalıdır."
        ),
        "simple_tr": (
            "Konuşurken sık nefes alma ve zorlanma belirtileri algılandı. Lütfen dinlenin, parmak ucu "
            "oksijen ölçümünüzü ve bulunduğunuz bölmenin havalandırma/CO2 değerlerini hemen kontrol edin."
        ),
        "clinical_en": (
            "Shortened breath groups, elevated inspiratory turbulence during pauses, and reduced HNR "
            "suggest dyspnea or cabin atmosphere stress (elevated ppCO2 / low ppO2). Verify SpO2 and ECLSS sensors."
        ),
        "simple_en": (
            "Your voice suggests shortness of breath while speaking. Rest immediately, check your oxygen "
            "sensor, and verify cabin air / CO2 levels."
        ),
    },
}


class VoiceAnalyzer:
    """Multimodal voice psychological & risk analyzer with automatic checkpoint loading."""

    def __init__(self, settings: Optional[Settings] = None, checkpoint_path: Optional[Path] = None, load_model: bool = True):
        self.settings = settings or get_settings()
        self.checkpoint_path = checkpoint_path or (PROJECT_ROOT / "checkpoints" / "astrovoice_v1.pt")
        self.model: Optional[AstroVoiceNet] = None
        if load_model and self.checkpoint_path.is_file():
            try:
                self.model = load_voice_checkpoint(self.checkpoint_path)
                log.info("Loaded AstroVoiceNet checkpoint %s", self.checkpoint_path)
            except Exception as exc:
                log.warning("Could not load AstroVoiceNet checkpoint (%s) - using acoustic rule mode", exc)

    def status(self) -> Dict[str, Any]:
        meta = getattr(self.model, "meta", {}) if self.model is not None else {}
        ver = VOICE_MODEL_VERSION if self.model is not None else "rule-1.0"
        loaded = self.model is not None
        return {
            "loaded": loaded,
            "model_loaded": loaded,
            "neural_loaded": loaded,
            "mode": VOICE_MODEL_VERSION if loaded else "acoustic-rule-screening",
            "version": ver,
            "model_version": ver,
            "validation_mcc": meta.get("test_mcc"),
            "states": [
                {"key": s, "tr": VOICE_STATE_META[s]["tr"], "en": VOICE_STATE_META[s]["en"]}
                for s in VOICE_STATES
            ],
            "metrics": {
                k: meta.get(k)
                for k in ("test_mcc", "test_macro_f1", "test_risk_mae", "params")
                if k in meta
            },
        }

    def analyze_waveform(
        self,
        waveform: np.ndarray,
        sr: int = 16000,
        record_name: str = "voice_log.wav",
        lang: str = "tr",
        role: str = "non_doctor",
    ) -> Dict[str, Any]:
        ext = extract_voice_features(waveform, sr)
        raw_bio = dict(ext["biomarkers"])
        # Normalised aliases so Web SPA, Streamlit UI, and LLM prompt all read consistent keys
        raw_bio["f0_cv"] = round(float(raw_bio.get("f0_cv_pct", 0.0)) / 100.0, 4)
        raw_bio["speech_rate_sps"] = float(raw_bio.get("speech_rate_syl_s", 0.0))
        raw_bio["pause_ratio"] = round(float(raw_bio.get("pause_ratio_pct", 0.0)) / 100.0, 4)
        rule = rule_voice_assessment(raw_bio)

        if self.model is not None:
            pred = self.model.predict(ext["features"], ext["mel_spec"], n_mc=20)
            nn_probs = pred["probs"][0]
            probs = 0.15 * rule["probs"] + 0.85 * nn_probs
            probs = probs / probs.sum()
            nn_risk = float(pred["risk_pct"][0])
            final_risk = 0.20 * rule["risk_pct"] + 0.80 * nn_risk
            if rule["risk_pct"] >= 70.0:
                final_risk = max(final_risk, rule["risk_pct"])
            half_ci = 0.80 * float(pred["ci_high"][0] - pred["ci_low"][0]) / 2.0
            ci_low = max(0.0, final_risk - half_ci)
            ci_high = min(100.0, final_risk + half_ci)
            epi_pct = round(float(pred.get("epistemic_std", [half_ci / 1.645])[0]), 2)
            mood_vec = 0.15 * rule["mood"] + 0.85 * pred["mood"][0]
            fw = pred["feature_weights"][0]
            source = VOICE_MODEL_VERSION
        else:
            probs = rule["probs"]
            final_risk = float(rule["risk_pct"])
            ci_low = max(0.0, final_risk - 8.0)
            ci_high = min(100.0, final_risk + 8.0)
            epi_pct = 4.5
            mood_vec = rule["mood"]
            fw = np.ones(len(VOICE_FEATURE_NAMES), dtype=np.float32) / len(VOICE_FEATURE_NAMES)
            source = "acoustic-rule-screening"

        final_risk = float(np.clip(final_risk, 0.0, 100.0))
        top_idx = int(np.argmax(probs))
        top_state = VOICE_STATES[top_idx]
        lv = risk_level(final_risk)

        top_fw = np.argsort(fw)[::-1][:5]
        drivers = []
        for idx in top_fw:
            fname = VOICE_FEATURE_NAMES[int(idx)]
            lbl = BIOMARKER_LABELS.get(fname, {"tr": fname, "en": fname})
            drivers.append({
                "key": fname,
                "label_tr": lbl["tr"],
                "label_en": lbl["en"],
                "weight_pct": round(float(fw[int(idx)]) * 100.0, 1),
            })

        prob_list = [
            {
                "state": s,
                "label_tr": VOICE_STATE_META[s]["tr"],
                "label_en": VOICE_STATE_META[s]["en"],
                "name_tr": VOICE_STATE_META[s]["tr"],
                "name_en": VOICE_STATE_META[s]["en"],
                "pct": round(float(probs[i]) * 100.0, 1),
                "probability_pct": round(float(probs[i]) * 100.0, 1),
            }
            for i, s in sorted(enumerate(VOICE_STATES), key=lambda kv: -probs[kv[0]])
        ]

        mood_dict = {
            "valence_pct": round(float(mood_vec[0]), 1),
            "arousal_pct": round(float(mood_vec[1]), 1),
            "stress_pct": round(float(mood_vec[2]), 1),
            "fatigue_pct": round(float(mood_vec[3]), 1),
        }
        g = STATE_GUIDANCE[top_state]
        active_summary = (
            (g["clinical_tr"] if role == "doctor" else g["simple_tr"])
            if lang == "tr"
            else (g["clinical_en"] if role == "doctor" else g["simple_en"])
        )

        return {
            "record_name": record_name,
            "source": source,
            "model_version": source,
            "duration_s": ext["duration_s"],
            "state": top_state,
            "state_key": top_state,
            "psychological_state": top_state,
            "label_tr": VOICE_STATE_META[top_state]["tr"],
            "label_en": VOICE_STATE_META[top_state]["en"],
            "label_ru": VOICE_STATE_META[top_state]["ru"],
            "state_tr": VOICE_STATE_META[top_state]["tr"],
            "state_en": VOICE_STATE_META[top_state]["en"],
            "state_label_tr": VOICE_STATE_META[top_state]["tr"],
            "state_label_en": VOICE_STATE_META[top_state]["en"],
            "confidence_pct": round(float(probs[top_idx]) * 100.0, 1),
            "voice_risk_pct": round(final_risk, 1),
            "risk_pct": round(final_risk, 1),
            "rule_risk_pct": round(float(rule["risk_pct"]), 1),
            "ci_low": round(ci_low, 1),
            "ci_high": round(ci_high, 1),
            "epistemic_pct": epi_pct,
            "level": lv,
            "color": level_color(lv),
            "mood": mood_dict,
            "mood_dimensions": mood_dict,
            "probabilities": prob_list,
            "state_probabilities": prob_list,
            "biomarkers": raw_bio,
            "acoustic_metrics": raw_bio,
            "top_drivers": drivers,
            "summary": active_summary,
            "clinical_summary_tr": g["clinical_tr"],
            "simple_summary_tr": g["simple_tr"],
            "clinical_summary_en": g["clinical_en"],
            "simple_summary_en": g["simple_en"],
            "clinical_note_tr": g["clinical_tr"],
            "simple_note_tr": g["simple_tr"],
            "clinical_note_en": g["clinical_en"],
            "simple_note_en": g["simple_en"],
            "summary_doctor_tr": g["clinical_tr"],
            "summary_simple_tr": g["simple_tr"],
            "summary_doctor_en": g["clinical_en"],
            "summary_simple_en": g["simple_en"],
            "waveform_preview": ext["waveform_preview"],
            "spectrogram_preview": ext["spectrogram_preview"],
            "mel_spectrogram_preview": ext["spectrogram_preview"],
        }

    def analyze_bytes(
        self,
        content: bytes,
        filename: str = "voice.wav",
        lang: str = "tr",
        role: str = "non_doctor",
    ) -> Dict[str, Any]:
        wav, sr = decode_audio_bytes(content, filename)
        return self.analyze_waveform(wav, sr, record_name=filename, lang=lang, role=role)

    def sample(
        self,
        state_name: str = "calm_regulated",
        sex: str = "female",
        lang: str = "tr",
        role: str = "non_doctor",
        seed: int = 104,
    ) -> Dict[str, Any]:
        if state_name not in VOICE_STATE_META:
            raise ValueError(f"Unknown voice sample preset '{state_name}'. Valid: {VOICE_STATES}")
        sex_norm = "male" if sex == "male" else "female"
        wav, _ = synthesize_astronaut_voice(
            state_name=state_name,
            seed=seed + VOICE_STATES.index(state_name) * 19 + (7 if sex_norm == "male" else 0),
            duration_s=4.4,
            speaker_sex=sex_norm,
        )
        res = self.analyze_waveform(wav, 16000, record_name=f"sample_{state_name}_{sex_norm}.wav", lang=lang, role=role)
        wav_bytes = encode_wav_bytes(wav, 16000)
        b64 = base64.b64encode(wav_bytes).decode("ascii")
        res["audio_wav_b64"] = b64
        res["audio_wav_base64"] = b64
        return res

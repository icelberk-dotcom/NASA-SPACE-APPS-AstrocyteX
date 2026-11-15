"""
Acoustic feature extraction and formant-prosodic astronaut speech synthesis for AstrocyteX.

Provides:
1. ``decode_audio_bytes``: decodes WAV (PCM 8/16/24/32-bit or IEEE float), NPY, or CSV audio
   into a clean mono 16 kHz float32 waveform in [-1, 1].
2. ``extract_voice_features``: extracts 42 interpretable clinical/prosodic biomarkers
   (pitch F0 stats, relative CV, jitter, shimmer, HNR, alpha ratio, speech cadence, pause
   dynamics, spectral descriptors, and 13 MFCC means + stds) plus a (48, 64) log-Mel
   spectrogram tensor for ``AstroVoiceNet``.
3. ``synthesize_astronaut_voice``: Klatt-inspired glottal-source + 3-formant syllabic
   speech-prosody generator across diverse male/female speakers and spacecraft cabin noise,
   used for training/evaluating ``AstroVoiceNet`` and providing playable reference clips.

Scientific references:
- Eyben et al. (2016), "The Geneva Minimalistic Acoustic Parameter Set (GeMAPS) for Voice
  Research and Affective Computing", IEEE Trans. Affective Computing.
- Klatt (1980), "Software for a cascade/parallel formant synthesizer", JASA.
- NASA-STD-3001 Vol. 1 & Behavioral Health & Performance (BHP) acoustic monitoring studies.
"""
from __future__ import annotations

import io
import math
import struct
import wave
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import fft as sp_fft
from scipy import signal as sp_signal

TARGET_SR = 16000
N_MELS = 48
SPEC_FRAMES = 64

VOICE_STATES: List[str] = [
    "calm_regulated",
    "cognitive_fatigue",
    "acute_stress",
    "low_mood_isolation",
    "dyspnea_hypoxic",
]

VOICE_STATE_META: Dict[str, Dict[str, Any]] = {
    "calm_regulated": {
        "en": "Calm & Regulated",
        "tr": "Dengeli ve Regüle",
        "ru": "Спокойное / Норма",
        "base_risk": 10.0,
        "valence": 76.0,
        "arousal": 34.0,
        "stress": 14.0,
        "fatigue": 16.0,
    },
    "cognitive_fatigue": {
        "en": "Cognitive Fatigue & Sleep Debt",
        "tr": "Bilişsel Yorgunluk ve Uyku Borcu",
        "ru": "Когнитивное утомление",
        "base_risk": 56.0,
        "valence": 42.0,
        "arousal": 24.0,
        "stress": 44.0,
        "fatigue": 82.0,
    },
    "acute_stress": {
        "en": "Elevated Sympathetic Stress",
        "tr": "Akut Stres ve Sempatik Uyarılma",
        "ru": "Острый стресс",
        "base_risk": 76.0,
        "valence": 28.0,
        "arousal": 86.0,
        "stress": 85.0,
        "fatigue": 48.0,
    },
    "low_mood_isolation": {
        "en": "Flat Affect / Isolation Strain",
        "tr": "Çökkün Duygudurum / İzolasyon Yükü",
        "ru": "Сниженный аффект / Изоляция",
        "base_risk": 66.0,
        "valence": 22.0,
        "arousal": 20.0,
        "stress": 58.0,
        "fatigue": 68.0,
    },
    "dyspnea_hypoxic": {
        "en": "Respiratory / Dyspneic Vocal Strain",
        "tr": "Solunumsal Zorlanma / Hipoksik Konuşma",
        "ru": "Дыхательное напряжение",
        "base_risk": 88.0,
        "valence": 24.0,
        "arousal": 78.0,
        "stress": 82.0,
        "fatigue": 72.0,
    },
}

VOICE_FEATURE_NAMES: List[str] = [
    "f0_mean_hz",
    "f0_std_hz",
    "f0_cv",
    "f0_iqr_hz",
    "f0_slope_hz_s",
    "jitter_local_pct",
    "shimmer_local_pct",
    "hnr_db",
    "rms_mean_db",
    "rms_std_db",
    "dynamic_range_db",
    "voiced_fraction",
    "pause_ratio",
    "pause_rate_hz",
    "mean_pause_sec",
    "speech_rate_syl_s",
    "alpha_ratio_db",
    "hammarberg_index_db",
    "spectral_centroid_hz",
    "spectral_rolloff_hz",
    "spectral_flatness",
    "spectral_flux",
    "zcr_mean",
    "zcr_std",
    "breath_noise_ratio",
    "crest_factor",
    *[f"mfcc_{i}_mean" for i in range(1, 9)],
    *[f"mfcc_{i}_std" for i in range(1, 9)],
]
N_VOICE_FEATURES = len(VOICE_FEATURE_NAMES)  # 42


def _mel_filterbank(sr: int, n_fft: int, n_mels: int = N_MELS, fmin: float = 50.0, fmax: float = 7600.0) -> np.ndarray:
    def hz_to_mel(f: np.ndarray | float) -> np.ndarray | float:
        return 2595.0 * np.log10(1.0 + np.asarray(f) / 700.0)

    def mel_to_hz(m: np.ndarray) -> np.ndarray:
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    mels = np.linspace(hz_to_mel(fmin), hz_to_mel(min(fmax, sr / 2.0 - 100.0)), n_mels + 2)
    hz = mel_to_hz(mels)
    bins = np.floor((n_fft + 1) * hz / sr).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for m in range(1, n_mels + 1):
        left, center, right = bins[m - 1], bins[m], bins[m + 1]
        if center <= left:
            center = left + 1
        if right <= center:
            right = center + 1
        for k in range(left, min(center, fb.shape[1])):
            fb[m - 1, k] = (k - left) / max(1, center - left)
        for k in range(center, min(right, fb.shape[1])):
            fb[m - 1, k] = (right - k) / max(1, right - center)
    return fb


_MEL_FB_512 = _mel_filterbank(TARGET_SR, 512, N_MELS)


def encode_wav_bytes(samples: np.ndarray, sr: int = TARGET_SR) -> bytes:
    """Encode a mono floatwaveform in [-1, 1] into standard 16-bit PCM RIFF WAV bytes."""
    arr = np.asarray(samples, dtype=np.float32).ravel()
    peak = float(np.max(np.abs(arr))) if arr.size else 1.0
    if peak > 0.99:
        arr = arr / peak * 0.96
    pcm16 = np.clip(arr * 32767.0, -32768, 32767).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sr))
        wf.writeframes(pcm16.tobytes())
    return buf.getvalue()


def decode_audio_bytes(content: bytes, filename: str = "voice.wav") -> Tuple[np.ndarray, int]:
    """Decode WAV / NPY / CSV audio bytes into (mono_16khz_float32, 16000)."""
    if not content or len(content) < 16:
        raise ValueError("Audio payload is empty or too short.")
    ext = (filename.rsplit(".", 1)[-1].lower() if "." in filename else "wav")

    raw: Optional[np.ndarray] = None
    sr: int = TARGET_SR

    if ext == "npy" or content[:6] == b"\x93NUMPY":
        arr = np.load(io.BytesIO(content), allow_pickle=False)
        raw = np.asarray(arr, dtype=np.float32)
        if raw.ndim == 2:
            raw = raw.mean(axis=1 if raw.shape[1] <= 8 else 0)
        sr = TARGET_SR
    elif ext in ("csv", "txt") and content[:4] != b"RIFF":
        txt = content.decode("utf-8", errors="ignore")
        vals = [float(x) for x in txt.replace(",", "\n").splitlines() if x.strip() and x.strip()[0] in "0123456789+-."]
        if len(vals) < 400:
            raise ValueError("CSV audio must contain at least 400 samples.")
        raw = np.asarray(vals, dtype=np.float32)
        sr = TARGET_SR
    elif content[:4] == b"RIFF" and content[8:12] == b"WAVE":
        raw, sr = _parse_riff_wav(content)
    else:
        try:
            import soundfile as sf  # type: ignore
            data, sr_read = sf.read(io.BytesIO(content), dtype="float32", always_2d=True)
            raw = data.mean(axis=1)
            sr = int(sr_read)
        except Exception as exc:
            raise ValueError(f"Unsupported audio format '{ext}'. Upload WAV (or record directly in browser).") from exc

    raw = np.nan_to_num(raw.astype(np.float32).ravel(), nan=0.0, posinf=0.0, neginf=0.0)
    if raw.size < int(0.4 * sr):
        raise ValueError("Voice recording must be at least 0.4 seconds long.")

    # Trim to max 25 seconds to keep real-time latency crisp
    if raw.size > 25 * sr:
        raw = raw[: 25 * sr]

    if sr != TARGET_SR:
        gcd = math.gcd(int(sr), TARGET_SR)
        raw = sp_signal.resample_poly(raw, TARGET_SR // gcd, int(sr) // gcd).astype(np.float32)

    # Remove DC offset and gently normalize peak if extremely quiet/loud while preserving relative RMS cues
    raw = raw - float(np.mean(raw))
    peak = float(np.max(np.abs(raw)))
    if peak > 1.0:
        raw = raw / peak
    elif 1e-4 < peak < 0.04:
        raw = raw * (0.12 / peak)
    return raw, TARGET_SR


def _parse_riff_wav(content: bytes) -> Tuple[np.ndarray, int]:
    """Pure-Python RIFF WAVE parser supporting PCM 8/16/24/32-bit and IEEE float 32/64-bit."""
    pos = 12
    fmt_tag = 1
    channels = 1
    sr = TARGET_SR
    bits = 16
    data_bytes = b""
    n = len(content)
    while pos + 8 <= n:
        chunk_id = content[pos : pos + 4]
        chunk_sz = struct.unpack_from("<I", content, pos + 4)[0]
        body = content[pos + 8 : min(n, pos + 8 + chunk_sz)]
        if chunk_id == b"fmt " and len(body) >= 16:
            fmt_tag, channels, sr, _byte_rate, _align, bits = struct.unpack_from("<HHIIHH", body, 0)
            if fmt_tag == 0xFFFE and len(body) >= 26:  # WAVE_FORMAT_EXTENSIBLE
                fmt_tag = struct.unpack_from("<H", body, 24)[0]
        elif chunk_id == b"data":
            data_bytes = body
            break
        pos += 8 + chunk_sz + (chunk_sz & 1)

    if not data_bytes:
        raise ValueError("WAV file contains no audio data chunk.")

    channels = max(1, int(channels))
    if fmt_tag == 3 and bits == 32:
        arr = np.frombuffer(data_bytes, dtype="<f4").astype(np.float32)
    elif fmt_tag == 3 and bits == 64:
        arr = np.frombuffer(data_bytes, dtype="<f8").astype(np.float32)
    elif bits == 8:
        arr = (np.frombuffer(data_bytes, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif bits == 16:
        arr = np.frombuffer(data_bytes[: len(data_bytes) - (len(data_bytes) % 2)], dtype="<i2").astype(np.float32) / 32768.0
    elif bits == 24:
        usable = (len(data_bytes) // 3) * 3
        b3 = np.frombuffer(data_bytes[:usable], dtype=np.uint8).reshape(-1, 3)
        i32 = (b3[:, 0].astype(np.int32) | (b3[:, 1].astype(np.int32) << 8) | (b3[:, 2].astype(np.int32) << 16))
        i32 = np.where(i32 & 0x800000, i32 - 0x1000000, i32)
        arr = i32.astype(np.float32) / 8388608.0
    elif bits == 32:
        arr = np.frombuffer(data_bytes[: len(data_bytes) - (len(data_bytes) % 4)], dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported WAV encoding (fmt={fmt_tag}, bits={bits}).")

    if channels > 1 and arr.size >= channels:
        arr = arr[: (arr.size // channels) * channels].reshape(-1, channels).mean(axis=1)
    return arr, int(sr)


def extract_voice_features(waveform: np.ndarray, sr: int = TARGET_SR) -> Dict[str, Any]:
    """Extract 42 prosodic/spectral features, a (48, 64) log-Mel spectrogram, and UI summary."""
    x = np.asarray(waveform, dtype=np.float32).ravel()
    if sr != TARGET_SR:
        gcd = math.gcd(int(sr), TARGET_SR)
        x = sp_signal.resample_poly(x, TARGET_SR // gcd, int(sr) // gcd).astype(np.float32)
        sr = TARGET_SR

    duration_s = float(len(x)) / sr
    # Pre-emphasis filter for spectral analysis
    x_pre = np.append(x[0], x[1:] - 0.95 * x[:-1]).astype(np.float32)

    frame_len = 512   # 32 ms at 16 kHz
    hop_len = 160     # 10 ms hop
    if len(x) < frame_len:
        x = np.pad(x, (0, frame_len - len(x)))
        x_pre = np.pad(x_pre, (0, frame_len - len(x_pre)))

    n_frames = 1 + (len(x) - frame_len) // hop_len
    idx = np.arange(frame_len)[None, :] + hop_len * np.arange(n_frames)[:, None]
    frames = x[idx]
    frames_pre = x_pre[idx]
    win = np.hanning(frame_len).astype(np.float32)[None, :]

    # Frame RMS & dB
    rms = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-10)
    rms_db = 20.0 * np.log10(rms + 1e-7)
    noise_floor_db = float(np.percentile(rms_db, 12))
    peak_db = float(np.percentile(rms_db, 96))
    vad_thresh_db = max(noise_floor_db + 7.5, peak_db - 24.0, -52.0)
    voiced_mask = rms_db > vad_thresh_db

    # Zero-crossing rate
    signs = np.signbit(frames)
    zcr = np.mean(signs[:, 1:] != signs[:, :-1], axis=1).astype(np.float32)

    # Pitch (F0) + Harmonicity (HNR) via normalized frame autocorrelation on 65..380 Hz
    min_lag = int(sr / 380.0)  # ~42
    max_lag = int(sr / 65.0)   # ~246
    f0_list: List[float] = []
    f0_times: List[float] = []
    hnr_list: List[float] = []
    voiced_rms: List[float] = []

    w_frames = frames * win
    fft_ac = sp_fft.rfft(w_frames, n=1024, axis=1)
    ac_full = sp_fft.irfft(np.abs(fft_ac) ** 2, n=1024, axis=1)[:, : max_lag + 2]
    ac_norm = ac_full / (ac_full[:, :1] + 1e-9)

    for i in range(n_frames):
        if not voiced_mask[i]:
            continue
        seg = ac_norm[i, min_lag : max_lag + 1]
        pk_idx = int(np.argmax(seg))
        r_max = float(seg[pk_idx])
        if r_max > 0.30 and zcr[i] < 0.28:
            lag = min_lag + pk_idx
            # Parabolic interpolation around lag
            if 0 < pk_idx < len(seg) - 1:
                y0, y1, y2 = float(seg[pk_idx - 1]), r_max, float(seg[pk_idx + 1])
                denom = y0 - 2.0 * y1 + y2
                if abs(denom) > 1e-7:
                    lag_f = lag + 0.5 * (y0 - y2) / denom
                else:
                    lag_f = float(lag)
            else:
                lag_f = float(lag)
            f0_hz = float(sr) / max(lag_f, 1.0)
            if 65.0 <= f0_hz <= 380.0:
                f0_list.append(f0_hz)
                f0_times.append(i * hop_len / sr)
                r_clip = min(max(r_max, 0.05), 0.992)
                hnr_list.append(10.0 * math.log10(r_clip / (1.0 - r_clip)))
                voiced_rms.append(float(rms[i]))

    if len(f0_list) >= 4:
        f0_arr = np.asarray(f0_list, dtype=np.float32)
        t_arr = np.asarray(f0_times, dtype=np.float32)
        f0_mean = float(np.mean(f0_arr))
        f0_std = float(np.std(f0_arr))
        f0_cv = float(f0_std / max(f0_mean, 1.0))
        f0_iqr = float(np.percentile(f0_arr, 75) - np.percentile(f0_arr, 25))
        slope = float(np.polyfit(t_arr - t_arr.mean(), f0_arr, 1)[0]) if len(t_arr) >= 5 else 0.0
        periods = 1.0 / np.maximum(f0_arr, 40.0)
        # Contiguous frame pairs only (dt <= 25 ms)
        dt = np.diff(t_arr)
        contig = dt <= 0.025
        if np.any(contig):
            jitter_pct = float(np.mean(np.abs(np.diff(periods)[contig])) / np.mean(periods) * 100.0)
            v_rms = np.asarray(voiced_rms, dtype=np.float32)
            shimmer_pct = float(np.mean(np.abs(np.diff(v_rms)[contig])) / (np.mean(v_rms) + 1e-7) * 100.0)
        else:
            jitter_pct = float(np.mean(np.abs(np.diff(periods))) / np.mean(periods) * 100.0)
            shimmer_pct = 4.0
        hnr_db = float(np.mean(hnr_list))
    else:
        f0_mean, f0_std, f0_cv, f0_iqr, slope = 135.0, 8.0, 0.06, 10.0, 0.0
        jitter_pct, shimmer_pct, hnr_db = 1.5, 4.5, 11.0

    jitter_pct = float(np.clip(jitter_pct, 0.1, 14.0))
    shimmer_pct = float(np.clip(shimmer_pct, 0.3, 28.0))
    hnr_db = float(np.clip(hnr_db, -2.0, 28.0))

    # Pause & speech cadence metrics
    voiced_fraction = float(np.mean(voiced_mask))
    pause_mask = ~voiced_mask
    pause_runs: List[int] = []
    cur = 0
    for is_p in pause_mask:
        if is_p:
            cur += 1
        elif cur > 0:
            if cur >= 14:  # >= 140 ms pause
                pause_runs.append(cur)
            cur = 0
    if cur >= 14:
        pause_runs.append(cur)

    pause_ratio = float(sum(pause_runs) / max(1, n_frames))
    pause_rate_hz = float(len(pause_runs) / max(duration_s, 0.5))
    mean_pause_sec = float((np.mean(pause_runs) * hop_len / sr) if pause_runs else 0.08)

    # Syllable rate from smoothed RMS envelope peaks (1.5 - 9 Hz syllabic band)
    b_lp, a_lp = sp_signal.butter(2, 9.0 / (0.5 * (sr / hop_len)), btype="low")
    env_smooth = sp_signal.filtfilt(b_lp, a_lp, rms) if len(rms) > 12 else rms
    prominence = max(float(np.std(env_smooth)) * 0.32, 0.004)
    syl_peaks, _ = sp_signal.find_peaks(env_smooth, distance=11, prominence=prominence)
    speech_rate_syl_s = float(np.clip(len(syl_peaks) / max(duration_s, 0.5), 0.5, 9.5))

    # Unvoiced high-frequency breathiness / gasp energy during non-voiced frames
    if np.any(pause_mask) and np.any(voiced_mask):
        breath_noise_ratio = float(np.mean(rms[pause_mask] * zcr[pause_mask]) / (np.mean(rms[voiced_mask]) + 1e-7))
    else:
        breath_noise_ratio = 0.04

    # Magnitude spectrum & Mel filterbank
    spec_mag = np.abs(sp_fft.rfft(frames_pre * win, n=512, axis=1)).astype(np.float32)  # (T, 257)
    power_spec = (spec_mag ** 2) / 512.0
    freqs = np.linspace(0.0, sr / 2.0, spec_mag.shape[1], dtype=np.float32)

    active_spec = power_spec[voiced_mask] if np.any(voiced_mask) else power_spec
    low_band = (freqs >= 50.0) & (freqs < 1000.0)
    high_band = (freqs >= 1000.0) & (freqs <= 5000.0)
    f0_2k = (freqs >= 50.0) & (freqs < 2000.0)
    f2_5k = (freqs >= 2000.0) & (freqs <= 5000.0)

    e_low = float(np.mean(np.sum(active_spec[:, low_band], axis=1)) + 1e-9)
    e_high = float(np.mean(np.sum(active_spec[:, high_band], axis=1)) + 1e-9)
    alpha_ratio_db = float(10.0 * math.log10(e_high / e_low))

    p_max_low = float(np.mean(np.max(active_spec[:, f0_2k], axis=1)) + 1e-9)
    p_max_high = float(np.mean(np.max(active_spec[:, f2_5k], axis=1)) + 1e-9)
    hammarberg_db = float(10.0 * math.log10(p_max_low / p_max_high))

    spec_sum = np.sum(spec_mag, axis=1, keepdims=True) + 1e-8
    centroid = np.sum(spec_mag * freqs[None, :], axis=1) / spec_sum[:, 0]
    cum_e = np.cumsum(spec_mag, axis=1)
    roll_idx = np.argmax(cum_e >= 0.85 * spec_sum, axis=1)
    rolloff = freqs[roll_idx]
    geo_mean = np.exp(np.mean(np.log(spec_mag + 1e-8), axis=1))
    arith_mean = np.mean(spec_mag, axis=1) + 1e-8
    flatness = geo_mean / arith_mean
    norm_mag = spec_mag / spec_sum
    flux = np.sqrt(np.mean(np.diff(norm_mag, axis=0) ** 2, axis=1)) if len(norm_mag) > 1 else np.zeros(1)

    # Mel spectrogram & MFCCs (1..8)
    mel_e = np.matmul(power_spec, _MEL_FB_512.T) + 1e-9  # (T, 48)
    log_mel = np.log(mel_e).astype(np.float32)
    mfcc = sp_fft.dct(log_mel, type=2, axis=1, norm="ortho")[:, 1:9]  # (T, 8)
    mfcc_v = mfcc[voiced_mask] if np.sum(voiced_mask) >= 4 else mfcc
    mfcc_mean = np.mean(mfcc_v, axis=0)
    mfcc_std = np.std(mfcc_v, axis=0)

    crest = float(np.max(np.abs(x)) / (np.sqrt(np.mean(x ** 2)) + 1e-7))

    feat_vec = np.array([
        f0_mean,
        f0_std,
        f0_cv,
        f0_iqr,
        slope,
        jitter_pct,
        shimmer_pct,
        hnr_db,
        float(np.mean(rms_db)),
        float(np.std(rms_db)),
        peak_db - noise_floor_db,
        voiced_fraction,
        pause_ratio,
        pause_rate_hz,
        mean_pause_sec,
        speech_rate_syl_s,
        alpha_ratio_db,
        hammarberg_db,
        float(np.mean(centroid[voiced_mask]) if np.any(voiced_mask) else np.mean(centroid)),
        float(np.mean(rolloff[voiced_mask]) if np.any(voiced_mask) else np.mean(rolloff)),
        float(np.mean(flatness)),
        float(np.mean(flux)),
        float(np.mean(zcr)),
        float(np.std(zcr)),
        breath_noise_ratio,
        crest,
        *mfcc_mean.tolist(),
        *mfcc_std.tolist(),
    ], dtype=np.float32)

    # Resize log_mel time axis to fixed SPEC_FRAMES = 64 for the 2D CNN branch
    t_orig = np.linspace(0.0, 1.0, log_mel.shape[0])
    t_grid = np.linspace(0.0, 1.0, SPEC_FRAMES)
    spec_fixed = np.empty((N_MELS, SPEC_FRAMES), dtype=np.float32)
    for m in range(N_MELS):
        spec_fixed[m] = np.interp(t_grid, t_orig, log_mel[:, m])
    spec_norm = (spec_fixed - float(np.mean(spec_fixed))) / (float(np.std(spec_fixed)) + 1e-5)

    # Downsampled waveform preview (240 points) for crisp frontend canvas rendering
    step_w = max(1, len(x) // 240)
    wave_preview = [round(float(v), 4) for v in x[::step_w][:240]]
    spec_preview = [[round(float(v), 2) for v in row[::2]] for row in spec_norm[::2]]  # (24, 32)

    return {
        "features": feat_vec,
        "mel_spec": spec_norm,
        "duration_s": round(duration_s, 2),
        "biomarkers": {
            "f0_mean_hz": round(f0_mean, 1),
            "f0_std_hz": round(f0_std, 1),
            "f0_cv_pct": round(f0_cv * 100.0, 1),
            "jitter_pct": round(jitter_pct, 2),
            "shimmer_pct": round(shimmer_pct, 2),
            "hnr_db": round(hnr_db, 1),
            "speech_rate_syl_s": round(speech_rate_syl_s, 2),
            "pause_ratio_pct": round(pause_ratio * 100.0, 1),
            "voiced_pct": round(voiced_fraction * 100.0, 1),
            "alpha_ratio_db": round(alpha_ratio_db, 1),
            "spectral_centroid_hz": round(float(feat_vec[18]), 0),
        },
        "waveform_preview": wave_preview,
        "spectrogram_preview": spec_preview,
    }


def _resonator_biquad(f_hz: float, bw_hz: float, sr: int = TARGET_SR) -> Tuple[np.ndarray, np.ndarray]:
    r = math.exp(-math.pi * bw_hz / sr)
    theta = 2.0 * math.pi * f_hz / sr
    b = np.array([1.0 - 2.0 * r * math.cos(theta) + r * r], dtype=np.float64)
    a = np.array([1.0, -2.0 * r * math.cos(theta), r * r], dtype=np.float64)
    return b, a


def synthesize_astronaut_voice(
    state_name: str = "calm_regulated",
    seed: int = 42,
    duration_s: float = 4.6,
    speaker_sex: Optional[str] = None,
    cabin_snr_db: Optional[float] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Generate a realistic Klatt-formant prosodic astronaut voice clip for a given psychological state."""
    if state_name not in VOICE_STATE_META:
        raise ValueError(f"Unknown voice state '{state_name}'. Choose from {VOICE_STATES}.")
    rng = np.random.default_rng(seed)
    sr = TARGET_SR
    n_total = int(duration_s * sr)

    sex = speaker_sex or ("female" if rng.random() < 0.45 else "male")
    base_f0 = float(rng.uniform(188.0, 238.0) if sex == "female" else rng.uniform(98.0, 145.0))
    vt_scale = 1.14 if sex == "female" else 1.00

    # State-specific prosodic, glottal, pause & respiratory parameters
    if state_name == "calm_regulated":
        syl_rate = rng.uniform(3.9, 4.9)
        pause_prob, pause_dur_range = 0.18, (0.16, 0.28)
        f0_shift, f0_mod_cv = rng.uniform(0.97, 1.04), rng.uniform(0.085, 0.135)
        jitter, shimmer = rng.uniform(0.004, 0.010), rng.uniform(0.02, 0.045)
        tilt_high, asp_noise, breath_burst = 1.0, 0.025, 0.004
        amp_gain = rng.uniform(0.42, 0.58)
    elif state_name == "cognitive_fatigue":
        syl_rate = rng.uniform(2.1, 3.0)
        pause_prob, pause_dur_range = 0.42, (0.34, 0.68)
        f0_shift, f0_mod_cv = rng.uniform(0.90, 0.96), rng.uniform(0.048, 0.082)
        jitter, shimmer = rng.uniform(0.013, 0.024), rng.uniform(0.075, 0.135)
        tilt_high, asp_noise, breath_burst = 0.52, 0.070, 0.012
        amp_gain = rng.uniform(0.24, 0.36)
    elif state_name == "acute_stress":
        syl_rate = rng.uniform(5.4, 6.8)
        pause_prob, pause_dur_range = 0.10, (0.10, 0.19)
        f0_shift, f0_mod_cv = rng.uniform(1.14, 1.28), rng.uniform(0.165, 0.255)
        jitter, shimmer = rng.uniform(0.022, 0.042), rng.uniform(0.065, 0.115)
        tilt_high, asp_noise, breath_burst = 2.05, 0.045, 0.015
        amp_gain = rng.uniform(0.60, 0.82)
    elif state_name == "low_mood_isolation":
        syl_rate = rng.uniform(2.3, 3.2)
        pause_prob, pause_dur_range = 0.46, (0.38, 0.78)
        f0_shift, f0_mod_cv = rng.uniform(0.88, 0.95), rng.uniform(0.018, 0.042)  # stark monopitch
        jitter, shimmer = rng.uniform(0.008, 0.016), rng.uniform(0.055, 0.095)
        tilt_high, asp_noise, breath_burst = 0.38, 0.050, 0.006
        amp_gain = rng.uniform(0.15, 0.25)
    else:  # dyspnea_hypoxic
        syl_rate = rng.uniform(3.4, 4.6)
        pause_prob, pause_dur_range = 0.48, (0.22, 0.42)
        f0_shift, f0_mod_cv = rng.uniform(1.06, 1.18), rng.uniform(0.130, 0.200)
        jitter, shimmer = rng.uniform(0.020, 0.038), rng.uniform(0.095, 0.165)
        tilt_high, asp_noise, breath_burst = 1.45, 0.125, 0.095  # audible gasping inspiratory noise in pauses
        amp_gain = rng.uniform(0.38, 0.54)

    vowel_formants = [
        (520, 1480, 2480),  # /ə/
        (710, 1150, 2540),  # /a/
        (390, 1980, 2650),  # /e/
        (310, 2220, 2850),  # /i/
        (440, 920, 2420),   # /o/
    ]

    out = np.zeros(n_total, dtype=np.float64)
    cursor = int(0.08 * sr)
    phase = 0.0
    mean_f0 = base_f0 * f0_shift

    while cursor < n_total - int(0.12 * sr):
        syl_dur = float(np.clip(rng.normal(1.0 / syl_rate, 0.035), 0.10, 0.52))
        n_syl = min(int(syl_dur * sr), n_total - cursor)
        if n_syl < 400:
            break

        t_rel = np.linspace(0.0, 1.0, n_syl)
        contour = math.sin(math.pi * rng.uniform(0.6, 1.4)) * f0_mod_cv * mean_f0
        f0_curve = mean_f0 + contour * np.sin(np.pi * t_rel + rng.uniform(-0.5, 0.5))
        # Cycle-level jitter perturbation
        jit_noise = sp_signal.lfilter([0.25], [1.0, -0.75], rng.normal(0.0, jitter * mean_f0, size=n_syl))
        f0_inst = np.clip(f0_curve + jit_noise, 65.0, 400.0)

        # Glottal source: band-limited Rosenberg-like harmonic sum + aspiration
        dphi = 2.0 * np.pi * f0_inst / sr
        phi = phase + np.cumsum(dphi)
        phase = float(phi[-1] % (2.0 * np.pi))

        glottal = np.zeros(n_syl, dtype=np.float64)
        max_h = min(18, int((sr * 0.45) / np.max(f0_inst)))
        for h in range(1, max_h + 1):
            roll = (h ** (-1.35 / (0.65 + 0.35 * tilt_high)))
            glottal += roll * np.sin(h * phi + (0.2 * h))
        glottal /= max(1.0, np.max(np.abs(glottal)))
        glottal += rng.normal(0.0, asp_noise, size=n_syl)

        # Syllable amplitude envelope + shimmer
        env = np.sin(np.pi * t_rel) ** 0.65
        shim_mod = 1.0 + sp_signal.lfilter([0.2], [1.0, -0.8], rng.normal(0.0, shimmer, size=n_syl))
        src = glottal * env * np.clip(shim_mod, 0.35, 1.75)

        # Cascade 3-formant vocal tract filter
        f1, f2, f3 = vowel_formants[int(rng.integers(0, len(vowel_formants)))]
        y_syl = src
        for fk, bw in ((f1 * vt_scale, 85.0), (f2 * vt_scale, 120.0), (f3 * vt_scale, 170.0)):
            b, a = _resonator_biquad(min(fk, sr * 0.44), bw, sr)
            y_syl = sp_signal.lfilter(b, a, y_syl)

        # Spectral tilt shaping (high-frequency boost in tense/stressed phonation)
        if tilt_high > 1.25:
            b_hi, a_hi = sp_signal.butter(1, 1300.0 / (0.5 * sr), btype="high")
            y_syl = y_syl + (tilt_high - 1.0) * 0.45 * sp_signal.lfilter(b_hi, a_hi, y_syl)
        elif tilt_high < 0.75:
            b_lo, a_lo = sp_signal.butter(2, 1600.0 / (0.5 * sr), btype="low")
            y_syl = (1.0 - tilt_high) * sp_signal.lfilter(b_lo, a_lo, y_syl) + tilt_high * y_syl

        pk = float(np.max(np.abs(y_syl))) + 1e-8
        out[cursor : cursor + n_syl] += (y_syl / pk) * amp_gain
        cursor += n_syl

        # Inter-syllabic gap or respiratory pause
        if rng.random() < pause_prob:
            p_dur = float(rng.uniform(*pause_dur_range))
            n_p = min(int(p_dur * sr), n_total - cursor)
            if n_p > 200 and breath_burst > 0.02:
                # Band-passed inspiratory airflow turbulence (1.2 - 4.8 kHz)
                b_br, a_br = sp_signal.butter(2, [1200.0 / (0.5 * sr), 4800.0 / (0.5 * sr)], btype="band")
                gasp = sp_signal.lfilter(b_br, a_br, rng.normal(0.0, 1.0, size=n_p))
                gasp_env = np.sin(np.pi * np.linspace(0.0, 1.0, n_p)) ** 1.2
                out[cursor : cursor + n_p] += gasp * gasp_env * breath_burst
            cursor += n_p
        else:
            cursor += int(rng.uniform(0.015, 0.045) * sr)

    # Spacecraft cabin ECLSS fan hum + broadband ambient floor
    snr = cabin_snr_db if cabin_snr_db is not None else float(rng.uniform(22.0, 36.0))
    sig_rms = float(np.sqrt(np.mean(out ** 2)) + 1e-6)
    noise_rms = sig_rms * (10.0 ** (-snr / 20.0))
    t_full = np.arange(n_total) / sr
    cabin_hum = 0.35 * np.sin(2.0 * np.pi * 120.0 * t_full) + 0.65 * rng.normal(0.0, 1.0, size=n_total)
    cabin_hum = cabin_hum / (np.sqrt(np.mean(cabin_hum ** 2)) + 1e-7) * noise_rms
    out = np.clip(out + cabin_hum, -0.98, 0.98).astype(np.float32)

    meta = VOICE_STATE_META[state_name]
    targets = {
        "state_idx": VOICE_STATES.index(state_name),
        "state": state_name,
        "speaker_sex": sex,
        "cabin_snr_db": round(snr, 1),
        "risk_pct": float(np.clip(meta["base_risk"] + rng.normal(0.0, 3.8), 2.0, 98.0)),
        "valence_pct": float(np.clip(meta["valence"] + rng.normal(0.0, 4.5), 2.0, 98.0)),
        "arousal_pct": float(np.clip(meta["arousal"] + rng.normal(0.0, 4.5), 2.0, 98.0)),
        "stress_pct": float(np.clip(meta["stress"] + rng.normal(0.0, 4.5), 2.0, 98.0)),
        "fatigue_pct": float(np.clip(meta["fatigue"] + rng.normal(0.0, 4.5), 2.0, 98.0)),
    }
    return out, targets

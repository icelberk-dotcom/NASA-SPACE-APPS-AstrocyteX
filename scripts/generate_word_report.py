r"""
Generates the comprehensive Turkish AI architecture, library attribution, citation, and
held-out performance benchmark report for AstrocyteX v3.0 in both Microsoft Word (.docx)
and Markdown (.md) formats. Zero emojis throughout.

Outputs:
  1. C:\Users\VICTUS\Desktop\AstrocyteX_Yapay_Zeka_Model_Ozeti_ve_Performans_Raporu.docx
  2. docs/ASTROCYTEX_AI_OZET_RAPORU.md

Usage:
    .\.venv\Scripts\python.exe scripts/generate_word_report.py
"""
from __future__ import annotations

import importlib.metadata as importlib_meta
import json
import os
import platform
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from docx import Document  # noqa: E402
from docx.enum.table import WD_ALIGN_VERTICAL, WD_TABLE_ALIGNMENT  # noqa: E402
from docx.enum.text import WD_ALIGN_PARAGRAPH  # noqa: E402
from docx.oxml import parse_xml  # noqa: E402
from docx.oxml.ns import nsdecls  # noqa: E402
from docx.shared import Cm, Inches, Pt, RGBColor  # noqa: E402

from backend.gamification import BADGES  # noqa: E402
from utils.environment import ENV_EVENTS, MISSION_PROFILES, derived_environment, environment_preset  # noqa: E402
from utils.reference_ranges import CATEGORIES, PARAMETERS  # noqa: E402
from utils.voice_synth import VOICE_STATE_META, VOICE_STATES  # noqa: E402


# --------------------------------------------------------------------------- citations
CITATIONS: list[tuple[str, str]] = [
    ("[1]", "T. Dao and A. Gu, 'Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality (Mamba-2),' Proc. 41st International Conference on Machine Learning (ICML), PMLR vol. 235, pp. 10041-10071, 2024."),
    ("[2]", "S. Bai, J. Z. Kolter, and V. Koltun, 'An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling (TCN),' arXiv preprint arXiv:1803.01271, 2018."),
    ("[3]", "B. Lim, S. O. Arik, N. Loeff, and T. Pfister, 'Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting,' International Journal of Forecasting, vol. 37, no. 4, pp. 1748-1764, 2021."),
    ("[4]", "A. Jaegle et al., 'Perceiver: General Perception with Iterative Attention,' Proc. International Conference on Machine Learning (ICML), PMLR vol. 139, pp. 4651-4664, 2021."),
    ("[5]", "A. Vaswani et al., 'Attention Is All You Need,' Advances in Neural Information Processing Systems (NeurIPS), vol. 30, pp. 5998-6008, 2017."),
    ("[6]", "E. Perez, F. Strub, H. de Vries, V. Dumoulin, and A. Courville, 'FiLM: Visual Reasoning with a General Conditioning Layer,' Proc. AAAI Conference on Artificial Intelligence, vol. 32, no. 1, 2018."),
    ("[7]", "N. Shazeer, 'GLU Variants Improve Transformer (SwiGLU),' arXiv preprint arXiv:2002.05202, 2020."),
    ("[8]", "R. T. Q. Chen, Y. Rubanova, J. Bettencourt, and D. K. Duvenaud, 'Neural Ordinary Differential Equations,' Advances in Neural Information Processing Systems (NeurIPS), vol. 31, pp. 6571-6583, 2018."),
    ("[9]", "M. Raissi, P. Perdikaris, and G. E. Karniadakis, 'Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations,' Journal of Computational Physics, vol. 378, pp. 686-707, 2019."),
    ("[10]", "C. Blundell, J. Cornebise, K. Kavukcuoglu, and D. Wierstra, 'Weight Uncertainty in Neural Networks (Bayes by Backprop),' Proc. International Conference on Machine Learning (ICML), PMLR vol. 37, pp. 1613-1622, 2015."),
    ("[11]", "Y. Gal and Z. Ghahramani, 'Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning,' Proc. International Conference on Machine Learning (ICML), PMLR vol. 48, pp. 1050-1059, 2016."),
    ("[12]", "A. Kendall and Y. Gal, 'What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?,' Advances in Neural Information Processing Systems (NeurIPS), vol. 30, pp. 5574-5584, 2017."),
    ("[13]", "F. Eyben et al., 'The Geneva Minimalistic Acoustic Parameter Set (GeMAPS) for Voice Research and Affective Computing,' IEEE Transactions on Affective Computing, vol. 7, no. 2, pp. 190-202, 2016."),
    ("[14]", "J. Hu, L. Shen, and G. Sun, 'Squeeze-and-Excitation Networks,' Proc. IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 7132-7141, 2018."),
    ("[15]", "D. H. Klatt, 'Software for a cascade/parallel formant synthesizer,' Journal of the Acoustical Society of America (JASA), vol. 67, no. 3, pp. 971-995, 1980."),
    ("[16]", "J. A. Russell, 'A circumplex model of affect,' Journal of Personality and Social Psychology, vol. 39, no. 6, pp. 1161-1178, 1980."),
    ("[17]", "P. E. McSharry, G. D. Clifford, L. Tarassenko, and L. A. Smith, 'A dynamical model for generating synthetic electrocardiogram signals,' IEEE Transactions on Biomedical Engineering, vol. 50, no. 3, pp. 289-294, 2003."),
    ("[18]", "W. Cao, V. Mirjalili, and S. Raschka, 'Rank consistent ordinal regression for neural networks with application to age estimation (CORAL),' Pattern Recognition Letters, vol. 140, pp. 325-331, 2020."),
    ("[19]", "I. Loshchilov and F. Hutter, 'Decoupled Weight Decay Regularization (AdamW),' Proc. International Conference on Learning Representations (ICLR), 2019; L. N. Smith and N. Topin, 'Super-Convergence: Very Fast Training of Neural Networks Using Large Learning Rates (OneCycleLR),' SPIE Defense + Commercial Sensing, 2019."),
    ("[20]", "B. T. Polyak and A. B. Juditsky, 'Acceleration of stochastic approximation by averaging (EMA),' SIAM Journal on Control and Optimization, vol. 30, no. 4, pp. 838-855, 1992."),
    ("[21]", "B. W. Matthews, 'Comparison of the predicted and observed secondary structure of T4 phage lysozyme,' Biochimica et Biophysica Acta, vol. 405, no. 2, pp. 442-451, 1975; J. Gorodkin, 'Comparing two K-category assignments by a K-category correlation coefficient,' Computational Biology and Chemistry, vol. 28, pp. 367-374, 2004; D. Chicco and G. Jurman, 'The advantages of the Matthews correlation coefficient (MCC) over F1 score and accuracy in binary classification evaluation,' BMC Genomics, vol. 21, art. 6, 2020."),
    ("[22]", "J. Cohen, 'Weighted kappa: Nominal scale agreement provision for scaled disagreement or partial credit,' Psychological Bulletin, vol. 70, no. 4, pp. 213-220, 1968."),
    ("[23]", "National Aeronautics and Space Administration (NASA), 'NASA Space Flight Human-System Standard, Volume 1: Crew Health (NASA-STD-3001 Vol. 1) & Volume 2: Human Factors, Habitability, and Environmental Health (NASA-STD-3001 Vol. 2),' Washington, DC, Technical Standards, 2022-2023."),
    ("[24]", "Z. S. Patel et al., 'Red risks for a journey to the red planet: The highest priority human health risks for a mission to Mars,' npj Microgravity, vol. 6, art. 33, 2020; C. Zeitlin et al., 'Measurements of energetic particle radiation in transit to Mars on the Mars Science Laboratory,' Science, vol. 340, pp. 1080-1084, 2013; D. M. Hassler et al., 'Mars surface radiation environment measured with the MSL Curiosity rover,' Science, vol. 343, 1244797, 2014."),
    ("[25]", "A. Paszke et al., 'PyTorch: An Imperative Style, High-Performance Deep Learning Library,' Advances in Neural Information Processing Systems (NeurIPS), vol. 32, pp. 8024-8035, 2019."),
    ("[26]", "F. Pedregosa et al., 'Scikit-learn: Machine Learning in Python,' Journal of Machine Learning Research (JMLR), vol. 12, pp. 2825-2830, 2011."),
    ("[27]", "C. R. Harris et al., 'Array programming with NumPy,' Nature, vol. 585, pp. 357-362, 2020; P. Virtanen et al., 'SciPy 1.0: Fundamental Algorithms for Scientific Computing in Python,' Nature Methods, vol. 17, pp. 261-272, 2020; W. McKinney, 'Data Structures for Statistical Computing in Python (pandas),' Proc. 9th Python in Science Conf., pp. 56-61, 2010."),
    ("[28]", "J. D. Hunter, 'Matplotlib: A 2D Graphics Environment,' Computing in Science & Engineering, vol. 9, no. 3, pp. 90-95, 2007; Plotly Technologies Inc., 'Collaborative data science (Plotly.py),' Montreal, QC, 2015; S. Ramirez, 'FastAPI: modern, high-performance web framework for building APIs with Python,' https://fastapi.tiangolo.com, 2019; Google DeepMind / Google Cloud, 'Google GenAI Python SDK (google-genai) & Gemini Models Technical Report,' 2024-2025; W3C Audio Working Group, 'Web Audio API Recommendation,' World Wide Web Consortium, 2021."),
]


def _pkg_ver(name: str, fallback: str = "installed") -> str:
    try:
        return importlib_meta.version(name)
    except Exception:
        return fallback


def collect_software_table() -> list[list[str]]:
    import torch

    cuda_str = f"CUDA {torch.version.cuda}" if torch.version.cuda else "CPU"
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "x86-64 CPU"
    return [
        ["Python CPython Runtime", platform.python_version(), "PSF-2.0", "Ana çalışma zamanı, RIFF PCM ses çözümleme (wave), eşzamanlı veri üretimi.", "-"],
        ["PyTorch (torch) + Torch CUDA", f"{_pkg_ver('torch')} ({cuda_str})", "BSD-3-Clause", "AstrocyteX-Net v3.0 ve AstroVoiceNet v1.0 derin sinir ağları, otomatik türev, GPU eğitimi, MC-Dropout Bayesçi çıkarım.", "[25]"],
        ["NVIDIA CUDA Donanım Hızlandırıcı", f"{cuda_str} / {gpu_name}", "NVIDIA EULA", "42.000 mürettebat-günü + 4.000 ses kaydı paralel tensör eğitimi ve 1.000x özyükleme (bootstrap) değerlendirmesi.", "[25]"],
        ["NumPy", _pkg_ver("numpy"), "BSD-3-Clause", "43-parametreli vektörize simülasyon, Hann pencereli STFT, 48-bant Mel filtre bankası ve matris tabanlı MCC/QWK hesabı.", "[27]"],
        ["SciPy", _pkg_ver("scipy"), "BSD-3-Clause", "Dijital sinyal işleme, polifaz yeniden örnekleme (resample_poly), DCT-II MFCC çıkarımı, Klatt formant rezonatör filtreleri ve EKG R-tepe algılama.", "[15, 27]"],
        ["scikit-learn", _pkg_ver("scikit-learn"), "BSD-3-Clause", "Ridge, çok katmanlı algılayıcı (MLPRegressor), RandomForest ve HistGradientBoosting karşılaştırma modelleri; MCC, Macro-F1 ve QWK doğrulaması.", "[21, 22, 26]"],
        ["pandas", _pkg_ver("pandas"), "BSD-3-Clause", "Tablosal klinik/EKG/ses CSV verilerinin ayrıştırılması ve yapılandırılmış dışa aktarım.", "[27]"],
        ["FastAPI & Starlette", f"{_pkg_ver('fastapi')} / {_pkg_ver('starlette')}", "MIT / BSD-3", "Asenkron REST API (/assess, /voice/analyze, /voice/sample, /ecg/*, /chat) ve statik Web SPA sunumu.", "[28]"],
        ["Pydantic v2", _pkg_ver("pydantic"), "MIT", "Girdi/çıktı şema doğrulaması (ReadingIn, Assessment, AdvisorOutput, user_role ve voice alanları).", "[28]"],
        ["Uvicorn", _pkg_ver("uvicorn"), "BSD-3-Clause", "Yüksek başarım ASGI uygulama sunucusu.", "[28]"],
        ["Google GenAI SDK (google-genai)", _pkg_ver("google-genai"), "Apache-2.0", "Gemini tabanlı rol-duyarlı ('Ben doktorum' / 'Doktor değilim') yapılandırılmış JSON karar destek ve sohbet motoru.", "[28]"],
        ["SQLite3 Yerleşik Veritabanı", sqlite3.sqlite_version, "Public Domain", "Astronot değerlendirme geçmişi, hafifletme görevleri, XP, rütbe ve rozet durumunun ACID kalıcılığı.", "-"],
        ["HTML5 / CSS3 / ES2022 + Web Audio API", "W3C Standard", "W3C", "Sıfır-emoji karanlık mor Samanyolu arayüzü, tarayıcı içi 16 kHz PCM .wav mikrofon kaydı, çoklu ses kodek çözümü ve canlı spektrogram çizimi.", "[28]"],
        ["Streamlit & Plotly.py", f"{_pkg_ver('streamlit')} / {_pkg_ver('plotly')}", "Apache-2.0 / MIT", "İkincil etkileşimli telemetri konsolu, gösterge kadranları ve zaman serisi grafikleri.", "[28]"],
        ["Matplotlib", _pkg_ver("matplotlib"), "PSF-based", "Koyu mor temalı karışıklık matrisi, eğitim yakınsama eğrisi, olay dilim analizi ve ses modeli değerlendirme grafikleri.", "[28]"],
        ["python-docx", _pkg_ver("python-docx"), "MIT", "Bu teknik özet ve performans değerlendirme raporunun otomatik Word (.docx) formatında üretilmesi.", "-"],
        ["Requests & python-dotenv", f"{_pkg_ver('requests')} / {_pkg_ver('python-dotenv')}", "Apache-2.0 / BSD-3", "REST istemci iletişimi, EKG servis köprüsü ve ortam değişkeni yönetimi.", "-"],
        ["pytest & HTTPX", f"{_pkg_ver('pytest')} / {_pkg_ver('httpx')}", "MIT / BSD-3", "Birim, entegrasyon, FastAPI TestClient, ses analizi, doktor/sivil rol ayrımı ve sıfır-emoji denetim testleri.", "-"],
        ["TEKNOFEST V23 EKG Sistemi", "OmniNet Apex-Titan v23", "Yerel Paket", "12-derivasyonlu 10-sınıflı kardiyak aritmi ve QTc analizi (çevrimdışı tarama yedekli).", "[17]"],
    ]


# --------------------------------------------------------------------------- docx helpers
def set_cell_bg(cell, hex_color: str) -> None:
    shading = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{hex_color}"/>')
    cell._tc.get_or_add_tcPr().append(shading)


def set_table_borders(table, border_color: str = "C5B3E6") -> None:
    tblPr = table._tbl.tblPr
    borders = parse_xml(
        f'<w:tblBorders {nsdecls("w")}>'
        f'<w:top w:val="single" w:sz="6" w:space="0" w:color="{border_color}"/>'
        f'<w:bottom w:val="single" w:sz="6" w:space="0" w:color="{border_color}"/>'
        f'<w:insideH w:val="single" w:sz="4" w:space="0" w:color="E4DCF5"/>'
        f'<w:insideV w:val="single" w:sz="4" w:space="0" w:color="E4DCF5"/>'
        f'<w:left w:val="none"/><w:right w:val="none"/>'
        f"</w:tblBorders>"
    )
    tblPr.append(borders)


def style_table(doc: Document, headers: list[str], rows: list[list[str]], col_widths: list[float] | None = None) -> None:
    table = doc.add_table(rows=1 + len(rows), cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(table)

    hdr_cells = table.rows[0].cells
    for j, h in enumerate(headers):
        hdr_cells[j].vertical_alignment = WD_ALIGN_VERTICAL.CENTER
        set_cell_bg(hdr_cells[j], "1E0E3A")
        p = hdr_cells[j].paragraphs[0]
        p.paragraph_format.space_before = Pt(3)
        p.paragraph_format.space_after = Pt(3)
        run = p.add_run(h)
        run.bold = True
        run.font.size = Pt(8.5)
        run.font.name = "Segoe UI"
        run.font.color.rgb = RGBColor(244, 240, 255)

    for i, r in enumerate(rows):
        row_cells = table.rows[i + 1].cells
        bg = "F6F3FC" if i % 2 == 1 else "FFFFFF"
        for j, val in enumerate(r):
            row_cells[j].vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            set_cell_bg(row_cells[j], bg)
            p = row_cells[j].paragraphs[0]
            p.paragraph_format.space_before = Pt(2.5)
            p.paragraph_format.space_after = Pt(2.5)
            run = p.add_run(str(val))
            run.font.size = Pt(8.3)
            run.font.name = "Segoe UI"
            if j == 0:
                run.bold = True
                run.font.color.rgb = RGBColor(38, 17, 74)
            else:
                run.font.color.rgb = RGBColor(34, 30, 46)

    if col_widths:
        for row in table.rows:
            for j, w_cm in enumerate(col_widths):
                row.cells[j].width = Cm(w_cm)
    doc.add_paragraph()


def add_h1(doc: Document, text: str) -> None:
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(14)
    p.paragraph_format.space_after = Pt(6)
    run = p.add_run(text)
    run.bold = True
    run.font.size = Pt(14.5)
    run.font.name = "Segoe UI"
    run.font.color.rgb = RGBColor(30, 14, 58)
    pBdr = parse_xml(f'<w:pBdr {nsdecls("w")}><w:bottom w:val="single" w:sz="10" w:space="3" w:color="6D42C7"/></w:pBdr>')
    p._p.get_or_add_pPr().append(pBdr)


def add_h2(doc: Document, text: str) -> None:
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(10)
    p.paragraph_format.space_after = Pt(4)
    run = p.add_run(text)
    run.bold = True
    run.font.size = Pt(11.5)
    run.font.name = "Segoe UI"
    run.font.color.rgb = RGBColor(74, 39, 148)


def add_body(doc: Document, text: str, bold_prefix: str = "") -> None:
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(5)
    p.paragraph_format.line_spacing = 1.18
    if bold_prefix:
        rb = p.add_run(bold_prefix)
        rb.bold = True
        rb.font.name = "Segoe UI"
        rb.font.size = Pt(9.8)
        rb.font.color.rgb = RGBColor(38, 17, 74)
    r = p.add_run(text)
    r.font.name = "Segoe UI"
    r.font.size = Pt(9.8)
    r.font.color.rgb = RGBColor(34, 30, 46)


def add_bullet(doc: Document, bold_prefix: str, text: str) -> None:
    p = doc.add_paragraph(style="List Bullet")
    p.paragraph_format.space_after = Pt(3)
    rb = p.add_run(bold_prefix)
    rb.bold = True
    rb.font.name = "Segoe UI"
    rb.font.size = Pt(9.5)
    rb.font.color.rgb = RGBColor(38, 17, 74)
    r = p.add_run(text)
    r.font.name = "Segoe UI"
    r.font.size = Pt(9.5)
    r.font.color.rgb = RGBColor(34, 30, 46)


def generate_training_plot(out_png: Path) -> bool:
    ckpt_path = ROOT / "checkpoints" / "astroguard_v2.pt"
    if not ckpt_path.is_file():
        return False
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import torch

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        meta_root = ckpt.get("meta", {})
        hist = meta_root.get("history", [])
        metrics = meta_root.get("metrics", {})
        if not hist:
            return False

        epochs = [h["epoch"] for h in hist]
        mcc = [h.get("band_mcc", 0) for h in hist]
        f1 = [h.get("band_macro_f1", 0) for h in hist]
        qwk = [h.get("band_qwk", 0) for h in hist]
        mae = [h.get("risk_mae_pct", 0) for h in hist]
        n_samp = metrics.get("samples", 42000)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.2, 4.1), facecolor="#120922")
        for ax in (ax1, ax2):
            ax.set_facecolor("#1a0e30")
            ax.tick_params(colors="#cbbce6")
            for spine in ax.spines.values():
                spine.set_color("#3d2569")
            ax.grid(color="#311d54", linestyle="--", alpha=0.6)

        ax1.plot(epochs, mcc, label="Validation MCC", color="#a78bfa", linewidth=2.2)
        ax1.plot(epochs, f1, label="Validation Macro-F1", color="#34d399", linewidth=1.9)
        ax1.plot(epochs, qwk, label="Validation QWK", color="#f472b6", linewidth=1.8, linestyle="--")
        ax1.set_xlabel("Epoch", color="#cbbce6")
        ax1.set_ylabel("Skor", color="#cbbce6")
        ax1.set_title(f"AstrocyteX-Net v3.0 - {len(epochs)} Epoch Dogrulama Yakinsamasi", fontsize=10, color="#f3edfc", fontweight="bold")
        leg1 = ax1.legend(facecolor="#1a0e30", edgecolor="#4a2e7a")
        for txt in leg1.get_texts():
            txt.set_color("#f3edfc")

        ax2.plot(epochs, mae, color="#f59e0b", linewidth=2.2, label="Validation Risk MAE (puan)")
        ax2.set_xlabel("Epoch", color="#cbbce6")
        ax2.set_ylabel("Ortalama Mutlak Hata (yuzde puan)", color="#cbbce6")
        ax2.set_title(f"Risk MAE Dusus Egrisi ({n_samp:,} Murettebat-Gunu, CUDA GPU)", fontsize=10, color="#f3edfc", fontweight="bold")
        leg2 = ax2.legend(facecolor="#1a0e30", edgecolor="#4a2e7a")
        for txt in leg2.get_texts():
            txt.set_color("#f3edfc")

        fig.tight_layout()
        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        return True
    except Exception:
        return False


def fmt_lim(pair) -> str:
    lo, hi = pair
    if lo is not None and hi is not None:
        return f"{lo} - {hi}"
    if lo is not None:
        return f">= {lo}"
    if hi is not None:
        return f"<= {hi}"
    return "-"


def build_reports(docx_path: Path, md_path: Path) -> tuple[Path, Path]:
    eval_json = json.loads((ROOT / "docs" / "model_evaluation.json").read_text(encoding="utf-8"))
    voice_json = json.loads((ROOT / "docs" / "voice_evaluation.json").read_text(encoding="utf-8"))

    res = eval_json["results"]
    boot = eval_json["bootstrap_95ci"]
    test_info = eval_json["test"]
    meta = eval_json["model_meta"]["metrics"]

    v_meta = voice_json["meta"]
    v_sys = voice_json["systems"]
    v_slices = voice_json["slices"]

    train_png = ROOT / "docs" / "img" / "training_convergence.png"
    has_train_png = generate_training_plot(train_png)
    cm_png = ROOT / "docs" / "img" / "confusion_matrix.png"
    ev_png = ROOT / "docs" / "img" / "mcc_by_event.png"
    voice_png = ROOT / "docs" / "img" / "voice_evaluation.png"

    sw_rows = collect_software_table()

    doc = Document()
    for sec in doc.sections:
        sec.top_margin = Cm(1.9)
        sec.bottom_margin = Cm(1.9)
        sec.left_margin = Cm(1.8)
        sec.right_margin = Cm(1.8)

    # ==================== KAPAK BASLIGI ====================
    banner = doc.add_table(rows=1, cols=1)
    banner.alignment = WD_TABLE_ALIGNMENT.CENTER
    c0 = banner.rows[0].cells[0]
    set_cell_bg(c0, "16092D")
    p_top = c0.paragraphs[0]
    p_top.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_top.paragraph_format.space_before = Pt(14)

    r_sub = p_top.add_run("ASTROCYTEX v3.0  |  OTONOM UZAY TIBBI VE TELEMETRI KONSOLU\n")
    r_sub.font.name = "Segoe UI"
    r_sub.font.size = Pt(9.5)
    r_sub.bold = True
    r_sub.font.color.rgb = RGBColor(198, 176, 255)

    r_title = p_top.add_run(
        "Yapay Zeka Model Mimarisi, Ses Psikoloji Analizi, Rol-Duyarli LLM,\n"
        "Kullanilan Yazilim/Kutuphaneler ve Performans Degerlendirme Raporu\n"
    )
    r_title.font.name = "Segoe UI"
    r_title.font.size = Pt(16.0)
    r_title.bold = True
    r_title.font.color.rgb = RGBColor(255, 255, 255)

    r_desc = p_top.add_run(
        "AstrocyteX-Net v3.0 (Mamba-2, TCN+TFT, FiLM, Cross-Attention, SwiGLU, Neural ODE+PINN, Bayesci UQ)  |  "
        "AstroVoiceNet v1.0 (Log-Mel Spektrogram + 42 Prozodik Biyo-Belirtec)  |  Doktor / Doktor Degilim Adaptif LLM"
    )
    r_desc.font.name = "Segoe UI"
    r_desc.font.size = Pt(8.9)
    r_desc.font.color.rgb = RGBColor(224, 212, 252)
    p_top.paragraph_format.space_after = Pt(14)

    doc.add_paragraph()

    uplift_mcc = res["model"]["health_params"]["mcc"] - res["model_no_env"]["health_params"]["mcc"]
    exec_rows = [
        ["Sistem Adi ve Surumu", "AstrocyteX v3.0 (Karanlik Mor Samanyolu Uzay Telemetri Konsolu, Sifir-Emoji Tasarim)"],
        ["Ana Cok-Modaliteli Ag", f"AstrocyteX-Net v3.0 — {meta['params']:,} egitilebilir parametre (d_model=80, 16 Perceiver latent, SwiGLU etkilesim bloklari) [1-12, 25]"],
        ["Ses ve Mood Analiz Agi", f"AstroVoiceNet v1.0 — {v_meta['params']:,} egitilebilir parametre (48-bant Log-Mel Conv+SE+Attention + 42 GeMAPS prozodik giris) [13-16]"],
        ["Izlenen Parametre Kapsami", "43 Telemetri Parametresi (31 Fizyolojik + 12 NASA HRP RIDGE Uzay Tehlikesi) + 42 Ses Biyo-Belirteci + 12-Kanal EKG [17, 23]"],
        ["Ekstra GPU Egitim Kapsami", f"AstrocyteX-Net: {meta['samples']:,} egitim / {meta['val_samples']:,} dogrulama gunu ({meta['epochs_run']} epoch, en iyi epoch {meta['best_epoch']}) | AstroVoiceNet: {v_meta['train_samples']:,} egitim / {v_meta['val_samples']:,} dogrulama ses kaydi ({v_meta['epochs_run']} epoch)"],
        ["Bagimsiz Test Performansi (43 Parametre)", f"MCC = {res['model']['all_params']['mcc']:.4f} | Macro-F1 = {res['model']['all_params']['macro_f1']:.4f} | QWK = {res['model']['all_params']['qwk']:.4f} | Kritik Recall = %{res['model']['all_params']['critical_recall']*100:.2f} | MAE = {res['model']['mae_pct']:.3f} puan [21, 22]"],
        ["Uzay Ortami Katkisi (31 Saglik Parametresi)", f"Ortam girdisiz MCC {res['model_no_env']['health_params']['mcc']:.4f} -> Tam model MCC {res['model']['health_params']['mcc']:.4f} (+{uplift_mcc:.4f}); Under-Triage %{res['model_no_env']['health_params']['under_triage_rate']*100:.2f} -> %{res['model']['health_params']['under_triage_rate']*100:.2f}"],
        ["Bagimsiz Ses Analizi Testi (1.200 Kayit)", f"Psikolojik Durum MCC = {v_sys['astrovoice_v1']['state_mcc']:.4f} | Macro-F1 = {v_sys['astrovoice_v1']['state_macro_f1']:.4f} | Ses Risk MAE = {v_sys['astrovoice_v1']['risk_mae_pct']:.3f} puan | Mood MAE = {v_sys['astrovoice_v1']['mood_mae_pct']:.3f} puan | Cikarim = {v_sys['astrovoice_v1']['inference_ms_per_clip']:.2f} ms/kayit"],
        ["Rol-Duyarli LLM Danisman", "'Ben doktorum' (detayli uzay tibbi patofizyolojisi ve nicel biyo-belirtecler) ve 'Doktor degilim' (sade, anlasilir gunluk rehberlik) cift modlu mimari [28]"],
    ]
    style_table(doc, ["Teknik Boyut", "Ozet Deger ve Mimari Nitelik"], exec_rows, col_widths=[5.6, 11.8])

    # ==================== BOLUM 1 ====================
    add_h1(doc, "1. Sistem Genel Bakisi ve Mimari Yenilikler")
    add_body(
        doc,
        "AstrocyteX v3.0, alcak Dunya yorungesi (ISS), Ay (Lunar Gateway ve yuzey) ve derin uzay (Mars transfer ve "
        "yuzey) gorevlerinde astronotlarin fizyolojik sagligini, uzay ortami tehlikelerini ve konusma sesindeki "
        "psikolojik/duygusal degisimleri tek bir belirsizlik-duyarli yapay zeka catisinda birlestiren otonom saglik "
        "izleme platformudur [23, 24]. Bu surumde bes temel muhendislik ve bilimsel gelistirme gerceklestirilmistir:"
    )
    add_bullet(
        doc, "1. AstrocyteX Yeniden Markalama ve Sifir-Emoji Karanlik Mor Uzay Arayuzu: ",
        "Arayuzdeki yapay ve abartili parlama efektleri ile tum emojiler kod tabanindan (%100 sifir-emoji denetimi ile) "
        "kaldirilmistir. Arka planda Samanyolu galaksisinin Dunya ufkundan gorunumunu tasiyan yuksek cozunurluklu "
        "karanlik mor astrofotografi katmani uzerine ince kenarli (1px), mat murdum telemetri panelleri insa edilmistir."
    )
    add_bullet(
        doc, "2. Astronot Ses Akustigi, Psikolojik Durum ve Mood Analizi (AstroVoiceNet v1.0): ",
        "Astronotlarin canli mikrofon kayitlarini veya ses dosyalarini (.wav, .mp3, .m4a, .ogg, .webm) analiz eden "
        "yeni bir derin sinir agi gelistirilmistir [13, 14]. Sistem 5 psikolojik durumu, 4 duygusal boyutu (valans, "
        "uyarilmislik, vokal stres, bilissel yorgunluk) [16] ve %0-%100 ses risk skorunu hesaplayarak genel saglik riskine entegre eder."
    )
    add_bullet(
        doc, "3. Rol-Duyarli LLM Iletisim Mimarisi ('Ben doktorum' / 'Doktor degilim'): ",
        "Hem cevrimici Gemini API katmaninda hem de cevrimdisi derin uzay yerel karar motorunda kullanici tarafindan "
        "secilebilen cift iletisim profili uygulanmistir [28]. Doktor modunda fizyopatolojik mekanizmalar, nicel F0/jitter/HNR "
        "ve QTc/biyokimya degerleri sunulurken; doktor olmayan modda tibbi jargon arindirilarak sakin ve uygulanabilir adimlar verilir."
    )
    add_bullet(
        doc, "4. Kapasitesi Artirilmis AstrocyteX-Net v3.0 Mimarisi ve Ekstra GPU Egitimi: ",
        "Gizli temsil boyutu d_model=80'e, Perceiver darboğazı 16 latente cikarilmis, artik baglantilar ve SwiGLU "
        "tabanli GatedInteractionBlock katmanlari [7] eklenmistir. Model 42.000 murettebat-gunu uzerinde 50 epoch ve "
        "kritik sinif agirlikli CORAL kaybi [18] ile yeniden egitilmistir."
    )
    add_bullet(
        doc, "5. Kural-Cipali Kalinti Ogrenme ve Klinik Guvenlik Tabani (Safety Floor): ",
        "Model her parametrenin seffaf referans araligi skorunu temel alir ve coklu etkilesimleri logit kalintisi olarak "
        "ogrenir [10, 12]. Referans araligina gore kirmizi veya siyah (>= %70) olan hicbir bulgu yapay zeka tarafindan yumusatilamaz."
    )

    # ==================== BOLUM 2 ====================
    add_h1(doc, "2. Izlenen 43 Fizyolojik/Uzay Parametresi ve 42 Ses Biyo-Belirteci")
    add_h2(doc, "2.1. Bes Bantli Risk Siniflandirma Olcegi")
    add_body(
        doc,
        "Tum fizyolojik parametreler, uzay tehlikeleri ve ses risk skoru ayni 5 siralama bandinda degerlendirilir [18, 23]:"
    )
    style_table(
        doc,
        ["Bant Kodu", "Risk Araligi", "Renk Kodu", "Klinik ve Operasyonel Karsiligi"],
        [
            ["GRAY (Gri)", "< %15", "#8e86a6", "Bazal / ideal fizyolojik durum veya henuz olcum girilmemis pasif parametre."],
            ["GREEN (Yesil)", "%15 - %40", "#3cbf88", "Normal fizyolojik aralik; dusuk risk ve rutin telemetri izlemi."],
            ["YELLOW (Sari)", "%40 - %70", "#d9a938", "Erken sapma / dikkat bolgesi; koruyucu karsi-onlem gorevleri (+10 XP) uretilir."],
            ["RED (Kirmizi)", "%70 - %90", "#e05260", "Yuksek risk; oncelikli mudahale gorevleri (+25 XP) ve yakin izlem baslatilir."],
            ["BLACK (Siyah)", "> %90", "#090512", "Kritik acil durum; acil protokol (+50 XP) ve ucus cerrahi / gorev kontrol eskalasyonu."],
        ],
        col_widths=[3.0, 2.5, 2.3, 9.6],
    )

    add_h2(doc, "2.2. Surekli Giyilebilir Sensor ve Uyku Parametreleri (8 Parametre)")
    seq_rows = [
        [f"{p.name_tr} ({p.key})", CATEGORIES[p.category]["tr"], p.unit or "-", str(p.ideal), fmt_lim(p.normal), fmt_lim(p.warn), fmt_lim(p.danger)]
        for p in PARAMETERS if p.kind == "continuous"
    ]
    style_table(doc, ["Parametre (Anahtar)", "Sistem", "Birim", "Ideal", "Normal", "Dikkat", "Tehlike"], seq_rows, col_widths=[4.4, 2.7, 1.5, 1.4, 2.4, 2.5, 2.5])

    add_h2(doc, "2.3. Gunluk Klinik, Biyokimya, Hemogram, Idrar/Tukuruk, Beslenme ve EKG Parametreleri (23 Parametre)")
    daily_rows = [
        [f"{p.name_tr} ({p.key})", CATEGORIES[p.category]["tr"], p.unit or "-", str(p.ideal), fmt_lim(p.normal), fmt_lim(p.warn), fmt_lim(p.danger)]
        for p in PARAMETERS if p.kind == "daily"
    ]
    style_table(doc, ["Parametre (Anahtar)", "Sistem", "Birim", "Ideal", "Normal", "Dikkat", "Tehlike"], daily_rows, col_widths=[4.4, 2.7, 1.5, 1.4, 2.4, 2.5, 2.5])

    add_h2(doc, "2.4. NASA HRP 'RIDGE' Uzay Ortami Tehlike Parametreleri (12 Parametre)")
    env_rows = [
        [f"{p.name_tr} ({p.key})", CATEGORIES[p.category]["tr"], p.unit or "-", str(p.ideal), fmt_lim(p.normal), fmt_lim(p.warn), fmt_lim(p.danger)]
        for p in PARAMETERS if p.kind == "environment"
    ]
    style_table(doc, ["Tehlike Parametresi", "RIDGE Temasi", "Birim", "Ideal", "Normal", "Dikkat", "Tehlike"], env_rows, col_widths=[4.5, 2.8, 1.5, 1.4, 2.4, 2.4, 2.4])

    add_h2(doc, "2.5. Astronot Ses Analizi: 5 Psikolojik Durum ve 42 Akustik/Prozodik Biyo-Belirtec")
    add_body(
        doc,
        "Ses analizi modulu (utils/voice_synth.py ve models/voice_model.py), 16 kHz mono ses dalgasindan 32 ms Hann "
        "pencereleri ve 10 ms adimlarla (48, 64) boyutlu log-Mel spektrogram ile Geneva Minimalistic Acoustic Parameter "
        "Set (GeMAPS) literatürüne dayanan 42 sayisal biyo-belirtec cikarir [13]: temel perde istatistikleri (F0 ortalama, "
        "standart sapma, bagil varyasyon katsayisi CV, IQR, egim), glottal mikro-perturbasyonlar (lokal jitter %, lokal "
        "shimmer %, harmonik-gurultu orani HNR dB), konuskanlik ve duraksama dinamikleri (hece/sn hizi, sesli cerceve "
        "orani, duraksama orani/sikligi/suresi), spektral egim (alpha orani, Hammarberg indeksi, spektral kutle merkezi, "
        "rolloff, akı, inspiratuvar solunum turbulansi) ve 13 MFCC ortalama + standart sapma katsayisi [13, 15]."
    )
    v_state_rows = []
    for sk in VOICE_STATES:
        sm = VOICE_STATE_META[sk]
        v_state_rows.append([
            f"{sm['tr']} ({sk})",
            sm["en"],
            f"%{sm['base_risk']:.0f}",
            f"Valans %{sm['valence']:.0f} | Uyarilma %{sm['arousal']:.0f} | Stres %{sm['stress']:.0f} | Yorgunluk %{sm['fatigue']:.0f}",
        ])
    style_table(doc, ["Psikolojik Durum (TR / Kod)", "Ingilizce Karsiligi", "Baz Risk", "4 Boyutlu Afektif Mood Profili [16]"], v_state_rows, col_widths=[4.6, 3.8, 1.8, 7.2])

    # ==================== BOLUM 3 ====================
    add_h1(doc, "3. Gelistirilmis Yapay Zeka Mimarileri ve Rol-Duyarli LLM")
    add_h2(doc, "3.1. AstrocyteX-Net v3.0 Cok-Modaliteli Mimari")
    add_body(
        doc,
        f"AstrocyteX-Net v3.0 (models/astroguard.py), toplam {meta['params']:,} egitilebilir parametreyle 5 farkli "
        "veri modalitesini birlestirir. onceki surume gore d_model 64'ten 80'e, Perceiver latent sayisi 12'den 16'ya "
        "cikarilmis ve cikis oncesine SwiGLU kapili GatedInteractionBlock katmanlari [7] eklenmistir:"
    )
    style_table(
        doc,
        ["Alt Modul", "Kaynak Dosya", "Girdi Boyutu", "Mimari Mekanizma ve Akademik Temel"],
        [
            ["Mamba-2 (SSD) Sensor Kodlayici", "models/mamba2.py", "(B, 96, 7)", "24 saatlik (15 dk aralikli) surekli giyilebilir sensor akisini yapilandirilmis durum-uzayi ikiligi (SSD), nedensel konvolusyon ve RMSNorm kapilama ile dogrusal karmasiklikta kodlar [1]."],
            ["TCN + Temporal Fusion Transformer", "models/tcn_tft.py", "(B, 14, 43)", "14 gunluk 43-parametreli gecmisi seyreltilmis TCN [2], vektorize degisken secim agi (GRN), LSTM ve cok basli dikkat ile isler; 7 gunluk risk tahmini uretir [3]."],
            ["EnvironmentEncoder + FiLM", "models/environment.py", "(B, 12, 2)", "12 NASA RIDGE uzay tehlikesini [z-skoru, tehlike riski] jetonlarina donusturur, 2 katmanli oz-dikkat [5] ve FiLM (gamma, beta) katmani ile fizyolojik temsili kosullar [6]."],
            ["5-Modaliteli Cross-Attention", "models/fusion.py", "5 Modalite -> (B, 80)", "Gunluk klinik ve semptom jetonlari; sensor, trend ve uzay ortami baglamini capraz dikkatle okur ve 16 Perceiver latent darbogazinda birlestirir [4, 5]."],
            ["SwiGLU GatedInteractionBlock", "models/astroguard.py", "(B, 240) -> (B, 224)", "Birlesik temsil, ham z-skorlari, 2 haftalik egilim farki ve fiziksel ODE durumunu cift katmanli SwiGLU kapili etkilesim blogundan gecirir [7]."],
            ["Neural ODE + PINN Fizyoloji", "models/physics.py", "(B, 80) + Kontroller", "Yercekimi, egzersiz, sivi ve gorev gunune bagli 4 durumlu (sefalik sivi, plazma hacmi, kemik BMD, kas kutlesi) diferansiyel denklemi 4. derece Runge-Kutta (RK4) ve PINN kisitlariyla +180 gun cozer [8, 9]."],
            ["BayesianRiskHead + RulePrior", "models/uncertainty.py", "(B, 224) -> 44 Cikis", "Bayes-by-Backprop varyasyonel agirliklar [10], MC-Dropout [11] ve heteroskedastik olasilik basligi [12] ile epistemik/aleatorik belirsizlik ve %90 guven araligi uretir."],
        ],
        col_widths=[3.8, 3.2, 2.6, 7.8],
    )

    add_h2(doc, "3.2. AstroVoiceNet v1.0 Akustik-Prozodik Hibrit Sinir Agi")
    add_body(
        doc,
        f"AstroVoiceNet v1.0 (models/voice_model.py, {v_meta['params']:,} parametre), konusma sinyalinin hem zaman-frekans "
        "dokusu hem de uzman akustik ozelliklerini cift koldan isler: (1) Spektrogram kolu, (1, 48, 64) log-Mel "
        "girdisini üç kademeli 2B derinlik-ayrilabilir konvolusyon, Squeeze-and-Excitation kanal dikkati [14] ve 4-basli "
        "zamansal Transformer oz-dikkat havuzlamasindan gecirir; (2) Prozodik kol (ProsodyGateEncoder), 42 boyutlu "
        "normalize GeMAPS vektoru [13] uzerinde ogrenilebilir ozellik secim kapisi (feature importance gate) uygular. "
        "Iki kol birlestirilerek MC-Dropout [11] uzerinden 5 sinifli psikolojik durum olasiligi, 4 boyutlu mood vektoru "
        "[16] ve kural-cipali ses risk skoru + %90 guven araligi hesaplanir. Yuksek vokal risk saptandiginda genel "
        "saglik riski (0.78 fizyolojik + 0.22 vokal; >=%75 vokal riskte eskalasyon tabani) guncellenir ve otomatik nefes/uyku/iletisim gorevi acilir."
    )

    add_h2(doc, "3.3. Rol-Duyarli LLM Karar Destek Sistemi ('Ben doktorum' / 'Doktor degilim')")
    add_body(
        doc,
        "backend/gemini_agent.py icinde hem Google Gemini API istemcisinde hem de cevrimdisi LocalReasoner motorunda "
        "kullanici rolu (user_role = 'doctor' | 'non_doctor') tam desteklenmektedir [28]. 'Ben doktorum' secildiginde "
        "sistem uzay tibbi terminolojisi (sefalik sivi kaymasi, baroreseptor yuk bosalmasi, sempatovagal denge, F0/jitter/shimmer/HNR "
        "vokal indeksleri, QTc/aritmi olasiligi ve ayirici fizyopatoloji) kullanir. 'Doktor degilim' secildiginde ise ayni "
        "klinik tablo tibbi kisaltmalardan arindirilmis, astronotun veya sivil kullanicinin dogrudan uygulayabilecegi "
        "sakin ve net adimlarla aktarilir."
    )

    # ==================== BOLUM 4 ====================
    add_h1(doc, "4. Ekstra Egitim Metodolojisi ve GPU Yakinsama Bulgulari")
    add_bullet(
        doc, "AstrocyteX-Net v3.0 Egitim Recetesi: ",
        f"{meta['samples']:,} egitim + {meta['val_samples']:,} dogrulama murettebat-gunu (onceki 32.000 gunden %31 artirildi), "
        f"batch_size=128, {meta['epochs_run']} epoch (en iyi dogrulama MCC epoch {meta['best_epoch']}), AdamW (lr=1.8e-3, "
        "weight_decay=1e-4) + OneCycleLR [19], ustel hareketli ortalama EMA (decay=0.9985) [20] ve kritik alarm odakli "
        "CORAL sirali bant kaybi ([1.0, 1.15, 1.85, 2.75] esik agirliklari, tau=0.016) [18] kullanilmistir."
    )
    add_bullet(
        doc, "AstroVoiceNet v1.0 Egitim Recetesi: ",
        f"Klatt formant-prozodi sentezleyicisi [15] ile kadin/erkek konusmacilar ve degisken kabin gurultusu (SNR 14-34 dB) "
        f"altinda uretilen {v_meta['train_samples']:,} egitim, {v_meta['val_samples']:,} dogrulama ve {v_meta['test_samples']:,} "
        f"bagimsiz test ses kaydi uzerinde Spektrogram SpecAugment maskelemesi, etiket yumusatmali capraz entropi, "
        "heteroskedastik risk kaybi ve CORAL bant kaybi ile 45 epoch GPU egitimi tamamlanmistir."
    )

    if has_train_png and train_png.is_file():
        p_img = doc.add_paragraph()
        p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_img.add_run().add_picture(str(train_png), width=Inches(6.4))
        add_body(
            doc,
            f"Sekil 1: AstrocyteX-Net v3.0 modelinin {meta['samples']:,} murettebat-gunu uzerinde GPU egitimi boyunca "
            "dogrulama MCC, Macro-F1, QWK ve Risk MAE yakinsama egrileri."
        )

    # ==================== BOLUM 5 ====================
    add_h1(doc, "5. Yeniden Yapilan Performans Testleri ve Karsilastirmali Sonuclar")
    add_h2(doc, "5.1. Degerlendirme Protokolu ve Metrik Secimi")
    add_body(
        doc,
        f"Ana model, egitim ve dogrulama tohumlarindan tamamen ayrik {test_info['crew_days']:,} gorulmemis murettebat-gunu "
        f"({test_info['cells']:,} parametre hucresi; gray: {test_info['band_distribution']['gray']:,}, green: "
        f"{test_info['band_distribution']['green']:,}, yellow: {test_info['band_distribution']['yellow']:,}, red: "
        f"{test_info['band_distribution']['red']:,}, black: {test_info['band_distribution']['black']:,}) uzerinde "
        f"{test_info['bootstrap_resamples']} tam murettebat-gunu bootstrap orneklemesi ile test edilmistir. Sinif "
        "dengesizligi nedeniyle birincil metrik olarak Cok Sinifli Matthews Korelasyon Katsayisi (MCC) [21], tamamlayici "
        "olarak Macro-F1, Quadratic Weighted Kappa (QWK) [22] ve klinik guvenlik icin Critical Recall / Under-Triage secilmistir."
    )

    add_h2(doc, "5.2. AstrocyteX-Net v3.0 — Tum 43 Parametre Bagimsiz Test Tablosu")
    labels_tr = {
        "rule": "Kural Bazli (Referans Araliklari)",
        "ridge": "Dogrusal ML Bazi (scikit-learn Ridge) [26]",
        "mlp": "Dogrusal-Olmayan ML Bazi (scikit-learn MLP) [26]",
        "model_no_env": "Ablasyon: Ortam Girdisiz AstrocyteX-Net v3.0",
        "model": "AstrocyteX-Net v3.0 (Tam Model)",
        "deployed": "Canli Dagitim (0.20 Kural + 0.80 v3.0 + Guvenlik Tabani)",
    }
    sys_order_all = [k for k in ("rule", "ridge", "mlp", "model_no_env", "model", "deployed") if k in res]
    all_rows = []
    for name in sys_order_all:
        a = res[name]["all_params"]
        b = boot[name]
        all_rows.append([
            labels_tr[name],
            f"{a['mcc']:.4f} [{b['mcc']['lo']:.4f}-{b['mcc']['hi']:.4f}]",
            f"{a['macro_f1']:.4f} [{b['macro_f1']['lo']:.4f}-{b['macro_f1']['hi']:.4f}]",
            f"{a['qwk']:.4f}",
            f"%{a['critical_recall']*100:.2f}",
            f"%{a['under_triage_rate']*100:.2f}",
            f"{a['actionable_f1']:.4f}",
            f"{res[name]['mae_pct']:.3f}",
        ])
    style_table(
        doc,
        ["Sistem", "MCC [%95 GA]", "Macro-F1 [%95 GA]", "QWK", "Kritik Recall", "Under-Triage", "Gorev F1", "MAE"],
        all_rows,
        col_widths=[4.2, 2.5, 2.5, 1.4, 1.7, 1.7, 1.7, 1.7],
    )

    add_h2(doc, "5.3. 31 Fizyolojik Saglik Parametresi Uzay Ortami Ablasyonu ve Istatistiksel Anlamlilik")
    sys_order_h = [k for k in ("rule", "model_no_env", "ridge", "mlp", "model", "deployed") if k in res]
    h_rows = []
    for name in sys_order_h:
        a = res[name]["health_params"]
        b = boot["health"][name]
        h_rows.append([
            labels_tr[name],
            f"{a['mcc']:.4f} [{b['mcc']['lo']:.4f}-{b['mcc']['hi']:.4f}]",
            f"{a['macro_f1']:.4f} [{b['macro_f1']['lo']:.4f}-{b['macro_f1']['hi']:.4f}]",
            f"{a['qwk']:.4f}",
            f"%{a['critical_recall']*100:.2f}",
            f"%{a['under_triage_rate']*100:.2f}",
            f"{res[name]['mae_health_pct']:.3f}",
        ])
    style_table(
        doc,
        ["Sistem (31 Fizyolojik Biyo-Belirtec)", "MCC [%95 GA]", "Macro-F1 [%95 GA]", "QWK", "Kritik Recall", "Under-Triage", "MAE"],
        h_rows,
        col_widths=[4.7, 2.7, 2.7, 1.7, 1.9, 1.9, 1.8],
    )

    sig_pairs = [
        ("model_vs_rule", "AstrocyteX-Net v3.0 - Kural Bazli (Tum 43)"),
        ("model_vs_ridge", "AstrocyteX-Net v3.0 - Ridge ML (Tum 43)"),
        ("model_vs_mlp", "AstrocyteX-Net v3.0 - MLP Sinir Agi Bazi (Tum 43)"),
        ("model_vs_no_env_health", "AstrocyteX-Net v3.0 - Ortam Girdisiz v3.0 (31 Saglik)"),
        ("deployed_vs_rule", "Canli Sistem - Kural Bazli (Tum 43)"),
    ]
    sig_rows = []
    for key, lbl in sig_pairs:
        if key not in boot:
            continue
        b = boot[key]
        sig_rows.append([
            lbl,
            f"{b['mcc']['diff_lo']:+.4f} .. {b['mcc']['diff_hi']:+.4f}",
            f"{b['macro_f1']['diff_lo']:+.4f} .. {b['macro_f1']['diff_hi']:+.4f}",
            f"{b['critical_recall']['diff_lo']:+.4f} .. {b['critical_recall']['diff_hi']:+.4f}",
            f"{b['mcc']['p_diff_le_0']:.3f}",
        ])
    style_table(
        doc,
        ["Eslestirilmis Bootstrap Karsilastirmasi", "Delta MCC %95 GA", "Delta Macro-F1 %95 GA", "Delta Kritik Recall %95 GA", "P(Delta MCC <= 0)"],
        sig_rows,
        col_widths=[5.4, 3.0, 3.0, 3.3, 2.7],
    )

    if cm_png.is_file():
        p_cm = doc.add_paragraph()
        p_cm.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_cm.add_run().add_picture(str(cm_png), width=Inches(6.4))
        add_body(doc, "Sekil 2: Kural bazli referans sistemi (sol) ve AstrocyteX-Net v3.0 (sag) normalize 5x5 karisiklik matrisleri.")

    if ev_png.is_file():
        p_ev = doc.add_paragraph()
        p_ev.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_ev.add_run().add_picture(str(ev_png), width=Inches(6.4))
        add_body(doc, "Sekil 3: Yedi olagan disi uzay olayi boyunca Kural Bazi, Ortam Girdisiz v3.0 ve Tam AstrocyteX-Net v3.0 fizyolojik MCC karsilastirmasi.")

    # ==================== BOLUM 5.4 SES MODELI ====================
    add_h2(doc, "5.4. AstroVoiceNet v1.0 — 1.200 Gorulmemis Astronot Ses Kaydi Performans Sonuclari")
    v_labels = {
        "naive_baseline": "Naif cogunluk / Ortalama Bazi",
        "rule_acoustic": "Kural Bazli Akustik Esikleme",
        "logistic_ridge": "Lojistik Regresyon + Ridge [26]",
        "hist_gradient_boosting": "HistGradientBoosting (scikit-learn) [26]",
        "random_forest": "RandomForest Toplulugu (220 Agac) [26]",
        "ablation_prosody_only": "Ablasyon: Sadece 42 Prozodik Ozellik (Spektrogramsiz)",
        "astrovoice_v1": "AstroVoiceNet v1.0 (Log-Mel Conv-SE-Attn + ProsodyGate)",
    }
    v_rows = []
    for vk in ("naive_baseline", "rule_acoustic", "logistic_ridge", "hist_gradient_boosting", "random_forest", "ablation_prosody_only", "astrovoice_v1"):
        s = v_sys[vk]
        v_rows.append([
            v_labels[vk],
            f"{s['state_mcc']:.4f} [{s['state_mcc_ci'][0]:.4f}-{s['state_mcc_ci'][1]:.4f}]",
            f"{s['state_macro_f1']:.4f}",
            f"{s['risk_mae_pct']:.3f} [{s['risk_mae_ci'][0]:.3f}-{s['risk_mae_ci'][1]:.3f}]",
            f"{s['risk_rmse_pct']:.3f}",
            f"{s['risk_band_qwk']:.4f}",
            f"%{s['critical_recall']*100:.2f}",
        ])
    style_table(
        doc,
        ["Ses Analiz Sistemi (n=1.200 Test Kaydi)", "Durum MCC [%95 GA]", "Durum Macro-F1", "Risk MAE [%95 GA]", "Risk RMSE", "Risk QWK", "Kritik Recall"],
        v_rows,
        col_widths=[4.8, 2.8, 2.0, 2.6, 1.7, 1.7, 1.8],
    )

    v_slice_rows = [
        ["Kadin Astronot Konusmacilar", str(v_slices["female_speakers"]["n"]), f"{v_slices['female_speakers']['mcc']:.4f}", f"{v_slices['female_speakers']['macro_f1']:.4f}", f"{v_slices['female_speakers']['risk_mae_pct']:.3f} puan"],
        ["Erkek Astronot Konusmacilar", str(v_slices["male_speakers"]["n"]), f"{v_slices['male_speakers']['mcc']:.4f}", f"{v_slices['male_speakers']['macro_f1']:.4f}", f"{v_slices['male_speakers']['risk_mae_pct']:.3f} puan"],
        ["Yuksek Kabin Gurultusu (SNR <= 22 dB)", str(v_slices["harsh_cabin_noise_snr_le_22db"]["n"]), f"{v_slices['harsh_cabin_noise_snr_le_22db']['mcc']:.4f}", f"{v_slices['harsh_cabin_noise_snr_le_22db']['macro_f1']:.4f}", f"{v_slices['harsh_cabin_noise_snr_le_22db']['risk_mae_pct']:.3f} puan"],
        ["Nominal Kabin Akustigi (SNR > 22 dB)", str(v_slices["nominal_cabin_snr_gt_22db"]["n"]), f"{v_slices['nominal_cabin_snr_gt_22db']['mcc']:.4f}", f"{v_slices['nominal_cabin_snr_gt_22db']['macro_f1']:.4f}", f"{v_slices['nominal_cabin_snr_gt_22db']['risk_mae_pct']:.3f} puan"],
    ]
    style_table(doc, ["Alt Grup Adillik ve Gurultu Dilimi", "Test Kaydi (n)", "Durum MCC", "Durum Macro-F1", "Ses Risk MAE"], v_slice_rows, col_widths=[5.8, 2.6, 3.0, 3.0, 3.0])

    if voice_png.is_file():
        p_vp = doc.add_paragraph()
        p_vp.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_vp.add_run().add_picture(str(voice_png), width=Inches(6.4))
        add_body(doc, "Sekil 4: AstroVoiceNet v1.0 5x5 psikolojik durum karisiklik matrisi ve model karsilastirma sonuclari.")

    # ==================== BOLUM 6 KUTUPHANELER VE PROGRAMLAR ====================
    add_h1(doc, "6. Kullanilan Kutuphaneler, Catilar (Frameworks) ve Yazilim Programlari")
    add_body(
        doc,
        "AstrocyteX v3.0 platformunun gelistirilmesi, derin sinir aglarinin GPU uzerinde egitilmesi, ses/EKG sinyal "
        "islemesi, web sunumu ve istatistiksel dogrulamasi icin asagidaki acik kaynakli kutuphaneler ve yazilim "
        "programlari kullanilmistir:"
    )
    style_table(
        doc,
        ["Yazilim / Kutuphane", "Surum", "Lisans", "AstrocyteX Mimarisindeki Gorevi", "Atif"],
        sw_rows,
        col_widths=[3.8, 2.5, 2.1, 7.6, 1.4],
    )

    # ==================== BOLUM 7 KAYNAKCA ====================
    add_h1(doc, "7. Akademik ve Yazilim Kaynakcasi (Citations)")
    add_body(
        doc,
        "Bu raporda ve AstrocyteX v3.0 kod tabaninda kullanilan algoritmalarin, uzay tibbi standartlarinin ve bilimsel "
        "yazilim kutuphanelerinin akademik atif listesi:"
    )
    for num, ref in CITATIONS:
        add_body(doc, ref, bold_prefix=f"{num} ")

    # ==================== BOLUM 8 SINIRLILIKLAR ====================
    add_h1(doc, "8. Sinirliliklar, Etik Ilkeler ve Klinik Guvenlik Notlari")
    add_bullet(doc, "Tibbi Cihaz Degildir: ", "AstrocyteX arastirma ve muhendislik gosterim amacli bir prototiptir; klinik tani koymaz ve ilac dozaji onermez.")
    add_bullet(doc, "Simulasyon Verisi Siniri: ", "43 uzay tibbi parametresini ve derin uzay ses senaryolarini eszamanli iceren acik bir insan ucus veri kumesi bulunmadigindan aglar zenginlestirilmis fizyolojik/akustik simulasyon kohortlari uzerinde egitilmistir; gercek ucus kullanimi icin klinik kalibrasyon sarttir.")
    add_bullet(doc, "Sifir-Emoji ve Cevrimdisi Dayaniklilik: ", "Tum kullanici arayuzu ve karar motoru %100 sifir-emoji kuralina uyar; internet veya bulut LLM erisimi kesildiginde devre kesici saniyeler icinde yerel karar motoruna gecerek gorev surekliligini korur.")

    docx_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(docx_path))

    # ==================== MARKDOWN OZET RAPORU ====================
    md_lines = [
        "# AstrocyteX v3.0 — Yapay Zeka Model Ozeti, Kutuphaneler ve Performans Raporu",
        "",
        "## 1. Yonetici Ozeti",
        "",
        "| Teknik Boyut | Ozet Deger ve Mimari Nitelik |",
        "|---|---|",
    ]
    for k, v in exec_rows:
        md_lines.append(f"| **{k}** | {v} |")

    md_lines += [
        "",
        "## 2. Gelistirilen Yapay Zeka Modelleri ve Yenilikler",
        "",
        f"- **AstrocyteX-Net v3.0 (`models/astroguard.py`, {meta['params']:,} parametre):** `d_model=80`, `16` Perceiver latent darbogazi ve `SwiGLU GatedInteractionBlock` [7] katmanlariyla guclendirilmis 5-modaliteli derin ag (Mamba-2 SSD [1], TCN+TFT [2, 3], EnvironmentEncoder+FiLM [6], Cross-Attention [4, 5], Neural ODE + PINN [8, 9], Bayes-by-Backprop + MC-Dropout [10-12]). {meta['samples']:,} egitim + {meta['val_samples']:,} dogrulama murettebat-gunu uzerinde {meta['epochs_run']} epoch GPU ile yeniden egitilmistir.",
        f"- **AstroVoiceNet v1.0 (`models/voice_model.py`, {v_meta['params']:,} parametre):** Astronot ses kayitlarindan `(48, 64)` log-Mel spektrogram + `42` GeMAPS esinli akustik/prozodik biyo-belirtec [13] cikaran, 2B Derinlik-Ayrilabilir Conv + Squeeze-and-Excitation [14] + Oz-Dikkat + `ProsodyGateEncoder` hibrit sinir agi. 5 psikolojik durumu, 4 mood boyutunu [16] ve `%0-%100` ses risk skorunu uretir.",
        "- **Rol-Duyarli LLM Danisman (`backend/gemini_agent.py`):** Kullanici **'Ben doktorum'** sectiginde uzay tibbi patofizyolojisi ve nicel F0/jitter/shimmer/HNR/QTc bulgulariyla detayli klinik dil; **'Doktor degilim'** sectiginde jargondan arindirilmis sade ve anlasilir gunluk dil kullanir [28].",
        "- **Sifir-Emoji Karanlik Mor Samanyolu Arayuzu (`web/`, `ui/`):** Tum kod tabanindan emojiler tamamen kaldirilmis, Samanyolu galaksisinin Dunya'dan gorunumunu iceren karanlik mor uzay telemetri tasarimi uygulanmistir.",
        "",
        f"## 3. AstrocyteX-Net v3.0 Bagimsiz Test Sonuclari ({test_info['crew_days']:,} Murettebat-Gunu, {test_info['cells']:,} Tahmin)",
        "",
        "### 3.1. Tum 43 Parametre (31 Fizyolojik + 12 NASA RIDGE Uzay Tehlikesi)",
        "",
        "| Sistem | MCC [%95 GA] | Macro-F1 [%95 GA] | QWK | Kritik Recall | Under-Triage | Gorev F1 | MAE (puan) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in all_rows:
        md_lines.append("| " + " | ".join(r) + " |")

    md_lines += [
        "",
        "### 3.2. Yalnizca 31 Fizyolojik Saglik Parametresi (Adil Uzay Ortami Ablasyonu)",
        "",
        "| Sistem | MCC [%95 GA] | Macro-F1 [%95 GA] | QWK | Kritik Recall | Under-Triage | MAE (puan) |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in h_rows:
        md_lines.append("| " + " | ".join(r) + " |")

    md_lines += [
        "",
        "## 4. AstroVoiceNet v1.0 Bagimsiz Ses Analizi Test Sonuclari (1.200 Test Ses Kaydi)",
        "",
        "| Sistem | Durum MCC [%95 GA] | Durum Macro-F1 | Risk MAE [%95 GA] | Risk RMSE | Risk QWK | Kritik Recall |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in v_rows:
        md_lines.append("| " + " | ".join(r) + " |")

    md_lines += [
        "",
        "### 4.1. Cinsiyet ve Kabin Gurultusu Dilim Analizi",
        "",
        "| Alt Grup Dilimi | n | Durum MCC | Durum Macro-F1 | Ses Risk MAE |",
        "|---|---|---|---|---|",
    ]
    for r in v_slice_rows:
        md_lines.append("| " + " | ".join(r) + " |")

    md_lines += [
        "",
        "## 5. Kullanilan Kutuphaneler, Catilar ve Yazilim Programlari",
        "",
        "| Yazilim / Kutuphane | Surum | Lisans | AstrocyteX Mimarisindeki Gorevi | Atif |",
        "|---|---|---|---|---|",
    ]
    for r in sw_rows:
        md_lines.append("| " + " | ".join(r) + " |")

    md_lines += [
        "",
        "## 6. Akademik ve Yazilim Kaynakcasi (Citations)",
        "",
    ]
    for num, ref in CITATIONS:
        md_lines.append(f"- **{num}** {ref}")

    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    return docx_path, md_path


if __name__ == "__main__":
    desktop = Path(os.environ.get("USERPROFILE", str(Path.home()))) / "Desktop"
    docx_target = desktop / "AstrocyteX_Yapay_Zeka_Model_Ozeti_ve_Performans_Raporu.docx"
    md_target = ROOT / "docs" / "ASTROCYTEX_AI_OZET_RAPORU.md"
    saved_docx, saved_md = build_reports(docx_target, md_target)
    print(f"SAVED_DOCX={saved_docx} ({saved_docx.stat().st_size:,} bytes)")
    print(f"SAVED_MD={saved_md} ({saved_md.stat().st_size:,} bytes)")

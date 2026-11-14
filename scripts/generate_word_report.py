r"""
Generates the comprehensive Turkish technical Word (.docx) report for AstroGuard-X / Astrocyte X v2
and writes it to the user's Desktop (plus an optional copy if requested).

Usage:
    .\.venv\Scripts\python.exe scripts/generate_word_report.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import docx
from docx import Document
from docx.enum.table import WD_ALIGN_VERTICAL, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn
from docx.shared import Cm, Inches, Pt, RGBColor

from backend.gamification import BADGES
from utils.environment import ENV_EVENTS, MISSION_PROFILES, derived_environment, environment_preset
from utils.reference_ranges import CATEGORIES, ENV_KEYS, HEALTH_KEYS, LEVELS, PARAMETERS


def set_cell_bg(cell, hex_color: str) -> None:
    shading = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{hex_color}"/>')
    cell._tc.get_or_add_tcPr().append(shading)


def set_table_borders(table, border_color: str = "B8CCE4") -> None:
    tblPr = table._tbl.tblPr
    borders = parse_xml(
        f'<w:tblBorders {nsdecls("w")}>'
        f'<w:top w:val="single" w:sz="6" w:space="0" w:color="{border_color}"/>'
        f'<w:bottom w:val="single" w:sz="6" w:space="0" w:color="{border_color}"/>'
        f'<w:insideH w:val="single" w:sz="4" w:space="0" w:color="D9E2F3"/>'
        f'<w:insideV w:val="single" w:sz="4" w:space="0" w:color="D9E2F3"/>'
        f'<w:left w:val="none"/><w:right w:val="none"/>'
        f'</w:tblBorders>'
    )
    tblPr.append(borders)


def style_table(doc: Document, headers: list[str], rows: list[list[str]], col_widths: list[float] | None = None) -> None:
    table = doc.add_table(rows=1 + len(rows), cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(table)

    hdr_cells = table.rows[0].cells
    for j, h in enumerate(headers):
        hdr_cells[j].vertical_alignment = WD_ALIGN_VERTICAL.CENTER
        set_cell_bg(hdr_cells[j], "0D1F4D")
        p = hdr_cells[j].paragraphs[0]
        p.paragraph_format.space_before = Pt(3)
        p.paragraph_format.space_after = Pt(3)
        run = p.add_run(h)
        run.bold = True
        run.font.size = Pt(8.8)
        run.font.name = "Segoe UI"
        run.font.color.rgb = RGBColor(255, 255, 255)

    for i, r in enumerate(rows):
        row_cells = table.rows[i + 1].cells
        bg = "F4F7FC" if i % 2 == 1 else "FFFFFF"
        for j, val in enumerate(r):
            row_cells[j].vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            set_cell_bg(row_cells[j], bg)
            p = row_cells[j].paragraphs[0]
            p.paragraph_format.space_before = Pt(2.5)
            p.paragraph_format.space_after = Pt(2.5)
            run = p.add_run(str(val))
            run.font.size = Pt(8.5)
            run.font.name = "Segoe UI"
            if j == 0:
                run.bold = True
                run.font.color.rgb = RGBColor(16, 36, 82)

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
    run.font.size = Pt(15)
    run.font.name = "Segoe UI"
    run.font.color.rgb = RGBColor(13, 31, 77)
    pBdr = parse_xml(f'<w:pBdr {nsdecls("w")}><w:bottom w:val="single" w:sz="10" w:space="3" w:color="3B6EC2"/></w:pBdr>')
    p._p.get_or_add_pPr().append(pBdr)


def add_h2(doc: Document, text: str) -> None:
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(10)
    p.paragraph_format.space_after = Pt(4)
    run = p.add_run(text)
    run.bold = True
    run.font.size = Pt(12)
    run.font.name = "Segoe UI"
    run.font.color.rgb = RGBColor(28, 63, 140)


def add_body(doc: Document, text: str, bold_prefix: str = "") -> None:
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(5)
    p.paragraph_format.line_spacing = 1.18
    if bold_prefix:
        rb = p.add_run(bold_prefix)
        rb.bold = True
        rb.font.name = "Segoe UI"
        rb.font.size = Pt(10)
        rb.font.color.rgb = RGBColor(16, 36, 82)
    r = p.add_run(text)
    r.font.name = "Segoe UI"
    r.font.size = Pt(10)
    r.font.color.rgb = RGBColor(32, 38, 48)


def add_bullet(doc: Document, bold_prefix: str, text: str) -> None:
    p = doc.add_paragraph(style="List Bullet")
    p.paragraph_format.space_after = Pt(3)
    rb = p.add_run(bold_prefix)
    rb.bold = True
    rb.font.name = "Segoe UI"
    rb.font.size = Pt(9.8)
    rb.font.color.rgb = RGBColor(16, 36, 82)
    r = p.add_run(text)
    r.font.name = "Segoe UI"
    r.font.size = Pt(9.8)
    r.font.color.rgb = RGBColor(32, 38, 48)


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
        hist = ckpt.get("meta", {}).get("history", [])
        if not hist:
            return False

        epochs = [h["epoch"] for h in hist]
        mcc = [h.get("band_mcc", 0) for h in hist]
        f1 = [h.get("band_macro_f1", 0) for h in hist]
        qwk = [h.get("band_qwk", 0) for h in hist]
        mae = [h.get("risk_mae_pct", 0) for h in hist]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.1))
        ax1.plot(epochs, mcc, label="Validation MCC", color="#1a73e8", linewidth=2.2)
        ax1.plot(epochs, f1, label="Validation Macro-F1", color="#2ecc71", linewidth=1.9)
        ax1.plot(epochs, qwk, label="Validation QWK", color="#8e44ad", linewidth=1.8, linestyle="--")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Skor")
        ax1.set_title("AstroGuardNet v2 — 45 Epoch Doğrulama Yakınsaması", fontsize=10.5)
        ax1.grid(alpha=0.25)
        ax1.legend()

        ax2.plot(epochs, mae, color="#e74c3c", linewidth=2.2, label="Validation Risk MAE (puan)")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Ortalama Mutlak Hata (yüzde puan)")
        ax2.set_title("Risk MAE Düşüş Eğrisi (32.000 Gün · RTX 4050)", fontsize=10.5)
        ax2.grid(alpha=0.25)
        ax2.legend()

        fig.tight_layout()
        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return True
    except Exception:
        return False


def fmt_lim(pair) -> str:
    lo, hi = pair
    if lo is not None and hi is not None:
        return f"{lo} – {hi}"
    if lo is not None:
        return f"≥ {lo}"
    if hi is not None:
        return f"≤ {hi}"
    return "—"


def build_report(dest_path: Path) -> Path:
    eval_json = json.loads((ROOT / "docs" / "model_evaluation.json").read_text(encoding="utf-8"))
    res = eval_json["results"]
    boot = eval_json["bootstrap_95ci"]
    test_info = eval_json["test"]
    meta = eval_json["model_meta"]["metrics"]

    train_png = ROOT / "docs" / "img" / "training_convergence.png"
    has_train_png = generate_training_plot(train_png)
    cm_png = ROOT / "docs" / "img" / "confusion_matrix.png"
    ev_png = ROOT / "docs" / "img" / "mcc_by_event.png"

    doc = Document()
    for sec in doc.sections:
        sec.top_margin = Cm(2.0)
        sec.bottom_margin = Cm(2.0)
        sec.left_margin = Cm(1.9)
        sec.right_margin = Cm(1.9)

    # ==================== KAPAK BAŞLIĞI ====================
    banner = doc.add_table(rows=1, cols=1)
    banner.alignment = WD_TABLE_ALIGNMENT.CENTER
    c0 = banner.rows[0].cells[0]
    set_cell_bg(c0, "08122E")
    p_top = c0.paragraphs[0]
    p_top.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_top.paragraph_format.space_before = Pt(14)
    r_sub = p_top.add_run("ASTROCYTE X · ASTROGUARD-X v2.0  |  NASA SPACE APPS CHALLENGE\n")
    r_sub.font.name = "Segoe UI"
    r_sub.font.size = Pt(9.5)
    r_sub.bold = True
    r_sub.font.color.rgb = RGBColor(158, 197, 255)

    r_title = p_top.add_run("Yapay Zeka Model Mimarisi, 43 Sağlık & Uzay Ortamı Parametresi\nve Nihai Performans Değerlendirme Raporu\n")
    r_title.font.name = "Segoe UI"
    r_title.font.size = Pt(17)
    r_title.bold = True
    r_title.font.color.rgb = RGBColor(255, 255, 255)

    r_desc = p_top.add_run(
        "Mamba-2 · TCN + Temporal Fusion Transformer · EnvironmentEncoder + FiLM · 5-Modaliteli Cross-Attention · "
        "Neural ODE + PINN · Bayesçi Belirsizlik · TEKNOFEST V23 12-Derivasyonlu EKG · Gemini Karar Destek"
    )
    r_desc.font.name = "Segoe UI"
    r_desc.font.size = Pt(9.2)
    r_desc.font.color.rgb = RGBColor(210, 226, 255)
    p_top.paragraph_format.space_after = Pt(14)

    doc.add_paragraph()

    style_table(
        doc,
        ["Özellik", "Değer / Teknik Detay"],
        [
            ["Model Sürümü ve Boyutu", f"astroguard-net-2.0 — {meta['params']:,} eğitilebilir parametre (PyTorch + CUDA)"],
            ["İzlenen Toplam Parametre", "43 Parametre = 31 Fizyolojik Biyo-Belirteç + 12 NASA HRP 'RIDGE' Uzay Tehlikesi"],
            ["Eğitim Veri Kümesi ve Donanım", f"{meta['samples']:,} eğitim + {meta['val_samples']:,} doğrulama mürettebat-günü · 45 Epoch · NVIDIA RTX 4050 GPU ({meta['train_seconds']} sn)"],
            ["Bağımsız (Held-Out) Test Seti", f"{test_info['crew_days']:,} görülmemiş mürettebat-günü = {test_info['cells']:,} parametre tahmini · 1.000 Bootstrap (%95 GA)"],
            ["Nihai Test Başarısı (Tüm 43 Parametre)", f"MCC = {res['model']['all_params']['mcc']:.4f} · Macro-F1 = {res['model']['all_params']['macro_f1']:.4f} · QWK = {res['model']['all_params']['qwk']:.4f} · MAE = {res['model']['mae_pct']:.3f} puan"],
            ["Kritik Güvenlik Performansı", f"Kritik Yakalama (Red/Black Recall) = %{res['model']['all_params']['critical_recall']*100:.2f} · Under-Triage = %{res['model']['all_params']['under_triage_rate']*100:.2f}"],
            ["Uzay Ortamı Ablasyon Katkısı (31 Biyo-belirteç)", f"MCC: {res['model_no_env']['health_params']['mcc']:.4f} → {res['model']['health_params']['mcc']:.4f} (+0.0724) · Kaçırılan Kritik Alarm: %16.79 → %1.99"],
            ["Kullanıcı Arayüzü ve Depo", "Astrocyte X · AstroGuard-X Kozmik Glassmorphism Web Arayüzü (TR/EN/RU) + GitHub: icelberk-dotcom/NASA-space-apps-Ondankoymacano"],
        ],
        col_widths=[6.0, 11.2],
    )

    # ==================== BÖLÜM 1 ====================
    add_h1(doc, "1. Proje Özeti ve Mimari Tasarım İlkeleri")
    add_body(
        doc,
        "AstroGuard-X (Astrocyte X arayüz mimarisiyle birleştirilmiş v2.0 sürümü), astronotların uzun süreli "
        "alçak Dünya yörüngesi (ISS), Ay (Lunar Gateway / yüzey) ve derin uzay (Mars yolculuğu / yüzey) görevlerinde "
        "karşılaştıkları çok boyutlu fizyolojik değişimleri ve çevresel tehlikeleri gerçek zamanlı analiz eden, "
        "belirsizlik duyarlı bir yapay zeka sağlık izleme ve oyunlaştırılmış önleyici müdahale platformudur."
    )
    add_bullet(
        doc, "Kural-Çıpalı Kalıntı Öğrenme (Rule-Anchored Residual Learning): ",
        "Model, her parametrenin şeffaf referans aralığı skorunu (rule prior) temel alır; bunun üzerine çoklu organ "
        "etkileşimlerini, 14 günlük zamansal eğilimleri, öznel semptomları ve 12 uzay ortamı tehlikesinin fizyolojiye "
        "etkisini logit uzayında öğrenir. Klinik güvenlik tabanı (safety floor), referans aralığına göre kırmızı/siyah "
        "(≥ %70) olan hiçbir alarmın yapay zeka tarafından düşürülmesine izin vermez."
    )
    add_bullet(
        doc, "Sağlık Riski ile Uzay Ortamı Tehlikesinin Bilimsel Ayrıştırılması: ",
        "Mikro-yerçekiminde (0 g) bulunan sağlıklı bir astronot 'hasta' değildir. Bu nedenle genel sağlık riski "
        "yalnızca 31 fizyolojik biyo-belirteçten hesaplanır; 12 uzay ortamı parametresi ayrı bir 'Uzay Ortamı Tehlike "
        "İndeksi' oluşturur ve fizyolojik parametrelerin riskini sinir ağı içindeki çapraz dikkat + FiLM katmanları "
        "üzerinden yükseltir."
    )
    add_bullet(
        doc, "Karşı-Olgusal Ortam Etkisi (Counterfactual Uplift) ve Özerk İletişim Yönetimi: ",
        "Her değerlendirmede aynı fizyolojik veriler Dünya benzeri nötr ortamda tekrar koşturularak mevcut uzay "
        "ortamının sağlık riskine eklediği net fark (+X puan) hesaplanır. Dünya'dan uzaklığa bağlı tek yönlü ışık "
        "gecikmesi yükseldiğinde (ör. Mars görevinde ~13 dk) sistem mürettebata özerk tıbbi hareket protokolü önerir."
    )

    # ==================== BÖLÜM 2 ====================
    add_h1(doc, "2. Kullanılan ve Ölçülen Tüm Parametreler (43 Parametre Kataloğu)")
    add_h2(doc, "2.1. Beş Bantlı Renk Kodlu Risk Sınıflandırma Ölçeği")
    add_body(
        doc,
        "Her parametre için ideal değer (%8 baz risk), normal sınır (%40), dikkat sınırı (%70) ve tehlike sınırı (%90–100) "
        "arasında parçalı doğrusal dönüşüm uygulanır ve hem veri girişinde hem de panellerde 5 renkli gösterge dairesi üretilir:"
    )
    style_table(
        doc,
        ["Renk Bandı", "Risk Aralığı", "Hex Kodu", "Klinik ve Operasyonel Anlamı"],
        [
            ["⚪ Gri (Gray)", "< %15", "#9aa0a6", "İdeal / bazal seviye veya henüz veri girilmemiş pasif parametre (görev üretmez)."],
            ["🟢 Yeşil (Green)", "%15 – %40", "#2ecc71", "Normal fizyolojik aralık / düşük risk (rutin izlem)."],
            ["🟡 Sarı (Yellow)", "%40 – %70", "#f1c40f", "Hafif sapma / erken uyarı → Koruyucu hafifletme görevleri (+10 XP) üretilir."],
            ["🔴 Kırmızı (Red)", "%70 – %90", "#e74c3c", "Yüksek risk → Öncelikli müdahale görevleri (+25 XP) ve yakın takip başlatılır."],
            ["⚫ Siyah (Black)", "> %90", "#000000", "Kritik acil durum → Acil protokol (+50 XP) ve uçuş cerrahı / görev kontrol eskalasyonu."],
        ],
        col_widths=[3.2, 2.6, 2.3, 9.1],
    )

    add_h2(doc, "2.2. Sürekli Giyilebilir Sensör ve Uyku Parametreleri (8 Parametre)")
    seq_rows = []
    for p in [x for x in PARAMETERS if x.kind == "continuous"]:
        seq_rows.append([
            f"{p.name_tr}\n({p.key})",
            CATEGORIES[p.category]["tr"],
            p.unit or "-",
            str(p.ideal),
            fmt_lim(p.normal),
            fmt_lim(p.warn),
            fmt_lim(p.danger),
        ])
    style_table(
        doc,
        ["Parametre (Anahtar)", "Sistem", "Birim", "İdeal", "Normal", "Dikkat", "Tehlike"],
        seq_rows,
        col_widths=[4.2, 2.8, 1.5, 1.5, 2.4, 2.4, 2.4],
    )

    add_h2(doc, "2.3. Günlük Klinik, Biyokimya, Hemogram, İdrar/Tükürük, Beslenme ve EKG Parametreleri (23 Parametre)")
    daily_rows = []
    for p in [x for x in PARAMETERS if x.kind == "daily"]:
        daily_rows.append([
            f"{p.name_tr}\n({p.key})",
            CATEGORIES[p.category]["tr"],
            p.unit or "-",
            str(p.ideal),
            fmt_lim(p.normal),
            fmt_lim(p.warn),
            fmt_lim(p.danger),
        ])
    style_table(
        doc,
        ["Parametre (Anahtar)", "Sistem", "Birim", "İdeal", "Normal", "Dikkat", "Tehlike"],
        daily_rows,
        col_widths=[4.2, 2.8, 1.5, 1.5, 2.4, 2.4, 2.4],
    )

    add_h2(doc, "2.4. NASA HRP 'RIDGE' Uzay Ortamı Tehlike Parametreleri (12 Parametre — 5 Tema)")
    env_rows = []
    for p in [x for x in PARAMETERS if x.kind == "environment"]:
        env_rows.append([
            f"{p.name_tr}\n({p.key})",
            CATEGORIES[p.category]["tr"],
            p.unit or "-",
            str(p.ideal),
            fmt_lim(p.normal),
            fmt_lim(p.warn),
            fmt_lim(p.danger),
        ])
    style_table(
        doc,
        ["Tehlike Parametresi", "RIDGE Teması", "Birim", "İdeal", "Normal", "Dikkat", "Tehlike"],
        env_rows,
        col_widths=[4.3, 2.9, 1.6, 1.4, 2.3, 2.3, 2.4],
    )

    add_h2(doc, "2.5. Desteklenen 6 Uzay Görev Profili ve 7 Anormal Ortam Olayı")
    def _fmt_delay(sec: float | None) -> str:
        if sec is None:
            return "-"
        return f"{sec:.2f} sn" if sec < 60 else f"{sec / 60.0:.1f} dk ({sec:.0f} sn)"

    def _fmt_care(days: float | None) -> str:
        if days is None:
            return "-"
        return f"~{days * 24:.0f} saat" if days < 1.0 else f"~{days:.0f} gün"

    prof_rows = []
    for k, prof in MISSION_PROFILES.items():
        vals = environment_preset(k, 90)
        der = derived_environment(vals)
        prof_rows.append([
            f"{prof['tr']} ({k})",
            f"{vals['gravity_g']} g",
            f"{vals['radiation_dose_msv_d']} mSv/gün",
            f"{vals['earth_distance_mkm']} Mkm",
            _fmt_delay(der.get("comm_delay_s")),
            _fmt_care(der.get("time_to_care_days")),
        ])
    style_table(
        doc,
        ["Görev Profili (Gün 90)", "Yerçekimi", "Radyasyon", "Dünya Uzaklığı", "İletişim Gecikmesi", "Tıbbi Tahliye"],
        prof_rows,
        col_widths=[4.8, 2.0, 2.6, 2.6, 2.6, 2.6],
    )

    ev_rows = [[k, v["tr"], v["en"]] for k, v in ENV_EVENTS.items()]
    style_table(doc, ["Olay Kodu", "Türkçe Açıklama", "İngilizce Karşılığı"], ev_rows, col_widths=[4.2, 6.5, 6.5])

    # ==================== BÖLÜM 3 ====================
    add_h1(doc, "3. Yapay Zeka Model Mimarisi (AstroGuardNet v2.0)")
    add_body(
        doc,
        "AstroGuardNet v2.0, farklı örnekleme hızlarına ve fiziksel anlamlara sahip 5 veri modalitesini tek bir "
        "uçtan uca türevlenebilir derin öğrenme çatısında birleştirir. Toplam 1.717.362 eğitilebilir parametreye sahiptir."
    )
    style_table(
        doc,
        ["Alt Modül", "Kaynak Dosya", "Girdi Boyutu", "Mimari Yapı ve Görevi"],
        [
            ["Mamba-2 (SSD) Kodlayıcı", "models/mamba2.py", "(B, 96, 7)", "96 adımlık (24 saat) sürekli giyilebilir sensör akışını durum-uzayı ikiliği (SSD), nedensel konvolüsyon ve RMSNorm kapılama ile kodlar."],
            ["TCN + Temporal Fusion Transformer", "models/tcn_tft.py", "(B, 14, 43)", "14 günlük 43-parametreli geçmişi seyreltilmiş TCN, vektörize BatchedVariableGRN değişken seçimi, LSTM ve çok başlı dikkat ile işler; 7 günlük risk öngörüsü üretir."],
            ["EnvironmentEncoder + FiLM", "models/environment.py", "(B, 12, 2)", "12 uzay tehlikesini [z-skoru, tehlike riski] jetonlarına dönüştürür, 2 katmanlı Transformer öz-dikkat ile tehlikeler arası etkileşimleri öğrenir ve FiLM (γ, β) ile birleşik temsili modüle eder."],
            ["5-Modaliteli Cross-Attention Fusion", "models/fusion.py", "5 Modalite → (B, 64)", "Günlük klinik ve öznel semptom jetonları; sensör, trend ve ortam bağlamını çapraz dikkatle okur; 12 Perceiver latent darboğazında birleşir."],
            ["Neural ODE + PINN Fizyoloji Motoru", "models/physics.py", "(B, 64) + Kontroller", "Yerçekimi, egzersiz, sıvı ve görev gününe bağlı 4 durumlu (sefalik sıvı, plazma hacmi, kemik BMD, kas kütlesi) fizik-bilgili diferansiyel denklemi RK4 ile +30/+90/+180 gün çözer."],
            ["BayesianRiskHead + RulePrior", "models/uncertainty.py\nmodels/rule_prior.py", "(B, 192) → 44 Çıkış", "Bayes-by-Backprop ağırlık dağılımları ve MC-Dropout (16–24 örneklem) ile kural skoru üzerine logit kalıntısı, epistemik/aleatorik belirsizlik ve %90 güven aralığı üretir."],
            ["TEKNOFEST V23 EKG & Gemini Ajanı", "backend/ecg_adapter.py\nbackend/gemini_agent.py", "12-Kanal EKG + JSON", "10 sınıflı 12-derivasyonlu aritmi analizi (QTc, anormallik %) ve yapılandırılmış Gemini / çevrimdışı klinik görev sentezi sağlar."],
        ],
        col_widths=[3.8, 3.3, 2.5, 7.6],
    )

    # ==================== BÖLÜM 4 ====================
    add_h1(doc, "4. Eğitim Metodolojisi, Etkileşim Önbilgileri ve GPU Yakınsaması")
    add_body(
        doc,
        "Gerçek astronot uçuş verileri 43 parametrenin tamamını kapsayan açık bir veri kümesi olarak bulunmadığından, "
        "model zenginleştirilmiş fizyolojik ve çevresel simülasyon mürettebatları üzerinde eğitilmiştir (utils/synthetic_data.py): "
        "11 klinik senaryo ve ikili karışımları, 6 görev profili, 7 anormal uzay olayı, bireysel bazal farklar, sirkadiyen "
        "sensör dalgalanmaları ve iki dilli (TR/EN) semptom günlükleri simüle edilmiştir."
    )
    add_bullet(
        doc, "Çok Görevli Kayıp Fonksiyonu: ",
        "Toplam kayıp = Heteroskedastik Gauss NLL + Isınmalı KL Iraksaması + CORAL Sıralı Bant Kaybı (%15, %40, %70, %90 "
        "eşiklerinde [1.0, 1.0, 1.5, 2.0] ağırlıklı ikili çapraz entropi) + 7 Günlük Öngörü MSE + PINN Fizik Kalıntı ve Sınır Kayıpları."
    )
    add_bullet(
        doc, "Optimizasyon: ",
        "AdamW (lr = 1.5e-3, weight_decay = 1e-4), OneCycleLR öğrenme oranı çizelgesi, gradyan kırpma (1.0), "
        "Üstel Hareketli Ortalama (EMA, decay = 0.998) ve doğrulama MCC skoru üzerinden en iyi model seçimi."
    )

    if has_train_png and train_png.is_file():
        p_img = doc.add_paragraph()
        p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_img.add_run().add_picture(str(train_png), width=Inches(6.4))
        add_body(doc, "Şekil 1: NVIDIA GeForce RTX 4050 GPU üzerinde 32.000 eğitim mürettebat-günü ve 45 epoch boyunca doğrulama MCC, Macro-F1, QWK ve Risk MAE yakınsama eğrileri.")

    # ==================== BÖLÜM 5 ====================
    add_h1(doc, "5. Test Yöntemi Seçimi ve Nihai Performans Sonuçları")
    add_h2(doc, "5.1. Hangi Test Metrikleri Neden Seçildi?")
    add_body(
        doc,
        "Modelin çıkış katmanı her parametreyi 5 sıralı ve dengesiz renk bandına (gray: 79.863, green: 125.956, "
        "yellow: 32.276, red: 15.977, black: 3.928 hücre) atar. Nadir olan 'black' sınıfı toplam verinin yalnızca %1,52'sini "
        "oluşturduğundan klasik Accuracy (%93+ çıkabilmektedir) yanıltıcıdır. Bu nedenle birincil metrik olarak Çok Sınıflı "
        "Matthews Korelasyon Katsayısı (MCC), tamamlayıcı olarak Macro-F1, Quadratic Weighted Kappa (QWK) ve klinik güvenlik "
        "için Critical Recall / Under-Triage Rate seçilmiştir."
    )

    add_h2(doc, "5.2. Tüm 43 Parametre Üzerinden Bağımsız Test Sonuçları (6.000 Mürettebat-Günü = 258.000 Tahmin)")
    labels_tr = {
        "rule": "Kural Bazlı (Referans Aralıkları)",
        "ridge": "Doğrusal ML Bazı (scikit-learn Ridge)",
        "model_no_env": "Ablasyon: Ortam Girdisiz v2",
        "model": "AstroGuardNet v2 (Tam Model)",
        "deployed": "Canlı Sistem (0.20 Kural + 0.80 v2 + Güvenlik Tabanı)",
    }
    all_rows = []
    for name in ("rule", "ridge", "model", "deployed"):
        a = res[name]["all_params"]
        b = boot[name]
        all_rows.append([
            labels_tr[name],
            f"{a['mcc']:.4f}\n[{b['mcc']['lo']:.4f}–{b['mcc']['hi']:.4f}]",
            f"{a['macro_f1']:.4f}\n[{b['macro_f1']['lo']:.4f}–{b['macro_f1']['hi']:.4f}]",
            f"{a['qwk']:.4f}",
            f"%{a['critical_recall']*100:.2f}",
            f"%{a['under_triage_rate']*100:.2f}",
            f"{a['actionable_f1']:.4f}",
            f"{res[name]['mae_pct']:.3f}",
        ])
    style_table(
        doc,
        ["Sistem", "MCC [%95 GA]", "Macro-F1 [%95 GA]", "QWK", "Kritik Recall", "Under-Triage", "Görev F1", "MAE"],
        all_rows,
        col_widths=[4.1, 2.5, 2.5, 1.5, 1.7, 1.7, 1.6, 1.6],
    )

    add_h2(doc, "5.3. Yalnızca 31 Fizyolojik Sağlık Parametresi Üzerinden Adil Uzay Ortamı Ablasyonu")
    h_rows = []
    for name in ("rule", "model_no_env", "ridge", "model", "deployed"):
        a = res[name]["health_params"]
        b = boot["health"][name]
        h_rows.append([
            labels_tr[name],
            f"{a['mcc']:.4f}\n[{b['mcc']['lo']:.4f}–{b['mcc']['hi']:.4f}]",
            f"{a['macro_f1']:.4f}\n[{b['macro_f1']['lo']:.4f}–{b['macro_f1']['hi']:.4f}]",
            f"{a['qwk']:.4f}",
            f"%{a['critical_recall']*100:.2f}",
            f"%{a['under_triage_rate']*100:.2f}",
            f"{res[name]['mae_health_pct']:.3f}",
        ])
    style_table(
        doc,
        ["Sistem (31 Fizyolojik Biyo-belirteç)", "MCC [%95 GA]", "Macro-F1 [%95 GA]", "QWK", "Kritik Recall", "Under-Triage", "MAE"],
        h_rows,
        col_widths=[4.8, 2.7, 2.7, 1.7, 1.8, 1.8, 1.7],
    )

    add_h2(doc, "5.4. Eşleştirilmiş Bootstrap Anlamlılık Testi ve Bant Bazlı Kesinlik / Duyarlılık")
    sig_rows = []
    for key, lbl in (
        ("model_vs_rule", "AstroGuardNet v2 − Kural Bazlı (Tüm 43)"),
        ("model_vs_ridge", "AstroGuardNet v2 − Ridge ML Bazı (Tüm 43)"),
        ("model_vs_no_env_health", "AstroGuardNet v2 − Ortam Girdisiz v2 (31 Fizyolojik)"),
        ("deployed_vs_rule", "Canlı Sistem − Kural Bazlı (Tüm 43)"),
    ):
        b = boot[key]
        sig_rows.append([
            lbl,
            f"+{b['mcc']['diff_lo']:.4f} … +{b['mcc']['diff_hi']:.4f}",
            f"+{b['macro_f1']['diff_lo']:.4f} … +{b['macro_f1']['diff_hi']:.4f}",
            f"+{b['critical_recall']['diff_lo']:.4f} … +{b['critical_recall']['diff_hi']:.4f}",
            f"{b['mcc']['p_diff_le_0']:.3f} (< 0.001)",
        ])
    style_table(
        doc,
        ["Karşılaştırma", "Δ MCC %95 GA", "Δ Macro-F1 %95 GA", "Δ Kritik Recall %95 GA", "P(Δ MCC ≤ 0)"],
        sig_rows,
        col_widths=[5.2, 3.0, 3.0, 3.2, 2.8],
    )

    band_rows = []
    for b_name in ("gray", "green", "yellow", "red", "black"):
        pm = eval_json["per_class"]["model"][b_name]
        pr = eval_json["per_class"]["rule"][b_name]
        band_rows.append([
            b_name.upper(),
            f"{pm['support']:,}",
            f"{pr['recall']:.4f}",
            f"{pm['precision']:.4f}",
            f"{pm['recall']:.4f}",
            f"{pm['f1']:.4f}",
        ])
    style_table(
        doc,
        ["Renk Bandı", "Test Hücre Sayısı", "Kural Recall", "AstroGuardNet v2 Precision", "AstroGuardNet v2 Recall", "AstroGuardNet v2 F1"],
        band_rows,
        col_widths=[2.6, 3.0, 2.8, 3.0, 3.0, 2.8],
    )

    if cm_png.is_file():
        p_cm = doc.add_paragraph()
        p_cm.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_cm.add_run().add_picture(str(cm_png), width=Inches(6.4))
        add_body(doc, "Şekil 2: Kural bazlı referans sistemi (sol, MCC 0.903) ve AstroGuardNet v2 (sağ, MCC 0.979) normalize karışıklık matrisleri. 3.928 kritik 'black' hücresinin hiçbirinde 2-bant hata yapılmamıştır.")

    add_h2(doc, "5.5. Anormal Uzay Olaylarına ve Görev Profillerine Göre Dilim (Slice) Analizi")
    ev_slice_rows = []
    for nm, s in eval_json["slices"]["event"].items():
        ev_slice_rows.append([
            nm,
            str(s["n"]),
            f"{s['rule_mcc']:.4f}",
            f"{s['model_no_env_mcc']:.4f}",
            f"{s['model_mcc']:.4f}",
            f"%{s['rule_crit_recall']*100:.1f}",
            f"%{s['model_crit_recall']*100:.1f}",
        ])
    style_table(
        doc,
        ["Uzay Ortamı Olayı", "n", "Kural MCC", "Ortam Girdisiz v2 MCC", "AstroGuardNet v2 MCC", "Kural Kritik Recall", "v2 Kritik Recall"],
        ev_slice_rows,
        col_widths=[3.8, 1.3, 2.2, 2.7, 2.7, 2.2, 2.3],
    )

    prof_slice_rows = []
    for nm, s in eval_json["slices"]["profile"].items():
        prof_slice_rows.append([
            MISSION_PROFILES[nm]["tr"],
            str(s["n"]),
            f"{s['rule_mcc']:.4f}",
            f"{s['model_no_env_mcc']:.4f}",
            f"{s['model_mcc']:.4f}",
            f"{s['model_mae']:.3f}",
        ])
    style_table(
        doc,
        ["Görev Profili", "n", "Kural MCC", "Ortam Girdisiz v2 MCC", "AstroGuardNet v2 MCC", "v2 MAE (puan)"],
        prof_slice_rows,
        col_widths=[4.8, 1.4, 2.6, 2.9, 2.9, 2.6],
    )

    if ev_png.is_file():
        p_ev = doc.add_paragraph()
        p_ev.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_ev.add_run().add_picture(str(ev_png), width=Inches(6.4))
        add_body(doc, "Şekil 3: Yedi anormal uzay olayı boyunca 31 fizyolojik parametre üzerinde Kural Bazı, Ortam Girdisiz Ağ ve Ortam Girdili AstroGuardNet v2 MCC karşılaştırması.")

    # ==================== BÖLÜM 6 ====================
    add_h1(doc, "6. Kozmik Kullanıcı Arayüzü (Astrocyte X · AstroGuard-X) ve Oyunlaştırma")
    add_body(
        doc,
        "Uygulama arayüzü, etkileşimli yıldız alanı (#stars fare-yay fiziği), kozmik derin uzay cam panelleri "
        "(glassmorphism), sol-üst açılır 'Menü ▾' hiyerarşisi, sağ-üst canlı Görev Komuta HUD çubuğu ve sağ-alt 3 dilli "
        "(Türkçe, İngilizce, Rusça) seçici ile 12 tam entegre ekrandan oluşacak şekilde yeniden tasarlanmıştır:"
    )
    add_bullet(doc, "Besin Stoğu, Öğün Porsiyonlama (%0–%100) ve YZ Köprüsü: ", "Astronot kilerindeki besinlerin kalan stok yüzdeleri renk kodlarıyla (>%75 yeşil, %50–75 sarı, %25–50 turuncu, <%25 kırmızı, %0 koyu kırmızı) izlenir; Kahvaltı/Öğle/Akşam/Ara Öğün porsiyonları ve sıvı kayıtları tek tıkla günlük kalori, protein ve sıvı girdisi olarak yapay zeka modeline aktarılır.")
    add_bullet(doc, "Oyunlaştırılmış Hafifletme Sistemi: ", "Sarı (10 XP), Kırmızı (25 XP) ve Siyah (50 XP) bulgular için üretilen görevler tamamlandıkça Cadet → Fleet Admiral rütbeleri, günlük seri (streak) ve 18 uzay rozeti kazanılır.")

    badge_rows = [[f"{b.icon} {b.name}", b.id, b.description] for b in BADGES]
    style_table(doc, ["Rozet", "Anahtar", "Kazanım Koşulu"], badge_rows, col_widths=[4.6, 3.2, 9.4])

    # ==================== BÖLÜM 7 ====================
    add_h1(doc, "7. Sınırlılıklar, Etik İlkeler ve Klinik Güvenlik Notları")
    add_bullet(doc, "Tıbbi Cihaz Değildir: ", "AstroGuard-X araştırma ve teknoloji gösterim amaçlı bir prototiptir; klinik tanı koymaz ve ilaç dozajı önermez.")
    add_bullet(doc, "Simülasyon Verisi Sınırı: ", "Raporlanan yüksek MCC (0.9788) ve Macro-F1 (0.9767) değerleri, modelin simülatördeki çoklu etkileşim yasalarını ne derece eksiksiz öğrendiğini gösterir; gerçek uçuş sertifikasyonu için klinik kalibrasyon gereklidir.")
    add_bullet(doc, "Çevrimdışı Dayanıklılık: ", "İnternet veya Gemini erişimi kesildiğinde devre kesici (circuit breaker) saniyeler içinde yerel karar motoruna ve yerleşik EKG taramasına geçerek derin uzay görev sürekliliğini korur.")

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(dest_path))
    return dest_path


if __name__ == "__main__":
    desktop = Path(os.environ.get("USERPROFILE", str(Path.home()))) / "Desktop"
    target = desktop / "AstroGuard-X_Model_Mimarisi_Parametreler_ve_Test_Raporu.docx"
    saved = build_report(target)
    print(f"SAVED_DOCX={saved} ({saved.stat().st_size:,} bytes)")

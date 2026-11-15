r"""
Generates a comprehensive Turkish academic & clinical reference dossier explaining
the exact scientific, clinical, and aerospace-medicine sources behind all 43 monitored
parameters (normal, warning, and danger thresholds) and the vocal prosodic biomarkers
in AstrocyteX v3.0.

Outputs:
  1. C:\Users\VICTUS\Desktop\AstrocyteX_Parametre_Referans_Araliklari_ve_Kaynakca.docx
  2. docs/PARAMETRE_KAYNAKCA.md

Usage:
    .\.venv\Scripts\python.exe scripts/generate_parameter_bibliography.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from docx import Document  # noqa: E402
from docx.enum.table import WD_ALIGN_VERTICAL, WD_TABLE_ALIGNMENT  # noqa: E402
from docx.enum.text import WD_ALIGN_PARAGRAPH  # noqa: E402
from docx.oxml import parse_xml  # noqa: E402
from docx.oxml.ns import nsdecls  # noqa: E402
from docx.shared import Cm, Pt, RGBColor  # noqa: E402

from utils.reference_ranges import CATEGORIES, PARAM_KEYS, PARAMETERS, Parameter  # noqa: E402


# --------------------------------------------------------------------------- #
# Verified Academic, Clinical & Institutional Bibliography (40 Sources)
# --------------------------------------------------------------------------- #

BIBLIOGRAPHY_GROUPS: List[Tuple[str, List[Tuple[str, str]]]] = [
    (
        "A. Uzay Tıbbı, NASA İnsan-Sistem Standartları ve Uzay Ortamı (RIDGE) Kaynakları",
        [
            (
                "[1]",
                "National Aeronautics and Space Administration (NASA), NASA Space Flight Human-System Standard, "
                "Volume 1: Crew Health (NASA-STD-3001, Vol. 1, Rev. B/C), Washington, DC: NASA Technical Standards, 2022–2023.",
            ),
            (
                "[2]",
                "National Aeronautics and Space Administration (NASA), NASA Space Flight Human-System Standard, "
                "Volume 2: Human Factors, Habitability, and Environmental Health (NASA-STD-3001, Vol. 2, Rev. C/D), "
                "Washington, DC: NASA Technical Standards, 2022–2023.",
            ),
            (
                "[3]",
                "National Academies of Sciences, Engineering, and Medicine (NASEM), Space Radiation and Astronaut Health: "
                "Managing and Communicating Cancer Risks, Washington, DC: The National Academies Press, 2021. "
                "(NASA'nın 600 mSv evrensel kariyer efektif radyasyon dozu sınırının bilimsel dayanak raporu).",
            ),
            (
                "[4]",
                "Z. S. Patel et al., 'Red risks for a journey to the red planet: The highest priority human health risks "
                "for a mission to Mars,' npj Microgravity, vol. 6, art. 33, 2020.",
            ),
            (
                "[5]",
                "C. Zeitlin et al., 'Measurements of energetic particle radiation in transit to Mars on the Mars Science "
                "Laboratory,' Science, vol. 340, no. 6136, pp. 1080–1084, 2013.",
            ),
            (
                "[6]",
                "D. M. Hassler et al., 'Mars surface radiation environment measured with the MSL Curiosity rover,' "
                "Science, vol. 343, no. 6169, art. 1244797, 2014.",
            ),
            (
                "[7]",
                "J. Law et al., 'Relationship between carbon dioxide levels and reported headaches on the International "
                "Space Station,' Journal of Occupational and Environmental Medicine, vol. 56, no. 5, pp. 477–483, 2014.",
            ),
            (
                "[8]",
                "National Research Council (Committee on Spacecraft Exposure Guidelines), Spacecraft Maximum Allowable "
                "Concentrations for Selected Airborne Contaminants (SMACs), Vol. 5, Washington, DC: The National Academies Press, 2008.",
            ),
            (
                "[9]",
                "A. Whitmire et al., Minimum Acceptable Net Habitable Volume for Long-Duration Exploration Missions, "
                "NASA/TM-2015-218564, Houston, TX: NASA Johnson Space Center, 2015.",
            ),
            (
                "[10]",
                "M. Basner et al., 'Psychological and behavioral changes during confinement in a 520-day simulated "
                "interplanetary mission to Mars (Mars-500),' Proceedings of the National Academy of Sciences (PNAS), "
                "vol. 111, no. 7, pp. 2553–2558, 2014.",
            ),
            (
                "[11]",
                "F. E. Garrett-Bakelman et al., 'The NASA Twins Study: A multidimensional analysis of a year-long human "
                "spaceflight,' Science, vol. 364, no. 6436, art. eaau8650, 2019.",
            ),
            (
                "[12]",
                "G. Clément, Fundamentals of Space Medicine, 2nd ed., New York, NY: Springer, 2011; "
                "M. F. Reschke and G. Clément, 'Vestibular and sensorimotor dysfunction during space flight,' "
                "Current Pathobiology Reports, vol. 6, pp. 177–183, 2018.",
            ),
            (
                "[13]",
                "National Institute for Occupational Safety and Health (NIOSH), Criteria for a Recommended Standard: "
                "Occupational Noise Exposure, DHHS (NIOSH) Pub. No. 98-126, Cincinnati, OH, 1998; "
                "J. G. Limardo et al., 'International Space Station (ISS) Acoustic Environment — A Status Report,' "
                "47th International Conference on Environmental Systems (ICES-2017-263), 2017.",
            ),
        ],
    ),
    (
        "B. Kardiyorespiratuvar, Otonomik, Uyku, Egzersiz ve Klinik Erken Uyarı Kılavuzları",
        [
            (
                "[14]",
                "Royal College of Physicians (RCP), National Early Warning Score (NEWS) 2: Standardising the Assessment "
                "of Acute-Illness Severity in the NHS, London, UK: RCP, 2017.",
            ),
            (
                "[15]",
                "P. K. Whelton et al., '2017 ACC/AHA/AAPA/ABC/ACPM/AGS/APhA/ASH/ASPC/NMA/PCNA Guideline for the "
                "Prevention, Detection, Evaluation, and Management of High Blood Pressure in Adults,' Journal of the "
                "American College of Cardiology (JACC), vol. 71, no. 19, pp. e127–e248, 2018; "
                "B. Williams et al., '2018 ESC/ESH Guidelines for the management of arterial hypertension,' "
                "European Heart Journal, vol. 39, no. 33, pp. 3021–3104, 2018.",
            ),
            (
                "[16]",
                "F. M. Kusumoto et al., '2018 ACC/AHA/HRS Guideline on the Evaluation and Management of Patients With "
                "Bradycardia and Cardiac Conduction Delay,' Circulation, vol. 140, no. 8, pp. e382–e482, 2019; "
                "R. L. Hughson, A. Helm, and M. Durante, 'Heart in space: effect of the extraterrestrial environment on "
                "the cardiovascular system,' Nature Reviews Cardiology, vol. 15, pp. 167–180, 2018.",
            ),
            (
                "[17]",
                "B. R. O'Driscoll et al., 'BTS guideline for oxygen use in adults in healthcare and emergency settings,' "
                "Thorax, vol. 72, Suppl. 1, pp. ii1–ii90, 2017; World Health Organization (WHO), Pulse Oximetry Training "
                "Manual, Geneva: WHO, 2011.",
            ),
            (
                "[18]",
                "P. A. Mackowiak, S. S. Wasserman, and M. M. Levine, 'A critical appraisal of 98.6°F, the upper limit of "
                "the normal body temperature, and other legacies of Carl Reinhold August Wunderlich,' JAMA, vol. 268, "
                "no. 12, pp. 1578–1580, 1992; A. C. Stahn et al., 'Increased core body temperature in astronauts during "
                "long-duration spaceflight,' Scientific Reports, vol. 7, art. 16180, 2017.",
            ),
            (
                "[19]",
                "W. Boucsein, Electrodermal Activity, 2nd ed., New York, NY: Springer, 2012.",
            ),
            (
                "[20]",
                "N. F. Watson et al., 'Recommended amount of sleep for a healthy adult: A joint consensus statement of "
                "the American Academy of Sleep Medicine and Sleep Research Society,' SLEEP, vol. 38, no. 6, pp. 843–844, "
                "2015; L. K. Barger et al., 'Prevalence of sleep deficiency and use of hypnotic drugs in astronauts before, "
                "during, and after spaceflight: an observational study,' The Lancet Neurology, vol. 13, no. 9, pp. 904–912, 2014.",
            ),
            (
                "[21]",
                "J. A. Loehr et al., 'Physical training for long-duration spaceflight,' Aerospace Medicine and Human "
                "Performance, vol. 86, no. 12 (Suppl.), pp. A14–A23, 2015; K. L. English et al., 'High intensity training "
                "during spaceflight: results from the NASA Sprint study,' npj Microgravity, vol. 6, art. 21, 2020.",
            ),
            (
                "[22]",
                "P. M. Rautaharju, B. Surawicz, and L. S. Gettes, 'AHA/ACCF/HRS recommendations for the standardization "
                "and interpretation of the electrocardiogram, Part IV: The ST segment, T and U waves, and the QT interval,' "
                "Circulation, vol. 119, no. 10, pp. e241–e250, 2009; D. S. D'Aunno et al., 'Effect of short- and "
                "long-duration spaceflight on QTc intervals in healthy astronauts,' American Journal of Cardiology, "
                "vol. 91, no. 4, pp. 494–497, 2003.",
            ),
            (
                "[23]",
                "P. Wagner et al., 'PTB-XL, a large publicly available electrocardiography dataset,' Scientific Data, "
                "vol. 7, art. 154, 2020.",
            ),
        ],
    ),
    (
        "C. Klinik Biyokimya, Beslenme, Nefroloji, Endokrinoloji ve Hematoloji Kaynakları",
        [
            (
                "[24]",
                "N. Rifai, A. R. Horvath, and C. T. Wittwer (Eds.), Tietz Textbook of Laboratory Medicine, 7th ed., "
                "St. Louis, MO: Elsevier, 2022.",
            ),
            (
                "[25]",
                "A. Kratz, M. Ferraro, P. M. Sluss, and K. B. Lewandrowski, 'Case records of the Massachusetts General "
                "Hospital: Laboratory reference values,' New England Journal of Medicine (NEJM), vol. 351, no. 15, "
                "pp. 1548–1563, 2004.",
            ),
            (
                "[26]",
                "G. D. Lundberg, 'When to panic over abnormal values,' Medical Laboratory Observer, vol. 4, pp. 47–54, "
                "1972; G. J. Kost, 'Critical limits for urgent clinician notification at US medical centers,' JAMA, "
                "vol. 263, no. 5, pp. 704–707, 1990.",
            ),
            (
                "[27]",
                "S. M. Smith, S. R. Zwart, and M. Heer, Human Adaptation to Spaceflight: The Role of Nutrition, 2nd ed., "
                "NASA/NP-2021-03-003-JSC, Houston, TX: NASA Lyndon B. Johnson Space Center, 2021.",
            ),
            (
                "[28]",
                "M. N. Sawka et al., 'American College of Sports Medicine position stand: Exercise and fluid replacement,' "
                "Medicine & Science in Sports & Exercise, vol. 39, no. 2, pp. 377–390, 2007; R. A. Pietrzyk et al., "
                "'Renal stone formation among astronauts,' Aviation, Space, and Environmental Medicine, vol. 78, no. 4 "
                "(Suppl.), pp. A9–A13, 2007.",
            ),
            (
                "[29]",
                "L. K. Nieman et al., 'The diagnosis of Cushing's syndrome: An Endocrine Society Clinical Practice "
                "Guideline,' Journal of Clinical Endocrinology & Metabolism (JCEM), vol. 93, no. 5, pp. 1526–1540, 2008; "
                "B. E. Crucian et al., 'Immune system dysregulation during spaceflight: potential countermeasures for "
                "deep space exploration missions,' Frontiers in Immunology, vol. 9, art. 1437, 2018.",
            ),
            (
                "[30]",
                "American Diabetes Association (ADA) Professional Practice Committee, '2. Diagnosis and Classification of "
                "Diabetes: Standards of Care in Diabetes—2024,' Diabetes Care, vol. 47, Suppl. 1, pp. S20–S42, 2024; "
                "R. L. Hughson et al., 'Increased postflight carotid artery stiffness and inflight insulin resistance "
                "resulting from 6-month spaceflight in male and female astronauts,' Diabetologia, vol. 59, pp. 1292–1296, 2016.",
            ),
            (
                "[31]",
                "Kidney Disease: Improving Global Outcomes (KDIGO), 'KDIGO Clinical Practice Guideline for Acute Kidney "
                "Injury,' Kidney International Supplements, vol. 2, no. 1, pp. 1–138, 2012 (ve KDIGO 2024 CKD Guideline, "
                "Kidney International, vol. 105, no. 4S, 2024).",
            ),
            (
                "[32]",
                "G. Spasovski et al., 'Clinical practice guideline on diagnosis and treatment of hyponatraemia,' "
                "European Journal of Endocrinology, vol. 170, no. 3, pp. G1–G47, 2014; R. H. Sterns, 'Disorders of plasma "
                "sodium—causes, consequences, and correction,' NEJM, vol. 372, pp. 55–65, 2015.",
            ),
            (
                "[33]",
                "T. A. Pearson et al., 'Markers of inflammation and cardiovascular disease: Application to clinical and "
                "public health practice (CDC/AHA Statement),' Circulation, vol. 107, no. 3, pp. 499–511, 2003.",
            ),
            (
                "[34]",
                "S. Vasikaran et al., 'Markers of bone turnover for the prediction of fracture risk and monitoring of "
                "osteoporosis treatment: A need for international reference standards (IOF-IFCC),' Osteoporosis "
                "International, vol. 22, no. 2, pp. 391–420, 2011; S. M. Smith et al., 'Benefits for bone from resistance "
                "exercise and nutrition in long-duration spaceflight,' Journal of Bone and Mineral Research (JBMR), "
                "vol. 27, no. 9, pp. 1896–1906, 2012.",
            ),
            (
                "[35]",
                "M. F. Holick et al., 'Evaluation, treatment, and prevention of vitamin D deficiency: An Endocrine "
                "Society clinical practice guideline,' Journal of Clinical Endocrinology & Metabolism (JCEM), vol. 96, "
                "no. 7, pp. 1911–1930, 2011.",
            ),
            (
                "[36]",
                "World Health Organization (WHO), Haemoglobin Concentrations for the Diagnosis of Anaemia and Assessment "
                "of Severity (WHO/NMH/NHD/MNM/11.1), Geneva: WHO, 2011 (ve 2024 güncellemesi); G. Trudel et al., "
                "'Hemolysis contributes to anemia during long-duration space flight,' Nature Medicine, vol. 28, "
                "pp. 59–62, 2022.",
            ),
        ],
    ),
    (
        "D. Astronot Ses Akustiği, Prosodik Biyo-Belirteçler ve Afektif Hesaplama Kaynakları",
        [
            (
                "[37]",
                "F. Eyben et al., 'The Geneva Minimalistic Acoustic Parameter Set (GeMAPS) for Voice Research and "
                "Affective Computing,' IEEE Transactions on Affective Computing, vol. 7, no. 2, pp. 190–202, 2016.",
            ),
            (
                "[38]",
                "P. Boersma, 'Accurate short-term analysis of the fundamental frequency and the harmonics-to-noise ratio "
                "of a sampled sound,' Proc. Institute of Phonetic Sciences (Univ. Amsterdam), vol. 17, pp. 97–110, 1993; "
                "J. P. Teixeira, C. Oliveira, and C. Lopes, 'Vocal acoustic analysis — Jitter, shimmer and HNR parameters,' "
                "Procedia Technology, vol. 9, pp. 1112–1122, 2013.",
            ),
            (
                "[39]",
                "A. P. Vogel, J. Fletcher, and P. Maruff, 'Acoustic analysis of the effects of sustained wakefulness on "
                "speech,' Journal of the Acoustical Society of America (JASA), vol. 128, no. 6, pp. 3747–3756, 2010.",
            ),
            (
                "[40]",
                "J. A. Russell, 'A circumplex model of affect,' Journal of Personality and Social Psychology, vol. 39, "
                "no. 6, pp. 1161–1178, 1980; D. H. Klatt, 'Software for a cascade/parallel formant synthesizer,' "
                "Journal of the Acoustical Society of America (JASA), vol. 67, no. 3, pp. 971–995, 1980.",
            ),
        ],
    ),
]


# Per-parameter clinical/aerospace rationale and exact citation mapping.
PARAM_SOURCES: Dict[str, Tuple[str, str]] = {
    # --- 8 Continuous wearable & sleep parameters ---
    "heart_rate": (
        "Yetişkin istirahat sinüs ritmi (60–100 bpm; kondisyonlu astronotlarda alt sınır ~55 bpm). "
        "45–110 bpm RCP NEWS2 erken uyarı sınırı; <=35 veya >=140 bpm ciddi bradikardi/taşikardi krizi.",
        "[14, 16]",
    ),
    "spo2": (
        "Normoksik kabinde fizyolojik SpO2 >=%95 (WHO/BTS/NEWS2). %92–95 hafif hipoksemi uyarısı; "
        "<=%88 akut solunum yetmezliği veya kabin hipoksisi acil eşiği.",
        "[2, 14, 17]",
    ),
    "systolic_bp": (
        "ACC/AHA ve ESC/ESH normal/yüksek-normal üst sınırı 130 mmHg; >=145 mmHg Evre 2 hipertansiyon, "
        ">=180 mmHg hipertansif kriz. <=90–80 mmHg ortostatik/hipovolemik şok sınırı (NEWS2).",
        "[14, 15, 16]",
    ),
    "diastolic_bp": (
        "ACC/AHA ve ESC/ESH dinlenim diyastolik üst sınırı 85 mmHg; >=92 mmHg belirgin hipertansiyon, "
        ">=110 mmHg hipertansif acil durum; <=50–40 mmHg ciddi diyastolik hipotansiyon.",
        "[15, 16]",
    ),
    "resp_rate": (
        "Öpneik yetişkin solunum hızı 12–18 soluk/dk (NEWS2). <=9 veya >=23 soluk/dk erken solunumsal "
        "kompansasyon/hiperkapni; <=6 (depresyon) veya >=30 soluk/dk akut solunum tehlikesi.",
        "[7, 14]",
    ),
    "body_temp": (
        "Normal diürnal kor/oral sıcaklık 36.1–37.2 °C (Mackowiak/JAMA). >=38.0 °C klinik ateş ve uzay "
        "termal yüklenmesi ('space fever'); >=39.5 °C hiperpireksi, <=34.5 °C hipotermi.",
        "[1, 14, 18]",
    ),
    "gsr": (
        "Boucsein elektrodermal aktivite referansı: sakin tonik deri iletkenliği (SCL) 1–8 µS; "
        "14–25+ µS akut sempatik deşarj, hareket hastalığı (SAS) veya termal stres.",
        "[19]",
    ),
    "sleep_hours": (
        "AASM/SRS ortak uzlaşı raporu yetişkin uyku süresi 7–9 saat/gece. ISS mürettebatında <5.5 saat "
        "bilişsel hata artışı; <=4 saat ağır uyku borcu (Barger et al., Lancet Neurol.).",
        "[1, 20]",
    ),
    # --- 10 Daily clinical, hydration, exercise, urine/saliva, nutrition & ECG ---
    "fluid_intake_ml": (
        "NASA-STD-3001 Vol. 2 ve NASA Beslenme Raporu: mikroyerçekiminde böbrek taşı riskini önlemek "
        "için 2.0–3.2 L/gün sıvı; <=800 mL ağır dehidratasyon, >=6 L su zehirlenmesi/hiponatremi.",
        "[2, 27, 28]",
    ),
    "exercise_min": (
        "ISS mürettebatı kemik/kas kaybını sınırlamak için ARED + T2 koşu bandı + CEVIS ile ~120 dk/gün "
        "(normal 90–180 dk) egzersiz yapar; <45–10 dk korumasız kas/kemik atrofisi riski.",
        "[1, 21]",
    ),
    "urine_sg": (
        "Tietz/NEJM ve ACSM hidrasyon kriteri: 1.005–1.025 öhidrasyon; >=1.030 konsantre idrar ve "
        "kalsiyum oksalat süper-satürasyonu, >=1.040 ağır hipovolemi.",
        "[24, 25, 28]",
    ),
    "urine_ph": (
        "Tietz/NEJM fizyolojik idrar pH aralığı 5.0–7.5. <4.6 ürik asit/oksalat kristalizasyonu; "
        ">8.0 brusit (kalsiyum fosfat) taşı veya üreaz pozitif idrar yolu enfeksiyonu riski.",
        "[24, 25, 28]",
    ),
    "urine_calcium": (
        "Tietz/NEJM 24 saatlik idrar kalsiyumu 50–250 mg/24s. Mikroyerçekimi osteolizi kalsiyum "
        "atılımını artırır; >320–450 mg/24s belirgin hiperkalsiüri ve nefrolitiyazis tehlikesi.",
        "[24, 25, 27, 34]",
    ),
    "saliva_cortisol": (
        "Endocrine Society ve Tietz gündüz/sabah tükürük kortizolü 3–15 nmol/L. >22–35 nmol/L "
        "uzay uçuşunda HPA ekseni hiperaktivasyonu ve bağışıklık baskılanması (Crucian et al.).",
        "[24, 29]",
    ),
    "calories_kcal": (
        "NASA uzay beslenme standardı: günlük egzersiz yapan astronot enerji gereksinimi ~2300–3300 "
        "kcal/gün (ideal 2800); <1800–1200 kcal katabolik kas ve kardiyak kütle kaybını hızlandırır.",
        "[1, 27]",
    ),
    "protein_g": (
        "NASA standardı kas koruma için 1.2–1.7 g/kg/gün (~80–140 g/gün). <55–30 g sarkopeni; "
        ">180–260 g aşırı kükürtlü aminoasit asit yükü nedeniyle kemik rezorpsiyonunu artırır.",
        "[27, 34]",
    ),
    "ecg_qtc_ms": (
        "AHA/ACCF/HRS standardı normal QTc 350–440 ms. Uzun süreli uzay uçuşunda QTc uzaması görülebilir "
        "(D'Aunno et al.); >=470 ms dikkat, >=500 ms Torsades de Pointes aritmi tehlikesi.",
        "[22]",
    ),
    "ecg_abnormal_pct": (
        "12-derivasyonlu EKG derin öğrenme sınıflandırıcısı ve PTB-XL aritmi taksonomisi üzerinden türetilen "
        "100 - P(Normal Sinüs Ritmi) olasılığı (<=%25 normal, >%50 dikkat, >=%80 yüksek patoloji).",
        "[22, 23]",
    ),
    # --- 13 Blood biochemistry & hemogram (CBC) parameters ---
    "glucose_mgdl": (
        "ADA 2024 açlık plazma glukozu normal 70–100 mg/dL; 100–126 mg/dL prediyabet/uzay insülin "
        "direnci, >=200 mg/dL belirgin hiperglisemi; <=60–45 mg/dL nöroglikopenik hipoglisemi.",
        "[25, 26, 30]",
    ),
    "creatinine_mgdl": (
        "KDIGO ve Tietz/NEJM yetişkin serum kreatinin 0.6–1.2 mg/dL. >=1.5 mg/dL KDIGO Evre 1 akut "
        "böbrek hasarı (ABH), >=3.0 mg/dL Evre 3 kritik ABH; <=0.4 mg/dL ileri kas erimesi.",
        "[24, 25, 31]",
    ),
    "sodium_mmol": (
        "Tietz/NEJM ve Avrupa Hiponatremi Kılavuzu normal Na+ 136–145 mmol/L. <=125 mmol/L beyin "
        "ödemi riskli ağır hiponatremi; >=158 mmol/L kritik hipernatremik dehidratasyon panik değeri.",
        "[24, 25, 26, 32]",
    ),
    "potassium_mmol": (
        "Tietz/NEJM normal K+ 3.5–5.0 mmol/L. Klinik laboratuvar panik sınırları (Lundberg/Kost): "
        "<=2.8 mmol/L hipokalemik aritmi, >=6.2 mmol/L kardiyotoksik hiperkalemi acil durumu.",
        "[24, 25, 26]",
    ),
    "calcium_mgdl": (
        "Tietz/NEJM serum total kalsiyum 8.6–10.2 mg/dL. <=7.0 mg/dL hipokalsemik tetani/laringospazm; "
        ">=12.5 mg/dL hiperkalsemik kriz (mikroyerçekimi kemik rezorpsiyonu + D vitamini fazlalığı).",
        "[24, 25, 26]",
    ),
    "crp_mgl": (
        "Tietz ve AHA/CDC klinik inflamasyon eşiği <=5.0 mg/L normal (ideal ~1 mg/L); 5–10 mg/L "
        "subklinik/hafif sistemik yanıt, >=50 mg/L akut bakteriyel enfeksiyon veya ağır doku hasarı.",
        "[24, 33]",
    ),
    "ctx_ngml": (
        "IOF-IFCC kemik dönüşüm referansı serum beta-CTX <=0.60 ng/mL. Mikroyerçekiminde osteoklastik "
        "yıkım hızla yükselir (Smith et al., JBMR); >0.90 dikkat, >=1.50 ng/mL hızlanmış osteoliz.",
        "[34]",
    ),
    "vitamin_d_ngml": (
        "Endocrine Society kılavuzu 25(OH)D yeterlilik hedefi 30–60 ng/mL (uzay aracında UVB eksiktir); "
        "<20 ng/mL eksiklik, <=10 ng/mL ağır osteomalazi; >=100 ng/mL hipervitaminoz D riski.",
        "[27, 35]",
    ),
    "hemoglobin_gdl": (
        "WHO ve Kratz/NEJM erişkin referansı 13.0–17.0 g/dL. Uzay uçuşunda hemoliz/neositoliz kaynaklı "
        "'uzay anemisi' gelişir (Trudel et al., Nature Med. 2022); <=8.5 ağır anemi, >=20 hemokonsantrasyon.",
        "[25, 36]",
    ),
    "hematocrit_pct": (
        "Kratz/NEJM ve Tietz normal hematokrit %38–50. <=%27 kritik anemi; >=%54–60 akut plazma "
        "hacmi kaybı / ağır dehidratasyon kaynaklı hiperviskozite.",
        "[24, 25, 36]",
    ),
    "rbc_mil": (
        "Kratz/NEJM eritrosit sayısı 4.3–5.8 x10^6/µL; <=3.8–3.0 x10^6/µL uzay hemolizi veya kemik "
        "iliği baskılanması, >=7.0 x10^6/µL polisitemi/hemokonsantrasyon.",
        "[24, 25, 36]",
    ),
    "wbc_k": (
        "Kratz/NEJM lökosit referansı 4.0–10.5 x10^3/µL. <=1.5 x10^3/µL akut radyasyon sendromu (ARS) "
        "hematolojik prodromu / ağır nötropeni panik değeri; >=25 x10^3/µL ağır enfeksiyon/lökomoid yanıt.",
        "[1, 25, 26]",
    ),
    "platelets_k": (
        "Kratz/NEJM trombosit referansı 150–400 x10^3/µL. <=50 x10^3/µL spontan kanama ve radyasyon "
        "kemik iliği baskılanması kritik sınırı (Lundberg/Kost); >=800 x10^3/µL tromboz riski.",
        "[1, 25, 26]",
    ),
    # --- 12 NASA HRP "RIDGE" space-environment hazard parameters ---
    "radiation_dose_msv_d": (
        "ISS alçak yörünge dozu ~0.5 mSv/gün (normal <=1.0). Derin uzay Mars transferi GCR dozu "
        "~1.84 mSv/gün (Zeitlin et al., Science 2013 -> dikkat <=3.0). Güneş parçacık fırtınası (SPE) >=20 mSv/gün.",
        "[1, 5, 6]",
    ),
    "radiation_cumulative_msv": (
        "NASA-STD-3001 Vol. 1 (2022) ve NASEM (2021) evrensel astronot kariyer efektif doz sınırı "
        "600 mSv'dir (tehlike = 600 mSv). <=150 mSv tipik 6 aylık ISS görevi, <=350 mSv uzun görev uyarısı.",
        "[1, 3, 5]",
    ),
    "isolation_days": (
        "NASA HRP Davranışsal Sağlık (BHP), Mars-500 (520 gün) ve NASA İkizler Çalışması (340 gün): "
        "<=90 gün standart çeyrek, >180 gün 'üçüncü çeyrek fenomeni', >365 gün kronik izolasyon yükü.",
        "[4, 10, 11]",
    ),
    "habitable_volume_m3": (
        "NASA-STD-3001 Vol. 2 ve NASA/TM-2015-218564 Net Yaşanabilir Hacim (NHV) çalışması: uzun "
        "keşif görevleri için minimum kabul edilebilir hacim >=25 m3/kişi (ISS ~55 m3; kapsüller <4 m3).",
        "[2, 9]",
    ),
    "earth_distance_mkm": (
        "ISS 0.0004, Ay 0.384 (<=1.0x10^6 km normal; tahliye ~4 gün), Mars yakınlaşma ~55 (dikkat <=50), "
        "Mars kavuşum ~401x10^6 km (tehlike = 400; tek yönlü 22 dk gecikme ve sıfır tahliye imkanı).",
        "[1, 4]",
    ),
    "gravity_g": (
        "Dünya 1.0 g, Mars 0.38 g (sarı dikkat bandı), Ay 0.166 g ve ISS/transfer 0.0 g (~%80 kırmızı "
        "kronik dekondisyon tehlikesi; günlük karşı-önlemle yönetildiğinden siyah acil durum tetiklemez).",
        "[1, 4, 12]",
    ),
    "g_transition_days": (
        "Fırlatma veya gezegen inişi sonrası ilk 1–4 gün Uzay Adaptasyon Sendromu (SAS), vestibüler "
        "ataksi ve ortostatik intoleransın zirve dönemidir; ~14 günde nörovestibüler adaptasyon sağlanır.",
        "[1, 12]",
    ),
    "cabin_co2_mmhg": (
        "Law et al. (JOEM 2014) ISS baş ağrısı eşiği ve NASA-STD-3001 Vol. 2 hedefi <=3.0 mmHg; "
        "NASA tarihsel uzun süreli SMAC uyarı sınırı 5.3 mmHg (%0.7); >=8.0 mmHg akut hiperkapni acili.",
        "[2, 7, 8]",
    ),
    "cabin_o2_kpa": (
        "NASA-STD-3001 Vol. 2 nominal kabin ppO2 kontrol aralığı 19.5–23.1 kPa (normal 19.5–23.5). "
        "<=17.0–14.0 kPa hipoksi tehlikesi; >=25.5–30.0 kPa kabin yangın/parlama ve O2 toksisitesi sınırı.",
        "[2]",
    ),
    "cabin_temp_c": (
        "NASA-STD-3001 Vol. 2 Bölüm 6.3 nominal mürettebat termal konfor aralığı birebir 18.0–27.0 °C'dir "
        "(ideal 22 °C). <=12 °C soğuk stresi, >=33 °C ECLSS soğutma arızası kaynaklı akut ısı bitkinliği.",
        "[2]",
    ),
    "cabin_humidity_pct": (
        "NASA-STD-3001 Vol. 2 nominal bağıl nem aralığı birebir %25–%70'tir (ideal %45). <%20 mukoza "
        "kuruluğu/statik elektrik; >%75–85 kabin yüzeylerinde yoğuşma, mantar üremesi ve terleme engeli.",
        "[2]",
    ),
    "cabin_noise_dba": (
        "NASA-STD-3001 Vol. 2 sürekli çalışma alanı akustik sınırı NC-50 (~60 dBA; uyku bölmesi ~48 dBA). "
        "NIOSH ve NASA işitme kaybı / kulak koruyucu zorunluluk tehlike eşiği >=85 dBA.",
        "[2, 13]",
    ),
}

assert set(PARAM_SOURCES.keys()) == set(PARAM_KEYS), (
    f"Missing parameter mappings: {set(PARAM_KEYS) - set(PARAM_SOURCES.keys())}"
)


VOICE_BIOMARKER_ROWS: List[List[str]] = [
    [
        "Bağıl Perde Varyasyonu (F0 CV %)",
        "%8.0 – %14.5",
        "<%5.2 (Monopitch) veya >%15.0",
        "İzolasyon/çökkün duygudurumda perde düzleşir (<%5.2); akut sempatik streste F0 dalgalanması >%15'e çıkar.",
        "[37, 40]",
    ],
    [
        "Lokal Jitter (Perde Pertürbasyonu %)",
        "<= %1.04 (Sakin ses < %1.2)",
        "> %1.80 (Stres / Disfoni)",
        "Praat/Boersma ve GeMAPS vokal kord mikro-titreşim referansı; otonom gerilim ve yorgunlukta yükselir.",
        "[37, 38]",
    ],
    [
        "Lokal Shimmer (Genlik Pertürbasyonu %)",
        "<= %3.81 (Sakin ses < %4.5)",
        "> %6.0 (Yorgunluk) / > %8.0 (Dispne)",
        "Glottal kapanma düzensizliği ve solunumsal zorlanmada genlik kararsızlığı belirgin şekilde artar.",
        "[37, 38]",
    ],
    [
        "Harmonik-Gürültü Oranı (HNR, dB)",
        "12.0 – 22.0 dB",
        "< 9.5 dB",
        "Sağlıklı modal fonasyonda harmonik yapı baskındır; <9.5 dB soluklu ses ve inspiratuvar dispne göstergesidir.",
        "[37, 38]",
    ],
    [
        "Konuşma Hızı (Hece/sn) ve Duraksama (%)",
        "3.8 – 5.0 hece/sn | Duraksama <%25",
        "<3.4 hece/sn & >%30 duraksama (Yorgunluk)\n>5.1 hece/sn (Akut Stres)",
        "Uzamış uyanıklık ve bilişsel yorgunlukta konuşma yavaşlar ve duraksamalar uzar (Vogel et al., JASA 2010).",
        "[37, 39]",
    ],
    [
        "Spektral Alfa Oranı (1–5 kHz / <1 kHz)",
        "-10.0 ile -7.0 dB arası",
        "> -6.5 dB (Gergin) | < -9.5 dB (Hipofonik)",
        "GeMAPS spektral eğim parametresi; akut streste yüksek frekans enerjisi artarken çökkünlükte düşer.",
        "[37]",
    ],
    [
        "4 Boyutlu Afektif Mood Uzayı",
        "Valans ~%76, Uyarılma ~%34,\nStres ~%14, Yorgunluk ~%16",
        "Stres/Yorgunluk >= %70 (Kırmızı)\nBaz Vokal Risk >= %75–88",
        "Russell'ın Dairesel Duygulanım Modeli (Valans x Uyarılmışlık) uzay psikolojisi eksenlerine uyarlanmıştır.",
        "[37, 40]",
    ],
]


# --------------------------------------------------------------------------- #
# Formatting Helpers
# --------------------------------------------------------------------------- #

def _f(v: Optional[float], dec: int) -> str:
    if v is None:
        return "–"
    return f"{v:.{dec}f}"


def format_parameter_bands(p: Parameter) -> Tuple[str, str, str]:
    """Return human-readable (ideal_and_normal, warn_band, danger_band) strings."""
    d = p.decimals
    lo_n, hi_n = p.normal
    lo_w, hi_w = p.warn
    lo_d, hi_d = p.danger

    if p.key == "gravity_g":
        return (
            f"İdeal: {_f(p.ideal, d)}\nNormal: >= {_f(lo_n, d)}",
            f"0.30 – 0.69\n(Mars: 0.38 g)",
            "0.00 – 0.29\n(0 g = ~%80 Kırmızı)",
        )

    if lo_n is not None and hi_n is not None:
        norm_s = f"İdeal: {_f(p.ideal, d)}\n{_f(lo_n, d)} – {_f(hi_n, d)}"
        warn_s = f"{_f(lo_w, d)} – {_f(lo_n, d)} veya\n{_f(hi_n, d)} – {_f(hi_w, d)}"
        dang_s = f"<= {_f(lo_d, d)} (Siyah)\n>= {_f(hi_d, d)} (Siyah)\n[<{_f(lo_w, d)} / >{_f(hi_w, d)} Kırmızı]"
    elif lo_n is None and hi_n is not None:
        norm_s = f"İdeal: {_f(p.ideal, d)}\n<= {_f(hi_n, d)}"
        warn_s = f"{_f(hi_n, d)} – {_f(hi_w, d)}"
        dang_s = f"> {_f(hi_w, d)} (Kırmızı)\n>= {_f(hi_d, d)} (Siyah)"
    else:
        norm_s = f"İdeal: {_f(p.ideal, d)}\n>= {_f(lo_n, d)}"
        warn_s = f"{_f(lo_w, d)} – {_f(lo_n, d)}"
        dang_s = f"< {_f(lo_w, d)} (Kırmızı)\n<= {_f(lo_d, d)} (Siyah)"
    return norm_s, warn_s, dang_s


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


def mark_repeat_header(row) -> None:
    trPr = row._tr.get_or_add_trPr()
    trPr.append(parse_xml(f"<w:tblHeader {nsdecls('w')}/>"))
    trPr.append(parse_xml(f"<w:cantSplit {nsdecls('w')}/>"))


def mark_cant_split(row) -> None:
    trPr = row._tr.get_or_add_trPr()
    trPr.append(parse_xml(f"<w:cantSplit {nsdecls('w')}/>"))


def style_table(doc: Document, headers: List[str], rows: List[List[str]], col_widths: List[float]) -> None:
    table = doc.add_table(rows=1 + len(rows), cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(table)

    hdr_row = table.rows[0]
    mark_repeat_header(hdr_row)
    for j, h in enumerate(headers):
        cell = hdr_row.cells[j]
        cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
        set_cell_bg(cell, "1E0E3A")
        p = cell.paragraphs[0]
        p.paragraph_format.space_before = Pt(3.5)
        p.paragraph_format.space_after = Pt(3.5)
        run = p.add_run(h)
        run.bold = True
        run.font.size = Pt(8.3)
        run.font.name = "Segoe UI"
        run.font.color.rgb = RGBColor(244, 240, 255)

    for i, r in enumerate(rows):
        row = table.rows[i + 1]
        mark_cant_split(row)
        bg = "F6F3FC" if i % 2 == 1 else "FFFFFF"
        for j, val in enumerate(r):
            cell = row.cells[j]
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            set_cell_bg(cell, bg)
            p = cell.paragraphs[0]
            p.paragraph_format.space_before = Pt(2.5)
            p.paragraph_format.space_after = Pt(2.5)
            p.paragraph_format.line_spacing = 1.12
            run = p.add_run(str(val))
            run.font.size = Pt(8.1)
            run.font.name = "Segoe UI"
            if j == 0:
                run.bold = True
                run.font.color.rgb = RGBColor(38, 17, 74)
            elif j == len(r) - 1 and str(val).startswith("["):
                run.bold = True
                run.font.color.rgb = RGBColor(74, 39, 148)
            else:
                run.font.color.rgb = RGBColor(34, 30, 46)

    for row in table.rows:
        for j, w_cm in enumerate(col_widths):
            row.cells[j].width = Cm(w_cm)
    doc.add_paragraph()


def add_h1(doc: Document, text: str) -> None:
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(14)
    p.paragraph_format.space_after = Pt(6)
    p.paragraph_format.keep_with_next = True
    run = p.add_run(text)
    run.bold = True
    run.font.size = Pt(13.5)
    run.font.name = "Segoe UI"
    run.font.color.rgb = RGBColor(30, 14, 58)
    pBdr = parse_xml(f'<w:pBdr {nsdecls("w")}><w:bottom w:val="single" w:sz="10" w:space="3" w:color="6D42C7"/></w:pBdr>')
    p._p.get_or_add_pPr().append(pBdr)


def add_h2(doc: Document, text: str) -> None:
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(10)
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.keep_with_next = True
    run = p.add_run(text)
    run.bold = True
    run.font.size = Pt(11.0)
    run.font.name = "Segoe UI"
    run.font.color.rgb = RGBColor(74, 39, 148)


def add_body(doc: Document, text: str, bold_prefix: str = "", hanging_cm: float = 0.0) -> None:
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(4.5)
    p.paragraph_format.line_spacing = 1.18
    if hanging_cm > 0:
        p.paragraph_format.left_indent = Cm(hanging_cm)
        p.paragraph_format.first_line_indent = Cm(-hanging_cm)
    if bold_prefix:
        rb = p.add_run(bold_prefix)
        rb.bold = True
        rb.font.name = "Segoe UI"
        rb.font.size = Pt(9.5)
        rb.font.color.rgb = RGBColor(38, 17, 74)
    r = p.add_run(text)
    r.font.name = "Segoe UI"
    r.font.size = Pt(9.5)
    r.font.color.rgb = RGBColor(34, 30, 46)


def add_bullet(doc: Document, bold_prefix: str, text: str) -> None:
    p = doc.add_paragraph(style="List Bullet")
    p.paragraph_format.space_after = Pt(3)
    p.paragraph_format.line_spacing = 1.15
    rb = p.add_run(bold_prefix)
    rb.bold = True
    rb.font.name = "Segoe UI"
    rb.font.size = Pt(9.4)
    rb.font.color.rgb = RGBColor(38, 17, 74)
    r = p.add_run(text)
    r.font.name = "Segoe UI"
    r.font.size = Pt(9.4)
    r.font.color.rgb = RGBColor(34, 30, 46)


def build_param_rows(subset: List[Parameter]) -> List[List[str]]:
    rows: List[List[str]] = []
    for p in subset:
        norm_s, warn_s, dang_s = format_parameter_bands(p)
        rationale, refs = PARAM_SOURCES[p.key]
        unit_lbl = f" [{p.unit}]" if p.unit else ""
        cat_tr = CATEGORIES[p.category]["tr"]
        rows.append([
            f"{p.name_tr}{unit_lbl}\n({p.key} | {cat_tr})",
            norm_s,
            warn_s,
            dang_s,
            f"{rationale} {refs}",
        ])
    return rows


def generate_documents(docx_path: Path, md_path: Path) -> Tuple[Path, Path]:
    seq_params = [p for p in PARAMETERS if p.kind == "continuous"]
    daily_non_blood_cats = {"hydration_renal", "sleep_activity", "bone_muscle", "stress_thermal", "nutrition", "ecg"}
    daily_clinical = [
        p for p in PARAMETERS
        if p.kind == "daily"
        and p.category in daily_non_blood_cats
        and p.key not in {"ctx_ngml", "vitamin_d_ngml"}
    ]
    blood_params = [
        p for p in PARAMETERS
        if p.kind == "daily" and (p.category in {"biochemistry", "hemogram"} or p.key in {"ctx_ngml", "vitamin_d_ngml"})
    ]
    env_params = [p for p in PARAMETERS if p.kind == "environment"]

    covered = [p.key for p in (*seq_params, *daily_clinical, *blood_params, *env_params)]
    assert len(covered) == 43 and set(covered) == set(PARAM_KEYS), "Parameter partition mismatch!"

    doc = Document()
    for sec in doc.sections:
        sec.top_margin = Cm(1.8)
        sec.bottom_margin = Cm(1.8)
        sec.left_margin = Cm(1.7)
        sec.right_margin = Cm(1.7)

    # ==================== KAPAK BANNER ====================
    banner = doc.add_table(rows=1, cols=1)
    banner.alignment = WD_TABLE_ALIGNMENT.CENTER
    c0 = banner.rows[0].cells[0]
    set_cell_bg(c0, "16092D")
    p_top = c0.paragraphs[0]
    p_top.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_top.paragraph_format.space_before = Pt(14)
    p_top.paragraph_format.space_after = Pt(14)

    r_sub = p_top.add_run("ASTROCYTEX v3.0  |  KLİNİK REFERANS ARALIKLARI, RİSK EŞİKLERİ VE BİLİMSEL KAYNAKÇA\n")
    r_sub.font.name = "Segoe UI"
    r_sub.font.size = Pt(9.2)
    r_sub.bold = True
    r_sub.font.color.rgb = RGBColor(198, 176, 255)

    r_title = p_top.add_run(
        "İzlenen 43 Fizyolojik/Uzay Parametresi ve Ses Biyo-Belirteçlerinin\n"
        "Normal, Dikkat ve Tehlike Değerlerinin Bilimsel Kaynakları ve Kaynakça\n"
    )
    r_title.font.name = "Segoe UI"
    r_title.font.size = Pt(15.0)
    r_title.bold = True
    r_title.font.color.rgb = RGBColor(255, 255, 255)

    r_desc = p_top.add_run(
        "NASA-STD-3001 Cilt 1 & 2  |  NASA HRP 'RIDGE' Raporları  |  RCP NEWS2 Erken Uyarı Ölçeği  |  "
        "Tietz & NEJM Klinik Laboratuvar Referansları  |  AHA/ACC, WHO, ADA, KDIGO, Endocrine Society, GeMAPS"
    )
    r_desc.font.name = "Segoe UI"
    r_desc.font.size = Pt(8.6)
    r_desc.font.color.rgb = RGBColor(224, 212, 252)

    doc.add_paragraph()

    # ==================== BÖLÜM 1: METODOLOJİ ====================
    add_h1(doc, "1. Metodoloji: Normal ve Riskli Değerler Hangi Kaynaklardan ve Nasıl Oluşturuldu?")
    add_body(
        doc,
        "AstrocyteX (utils/reference_ranges.py ve models/voice_model.py) içinde tanımlanan 43 temel telemetri "
        "parametresi (31 fizyolojik sağlık biyo-belirteci + 12 NASA uzay ortamı tehlikesi) ve 42 vokal akustik "
        "özelliğin eşik değerleri tek bir kaynaktan değil; erişkin klinik tıp kılavuzları, klinik laboratuvar "
        "referans/panik cetvelleri ve yayınlanmış uzay tıbbı standartlarının hibrit senteziyle oluşturulmuştur [1, 2, 14, 24, 25]."
    )
    add_bullet(
        doc,
        "1. Normal Aralıklar (Yeşil Bant — %15–%40 Risk): ",
        "Fizyolojik ve biyokimyasal parametrelerde doğrudan uluslararası tıbbi referans aralıklarına "
        "(Tietz Textbook of Laboratory Medicine [24], Massachusetts General Hospital / NEJM Referans Tabloları [25], "
        "AHA/ACC Kardiyoloji Kılavuzları [15, 16, 22], WHO Anemi ve Oksimetri Kriterleri [17, 36], ADA Diyabet Standartları [30], "
        "KDIGO Böbrek Kılavuzu [31], Endocrine Society [29, 35] ve AASM Uyku Uzlaşısı [20]); kabin ve uzay ortamı "
        "parametrelerinde ise birebir NASA-STD-3001 Cilt 1 & Cilt 2 [1, 2] nominal operasyonel sınırlarına "
        "(örneğin kabin sıcaklığı 18–27 °C, bağıl nem %25–%70, sürekli gürültü <=60 dBA, yaşanabilir hacim >=25 m3/kişi) dayanır.",
    )
    add_bullet(
        doc,
        "2. Dikkat Eşikleri (Warn / Sarı Bant — %40–%70 Risk): ",
        "Erken fizyolojik sapmayı ve otonom karşı-önlem (su içme, solunum egzersizi, ARED direnç antrenmanı, ECLSS "
        "havalandırma kontrolü) gereksinimini temsil eder. Yatak başı erken uyarı sistemleri (Royal College of "
        "Physicians NEWS2 orta-skor eşikleri [14]), prediyabet/evre-1 hipertansiyon/hafif elektrolit bozukluğu sınırları "
        "ve NASA'nın tarihsel uzay aracı maksimum izin verilebilir konsantrasyon (SMAC) uyarı değerlerinden "
        "(örneğin ISS kabin CO2 için 5.3 mmHg SMAC sınırı [7, 8] veya Mars transferi ~1.84 mSv/gün GCR dozu [5]) türetilmiştir.",
    )
    add_bullet(
        doc,
        "3. Tehlike ve Kritik Acil Durum Eşikleri (Danger / Kırmızı %70–%90 ve Siyah >%90 Bant): ",
        "Klinik laboratuvar 'panik / acil bildirim' değerlerine (Lundberg ve Kost JAMA kritik limitleri [26]: "
        "potasyum <=2.8 veya >=6.2 mmol/L, sodyum <=125 veya >=158 mmol/L, kalsiyum <=7.0 veya >=12.5 mg/dL, "
        "trombosit <=50x10^3/µL, lökosit <=1.5x10^3/µL, glukoz <=45 mg/dL), aritmi ve şok sınırlarına "
        "(QTc >=500 ms [22], sistolik KB <=80 veya >=180 mmHg [14, 15], SpO2 <=%88 [17]) ve NASA yasal/acil üst "
        "sınırlarına (600 mSv kariyer radyasyon limiti [1, 3], >=85 dBA NIOSH/NASA işitme tehlikesi [2, 13], "
        ">=8.0 mmHg akut hiperkapni [8]) sabitlenmiştir.",
    )
    add_bullet(
        doc,
        "4. Parçalı-Doğrusal Kalibrasyon ve Gösterim Notu: ",
        "utils/reference_ranges.py içinde her parametre ideal -> %8, normal sınırı -> %40, dikkat sınırı -> %70 ve "
        "tehlike sınırı -> %90 risk puanına parçalı-doğrusal (piecewise-linear) olarak eşlenir; tehlike sınırının "
        "ötesinde skor asimptotik olarak %100'e yaklaşır. Kod tabanında açıkça belirtildiği üzere bu sayısal eşikler "
        "araştırma ve mühendislik simülasyonu amacıyla harmonize edilmiş temsili (illustrative) referanslardır; "
        "gerçek uçuş operasyonlarında uçuş cerrahı onayı gerektirir.",
    )

    style_table(
        doc,
        ["Risk Bandı", "Skor Aralığı", "Kod Çapası (Anchor)", "Klinik / Operasyonel Anlamı ve Kaynak Mantığı"],
        [
            ["GRAY (Gri)", "< %15", "ideal -> %8 (Tek yönlü taban %6)", "Astronotun fizyolojik bazal/ideal noktası veya henüz ölçüm girilmemiş pasif parametre."],
            ["GREEN (Yeşil)", "%15 – %40", "ideal .. normal_lo / normal_hi", "Standart yetişkin klinik referans aralığı veya NASA-STD-3001 nominal kabin aralığı."],
            ["YELLOW (Sarı)", "%40 – %70", "normal sınırı (%40) .. warn sınırı (%70)", "Erken sapma / dikkat bölgesi (NEWS2 erken uyarı, SMAC uyarı eşiği); koruyucu görev üretir."],
            ["RED (Kırmızı)", "%70 – %90", "warn sınırı (%70) .. danger sınırı (%90)", "Yüksek klinik veya çevresel risk; öncelikli karşı-önlem ve seri vital izlem gerektirir."],
            ["BLACK (Siyah)", "> %90", "danger_lo / danger_hi ötesi (%90 -> %100)", "Kritik laboratuvar panik değeri veya acil kabin/radyasyon tehlikesi; uçuş cerrahı eskalasyonu."],
        ],
        col_widths=[2.8, 2.3, 4.3, 8.2],
    )

    # ==================== BÖLÜM 2: SÜREKLİ SENSÖRLER ====================
    col_w = [3.6, 2.5, 2.5, 2.7, 6.3]
    hdrs = [
        "Parametre [Birim]\n(Kod | Sistem)",
        "İdeal ve Normal\n(Yeşil <=%40)",
        "Dikkat Aralığı\n(Sarı %40–%70)",
        "Tehlike / Kritik\n(Kırmızı/Siyah)",
        "Alındığı Klinik / Uzay Tıbbı Kaynağı ve Gerekçe",
    ]

    add_h1(doc, "2. Sürekli Giyilebilir Sensör ve Uyku Parametreleri (8 Parametre)")
    style_table(doc, hdrs, build_param_rows(seq_params), col_widths=col_w)

    # ==================== BÖLÜM 3: GÜNLÜK KLİNİK, BESLENME, İDRAR, EKG ====================
    add_h1(doc, "3. Günlük Klinik, Hidrasyon, Egzersiz, İdrar/Tükürük, Beslenme ve EKG (10 Parametre)")
    style_table(doc, hdrs, build_param_rows(daily_clinical), col_widths=col_w)

    # ==================== BÖLÜM 4: KAN BİYOKİMYASI VE HEMOGRAM ====================
    add_h1(doc, "4. Kan Biyokimyası, Kemik Dönüşüm ve Hemogram / CBC Parametreleri (13 Parametre)")
    style_table(doc, hdrs, build_param_rows(blood_params), col_widths=col_w)

    # ==================== BÖLÜM 5: NASA RIDGE UZAY ORTAMI ====================
    add_h1(doc, "5. NASA HRP 'RIDGE' Uzay Ortamı ve Kabin (ECLSS) Tehlike Parametreleri (12 Parametre)")
    style_table(doc, hdrs, build_param_rows(env_params), col_widths=col_w)

    # ==================== BÖLÜM 6: SES BİYO-BELİRTEÇLERİ ====================
    add_h1(doc, "6. Astronot Ses Akustiği ve Mood Biyo-Belirteçleri (AstroVoiceNet v1.0)")
    add_body(
        doc,
        "Konuşma analizi alt sistemi (utils/voice_synth.py ve models/voice_model.py), Geneva Minimalistic Acoustic "
        "Parameter Set (GeMAPS) [37], Praat/Boersma glottal pertürbasyon ölçümleri [38], uyku yoksunluğunun konuşma "
        "üzerindeki etkileri [39] ve Russell'ın Dairesel Duygulanım Modeli [40] literatürüne dayanır:"
    )
    style_table(
        doc,
        ["Vokal Akustik Biyo-Belirteç", "Nominal / Dengeli Ses", "Riskli / Patolojik Eşik", "Fizyopsikolojik Anlamı", "Atıf"],
        VOICE_BIOMARKER_ROWS,
        col_widths=[3.6, 3.0, 3.3, 6.2, 1.5],
    )

    # ==================== BÖLÜM 7: TAM KAYNAKÇA ====================
    add_h1(doc, "7. Kaynakça (Akademik Makaleler, Klinik Kılavuzlar ve NASA Standartları)")
    add_body(
        doc,
        "Aşağıdaki kaynakça, AstrocyteX v3.0 içindeki tüm normal referans aralıklarının, dikkat/uyarı sınırlarının "
        "ve kritik tehlike eşiklerinin dayandığı birincil bilimsel yayınları ve kurumsal standartları listeler:"
    )

    for group_title, refs in BIBLIOGRAPHY_GROUPS:
        add_h2(doc, group_title)
        for num, citation in refs:
            add_body(doc, citation, bold_prefix=f"{num} ", hanging_cm=0.95)

    docx_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(docx_path))

    # ==================== MARKDOWN MIRROR ====================
    md: List[str] = [
        "# AstrocyteX v3.0 — Parametre Referans Aralıkları, Risk Eşikleri ve Bilimsel Kaynakça",
        "",
        "Bu doküman, `utils/reference_ranges.py` ve `models/voice_model.py` içinde izlenen **43 telemetri parametresi** "
        "(31 fizyolojik biyo-belirteç + 12 NASA HRP RIDGE uzay ortamı tehlikesi) ile **vokal prosodik biyo-belirteçlerin** "
        "normal, dikkat (sarı) ve tehlike (kırmızı/siyah) değerlerinin hangi bilimsel kaynaklardan alındığını belgeler.",
        "",
        "## 1. Metodolojik Özet",
        "",
        "- **Normal Aralıklar (`%15–%40`, Yeşil):** Uluslararası erişkin klinik referans aralıklarından (*Tietz Textbook of Laboratory Medicine* [24], *Kratz et al. NEJM* [25], AHA/ACC, WHO, ADA, KDIGO, Endocrine Society, AASM) ve **NASA-STD-3001 Cilt 1 & Cilt 2** [1, 2] nominal kabin/sağlık standartlarından alınmıştır.",
        "- **Dikkat Eşikleri (`%40–%70`, Sarı):** Yatak başı fizyolojik erken uyarı kriterleri (**RCP NEWS2** [14]), sınırda klinik patoloji eşikleri ve NASA SMAC / derin uzay GCR uyarı değerlerinden [5, 7, 8] oluşturulmuştur.",
        "- **Tehlike Eşikleri (`%70–%90` Kırmızı ve `>%90` Siyah):** Klinik laboratuvar kritik 'panik' bildirim tabloları (**Lundberg & Kost, JAMA** [26]), malign aritmi/şok sınırları (`QTc >= 500 ms` [22], `SpO2 <= %88` [17]) ve NASA yasal/acil limitlerinden (`600 mSv` kariyer radyasyon limiti [1, 3], `>=85 dBA` akustik tehlike [2, 13], `>=8.0 mmHg` kabin CO2 [8]) kalibre edilmiştir.",
        "",
    ]

    sections_md = [
        ("2. Sürekli Giyilebilir Sensör ve Uyku Parametreleri (8 Parametre)", seq_params),
        ("3. Günlük Klinik, Hidrasyon, Egzersiz, İdrar/Tükürük, Beslenme ve EKG (10 Parametre)", daily_clinical),
        ("4. Kan Biyokimyası, Kemik Dönüşüm ve Hemogram Parametreleri (13 Parametre)", blood_params),
        ("5. NASA HRP 'RIDGE' Uzay Ortamı ve Kabin Tehlike Parametreleri (12 Parametre)", env_params),
    ]
    for sec_title, plist in sections_md:
        md += [
            f"## {sec_title}",
            "",
            "| Parametre (Kod) | İdeal & Normal | Dikkat (Sarı) | Tehlike (Kırmızı/Siyah) | Klinik / Uzay Tıbbı Kaynağı |",
            "|---|---|---|---|---|",
        ]
        for r in build_param_rows(plist):
            clean_cells = [c.replace("\n", " <br> ") for c in r]
            md.append("| " + " | ".join(clean_cells) + " |")
        md.append("")

    md += [
        "## 6. Astronot Ses Akustiği ve Mood Biyo-Belirteçleri",
        "",
        "| Vokal Biyo-Belirteç | Nominal / Dengeli | Riskli Eşik | Fizyopsikolojik Anlamı | Atıf |",
        "|---|---|---|---|---|",
    ]
    for r in VOICE_BIOMARKER_ROWS:
        clean_cells = [c.replace("\n", " <br> ") for c in r]
        md.append("| " + " | ".join(clean_cells) + " |")
    md.append("")

    md += ["## 7. Kaynakça", ""]
    for group_title, refs in BIBLIOGRAPHY_GROUPS:
        md.append(f"### {group_title}")
        md.append("")
        for num, citation in refs:
            md.append(f"- **{num}** {citation}")
        md.append("")

    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")
    return docx_path, md_path


if __name__ == "__main__":
    desktop = Path(os.environ.get("USERPROFILE", str(Path.home()))) / "Desktop"
    out_docx = desktop / "AstrocyteX_Parametre_Referans_Araliklari_ve_Kaynakca.docx"
    out_md = ROOT / "docs" / "PARAMETRE_KAYNAKCA.md"
    saved_docx, saved_md = generate_documents(out_docx, out_md)
    print(f"SAVED_DOCX={saved_docx} ({saved_docx.stat().st_size:,} bytes)")
    print(f"SAVED_MD={saved_md} ({saved_md.stat().st_size:,} bytes)")

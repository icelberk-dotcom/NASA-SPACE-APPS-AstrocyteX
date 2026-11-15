# AstrocyteX v3.0 — Yapay Zeka Model Ozeti, Kutuphaneler ve Performans Raporu

## 1. Yonetici Ozeti

| Teknik Boyut | Ozet Deger ve Mimari Nitelik |
|---|---|
| **Sistem Adi ve Surumu** | AstrocyteX v3.0 (Karanlik Mor Samanyolu Uzay Telemetri Konsolu, Sifir-Emoji Tasarim) |
| **Ana Cok-Modaliteli Ag** | AstrocyteX-Net v3.0 — 2,976,783 egitilebilir parametre (d_model=80, 16 Perceiver latent, SwiGLU etkilesim bloklari) [1-12, 25] |
| **Ses ve Mood Analiz Agi** | AstroVoiceNet v1.0 — 259,645 egitilebilir parametre (48-bant Log-Mel Conv+SE+Attention + 42 GeMAPS prozodik giris) [13-16] |
| **Izlenen Parametre Kapsami** | 43 Telemetri Parametresi (31 Fizyolojik + 12 NASA HRP RIDGE Uzay Tehlikesi) + 42 Ses Biyo-Belirteci + 12-Kanal EKG [17, 23] |
| **Ekstra GPU Egitim Kapsami** | AstrocyteX-Net: 42,000 egitim / 4,200 dogrulama gunu (50 epoch, en iyi epoch 48) | AstroVoiceNet: 4,000 egitim / 700 dogrulama ses kaydi (45 epoch) |
| **Bagimsiz Test Performansi (43 Parametre)** | MCC = 0.9894 | Macro-F1 = 0.9899 | QWK = 0.9958 | Kritik Recall = %99.53 | MAE = 0.144 puan [21, 22] |
| **Uzay Ortami Katkisi (31 Saglik Parametresi)** | Ortam girdisiz MCC 0.9040 -> Tam model MCC 0.9850 (+0.0810); Under-Triage %16.37 -> %1.05 |
| **Bagimsiz Ses Analizi Testi (1.200 Kayit)** | Psikolojik Durum MCC = 0.9957 | Macro-F1 = 0.9963 | Ses Risk MAE = 3.168 puan | Mood MAE = 3.683 puan | Cikarim = 2.41 ms/kayit |
| **Rol-Duyarli LLM Danisman** | 'Ben doktorum' (detayli uzay tibbi patofizyolojisi ve nicel biyo-belirtecler) ve 'Doktor degilim' (sade, anlasilir gunluk rehberlik) cift modlu mimari [28] |

## 2. Gelistirilen Yapay Zeka Modelleri ve Yenilikler

- **AstrocyteX-Net v3.0 (`models/astroguard.py`, 2,976,783 parametre):** `d_model=80`, `16` Perceiver latent darbogazi ve `SwiGLU GatedInteractionBlock` [7] katmanlariyla guclendirilmis 5-modaliteli derin ag (Mamba-2 SSD [1], TCN+TFT [2, 3], EnvironmentEncoder+FiLM [6], Cross-Attention [4, 5], Neural ODE + PINN [8, 9], Bayes-by-Backprop + MC-Dropout [10-12]). 42,000 egitim + 4,200 dogrulama murettebat-gunu uzerinde 50 epoch GPU ile yeniden egitilmistir.
- **AstroVoiceNet v1.0 (`models/voice_model.py`, 259,645 parametre):** Astronot ses kayitlarindan `(48, 64)` log-Mel spektrogram + `42` GeMAPS esinli akustik/prozodik biyo-belirtec [13] cikaran, 2B Derinlik-Ayrilabilir Conv + Squeeze-and-Excitation [14] + Oz-Dikkat + `ProsodyGateEncoder` hibrit sinir agi. 5 psikolojik durumu, 4 mood boyutunu [16] ve `%0-%100` ses risk skorunu uretir.
- **Rol-Duyarli LLM Danisman (`backend/gemini_agent.py`):** Kullanici **'Ben doktorum'** sectiginde uzay tibbi patofizyolojisi ve nicel F0/jitter/shimmer/HNR/QTc bulgulariyla detayli klinik dil; **'Doktor degilim'** sectiginde jargondan arindirilmis sade ve anlasilir gunluk dil kullanir [28].
- **Sifir-Emoji Karanlik Mor Samanyolu Arayuzu (`web/`, `ui/`):** Tum kod tabanindan emojiler tamamen kaldirilmis, Samanyolu galaksisinin Dunya'dan gorunumunu iceren karanlik mor uzay telemetri tasarimi uygulanmistir.

## 3. AstrocyteX-Net v3.0 Bagimsiz Test Sonuclari (6,000 Murettebat-Gunu, 258,000 Tahmin)

### 3.1. Tum 43 Parametre (31 Fizyolojik + 12 NASA RIDGE Uzay Tehlikesi)

| Sistem | MCC [%95 GA] | Macro-F1 [%95 GA] | QWK | Kritik Recall | Under-Triage | Gorev F1 | MAE (puan) |
|---|---|---|---|---|---|---|---|
| Kural Bazli (Referans Araliklari) | 0.9032 [0.9016-0.9047] | 0.8839 [0.8808-0.8872] | 0.9599 | %88.03 | %11.97 | 0.9509 | 1.494 |
| Dogrusal ML Bazi (scikit-learn Ridge) [26] | 0.9550 [0.9540-0.9560] | 0.9485 [0.9466-0.9504] | 0.9824 | %98.90 | %1.10 | 0.9775 | 0.622 |
| Dogrusal-Olmayan ML Bazi (scikit-learn MLP) [26] | 0.9548 [0.9538-0.9559] | 0.9578 [0.9561-0.9594] | 0.9822 | %98.06 | %1.94 | 0.9792 | 0.641 |
| Ablasyon: Ortam Girdisiz AstrocyteX-Net v3.0 | 0.6613 [0.6595-0.6630] | 0.6464 [0.6416-0.6510] | 0.5161 | %36.89 | %63.11 | 0.6837 | 8.296 |
| AstrocyteX-Net v3.0 (Tam Model) | 0.9894 [0.9889-0.9899] | 0.9899 [0.9891-0.9907] | 0.9958 | %99.53 | %0.47 | 0.9956 | 0.144 |
| Canli Dagitim (0.20 Kural + 0.80 v3.0 + Guvenlik Tabani) | 0.9751 [0.9744-0.9759] | 0.9696 [0.9680-0.9712] | 0.9901 | %97.00 | %3.00 | 0.9904 | 0.368 |

### 3.2. Yalnizca 31 Fizyolojik Saglik Parametresi (Adil Uzay Ortami Ablasyonu)

| Sistem | MCC [%95 GA] | Macro-F1 [%95 GA] | QWK | Kritik Recall | Under-Triage | MAE (puan) |
|---|---|---|---|---|---|---|
| Kural Bazli (Referans Araliklari) | 0.8588 [0.8564-0.8609] | 0.7809 [0.7754-0.7868] | 0.9306 | %72.86 | %27.14 | 2.061 |
| Ablasyon: Ortam Girdisiz AstrocyteX-Net v3.0 | 0.9040 [0.9022-0.9059] | 0.8625 [0.8578-0.8674] | 0.9546 | %83.63 | %16.37 | 1.363 |
| Dogrusal ML Bazi (scikit-learn Ridge) [26] | 0.9358 [0.9343-0.9372] | 0.9141 [0.9109-0.9170] | 0.9713 | %97.92 | %2.08 | 0.837 |
| Dogrusal-Olmayan ML Bazi (scikit-learn MLP) [26] | 0.9374 [0.9359-0.9389] | 0.9329 [0.9302-0.9356] | 0.9718 | %96.12 | %3.88 | 0.833 |
| AstrocyteX-Net v3.0 (Tam Model) | 0.9850 [0.9843-0.9857] | 0.9839 [0.9825-0.9852] | 0.9932 | %98.95 | %1.05 | 0.185 |
| Canli Dagitim (0.20 Kural + 0.80 v3.0 + Guvenlik Tabani) | 0.9640 [0.9630-0.9652] | 0.9461 [0.9432-0.9491] | 0.9834 | %93.19 | %6.81 | 0.497 |

## 4. AstroVoiceNet v1.0 Bagimsiz Ses Analizi Test Sonuclari (1.200 Test Ses Kaydi)

| Sistem | Durum MCC [%95 GA] | Durum Macro-F1 | Risk MAE [%95 GA] | Risk RMSE | Risk QWK | Kritik Recall |
|---|---|---|---|---|---|---|
| Naif cogunluk / Ortalama Bazi | 0.0000 [0.0000-0.0000] | 0.0974 | 26.185 [25.436-26.973] | 29.538 | 0.0000 | %0.00 |
| Kural Bazli Akustik Esikleme | 0.3623 [0.3278-0.3955] | 0.3864 | 25.926 [25.026-26.765] | 30.303 | 0.1289 | %0.26 |
| Lojistik Regresyon + Ridge [26] | 0.9968 [0.9925-1.0000] | 0.9975 | 10.471 [9.980-10.965] | 13.423 | 0.7958 | %57.85 |
| HistGradientBoosting (scikit-learn) [26] | 0.9828 [0.9750-0.9904] | 0.9852 | 4.009 [3.757-4.262] | 5.983 | 0.9531 | %90.58 |
| RandomForest Toplulugu (220 Agac) [26] | 0.9710 [0.9599-0.9808] | 0.9753 | 3.276 [3.110-3.461] | 4.502 | 0.9656 | %93.46 |
| Ablasyon: Sadece 42 Prozodik Ozellik (Spektrogramsiz) | 0.9861 [0.9785-0.9935] | 0.9854 | 3.334 [3.157-3.501] | 4.504 | 0.9655 | %92.67 |
| AstroVoiceNet v1.0 (Log-Mel Conv-SE-Attn + ProsodyGate) | 0.9957 [0.9914-0.9989] | 0.9963 | 3.168 [3.027-3.296] | 3.948 | 0.9674 | %93.19 |

### 4.1. Cinsiyet ve Kabin Gurultusu Dilim Analizi

| Alt Grup Dilimi | n | Durum MCC | Durum Macro-F1 | Ses Risk MAE |
|---|---|---|---|---|
| Kadin Astronot Konusmacilar | 580 | 0.9933 | 0.9935 | 3.283 puan |
| Erkek Astronot Konusmacilar | 620 | 0.9979 | 0.9984 | 3.059 puan |
| Yuksek Kabin Gurultusu (SNR <= 22 dB) | 420 | 0.9938 | 0.9947 | 3.073 puan |
| Nominal Kabin Akustigi (SNR > 22 dB) | 780 | 0.9967 | 0.9971 | 3.219 puan |

## 5. Kullanilan Kutuphaneler, Catilar ve Yazilim Programlari

| Yazilim / Kutuphane | Surum | Lisans | AstrocyteX Mimarisindeki Gorevi | Atif |
|---|---|---|---|---|
| Python CPython Runtime | 3.13.7 | PSF-2.0 | Ana çalışma zamanı, RIFF PCM ses çözümleme (wave), eşzamanlı veri üretimi. | - |
| PyTorch (torch) + Torch CUDA | 2.8.0+cu126 (CUDA 12.6) | BSD-3-Clause | AstrocyteX-Net v3.0 ve AstroVoiceNet v1.0 derin sinir ağları, otomatik türev, GPU eğitimi, MC-Dropout Bayesçi çıkarım. | [25] |
| NVIDIA CUDA Donanım Hızlandırıcı | CUDA 12.6 / NVIDIA GeForce RTX 4050 Laptop GPU | NVIDIA EULA | 42.000 mürettebat-günü + 4.000 ses kaydı paralel tensör eğitimi ve 1.000x özyükleme (bootstrap) değerlendirmesi. | [25] |
| NumPy | 2.1.2 | BSD-3-Clause | 43-parametreli vektörize simülasyon, Hann pencereli STFT, 48-bant Mel filtre bankası ve matris tabanlı MCC/QWK hesabı. | [27] |
| SciPy | 1.16.2 | BSD-3-Clause | Dijital sinyal işleme, polifaz yeniden örnekleme (resample_poly), DCT-II MFCC çıkarımı, Klatt formant rezonatör filtreleri ve EKG R-tepe algılama. | [15, 27] |
| scikit-learn | 1.8.0 | BSD-3-Clause | Ridge, çok katmanlı algılayıcı (MLPRegressor), RandomForest ve HistGradientBoosting karşılaştırma modelleri; MCC, Macro-F1 ve QWK doğrulaması. | [21, 22, 26] |
| pandas | 2.3.3 | BSD-3-Clause | Tablosal klinik/EKG/ses CSV verilerinin ayrıştırılması ve yapılandırılmış dışa aktarım. | [27] |
| FastAPI & Starlette | 0.162.7 / 1.10.0 | MIT / BSD-3 | Asenkron REST API (/assess, /voice/analyze, /voice/sample, /ecg/*, /chat) ve statik Web SPA sunumu. | [28] |
| Pydantic v2 | 2.14.3 | MIT | Girdi/çıktı şema doğrulaması (ReadingIn, Assessment, AdvisorOutput, user_role ve voice alanları). | [28] |
| Uvicorn | 0.54.0 | BSD-3-Clause | Yüksek başarım ASGI uygulama sunucusu. | [28] |
| Google GenAI SDK (google-genai) | 3.3.0 | Apache-2.0 | Gemini tabanlı rol-duyarlı ('Ben doktorum' / 'Doktor değilim') yapılandırılmış JSON karar destek ve sohbet motoru. | [28] |
| SQLite3 Yerleşik Veritabanı | 3.50.4 | Public Domain | Astronot değerlendirme geçmişi, hafifletme görevleri, XP, rütbe ve rozet durumunun ACID kalıcılığı. | - |
| HTML5 / CSS3 / ES2022 + Web Audio API | W3C Standard | W3C | Sıfır-emoji karanlık mor Samanyolu arayüzü, tarayıcı içi 16 kHz PCM .wav mikrofon kaydı, çoklu ses kodek çözümü ve canlı spektrogram çizimi. | [28] |
| Streamlit & Plotly.py | 1.53.0 / 6.5.2 | Apache-2.0 / MIT | İkincil etkileşimli telemetri konsolu, gösterge kadranları ve zaman serisi grafikleri. | [28] |
| Matplotlib | 3.10.6 | PSF-based | Koyu mor temalı karışıklık matrisi, eğitim yakınsama eğrisi, olay dilim analizi ve ses modeli değerlendirme grafikleri. | [28] |
| python-docx | 1.2.0 | MIT | Bu teknik özet ve performans değerlendirme raporunun otomatik Word (.docx) formatında üretilmesi. | - |
| Requests & python-dotenv | 2.32.5 / 1.2.4 | Apache-2.0 / BSD-3 | REST istemci iletişimi, EKG servis köprüsü ve ortam değişkeni yönetimi. | - |
| pytest & HTTPX | 9.2.0 / 0.28.1 | MIT / BSD-3 | Birim, entegrasyon, FastAPI TestClient, ses analizi, doktor/sivil rol ayrımı ve sıfır-emoji denetim testleri. | - |
| TEKNOFEST V23 EKG Sistemi | OmniNet Apex-Titan v23 | Yerel Paket | 12-derivasyonlu 10-sınıflı kardiyak aritmi ve QTc analizi (çevrimdışı tarama yedekli). | [17] |

## 6. Akademik ve Yazilim Kaynakcasi (Citations)

- **[1]** T. Dao and A. Gu, 'Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality (Mamba-2),' Proc. 41st International Conference on Machine Learning (ICML), PMLR vol. 235, pp. 10041-10071, 2024.
- **[2]** S. Bai, J. Z. Kolter, and V. Koltun, 'An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling (TCN),' arXiv preprint arXiv:1803.01271, 2018.
- **[3]** B. Lim, S. O. Arik, N. Loeff, and T. Pfister, 'Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting,' International Journal of Forecasting, vol. 37, no. 4, pp. 1748-1764, 2021.
- **[4]** A. Jaegle et al., 'Perceiver: General Perception with Iterative Attention,' Proc. International Conference on Machine Learning (ICML), PMLR vol. 139, pp. 4651-4664, 2021.
- **[5]** A. Vaswani et al., 'Attention Is All You Need,' Advances in Neural Information Processing Systems (NeurIPS), vol. 30, pp. 5998-6008, 2017.
- **[6]** E. Perez, F. Strub, H. de Vries, V. Dumoulin, and A. Courville, 'FiLM: Visual Reasoning with a General Conditioning Layer,' Proc. AAAI Conference on Artificial Intelligence, vol. 32, no. 1, 2018.
- **[7]** N. Shazeer, 'GLU Variants Improve Transformer (SwiGLU),' arXiv preprint arXiv:2002.05202, 2020.
- **[8]** R. T. Q. Chen, Y. Rubanova, J. Bettencourt, and D. K. Duvenaud, 'Neural Ordinary Differential Equations,' Advances in Neural Information Processing Systems (NeurIPS), vol. 31, pp. 6571-6583, 2018.
- **[9]** M. Raissi, P. Perdikaris, and G. E. Karniadakis, 'Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations,' Journal of Computational Physics, vol. 378, pp. 686-707, 2019.
- **[10]** C. Blundell, J. Cornebise, K. Kavukcuoglu, and D. Wierstra, 'Weight Uncertainty in Neural Networks (Bayes by Backprop),' Proc. International Conference on Machine Learning (ICML), PMLR vol. 37, pp. 1613-1622, 2015.
- **[11]** Y. Gal and Z. Ghahramani, 'Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning,' Proc. International Conference on Machine Learning (ICML), PMLR vol. 48, pp. 1050-1059, 2016.
- **[12]** A. Kendall and Y. Gal, 'What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?,' Advances in Neural Information Processing Systems (NeurIPS), vol. 30, pp. 5574-5584, 2017.
- **[13]** F. Eyben et al., 'The Geneva Minimalistic Acoustic Parameter Set (GeMAPS) for Voice Research and Affective Computing,' IEEE Transactions on Affective Computing, vol. 7, no. 2, pp. 190-202, 2016.
- **[14]** J. Hu, L. Shen, and G. Sun, 'Squeeze-and-Excitation Networks,' Proc. IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 7132-7141, 2018.
- **[15]** D. H. Klatt, 'Software for a cascade/parallel formant synthesizer,' Journal of the Acoustical Society of America (JASA), vol. 67, no. 3, pp. 971-995, 1980.
- **[16]** J. A. Russell, 'A circumplex model of affect,' Journal of Personality and Social Psychology, vol. 39, no. 6, pp. 1161-1178, 1980.
- **[17]** P. E. McSharry, G. D. Clifford, L. Tarassenko, and L. A. Smith, 'A dynamical model for generating synthetic electrocardiogram signals,' IEEE Transactions on Biomedical Engineering, vol. 50, no. 3, pp. 289-294, 2003.
- **[18]** W. Cao, V. Mirjalili, and S. Raschka, 'Rank consistent ordinal regression for neural networks with application to age estimation (CORAL),' Pattern Recognition Letters, vol. 140, pp. 325-331, 2020.
- **[19]** I. Loshchilov and F. Hutter, 'Decoupled Weight Decay Regularization (AdamW),' Proc. International Conference on Learning Representations (ICLR), 2019; L. N. Smith and N. Topin, 'Super-Convergence: Very Fast Training of Neural Networks Using Large Learning Rates (OneCycleLR),' SPIE Defense + Commercial Sensing, 2019.
- **[20]** B. T. Polyak and A. B. Juditsky, 'Acceleration of stochastic approximation by averaging (EMA),' SIAM Journal on Control and Optimization, vol. 30, no. 4, pp. 838-855, 1992.
- **[21]** B. W. Matthews, 'Comparison of the predicted and observed secondary structure of T4 phage lysozyme,' Biochimica et Biophysica Acta, vol. 405, no. 2, pp. 442-451, 1975; J. Gorodkin, 'Comparing two K-category assignments by a K-category correlation coefficient,' Computational Biology and Chemistry, vol. 28, pp. 367-374, 2004; D. Chicco and G. Jurman, 'The advantages of the Matthews correlation coefficient (MCC) over F1 score and accuracy in binary classification evaluation,' BMC Genomics, vol. 21, art. 6, 2020.
- **[22]** J. Cohen, 'Weighted kappa: Nominal scale agreement provision for scaled disagreement or partial credit,' Psychological Bulletin, vol. 70, no. 4, pp. 213-220, 1968.
- **[23]** National Aeronautics and Space Administration (NASA), 'NASA Space Flight Human-System Standard, Volume 1: Crew Health (NASA-STD-3001 Vol. 1) & Volume 2: Human Factors, Habitability, and Environmental Health (NASA-STD-3001 Vol. 2),' Washington, DC, Technical Standards, 2022-2023.
- **[24]** Z. S. Patel et al., 'Red risks for a journey to the red planet: The highest priority human health risks for a mission to Mars,' npj Microgravity, vol. 6, art. 33, 2020; C. Zeitlin et al., 'Measurements of energetic particle radiation in transit to Mars on the Mars Science Laboratory,' Science, vol. 340, pp. 1080-1084, 2013; D. M. Hassler et al., 'Mars surface radiation environment measured with the MSL Curiosity rover,' Science, vol. 343, 1244797, 2014.
- **[25]** A. Paszke et al., 'PyTorch: An Imperative Style, High-Performance Deep Learning Library,' Advances in Neural Information Processing Systems (NeurIPS), vol. 32, pp. 8024-8035, 2019.
- **[26]** F. Pedregosa et al., 'Scikit-learn: Machine Learning in Python,' Journal of Machine Learning Research (JMLR), vol. 12, pp. 2825-2830, 2011.
- **[27]** C. R. Harris et al., 'Array programming with NumPy,' Nature, vol. 585, pp. 357-362, 2020; P. Virtanen et al., 'SciPy 1.0: Fundamental Algorithms for Scientific Computing in Python,' Nature Methods, vol. 17, pp. 261-272, 2020; W. McKinney, 'Data Structures for Statistical Computing in Python (pandas),' Proc. 9th Python in Science Conf., pp. 56-61, 2010.
- **[28]** J. D. Hunter, 'Matplotlib: A 2D Graphics Environment,' Computing in Science & Engineering, vol. 9, no. 3, pp. 90-95, 2007; Plotly Technologies Inc., 'Collaborative data science (Plotly.py),' Montreal, QC, 2015; S. Ramirez, 'FastAPI: modern, high-performance web framework for building APIs with Python,' https://fastapi.tiangolo.com, 2019; Google DeepMind / Google Cloud, 'Google GenAI Python SDK (google-genai) & Gemini Models Technical Report,' 2024-2025; W3C Audio Working Group, 'Web Audio API Recommendation,' World Wide Web Consortium, 2021.

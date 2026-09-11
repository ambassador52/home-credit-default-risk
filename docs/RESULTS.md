# Final Sonuçlar

## Model karşılaştırması — validation

| Model | ROC-AUC | Average Precision | F1 @ 0.10 |
|---|---:|---:|---:|
| LightGBM | 0.757545 | 0.246972 | 0.281696 |
| XGBoost | 0.757086 | 0.248459 | 0.282323 |
| Logistic Regression | 0.748655 | 0.231783 | 0.275373 |
| Random Forest | 0.739175 | 0.231369 | 0.274030 |
| Dummy | 0.500000 | 0.080734 | 0.000000 |

Model seçimi / tuning değerlendirme metriği ROC-AUC'tır.

## LightGBM tuning

- RandomizedSearchCV
- 30 hiperparametre kombinasyonu
- 5-fold StratifiedKFold
- Toplam 150 fit
- Eğitim bölümü üzerinde çalıştırıldı
- En iyi CV ROC-AUC: **0.758392**

## Validation eşik karşılaştırması

| Eşik | Precision | Recall | F1 | TP | FP | TN | FN | İşaretlenen |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.05 | 0.129338 | 0.841568 | 0.224217 | 3134 | 21097 | 21306 | 590 | 24231 |
| 0.10 | 0.186126 | 0.584318 | **0.282322** | 2176 | 9515 | 32888 | 1548 | 11691 |
| 0.50 | 0.600000 | 0.020945 | 0.040477 | 78 | 52 | 42351 | 3646 | 130 |

Bu üç aday arasından F1 en yüksek olduğu için **0.10** seçildi. Bu eşik ekonomik optimum olarak yorumlanmamalıdır.

## Oran özellikleri — ablation

| Oran özellikleri | ROC-AUC | Average Precision |
|---|---:|---:|
| Yok | 0.757352 | 0.248439 |
| Var | 0.758891 | 0.250619 |

ROC-AUC farkı: **+0.001539**  
Average Precision farkı: **+0.002181**

## Final test

| Metrik | Değer |
|---|---:|
| ROC-AUC | **0.762784** |
| Average Precision | **0.249359** |
| Precision | 0.193096 |
| Recall | 0.606874 |
| F1 | 0.292974 |
| TP | 2260 |
| FP | 9444 |
| TN | 32959 |
| FN | 1464 |
| İşaretlenen başvuru | 11704 |

Test, model ve eşik sabitlendikten sonra değerlendirilmiştir.

## Feature importance

Gain tabanlı ilk üç özellik `EXT_SOURCE_3`, `EXT_SOURCE_2` ve `EXT_SOURCE_1` olmuştur. Oluşturulan `CREDIT_GOODS_RATIO` özelliği de ilk 10 içine girmiştir.

Gain importance, nedensellik veya etkinin yönünü göstermez.

# Final Sonuçlar

## Model karşılaştırması — validation

| Model | ROC-AUC | Average Precision | F1 @ 0.10 |
|---|---:|---:|---:|
| LightGBM | 0.757545 | 0.246972 | 0.281696 |
| XGBoost | 0.757086 | 0.248481 | 0.282323 |
| Logistic Regression | 0.748655 | 0.231819 | 0.275409 |
| Random Forest | 0.739175 | 0.231351 | 0.274025 |
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

İlk aşamada optimize edilmiş LightGBM modelinin validation olasılıkları `0.05`, `0.10` ve `0.50` referans eşiklerinde karşılaştırılmıştır.

| Eşik | Precision | Recall | F1 | TP | FP | TN | FN | İşaretlenen |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.05 | 0.129338 | 0.841568 | 0.224217 | 3134 | 21097 | 21306 | 590 | 24231 |
| 0.10 | 0.186126 | 0.584318 | **0.282322** | 2176 | 9515 | 32888 | 1548 | 11691 |
| 0.50 | 0.600000 | 0.020945 | 0.040477 | 78 | 52 | 42351 | 3646 | 130 |

Bu üç referans eşik arasında en yüksek F1 değeri `0.10` eşiğinde elde edilmiştir.

### Genişletilmiş eşik analizi

Ek bir duyarlılık analizi olarak validation verisinde `0.01–0.50` aralığı `0.01` adımlarla taranmıştır. Bu taramada F1 skorunu maksimum yapan eşik `0.15` olarak bulunmuştur.

| Eşik | Precision | Recall | F1 | TP | FP | TN | FN | İşaretlenen |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.10 | 0.186126 | 0.584318 | 0.282322 | 2176 | 9515 | 32888 | 1548 | 11691 |
| **0.15** | **0.245820** | 0.422395 | **0.310777** | 1573 | **4826** | 37577 | 2151 | 6399 |

`0.15` eşiği F1 açısından daha yüksek sonuç vermektedir. Bununla birlikte `0.10` eşiği 2176 gerçek `TARGET=1` başvuruyu yakalarken, `0.15` eşiği 1573 gerçek `TARGET=1` başvuruyu yakalamaktadır.

`0.15`, `0.10` ile karşılaştırıldığında 4689 yanlış alarmı azaltırken 603 ek gerçek `TARGET=1` başvurunun kaçırılmasına yol açmaktadır.

Bu projede riskli olarak işaretlenen başvurular otomatik kredi reddi olarak yorumlanmamaktadır. Model çıktısı, daha ayrıntılı incelenecek başvuruların önceliklendirilmesi amacıyla ele alınmıştır. Bu kullanım varsayımı altında daha yüksek recall ile daha fazla gerçek riskli başvuruyu yakalayan `0.10`, operasyonel karar eşiği olarak korunmuştur.

`0.15` ise validation verisinde F1 açısından en iyi istatistiksel alternatif olarak raporlanmıştır. Finansal maliyet bilgisi bulunmadığından `0.10` veya `0.15` için ekonomik olarak optimum eşik iddiasında bulunulmamaktadır.

## Oran özellikleri — ablation

| Oran özellikleri | ROC-AUC | Average Precision |
|---|---:|---:|
| Yok | 0.757352 | 0.248439 |
| Var | 0.758891 | 0.250619 |

ROC-AUC farkı: **+0.001539**  
Average Precision farkı: **+0.002181**

## Final test

Model, hiperparametreler ve operasyonel `0.10` eşiği sabitlendikten sonra test bölümü bir kez değerlendirilmiştir.

| Metrik | Değer |
|---|---:|
| ROC-AUC | **0.762784** |
| Average Precision | **0.249359** |
| Precision @ 0.10 | 0.193096 |
| Recall @ 0.10 | 0.606874 |
| F1 @ 0.10 | 0.292974 |
| TP | 2260 |
| FP | 9444 |
| TN | 32959 |
| FN | 1464 |
| İşaretlenen başvuru | 11704 |

Test bölümü model, hiperparametreler ve operasyonel eşik sabitlendikten sonra değerlendirilmiştir; model veya eşik seçimi amacıyla kullanılmamıştır.

## Feature importance

Gain tabanlı ilk üç özellik `EXT_SOURCE_3`, `EXT_SOURCE_2` ve `EXT_SOURCE_1` olmuştur. Oluşturulan `CREDIT_GOODS_RATIO` özelliği de ilk 10 içine girmiştir.

Gain importance, nedensellik veya etkinin yönünü göstermez.

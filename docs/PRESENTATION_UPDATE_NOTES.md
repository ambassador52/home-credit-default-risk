# Sunum Güncelleme Notları

Mevcut sunumdaki eski model sonuçlarını aşağıdaki doğrulanmış sonuçlarla değiştirin.

## Model karşılaştırması slaytı

- Logistic Regression ROC-AUC: 0.748655
- Random Forest ROC-AUC: 0.739175
- XGBoost ROC-AUC: 0.757086
- LightGBM ROC-AUC: 0.757545
- Dummy ROC-AUC: 0.500000

Ana mesaj: ROC-AUC açısından validation üzerinde en iyi temel model LightGBM'dir.

## Hyperparameter tuning slaytı

- RandomizedSearchCV
- 30 kombinasyon
- 5-fold StratifiedKFold
- Scoring: ROC-AUC
- Toplam 150 fit
- En iyi CV ROC-AUC: 0.758392

Tuning yalnızca train bölümünde yapılmıştır.

## Threshold slaytı

İlk referans karşılaştırmasında 0.05, 0.10 ve 0.50 eşikleri değerlendirilmiştir.

### İlk eşik karşılaştırması

| Eşik | Precision | Recall | F1 |
|---:|---:|---:|---:|
| 0.05 | 0.129338 | 0.841568 | 0.224217 |
| 0.10 | 0.186126 | 0.584318 | 0.282322 |
| 0.50 | 0.600000 | 0.020945 | 0.040477 |

Bu üç referans eşik arasında en yüksek F1 değeri 0.10 eşiğinde elde edilmiştir.

### Genişletilmiş eşik analizi

Ek bir duyarlılık analizi olarak validation verisinde 0.01–0.50 aralığı 0.01 adımlarla taranmıştır.

F1 açısından en yüksek sonuç 0.15 eşiğinde elde edilmiştir.

0.10:
- Precision: 0.186126
- Recall: 0.584318
- F1: 0.282322
- TP: 2176
- FP: 9515
- FN: 1548
- İşaretlenen: 11,691

0.15:
- Precision: 0.245820
- Recall: 0.422395
- F1: 0.310777
- TP: 1573
- FP: 4826
- FN: 2151
- İşaretlenen: 6,399

0.15 eşiği, 0.10'a kıyasla 4,689 yanlış alarmı azaltırken 603 ek gerçek TARGET=1 başvurunun kaçırılmasına yol açmaktadır.

Bu projede riskli olarak işaretlenen başvurular otomatik kredi reddi olarak değerlendirilmemektedir. Model çıktısı, daha ayrıntılı incelenecek başvuruların önceliklendirilmesi amacıyla ele alınmıştır.

Bu kullanım varsayımı altında daha yüksek recall ile daha fazla gerçek riskli başvuruyu yakalayan 0.10, operasyonel karar eşiği olarak korunmuştur. 0.15 ise validation verisinde F1 açısından en iyi istatistiksel alternatif olarak raporlanmıştır.

Finansal maliyet bilgisi bulunmadığından 0.10 veya 0.15 ekonomik olarak optimum eşik şeklinde yorumlanmamalıdır.

## Feature engineering / ablation slaytı

- Oransız validation ROC-AUC: 0.757352
- Oranlı validation ROC-AUC: 0.758891
- Fark: +0.001539
- Oransız AP: 0.248439
- Oranlı AP: 0.250619
- Fark: +0.002181

Ana mesaj: Dört oran özelliği kontrollü karşılaştırmada küçük fakat pozitif katkı sağlamıştır.

## Feature importance slaytı

`reports/figures/lightgbm_gain_top10.png` grafiğini kullanın.

İlk üç:
1. EXT_SOURCE_3
2. EXT_SOURCE_2
3. EXT_SOURCE_1

Oluşturulan CREDIT_GOODS_RATIO ilk 10 içerisindedir.

Not: Gain importance nedensellik veya ilişkinin yönü değildir.

## Final test slaytı

Operasyonel 0.10 eşiği ile final test sonuçları:

- ROC-AUC: **0.762784**
- Average Precision: **0.249359**
- Precision @ 0.10: 0.193096
- Recall @ 0.10: 0.606874
- F1 @ 0.10: 0.292974
- TP: 2260
- FP: 9444
- TN: 32959
- FN: 1464
- İşaretlenen: 11,704

Ana mesaj: Operasyonel 0.10 eşiğinde model test bölümündeki gerçek pozitiflerin yaklaşık %60.7'sini yakalamaktadır; riskli olarak işaretlenen başvuruların yaklaşık %19.3'ü gerçekten TARGET=1'dir.

Test bölümü model veya eşik seçimi amacıyla kullanılmamıştır.

## Sunumda kaçınılacak ifadeler

- “ROC-AUC = %76 doğruluk” demeyin.
- “0.10 F1 açısından tüm olası eşiklerin en iyisidir” demeyin.
- “0.10 en kârlı eşiktir” demeyin.
- “0.15 bankacılık açısından kesin en iyi eşiktir” demeyin.
- Riskli işaretlenen başvuruların otomatik olarak reddedildiğini söylemeyin.
- “Bu özellik temerrüde neden olur” demeyin.
- `EMPLOYED_BIRTH_RATIO` için “toplam çalışma hayatı / yaş” demeyin; mevcut işte geçen süre / yaş olarak açıklayın.

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
Karşılaştırılan eşikler: 0.05, 0.10, 0.50.

Seçilen eşik: **0.10**
- Validation precision: 0.186126
- Validation recall: 0.584318
- Validation F1: 0.282322
- Validation işaretlenen: 11,691

Ana mesaj: Üç aday eşik içinde F1 en yüksek olduğu için 0.10 seçilmiştir. Finansal maliyet verisi olmadığı için bu eşik ekonomik optimum olarak sunulmamalıdır.

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
- ROC-AUC: **0.762784**
- Average Precision: **0.249359**
- Precision: 0.193096
- Recall: 0.606874
- F1: 0.292974
- TP: 2260
- FP: 9444
- TN: 32959
- FN: 1464
- İşaretlenen: 11,704

Ana mesaj: Seçilen eşikte model gerçek pozitiflerin yaklaşık %60.7'sini yakalamaktadır; işaretlenen örneklerin yaklaşık %19.3'ü pozitiftir.

## Sunumda kaçınılacak ifadeler
- “ROC-AUC = %76 doğruluk” demeyin.
- “0.10 en kârlı eşiktir” demeyin.
- “Bu özellik temerrüde neden olur” demeyin.
- `EMPLOYED_BIRTH_RATIO` için “toplam çalışma hayatı / yaş” demeyin; mevcut işte geçen süre / yaş olarak açıklayın.

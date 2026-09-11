# Veri

Bu repoya ham Home Credit verisi eklenmemiştir.

Çalıştırmak için `application_train.csv` dosyasını aşağıdaki konumlardan birine koyun:

```text
proje_koku/application_train.csv
```

veya

```text
proje_koku/data/raw/application_train.csv
```

Kod, beklenen ana eğitim verisini `(307511, 122)` boyutuyla doğrular.

`TARGET` hedef değişkendir ve `SK_ID_CURR` yalnızca kimlik olarak kullanılır; ikisi de model girdisine dahil edilmez.

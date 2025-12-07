# X-Ray Insights: Proje ve Model Eğitim Rehberi

Bu rehber, **X-Ray Insights** projesinin nasıl çalıştığını, hangi teknolojilerin kullanıldığını ve model eğitim sürecinin her adımını detaylı bir şekilde açıklar.

## 1. Projenin Amacı
Bu projenin temel amacı, göğüs röntgeni (X-Ray) görüntülerini analiz ederek hastanın durumunu üç sınıftan birine ayırmaktır:
1.  **NORMAL:** Sağlıklı akciğer.
2.  **PNEUMONIA (Zatürre):** Bakteriyel veya viral zatürre.
3.  **COVID-19:** Koronavirüs kaynaklı enfeksiyon.

## 2. Veri Seti ve Veri Yükleme (`src/data_loader.py`)

Modelin başarısı veriye bağlıdır. Bu projede "Hibrit" bir veri seti yaklaşımı kullandık.

### Kullanılan Kaynaklar
Tek bir veri seti yeterli olmadığı için (özellikle COVID-19 görüntüleri az olduğu için) birden fazla kaynağı birleştirdik:
*   **Paul Mooney (Kaggle):** Normal ve Zatürre görüntüleri için ana kaynak.
*   **Tawsifur Rahman (Kaggle):** COVID-19 görüntüleri için ana kaynak.
*   **Prashant268 & NIH & Bachrr:** Ekstra COVID ve Normal görüntüleri için destekleyici kaynaklar.

### Veri İşleme Adımları
1.  **Görüntü Okuma:** Görüntüler diskten okunur ve RGB formatına çevrilir.
2.  **Veri Artırma (Data Augmentation):** Eğitim setindeki görüntülerin sayısı yapay olarak artırılmaz, ancak her epoch'ta görüntüler rastgele değiştirilir (döndürme, parlaklık değişimi, gürültü ekleme). Bu, modelin ezber yapmasını (overfitting) engeller.
    *   *Kütüphane:* `albumentations`
3.  **Normalizasyon:** Görüntü piksel değerleri, ImageNet standartlarına (ortalama ve standart sapma) göre ölçeklenir.

### Eğitim ve Doğrulama Ayrımı
*   **Eğitim Seti (Train):** Modelin ağırlıklarını güncellemek için kullandığı büyük veri havuzu (~25.000 görüntü).
*   **Doğrulama Seti (Validation):** Modelin eğitim sırasında ne kadar iyi öğrendiğini test etmek için ayrılan kısım (~3.156 görüntü). Model bu görüntüleri *asla* eğitim için kullanmaz.

## 3. Model Mimarisi (`src/model.py`)

Sıfırdan bir model eğitmek yerine **Transfer Learning (Transfer Öğrenme)** yöntemini kullandık.

### Kullanılan Model: ResNet50
*   **Nedir?** Microsoft tarafından geliştirilen, 50 katmanlı derin bir sinir ağıdır.
*   **Neden Seçtik?** ResNet50, ImageNet veri setindeki milyonlarca görüntü üzerinde önceden eğitilmiştir. Yani kenarları, şekilleri ve dokuları tanımayı zaten bilir.
*   **Uyarlama:** Modelin son katmanını (1000 sınıf tahmin eden kısmı) çıkardık ve yerine bizim 3 sınıfımızı (Normal, Pnömoni, COVID) tahmin eden yeni bir katman ekledik.

## 4. Eğitim Süreci (`src/train.py`)

Eğitim döngüsü şu şekilde işler:

### A. İleri Yayılım (Forward Pass)
Model bir grup (batch) röntgen görüntüsünü alır ve tahminlerde bulunur.

### B. Kayıp Hesaplama (Loss Calculation)
Modelin tahmini ile gerçek etiket karşılaştırılır.
*   **Kullanılan Fonksiyon:** `CrossEntropyLoss`
*   **Weighted Loss (Ağırlıklı Kayıp):** Veri setimizde Normal görüntüleri çok fazla, COVID görüntüleri daha azdı. Bu dengesizliği çözmek için COVID hatasına daha yüksek ceza (ağırlık) veren bir sistem kurduk.

### C. Geri Yayılım (Backward Pass & Optimization)
Hesaplanan hata geriye doğru yayılır ve modelin ağırlıkları güncellenir.
*   **Optimizasyon Algoritması:** `Adam` (Adaptive Moment Estimation). Hızlı ve etkili öğrenme sağlar.

### D. Öğrenme Oranı Planlayıcı (Scheduler)
Eğer modelin başarısı (Validation Loss) bir süre iyileşmezse, öğrenme oranı (`learning_rate`) otomatik olarak düşürülür. Bu, modelin daha hassas ayar yapmasını sağlar.

### E. Erken Durdurma (Early Stopping)
Eğer model belirli bir süre (örneğin 5 epoch) boyunca gelişme göstermezse, eğitim otomatik olarak durdurulur. Bu, zaman kaybını ve aşırı öğrenmeyi (overfitting) önler.

## 5. Değerlendirme Metrikleri

Modelin başarısını ölçmek için şu metrikleri kullanıyoruz:
*   **Accuracy (Doğruluk):** Toplam doğru tahmin oranı.
*   **Precision (Kesinlik):** "COVID" dediğimiz hastaların kaçı gerçekten COVID?
*   **Recall (Duyarlılık):** Gerçek COVID hastalarının kaçını yakalayabildik? (Tıbbi projelerde en kritik metrik budur).
*   **F1-Score:** Precision ve Recall'un harmonik ortalaması.

## 6. Proje Klasör Yapısı
*   `src/`: Tüm kaynak kodlar (veri yükleyici, model, eğitim döngüsü).
*   `notebooks/`: Veri analizi ve denemeler için Jupyter not defterleri.
*   `models/`: Eğitilen model dosyalarının (`.pth`) kaydedildiği yer.
*   `mlruns/`: MLflow tarafından tutulan deney kayıtları.
*   `web_app/`: Modeli canlı olarak kullanmak için yapılan arayüz (Streamlit).

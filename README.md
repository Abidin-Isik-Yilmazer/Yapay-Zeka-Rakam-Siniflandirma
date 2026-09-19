# 🧠 Yapay Zeka Tabanlı El Yazısı Rakam Sınıflandırma Sistemi (CNN)

Bu proje, MNIST veri setini kullanarak 0'dan 9'a kadar olan el yazısı rakamları yüksek doğrulukla (%99+) tahmin edebilen, 
uçtan uca (end-to-end) tasarlanmış bir makine öğrenmesi ve görüntü işleme sistemidir. 
Sistem; veri ön işleme, model eğitimi ve gerçek zamanlı tahmin (inference) süreçlerini modüler bir Nesne Yönelimli Programlama (OOP) mimarisiyle tek bir masaüstü uygulamasında birleştirir.

## 🛠️ Kullanılan Teknolojiler ve Bağımlılıklar

*   **Programlama Dili:** Python 3.12
*   **Yapay Zeka & Derin Öğrenme:** TensorFlow, Keras (CNN Mimarisi)
*   **Görüntü İşleme & Matris Veri Yapıları:** Pillow (PIL), NumPy
*   **Grafik Kullanıcı Arayüzü (GUI):** Tkinter, CustomTkinter

## 📐 Mimari Tasarım ve Dosya Yapısı

Proje, yazılım mühendisliği prensiplerine uygun olarak "Gereksinimlerin Ayrılması" (Separation of Concerns) kuralı gözetilerek dört ana modüle ayrılmıştır:

*   **`data_loader.py` (Data Pipeline):** Ham MNIST veri setinin sisteme entegre edilmesi, tensör boyutlandırması (reshaping) ve piksel normalizasyonu işlemlerini yöneten sınıf yapısıdır.
*   **`model.py` (Core AI Engine):** Evrişimli Sinir Ağı (CNN) mimarisinin inşa edildiği, katmanların (layers) tanımlandığı ve optimizasyon parametrelerinin derlendiği sınıftır.
*   **`gui_app.py` (User Interface & Asynchronous Tasks):** Kullanıcı etkileşimlerini yöneten, CustomTkinter tabanlı modern arayüz modülüdür. Arayüz kilitlenmelerini önlemek amacıyla model eğitimi için eşzamanlılık (Threading) mekanizmalarını kullanır.
*   **`main.py` (Orchestration Layer):** Tüm alt modülleri bir araya getiren ve uygulamanın yaşam döngüsünü başlatan ana yürütücü (entry point) dosyasıdır.

## ⚙️ Model Mimarisi (CNN)

Derin öğrenme modeli, ardışık (Sequential) katmanlardan oluşan özellik çıkarımı (feature extraction) odaklı bir yapıya sahiptir:

1.  **Evrişim Katmanları (Conv2D):** Uzamsal özellikleri (kenar, köşe, eğri) yakalamak için sırasıyla 32 ve 64 filtreli (3x3 kernel) iki adet evrişim katmanı kullanılmıştır.
2.  **Ortaklama Katmanları (MaxPooling2D):** Boyut indirgeme ve hesaplama maliyetini düşürmek amacıyla 2x2 matrislerle en belirgin özellikleri (downsampling) süzer.
3.  **Düzleştirme ve Tam Bağlı Ağ (Flatten & Dense):** İki boyutlu özellik haritaları tek boyutlu vektöre dönüştürülür ve 128 nöronlu (ReLU aktivasyonlu) gizli katmana aktarılır.
4.  **Çıkış Katmanı:** 10 sınıfı (0-9 rakamlar) temsil eden ve olasılık dağılımı üreten Softmax aktivasyonlu çıkış katmanı.
5.  **Optimizasyon:** Kayıp fonksiyonu olarak `sparse_categorical_crossentropy` ve ağırlık güncellemeleri için `Adam` optimizasyon algoritması tercih edilmiştir.

## ✨ Temel Özellikler

*   **Gerçek Zamanlı Çıkarım (Real-time Inference):** Kullanıcının tuval (canvas) üzerine çizdiği vektörel grafikler anlık olarak 28x28 piksellik matrislere dönüştürülür, normalize edilir ve eğitilmiş modele beslenerek mili-saniyeler içinde tahmin üretilir.
*   **Asenkron Model Eğitimi:** Arayüz üzerinden başlatılan model eğitimi (training) bağımsız bir iş parçacığına (Thread) atanarak ana programın (Main Thread) kilitlenmesi engellenmiştir.
*   **Kalıcı Model Durumu:** Eğitilen modelin ağırlıkları ve topolojisi `.h5` formatında diske kaydedilir, böylece uygulama her başlatıldığında sıfırdan eğitim gerektirmez.
*   **Gelişmiş UX/UI:** Standart arayüzler yerine CustomTkinter kütüphanesi kullanılarak koyu tema (Dark Mode) destekli, estetik ve duyarlı (responsive) bir masaüstü uygulaması tasarlanmıştır.

## 🚀 Kurulum ve Çalıştırma Yönergesi

1. Depoyu Klonlayın:
```bash
git clone https://github.com/Abidin-Isik-Yilmazer/Yapay-Zeka-Rakam-Siniflandirma.git
cd Yapay-Zeka-Rakam-Siniflandirma
```

2. Gerekli Kütüphaneleri Yükleyin:
```bash
pip install tensorflow numpy pillow customtkinter
```

3. Uygulamayı Başlatın:
```bash
python main.py
```

4. Kullanım Senaryosu:
* Sistemde halihazırda eğitilmiş bir model yoksa, sol taraftaki işlem menüsünden "1 - Modeli Eğit" butonuna basarak eğitim sürecini başlatın.
* Eğitim tamamlandıktan sonra ana ekrandaki siyah tuval üzerine farenizle net ve kalın çizgilerle bir rakam çizin.
* "Tahmin Et" butonuna tıklayarak modelin çıkarım sonucunu ve güven skorunu görüntüleyin. Ekranı sıfırlamak için "Temizle" butonunu kullanabilirsiniz.

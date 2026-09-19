from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense


class CnnModel:
    """MNIST rakamlarını tanımak için Evrişimli Sinir Ağı (CNN) mimarisi."""

    def __init__(self):
        # Sıralı (Sequential) model yapısını başlatıyoruz
        self.model = Sequential()

    def build_model(self):
        # 1. Evrişim (Convolution) Katmanı
        self.model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))

        # 2. Ortaklama (Pooling) Katmanı
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        # Daha iyi öğrenme için 2. Evrişim ve Ortaklama Katmanları
        self.model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        # 3. Düzleştirme (Flatten) Katmanı
        self.model.add(Flatten())

        # 4. Tam Bağlı (Dense/Fully Connected) Gizli Katman
        self.model.add(Dense(128, activation='relu'))

        # 5. Çıkış Katmanı (0-9 arası 10 rakam olduğu için 10 nöron)
        self.model.add(Dense(10, activation='softmax'))

        return self.model

    def compile_model(self):
        # Modeli eğitilmeye hazır hale getirme (Derleme)
        self.model.compile(optimizer='adam',
                           loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])
        print("CNN Modeli başarıyla oluşturuldu ve derlendi.")
from tensorflow.keras.datasets import mnist

class MnistDataLoader:
    """MNIST veri setini yüklemek ve CNN için ön işleme tabi tutmak için kullanılan sınıf."""

    def __init__(self):
        # Eğitim ve test verilerini tutacak nitelikler (attributes)
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None

    def load_and_preprocess(self):
        # 1. Keras üzerinden ham veri setini indir ve yükle
        (self.x_train, self.y_train), (self.x_test, self.y_test) = mnist.load_data()

        # 2. CNN mimarisi için veriyi yeniden boyutlandır (Reshape)
        self.x_train = self.x_train.reshape((self.x_train.shape[0], 28, 28, 1))
        self.x_test = self.x_test.reshape((self.x_test.shape[0], 28, 28, 1))

        # 3. Piksel değerlerini normalize et (0-1 aralığına çek)
        self.x_train = self.x_train.astype('float32') / 255.0
        self.x_test = self.x_test.astype('float32') / 255.0

        print("Veri başarıyla yüklendi ve CNN için hazır hale getirildi.")
        return (self.x_train, self.y_train), (self.x_test, self.y_test)
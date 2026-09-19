import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageDraw
import numpy as np
import os
import threading
from tensorflow.keras.models import load_model
from data_loader import MnistDataLoader
from model import CnnModel


class ModernDigitApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Yapay Zeka Rakam Tanıma - Modern Arayüz")
        self.root.geometry("750x450")
        self.root.configure(bg="#2b2d42")  # Modern koyu arka plan
        self.root.resizable(False, False)

        # A. SOL MENÜ PANELİ (Sidebar)
        self.menu_frame = tk.Frame(self.root, bg="#1a1b26", width=220)
        self.menu_frame.pack(side="left", fill="y")
        self.menu_frame.pack_propagate(False)  # Panelin boyutunu sabitle

        # Menü Başlığı
        tk.Label(self.menu_frame, text="ANA MENÜ", fg="white", bg="#1a1b26",
                 font=("Segoe UI", 16, "bold")).pack(pady=(30, 20))

        # Menü Butonları (Flat Design)
        self.btn_train = self.create_menu_button("1. Modeli Eğit", self.start_training, "#4CAF50")
        self.btn_clear = self.create_menu_button("2. Ekranı Temizle", self.clear_canvas, "#f1fa8c", )
        self.btn_exit = self.create_menu_button("3. Çıkış Yap", self.root.quit, "#ff5555")

        # B. SAĞ İÇERİK PANELİ (Çizim Ekranı)
        self.content_frame = tk.Frame(self.root, bg="#2b2d42")
        self.content_frame.pack(side="right", fill="both", expand=True)

        self.build_draw_screen()

        # C. BAŞLANGIÇ KONTROLLERİ
        self.model = None
        if os.path.exists("mnist_cnn_model.h5"):
            self.model = load_model("mnist_cnn_model.h5")
            self.lbl_info.config(text="Durum: Model hazır, çizim yapabilirsiniz.", fg="#50fa7b")
        else:
            self.lbl_info.config(text="Durum: Model yüklü değil! Önce 1. seçeneği kullanın.", fg="#ff5555")

    def create_menu_button(self, text, command, hover_color, text_color="white"):
        """Estetik menü butonları oluşturmak için yardımcı fonksiyon"""
        btn = tk.Button(self.menu_frame, text=text, command=command, bg="#414868", fg=text_color,
                        font=("Segoe UI", 11, "bold"), relief="flat", cursor="hand2")
        btn.pack(fill="x", padx=15, pady=8, ipady=5)
        # Butonun üzerine gelince renk değiştirme efekti (Hover)
        btn.bind("<Enter>", lambda e: btn.config(bg=hover_color))
        btn.bind("<Leave>", lambda e: btn.config(bg="#414868"))
        return btn

    def build_draw_screen(self):
        """Sağ taraftaki çizim ve tahmin alanını inşa eder"""
        tk.Label(self.content_frame, text="Fareye basılı tutarak bir rakam çizin",
                 fg="white", bg="#2b2d42", font=("Segoe UI", 12)).pack(pady=(20, 10))

        # Çizim Alanı (Canvas) - Etrafında modern bir mavi çerçeve ile
        self.canvas_width, self.canvas_height = 280, 280
        self.canvas = tk.Canvas(self.content_frame, width=self.canvas_width, height=self.canvas_height,
                                bg='black', cursor="crosshair", highlightthickness=2, highlightbackground="#8be9fd")
        self.canvas.pack()
        self.canvas.bind("<B1-Motion>", self.paint)

        # Arka plandaki görünmez PIL resmi
        self.image = Image.new("L", (self.canvas_width, self.canvas_height), color=0)
        self.draw = ImageDraw.Draw(self.image)

        # Tahmin Butonu
        tk.Button(self.content_frame, text="TAHMİN ET", command=self.predict_digit, bg="#8be9fd", fg="black",
                  font=("Segoe UI", 14, "bold"), relief="flat", cursor="hand2").pack(pady=15, ipadx=20)

        # Sonuç Yazısı
        self.lbl_result = tk.Label(self.content_frame, text="", font=("Segoe UI", 18, "bold"), bg="#2b2d42",
                                   fg="#f1fa8c")
        self.lbl_result.pack()

        # En alttaki bilgi/durum çubuğu
        self.lbl_info = tk.Label(self.content_frame, text="Sistem başlatılıyor...", font=("Segoe UI", 10), bg="#2b2d42")
        self.lbl_info.pack(side="bottom", pady=15)

    def paint(self, event):
        """Fırça ile çizim yapma fonksiyonu"""
        brush = 12
        x1, y1, x2, y2 = (event.x - brush), (event.y - brush), (event.x + brush), (event.y + brush)
        self.canvas.create_oval(x1, y1, x2, y2, fill="white", outline="white")
        self.draw.ellipse([x1, y1, x2, y2], fill=255)

    def clear_canvas(self):
        """Ekranı sıfırlama"""
        self.canvas.delete("all")
        self.image = Image.new("L", (self.canvas_width, self.canvas_height), color=0)
        self.draw = ImageDraw.Draw(self.image)
        self.lbl_result.config(text="")

    def predict_digit(self):
        """Çizimi modele gönderip tahmin alma"""
        if self.model is None:
            messagebox.showwarning("Uyarı", "Model henüz eğitilmedi! Soldaki menüden '1. Modeli Eğit' butonuna basın.")
            return

        img_array = np.array(self.image.resize((28, 28))).astype('float32') / 255.0
        img_array = img_array.reshape(1, 28, 28, 1)

        predictions = self.model.predict(img_array)
        predicted_digit = np.argmax(predictions)
        confidence = np.max(predictions) * 100

        self.lbl_result.config(text=f"Sonuç: {predicted_digit} (Eminlik: %{confidence:.1f})")

    # --- MODEL EĞİTİM BÖLÜMÜ (Arka Planda Çalıştırma) ---
    def start_training(self):
        """Eğitimi başlatır ama arayüzün donmaması için arka plan işlemi (Thread) açar"""
        self.lbl_info.config(text="Durum: Model eğitiliyor, lütfen bekleyin... (Bu işlem biraz sürebilir)",
                             fg="#ffb86c")
        self.btn_train.config(state="disabled")  # Eğitilirken butonu kilitle

        # Threading (İş Parçacığı) Kullanımı: Eğitim arayüzü dondurmasın diye
        threading.Thread(target=self.train_process, daemon=True).start()

    def train_process(self):
        """Modelin eğitim aşamaları"""
        try:
            data_loader = MnistDataLoader()
            (x_train, y_train), (x_test, y_test) = data_loader.load_and_preprocess()

            cnn = CnnModel()
            cnn.build_model()
            cnn.compile_model()
            cnn.model.fit(x_train, y_train, epochs=5, batch_size=64, validation_split=0.2)

            cnn.model.save("mnist_cnn_model.h5")

            # İşlem bitince arayüzü güncellemesi için ana thread'e mesaj gönder
            self.root.after(0, self.training_finished)
        except Exception as e:
            print("Hata:", e)

    def training_finished(self):
        """Eğitim bitince arayüzü normale döndürür"""
        self.model = load_model("mnist_cnn_model.h5")
        self.lbl_info.config(text="Durum: Eğitim tamamlandı! Artık çizim yapabilirsiniz.", fg="#50fa7b")
        self.btn_train.config(state="normal")
        messagebox.showinfo("Başarılı", "Yapay Zeka modeli başarıyla eğitildi ve kaydedildi!")
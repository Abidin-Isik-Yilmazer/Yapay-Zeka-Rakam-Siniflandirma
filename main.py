import tkinter as tk
from gui_app import ModernDigitApp


def main():
    # Tkinter ana penceresini oluştur
    root = tk.Tk()

    # Modern arayüz sınıfımıza pencereyi gönder
    app = ModernDigitApp(root)

    # Uygulamayı sonsuz döngüde ekranda tut
    root.mainloop()


if __name__ == "__main__":
    main()
import numpy as np
import cv2
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
import os

# ============================
# Model Ayarları
# ============================
alphabets = u"ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-' '"
num_of_characters = len(alphabets) + 1  # CTC için +1


# Harf - Sayı dönüşümleri
def decode_prediction(pred):
    out = K.ctc_decode(pred, input_length=np.ones(pred.shape[0]) * pred.shape[1], greedy=True)[0][0]
    out = K.get_value(out)
    result_text = ""
    for ch in out[0]:
        if ch == -1:
            break
        result_text += alphabets[ch]
    return result_text

def preprocess_canvas(img):
    """
    Tuvalden alınan resmi model girişine uygun şekilde
    64x256 boyutuna getirir.
    Yazının yüksekliği 15px civarında olacak şekilde yukarıdan ve aşağıdan boşluk bırakır.
    """

    # Önce resmi 64x256'ya küçült
    resized = cv2.resize(img, (256, 64), interpolation=cv2.INTER_AREA)


    # Siyah pikselleri (yazı) bul
    _, thresh = cv2.threshold(resized, 200, 255, cv2.THRESH_BINARY_INV)

    # Yazının yatayda sınırlarını bul (siyah alan)
    coords = cv2.findNonZero(thresh)

    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        # h yükseklik yazı kısmı

        # Eğer yükseklik 15'ten büyükse küçült
        if h > 15:
            scale_factor = 15 / h
            new_h = 15
            new_w = int(w * scale_factor)
            roi = resized[y:y+h, x:x+w]
            roi_resized = cv2.resize(roi, (new_w, new_h), interpolation=cv2.INTER_AREA)

            # Yeni boş resim oluştur 15x256
            new_img = 255 * np.ones((15, 256), dtype=np.uint8)

            # ortala yatayda
            start_x = (256 - new_w) // 2
            new_img[:, start_x:start_x+new_w] = roi_resized

            # Sonra 64x256'ya üstten ve alttan boşlukla genişlet (yani siyah alan 15px)
            final_img = 255 * np.ones((64, 256), dtype=np.uint8)
            start_y = (64 - 15) // 2
            final_img[start_y:start_y+15, :] = new_img

            # Döndürmek gerekiyorsa döndür (senin preprocess fonksiyonunda var)
            return cv2.rotate(final_img, cv2.ROTATE_90_CLOCKWISE)

        else:
            # 15px'in altındaysa direkt 64x256 yap ve döndür
            final_img = 255 * np.ones((64, 256), dtype=np.uint8)
            start_y = (64 - h) // 2
            final_img[start_y:start_y+h, :] = resized[y:y+h, :]
            return cv2.rotate(final_img, cv2.ROTATE_90_CLOCKWISE)

    else:
        # Yazı bulunmazsa direkt döndürülmüş boş resim döndür
        final_img = 255 * np.ones((64, 256), dtype=np.uint8)
        return cv2.rotate(final_img, cv2.ROTATE_90_CLOCKWISE)

# Preprocessing
def preprocess(img):

    (h, w) = img.shape
    final_img = np.ones([64, 256]) * 255
    if w > 256:
        img = img[:, :256]
    if h > 64:
        img = img[:64, :]
    final_img[:h, :w] = img
    return cv2.rotate(final_img, cv2.ROTATE_90_CLOCKWISE)


# Model Yükle
model_path = "CRNNFold4Model.keras"  # model yolunu kendi sistemine göre değiştir
model = load_model(model_path)


# ============================
# GUI Sınıfı
# ============================
class CRNNApp:
    def __init__(self, master):
        self.master = master
        self.master.title("El Yazısı Tanıma - CRNN")

        # Ana çerçeve
        main_frame = tk.Frame(master)
        main_frame.pack(padx=10, pady=10)

        # Sol taraf: Tuval ve butonlar
        left_frame = tk.Frame(main_frame)
        left_frame.grid(row=0, column=0, sticky="n")

        self.canvas_width = 512
        self.canvas_height = 128

        self.canvas = tk.Canvas(left_frame, width=self.canvas_width, height=self.canvas_height, bg='white',
                                cursor="cross")
        self.canvas.pack()
        self.canvas.bind("<B1-Motion>", self.draw)

        button_frame = tk.Frame(left_frame)
        button_frame.pack(pady=10)

        tk.Button(button_frame, text="Tuvalden Tahmin", command=self.predict_canvas).grid(row=0, column=0, padx=5)
        tk.Button(button_frame, text="Dosyadan Tahmin", command=self.predict_file).grid(row=0, column=1, padx=5)
        tk.Button(button_frame, text="Temizle", command=self.clear_canvas).grid(row=0, column=2, padx=5)

        # Sağ taraf: Dosyadan seçilen resim gösterme ve sonuç
        right_frame = tk.Frame(main_frame)
        right_frame.grid(row=0, column=1, padx=20, sticky="n")

        self.image_label = tk.Label(right_frame, text="Dosyadan Seçilen Resim", font=("Arial", 12))
        self.image_label.pack(pady=5)

        self.display_width = 256
        self.display_height = 64
        self.display_canvas = tk.Canvas(right_frame, width=self.display_width, height=self.display_height,
                                        bg='lightgray')
        self.display_canvas.pack()

        self.result_label = tk.Label(right_frame, text="Sonuç: ", font=("Arial", 16))
        self.result_label.pack(pady=15)

        # Tuval için resim (PIL)
        self.image = Image.new("L", (self.canvas_width, self.canvas_height), color=255)
        self.draw_image = self.image.copy()

    def draw(self, event):
        x, y = event.x, event.y
        r = 3  # çizim kalınlığı
        self.canvas.create_oval(x - r, y - r, x + r, y + r, fill='black', outline='black')

        # PIL resim üzerine çizim
        draw = Image.fromarray(np.array(self.image))
        cv_img = np.array(self.image)
        cv2.circle(cv_img, (x, y), r, (0,), -1)
        self.image = Image.fromarray(cv_img)

    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (self.canvas_width, self.canvas_height), color=255)
        self.result_label.config(text="Sonuç: ")
        self.display_canvas.delete("all")
        self.display_canvas.create_text(self.display_width // 2, self.display_height // 2,
                                        text="Dosyadan Seçilen Resim", fill="black")

    def predict_canvas(self):
        img_array = np.array(self.image)
        preprocessed = preprocess_canvas(img_array) / 255.0
        preprocessed = np.expand_dims(preprocessed, axis=(0, -1))

        pred = model.predict(preprocessed)
        decoded = decode_prediction(pred)
        self.result_label.config(text=f"Sonuç: {decoded}")

    def predict_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Resim Dosyaları", "*.png;*.jpg;*.jpeg;*.bmp")])
        if not file_path or not os.path.exists(file_path):
            messagebox.showerror("Hata", "Dosya bulunamadı veya okunamadı.")
            return

        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            messagebox.showerror("Hata", "Resim yüklenemedi.")
            return

        # Resmi GUI'de göster
        img_for_show = cv2.resize(img, (self.display_width, self.display_height))
        img_pil = Image.fromarray(img_for_show)
        img_tk = ImageTk.PhotoImage(img_pil)

        self.display_canvas.image = img_tk
        self.display_canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)

        # Tahmin
        preprocessed = preprocess(img) / 255.0
        preprocessed = np.expand_dims(preprocessed, axis=(0, -1))

        pred = model.predict(preprocessed)
        decoded = decode_prediction(pred)
        self.result_label.config(text=f"Sonuç: {decoded}")


# ============================
# Ana Program
# ============================
if __name__ == "__main__":
    root = tk.Tk()
    app = CRNNApp(root)
    root.mainloop()

import tkinter as tk
from tkinter import filedialog, Label, Button
from PIL import Image, ImageTk
import cv2
import numpy as np
from ultralytics import YOLO

# YOLO modelini yükle
model = YOLO("best.pt")  # YOLO modelinizi burada belirtebilirsiniz.

# Kamera durumu ve global değişkenler
camera_on = False
selected_image = None
cap = None

# Tkinter ana pencere
root = tk.Tk()
root.title("YOLO Nesne Algılama Arayüzü")
root.geometry("900x700")
root.config(bg="#2C3E50")

# Arka plan resmi
background_image = Image.open("background_Foto.jpg")  # Yüklediğiniz arka plan resmini kullanın
background_image = background_image.resize((900, 700))
bg_image_tk = ImageTk.PhotoImage(background_image)
bg_label = Label(root, image=bg_image_tk)
bg_label.place(relwidth=1, relheight=1)

# Kamera görüntüsü veya yüklenen resim için bir Label
display_label = Label(root)
display_label.place(relx=0.05, rely=0.1, relwidth=0.8, relheight=0.7)

# Resim yükleme ve algılama
def load_image():
    global selected_image
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.png;*.jpeg")])
    if file_path:
        selected_image = cv2.imread(file_path)
        detect_objects(selected_image)

def detect_objects(image):
    # Resmi 640x640 boyutlarına yeniden boyutlandır
    image_resized = cv2.resize(image, (640, 640))

    # YOLO ile algılama
    results = model(image_resized)
    annotated_image = results[0].plot()  # Algılanan nesnelerle anotasyon ekler

    # Algılanan sınıf ve doğruluk oranlarını konsola yazdır
    for i, box in enumerate(results[0].boxes):
        class_name = results[0].names[int(box.cls)]  # Sınıf adı
        confidence = box.conf  # Doğruluk oranı
        print(f"Class: {class_name}, Confidence: {confidence:.2f}")

    # Algılanan resmi Tkinter'de göster
    image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(image_rgb)
    img_tk = ImageTk.PhotoImage(img_pil)
    display_label.imgtk = img_tk
    display_label.config(image=img_tk)

# Çıkış fonksiyonu
def exit_program():
    global cap
    if cap:
        cap.release()
    root.quit()

# Butonlar
button_load = Button(root, text="Upload Image", command=load_image, font=("Helvetica", 14),
                     bg="#3498DB", fg="white", activebackground="#2980B9", width=20, height=2)
button_load.place(relx=0.3, rely=0.85)

button_exit = Button(root, text="Exit", command=exit_program, font=("Helvetica", 14),
                     bg="#E74C3C", fg="white", activebackground="#C0392B", width=20, height=2)
button_exit.place(relx=0.7, rely=0.85)

# Tkinter ana döngüsü
root.mainloop()

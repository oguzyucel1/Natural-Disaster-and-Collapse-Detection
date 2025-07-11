import tkinter as tk
from tkinter import filedialog, Label, Button, ttk
from PIL import Image, ImageTk
import cv2
from ultralytics import YOLO

# YOLO modelini yükle
model = YOLO("Collapse.pt")  # YOLO modelinizi burada belirtebilirsiniz.

# Kamera durumu ve global değişkenler
camera_on = False
selected_image = None
cap = None

# Tespit edilen sınıflar için sayaçlar
class_counters = {"Global Collapse": 0, "Non-Collapse": 0, "Partial Collapse": 0}

# Tkinter ana pencere
root = tk.Tk()
root.title("YOLO11 Fire_Smoke_Detection")
root.geometry("1200x700")
root.config(bg="#2C3E50")

# Arka plan resmini yüklemek için
background_image_path = "Collapse_BackgoundFoto.jpeg"  # Resmin tam yolu
background_image = Image.open(background_image_path)
background_image = background_image.resize((1200, 700))  # Boyutunu pencereye uygun hale getir
bg_image_tk = ImageTk.PhotoImage(background_image)

# Resmi Tkinter penceresinde görüntüle
bg_label = Label(root, image=bg_image_tk)
bg_label.place(relwidth=1, relheight=1)

# Kamera görüntüsü veya yüklenen resim için bir Label
display_label = Label(root)
display_label.place(relx=0.04, rely=0.04, relwidth=0.7, relheight=0.7)

# Tespit edilen sınıflar için tablo
tree = ttk.Treeview(root, columns=("Class", "Count"), show="headings", height=10)
tree.heading("Class", text="Class")
tree.heading("Count", text="Count")
tree.column("Class", width=150)
tree.column("Count", width=100)
tree.place(relx=0.75, rely=0.2)

# Tabloyu başlangıçta doldur
for cls in class_counters:
    tree.insert("", "end", values=(cls, class_counters[cls]))

# Tabloyu güncelleme fonksiyonu
def update_table():
    for i, cls in enumerate(class_counters):
        tree.item(tree.get_children()[i], values=(cls, class_counters[cls]))

# Kamera başlatma
def toggle_camera():
    global camera_on, cap
    camera_on = not camera_on
    if camera_on:
        cap = cv2.VideoCapture(0)  # Kamerayı başlat
        show_webcam()
        button_camera.config(text="Turn Off Camera")
    else:
        if cap:
            cap.release()
        display_label.config(image="")
        button_camera.config(text="Turn On Camera")

def show_webcam():
    global cap
    if camera_on:
        ret, frame = cap.read()
        if ret:
            # YOLO ile algılama
            results = model(frame)
            annotated_frame = results[0].plot()  # Algılanan nesnelerle anotasyon ekler

            # Tespit edilen sınıfları güncelle
            detected_classes = [result.names[cls] for cls in results[0].boxes.cls.cpu().numpy().astype(int)]
            for cls in detected_classes:
                if cls in class_counters:
                    class_counters[cls] += 1

            # Tabloyu güncelle
            update_table()

            # Görüntüyü Tkinter'de göster
            frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(frame_rgb)
            img_tk = ImageTk.PhotoImage(img_pil)
            display_label.imgtk = img_tk
            display_label.config(image=img_tk)

        # Kamera akışını yenile
        display_label.after(10, show_webcam)

# Resim yükleme ve algılama
def load_image():
    global selected_image
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.png;*.jpeg")])
    if file_path:
        selected_image = cv2.imread(file_path)

        # Resmi 640x640 boyutuna getir
        resized_image = cv2.resize(selected_image, (640, 640))

        detect_objects(resized_image)

def detect_objects(image):
    # YOLO ile algılama
    results = model(image)
    annotated_image = results[0].plot()  # Algılanan nesnelerle anotasyon ekler

    # Tespit edilen sınıfları güncelle
    detected_classes = [result.names[cls] for cls in results[0].boxes.cls.cpu().numpy().astype(int)]
    for cls in detected_classes:
        if cls in class_counters:
            class_counters[cls] += 1

    # Tabloyu güncelle
    update_table()

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
button_camera = Button(root, text="Turn On Camera", command=toggle_camera, font=("Helvetica", 14),
                       bg="#3498DB", fg="white", activebackground="#2980B9", width=20, height=2)
button_camera.place(relx=0.05, rely=0.85)

button_load = Button(root, text="Upload Image", command=load_image, font=("Helvetica", 14),
                     bg="#3498DB", fg="white", activebackground="#2980B9", width=20, height=2)
button_load.place(relx=0.3, rely=0.85)

button_exit = Button(root, text="Exit", command=exit_program, font=("Helvetica", 14),
                     bg="#E74C3C", fg="white", activebackground="#C0392B", width=20, height=2)
button_exit.place(relx=0.7, rely=0.85)

# Tkinter ana döngüsü
root.mainloop()

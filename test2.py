import tkinter as tk
from tkinter import Label, Button, filedialog
from PIL import Image, ImageOps, ImageTk
import numpy as np
from keras.models import load_model
import cv2
from tensorflow.keras.models import load_model


# Load the model and class labels
model = load_model("keras_Model.h5", compile=False)
class_names = open("labels.txt", "r").readlines()

# Global variables to control the camera and image status
camera_on = False
selected_image = None
is_image_uploaded = False

# Create the main Tkinter window
root = tk.Tk()
root.title("Natural Disaster Detection with Keras Model")
root.geometry("900x650")
root.config(bg="#2C3E50")  # Dark blue background color

# Load the background image
background_image = Image.open("Kapak_foto.jpeg")  # Replace with your image file
background_image = background_image.resize((900, 650))  # Resize to fit the window
bg_image_tk = ImageTk.PhotoImage(background_image)

# Create a Label to display the background image
bg_label = Label(root, image=bg_image_tk)
bg_label.place(relwidth=1, relheight=1)  # Stretch the image to cover the window

# Create webcam label (placed on top of the background)
webcam_label = Label(root)
webcam_label.place(relx=0.05, rely=0.15, relwidth=0.6, relheight=0.6)

# Styling for buttons and labels
button_style = {
    'font': ("Helvetica", 14),
    'bg': "#3498DB",  # Blue
    'fg': "white",
    'activebackground': "#2980B9",
    'bd': 0,
    'relief': "flat",
    'width': 20,
    'height': 2
}

label_style = {
    'font': ("Helvetica", 14),
    'bg': "white",
    'fg': "#34495E",
    'width': 20,
    'height': 2
}

# Styling for class and confidence labels with color change
class_label_style = {
    'font': ("Helvetica", 14),
    'bg': "lightblue",
    'fg': "#2ECC71",  # Green color
    'width': 20,
    'height': 2
}

confidence_label_style = {
    'font': ("Helvetica", 14),
    'bg': "lightyellow",
    'fg': "#E74C3C",  # Red color
    'width': 20,
    'height': 2
}

# Toggle camera button function
def toggle_camera():
    global camera_on
    camera_on = not camera_on
    if camera_on:
        button_camera.config(text="Turn Off Camera")
        show_webcam()
    else:
        button_camera.config(text="Turn On Camera")
        webcam_label.config(image='')

button_camera = Button(root, text="Turn On Camera", command=toggle_camera, **button_style)

# Load and classify an image
def load_image():
    global selected_image, is_image_uploaded
    file_path = filedialog.askopenfilename()
    if file_path:
        selected_image = Image.open(file_path).convert("RGB")
        is_image_uploaded = True
        classify_image(selected_image)  # Classify the uploaded image
        update_image_on_webcam(selected_image)  # Display the image in webcam area

def update_image_on_webcam(image):
    img = image.resize((int(900 * 0.6), int(650 * 0.6)))
    imgtk = ImageTk.PhotoImage(image=img)
    webcam_label.imgtk = imgtk
    webcam_label.config(image=imgtk)

button_load_image = Button(root, text="Upload Image", command=load_image, **button_style)

# Labels to display class name and confidence score
class_name_label = Label(root, text="Class: ", **class_label_style)
class_name_label.place(relx=0.7, rely=0.1)

confidence_score_label = Label(root, text="Confidence Score: ", **confidence_label_style)
confidence_score_label.place(relx=0.7, rely=0.3)

# Exit button
def exit_program():
    cap.release()  # Release the webcam
    root.quit()

exit_button = Button(root, text="Exit", command=exit_program, **button_style)
exit_button.place(relx=0.7, rely=0.85)  # Move Exit button to the left

# Webcam functionality
cap = cv2.VideoCapture(0)

def show_webcam():
    if camera_on:
        ret, frame = cap.read()
        if ret and not is_image_uploaded:  # Only show webcam if no image is uploaded
            frame_resized = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)

            # Classification
            image_array = np.asarray(frame_resized, dtype=np.float32).reshape(1, 224, 224, 3)
            image_array = (image_array / 127.5) - 1  # Normalize

            prediction = model.predict(image_array)
            index = np.argmax(prediction)
            class_name = class_names[index].strip()
            confidence_score = prediction[0][index]

            # Update class name and confidence score
            class_name_label.config(text=f"Class: {class_name[2:]}")
            confidence_score_label.config(text=f"Confidence Score: {confidence_score:.2f}")

            # Display the webcam frame in the Tkinter window
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(frame_rgb)
            img_pil = img_pil.resize((int(900 * 0.6), int(650 * 0.6)))
            imgtk = ImageTk.PhotoImage(image=img_pil)
            webcam_label.imgtk = imgtk
            webcam_label.config(image=imgtk)

        webcam_label.after(50, show_webcam)  # Refresh webcam feed every 50 ms

# Function to classify uploaded image
def classify_image(image):
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]
    class_name_label.config(text=f"Class: {class_name[2:]}")
    confidence_score_label.config(text=f"Confidence Score: {confidence_score:.2f}")

# Function to toggle between Webcam and Folder mode
def toggle_mode(mode):
    global camera_on, is_image_uploaded
    if mode == "Folder":
        if camera_on:
            toggle_camera()  # Turn off the camera
        button_camera.place_forget()
        button_load_image.place(relx=0.3, rely=0.85)  # Show the upload button
        class_name_label.config(text="Class: ")
        confidence_score_label.config(text="Confidence Score: ")

    elif mode == "Webcam":
        if is_image_uploaded:
            webcam_label.config(image='')  # Clear uploaded image
            is_image_uploaded = False
        button_load_image.place_forget()
        button_camera.place(relx=0.05, rely=0.85)  # Show the camera button
        toggle_camera()  # Turn on the camera
        class_name_label.config(text="Class: ")
        confidence_score_label.config(text="Confidence Score: ")

# Load the icons for Webcam and Folder modes
webcam_icon = Image.open("webcam_icon.jpg").resize((50, 50))
folder_icon = Image.open("folder_icon.jpeg").resize((50, 50))
webcam_icon_tk = ImageTk.PhotoImage(webcam_icon)
folder_icon_tk = ImageTk.PhotoImage(folder_icon)

# Create Buttons with icons for Webcam and Folder mode
button_webcam = Button(root, image=webcam_icon_tk, command=lambda: toggle_mode("Webcam"), bg="white", bd=0)
button_webcam.place(relx=0.3, rely=0.067)

button_folder = Button(root, image=folder_icon_tk, command=lambda: toggle_mode("Folder"), bg="white", bd=0)
button_folder.place(relx=0.4, rely=0.067)

# Start the Tkinter main loop
root.mainloop()

# Release camera when the program is closed
cap.release()
cv2.destroyAllWindows()

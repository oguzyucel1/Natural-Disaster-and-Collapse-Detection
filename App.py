import tkinter as tk
from tkinter import Label, Button, filedialog, Radiobutton, StringVar
from PIL import Image, ImageOps, ImageTk
import numpy as np
from keras.models import load_model
import cv2

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
root.geometry("800x600")


webcam_label = Label(root)
webcam_label.place(relwidth=0.7, relheight=0.7)

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

button_camera = Button(root, text="Turn On Camera", command=toggle_camera)

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
    img = image.resize((int(800 * 0.7), int(600 * 0.7)))
    imgtk = ImageTk.PhotoImage(image=img)
    webcam_label.imgtk = imgtk
    webcam_label.config(image=imgtk)

button_load_image = Button(root, text="Upload Image", command=load_image)

# Labels to display class name and confidence score
class_name_label = Label(root, text="Class: ", font=("Arial", 16))
class_name_label.place(relx=0.7, rely=0.)

confidence_score_label = Label(root, text="Confidence Score: ", font=("Arial", 16))
confidence_score_label.place(relx=0.7, rely=0.5)

# Exit button
def exit_program():
    root.quit()

exit_button = Button(root, text="Exit", command=exit_program)
exit_button.place(relx=0.85, rely=0.85)

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
            img_pil = img_pil.resize((int(800 * 0.7), int(600 * 0.7)))
            imgtk = ImageTk.PhotoImage(image=img_pil)
            webcam_label.imgtk = imgtk
            webcam_label.config(image=imgtk)

        webcam_label.after(10, show_webcam)  # Refresh webcam feed every 10 ms

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
def toggle_mode():
    global camera_on, is_image_uploaded
    mode = selected_mode.get()

    # If switching to Folder mode, stop the webcam and reset class and confidence
    if mode == "Folder":
        if camera_on:
            toggle_camera()  # Turn off the camera
        button_camera.place_forget()
        button_load_image.place(relx=0.3, rely=0.85)  # Show the upload button
        class_name_label.config(text="Class: ")
        confidence_score_label.config(text="Confidence Score: ")

    # If switching to Webcam mode, clear uploaded image and turn on the webcam
    elif mode == "Webcam":
        if is_image_uploaded:
            webcam_label.config(image='')  # Clear uploaded image
            is_image_uploaded = False
        button_load_image.place_forget()
        button_camera.place(relx=0.05, rely=0.85)  # Show the camera button
        toggle_camera()  # Turn on the camera
        class_name_label.config(text="Class: ")
        confidence_score_label.config(text="Confidence Score: ")

# Options for Webcam and Folder mode
selected_mode = StringVar(value="Webcam")

radiobutton_webcam = Radiobutton(root, text="Webcam", variable=selected_mode, value="Webcam", command=toggle_mode)
radiobutton_webcam.place(relx=0.01, rely=0.01)

radiobutton_folder = Radiobutton(root, text="Folder", variable=selected_mode, value="Folder", command=toggle_mode)
radiobutton_folder.place(relx=0.1, rely=0.01)

# Start the Tkinter main loop
root.mainloop()

# Release camera when the program is closed
cap.release()
cv2.destroyAllWindows()

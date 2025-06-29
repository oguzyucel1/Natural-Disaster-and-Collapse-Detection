from keras.models import load_model
import cv2
import numpy as np

np.set_printoptions(suppress=True)

model = load_model("keras_model.h5", compile=False)
class_names = open("labels.txt", "r").readlines()

camera = cv2.VideoCapture(0)

# Set the window size
cv2.namedWindow("Webcam Image", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Webcam Image", 800, 600)  # Set the desired window size

while True:
    ret, image = camera.read()
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
    cv2.imshow("Webcam Image", image)

    image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)
    image = (image / 127.5) - 1

    prediction = model.predict(image)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    print("Class:", class_name[2:], end="")
    print("Confidence Score:", str(np.round(confidence_score * 100))[:-2], "%")

    # Listen to the keyboard for presses
    keyboard_input = cv2.waitKey(1)

    # 27 is the ASCII for the esc key on your keyboard.
    # Use 'q' key to exit
    if keyboard_input == ord('q') or keyboard_input == 27:
        break

camera.release()
cv2.destroyAllWindows()

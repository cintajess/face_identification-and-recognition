import os
import cv2
import numpy as np
from random import shuffle
from tqdm import tqdm

def my_label(image_name):
    if "jecinta" in image_name:
        return "jecinta"

def generate_dataset():
    face_classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    def face_cropped(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, 1.3, 5)
       
        if len(faces) == 0:
            return None
        for (x, y, w, h) in faces:
            cropped_face = img[y:y+h, x:x+w]
            return cropped_face

    cap = cv2.VideoCapture(0)
    img_id = 0
    
    data = []
    for img in tqdm(os.listdir("data")):
        path = os.path.join("data", img)
        img_data = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img_data = cv2.resize(img_data, (50, 50))
        data.append([np.array(img_data), my_label(img)])

    shuffle(data)

    for img_id, (face, label) in enumerate(data):
        file_name_path = "data/jecinta." + str(img_id) + ".jpg"
        cv2.imwrite(file_name_path, face)
        cv2.putText(face, str(img_id), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
        print("Label for", file_name_path, "is:", label)  # Print the label
        if cv2.waitKey(1) == 13 or int(img_id) == 1000:
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Collecting samples is completed")

generate_dataset()

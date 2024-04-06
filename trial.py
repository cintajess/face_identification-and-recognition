import cv2
import numpy as np
from tensorflow.keras.models import load_model
import time

# Load the saved model
loaded_model = load_model("trained_model.h5")

# Load the Haar cascade classifier
face_classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Load the registration numbers and corresponjeding labels
registered_faces = {
    "jecinta": "jecinta"
}


def compare_faces(captured_face):
    # Resize the captured face image to the required size (50x50)
    resized_captured_face = cv2.resize(captured_face, (50, 50))
    
    # Convert the resized face image to grayscale
    resized_captured_face_gray = cv2.cvtColor(resized_captured_face, cv2.COLOR_BGR2GRAY)
    
    # Preprocess the image for model input
    resized_captured_face_gray = np.array(resized_captured_face_gray).reshape(-1, 50, 50, 1)
    
    # Perform prediction
    prediction = loaded_model.predict(resized_captured_face_gray)
    
    # Determine the label based on prediction
    label = "jecinta" if prediction > 0.5 else "other"
    
    return label


def capture_and_verify():
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        
        # Convert the captured frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Use the Haar cascade classifier to detect faces
        faces = face_classifier.detectMultiScale(gray, 1.3, 5)
        
        # Loop through the detected faces
        for (x, y, w, h) in faces:
            # Crop the detected face region
            captured_face = frame[y:y+h, x:x+w]
            
            # Compare the captured face with faces in the trained model
            label = compare_faces(captured_face)
            
            # If the captured face matches any face in the trained model
            if label == "jecinta":
                # Draw a rectangle around the detected face
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                
                # Display the captured frame with detected face
                cv2.imshow('Face Detection', frame)
                
                # Prompt for registration number
                registration_number = input("Face detected! Enter registration number: ")
                
                # Check if the registration number matches the registered number for the recognized face
                if registration_number == registered_faces[label]:
                    print("Verification successful: Face recognized as Jecinta.")
                else:
                    print("Verification failed: Registration number does not match the recognized face.")
                
                # Reset the frame and break from the loop
                frame = np.zeros_like(frame)
                break
        else:
            # If no face is recognized, directly output that the verification failed
            print("Verification failed: Face not recognized.")
        
        # Wait for 15 seconds before capturing the next picture
        time.sleep(5)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# Capture image and verify
capture_and_verify()

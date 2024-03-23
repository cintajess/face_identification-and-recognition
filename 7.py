import cv2
import numpy as np
from random import shuffle
from tqdm import tqdm
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import time

def capture_individual(person_name, num_images=100, delay=0, cap=None):
    if cap is None:
        cap = cv2.VideoCapture(0)
    face_classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    img_id = 0
    
    while img_id < num_images:
        ret, frame = cap.read()
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, 1.3, 5)
        
        for (x, y, w, h) in faces:
            captured_face = gray[y:y+h, x:x+w]
            captured_face = cv2.resize(captured_face, (50, 50))  # Resizing to 50x50
            
            img_id += 1
            file_name_path = f"data/{person_name}.{img_id}.jpg"
            label = person_name
            cv2.imwrite(file_name_path, captured_face)
            
            print("Label for", file_name_path, "is:", label)  # Print the label
            
            # Display the captured image
            cv2.imshow('Captured Image', captured_face)
            cv2.waitKey(1000)  # Wait for 1 second
            
            if img_id == num_images:
                break
            elif cv2.waitKey(1) == 13:
                break
                
        time.sleep(delay)  # Pause before capturing next image
        if cv2.waitKey(1) == 13:
            break

    if cap is None:
        cap.release()
        cv2.destroyAllWindows()
    print(f"Image capture for {person_name} is completed")

def generate_dataset():
    def my_label(image_name):
        if "jecinta" in image_name:
            return "jecinta"
        elif "diana" in image_name:
            return "diana"
        else:
            return "other"

    def face_cropped(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, 1.3, 5)
       
        if len(faces) == 0:
            return None
        for (x, y, w, h) in faces:
            cropped_face = img[y:y+h, x:x+w]
            return cropped_face

    face_classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    data = []
    
    cap = cv2.VideoCapture(0)  # Create a single VideoCapture object
    
    # Capture images for jecinta
    capture_individual("jecinta", num_images=100, cap=cap)

    # Introduce a delay before capturing images for the next individual
    time.sleep(10)  # Wait for 10 seconds before capturing images for the next individual

    # Capture images for diana
    capture_individual("diana", num_images=100, cap=cap)
    
    cap.release()  # Release the VideoCapture object after capturing images for both individuals

    for person_name in ["jecinta", "diana"]:
        img_id = 0
        while img_id < 100:
            ret, frame = cap.read()
            if face_cropped(frame) is not None:
                img_id += 1
                face = cv2.resize(face_cropped(frame), (50, 50))  # Resizing to 50x50
                face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                file_name_path = f"data/{person_name}.{img_id}.jpg"
                label = my_label(file_name_path)
                cv2.imwrite(file_name_path, face)
                cv2.putText(face, str(img_id), (20, 20), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
                data.append((face, label))
                print("Label for", file_name_path, "is:", label)  # Print the label

    cv2.destroyAllWindows()
    print("Image capture for jecinta and diana is completed")

    shuffle(data)  # Shuffle the data list
    shuffled_data = list(tqdm(data, desc="Shuffling data"))  # Use tqdm to visualize shuffling progress
    
    # Split data into training and testing sets
    train_data = shuffled_data[:int(len(shuffled_data)*0.85)]  # 85% for training
    test_data = shuffled_data[int(len(shuffled_data)*0.85):]   # 15% for testing
    
    # Prepare training and testing sets
    x_train = np.array([i[0] for i in train_data]).reshape(-1, 50, 50, 1)  # Reshape for CNN input shape (-1, 50, 50, 1)
    y_train = np.array([encode_label(i[1]) for i in train_data])

    x_test = np.array([i[0] for i in test_data]).reshape(-1, 50, 50, 1)  # Reshape for CNN input shape (-1, 50, 50, 1)
    y_test = np.array([encode_label(i[1]) for i in test_data])
    
    return x_train, y_train, x_test, y_test

# Generate dataset
x_train, y_train, x_test, y_test = generate_dataset()

print("Shape of training images:", x_train.shape)
print("Shape of testing images:", x_test.shape)

# Clearing any previous models from memory
import tensorflow as tf
tf.keras.backend.clear_session()

# Define the CNN model
model = Sequential([
    Conv2D(32, (5, 5), activation='relu', padding='same', input_shape=(50, 50, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (5, 5), activation='relu', padding='same'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2)),
    Conv2D(32, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # Sigmoid activation for binary classification
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=12, validation_data=(x_test, y_test))

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=1)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_acc)

# Visualize some sample images from the training dataset
plt.figure(figsize=(10, 10))
for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.imshow(x_train[i].reshape(50, 50), cmap='gray')
    plt.title("Label: {}".format("jecinta" if y_train[i] == 1 else "other"))
    plt.axis('off')
plt.suptitle("Sample Images from Training Dataset")
plt.show()

# Perform predictions on test data
predictions = model.predict(x_test)
predicted_labels = ["jecinta" if pred > 0.5 else "other" for pred in predictions]

# Visualize some predictions
plt.figure(figsize=(10, 10))
for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.imshow(x_test[i].reshape(50, 50), cmap='gray')
    plt.title("Predicted: {}".format(predicted_labels[i]))
    plt.axis('off')
plt.suptitle("Predictions on Test Dataset")
plt.show()

# Save the trained model
model.save("trained_model.h5")
print("Model saved successfully!")

# Load the saved model
loaded_model = load_model("trained_model.h5")

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import cv2
import numpy as np
from random import shuffle
from tqdm import tqdm
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import time

# Function to encode labels
def encode_label(label):
    if label == "jecinta":
        return 0
    elif label == "john":
        return 1
    else:
        return 2

def my_label(image_name):
    if "jecinta" in image_name:
        return "jecinta"
    elif "john" in image_name:
        return "john"
    else:
        return "other"

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
    while True:
        ret, frame = cap.read()
        if face_cropped(frame) is not None:
            img_id += 1
            face = cv2.resize(face_cropped(frame), (50, 50))  # Resizing to 50x50
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            if img_id <= 50:
                file_name_path = "data/" + my_label("jecinta") + "." + str(img_id) + ".jpg"
            else:
                file_name_path = "data/" + my_label("john") + "." + str(img_id) + ".jpg"
            
            label = my_label(file_name_path)
            cv2.imwrite(file_name_path, face)
            cv2.putText(face, str(img_id), (20, 20), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
            data.append((face, label))
            print("Label for", file_name_path, "is:", label)  # Print the label
            # Show the captured image
            cv2.imshow("Captured Image", face)
            if img_id == 50:
                print("Taking a break for 10 seconds...")
                time.sleep(10)  # Delay for 10 seconds
            if cv2.waitKey(1) == 13 or int(img_id) == 100:
                break

    cap.release()
    cv2.destroyAllWindows()
    print("Collecting samples is completed")
    
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
    Dense(3, activation='softmax')  # Three neurons for three classes, using softmax activation
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

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
    plt.title("Label: {}".format("jecinta" if y_train[i] == 0 else ("john" if y_train[i] == 1 else "other")))
    plt.axis('off')
plt.suptitle("Sample Images from Training Dataset")
plt.show()

# Perform predictions on test data
predictions = model.predict(x_test)
predicted_labels = ["jecinta" if pred.argmax() == 0 else ("john" if pred.argmax() == 1 else "other") for pred in predictions]

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

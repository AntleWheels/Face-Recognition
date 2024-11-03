import cv2
import numpy as np
import os

# Path to Haar cascade for face detection
haar_file = 'haarcascade_frontalface_default.xml'
datasets = 'datasets'  # Path to the datasets directory
print('Training...')

# Initializing variables
(images, labels, names, id) = ([], [], {}, 0)

# Iterate over each subdirectory in the dataset
for (subdirs, dirs, files) in os.walk(datasets):
    for subdir in dirs:
        names[id] = subdir
        subjectpath = os.path.join(datasets, subdir)
        for filename in os.listdir(subjectpath):
            path = os.path.join(subjectpath, filename)
            label = id
            img = cv2.imread(path, 0)  # Read image in grayscale
            if img is not None:
                img_resized = cv2.resize(img, (130, 100))  # Ensure consistent image size
                images.append(img_resized)
                labels.append(label)
        id += 1

# Convert lists to NumPy arrays
(images, labels) = (np.array(images), np.array(labels))

# Create the recognizer model (using LBPH for better compatibility)
model = cv2.face.LBPHFaceRecognizer_create()
model.train(images, labels)

# Load the Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(haar_file)

# Start the webcam
webcam = cv2.VideoCapture(0)
cnt = 0

while True:
    (_, img) = webcam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 4)
    
    for (x, y, w, h) in faces:
        face = gray[y:y + h, x:x + w]  # Crop the detected face
        face_resize = cv2.resize(face, (130, 100))  # Resize for prediction
        prediction = model.predict(face_resize)  # Predict using the trained model
        
        # Draw a rectangle around the face
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
        
        if prediction[1] < 500:
            cv2.putText(img, '%s - %.0f' % (names[prediction[0]], prediction[1]), (x - 10, y - 10),
                        cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))
            print(names[prediction[0]])
            cnt = 0
        else:
            cnt += 1
            cv2.putText(img, 'unknown - 0', (x - 10, y - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))
            if cnt > 100:
                print("Unknown Person")
                cv2.imwrite("unknown.jpg", img)
    
    cv2.imshow('OpenCV', img)
    key = cv2.waitKey(10)
    if key == 27:  # Press 'Esc' to exit
        break

webcam.release()
cv2.destroyAllWindows()

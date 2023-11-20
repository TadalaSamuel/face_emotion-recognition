import cv2
import numpy as np
import face_recognition
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense
from keras.regularizers import l2
from keras.preprocessing.image import ImageDataGenerator
import optuna
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.models import load_model
import cv2
import time

model = load_model(r"C:\Users\mitvo\Documents\Facial_Image_Dataset\Face_emotion_recognition_trail6_4091_IMAGES_LANCOSZ_cnn2_normalised_layers.h5")
# model=load_model(r"c:\users\mitvo\Documents\Facial_Image_Dataset\Face_emotion_recognition_trail6_4091_IMAGES_LANCOSZ_cnn_with equal_dist1.h5")
# Open the video capture
video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    if not ret:
        print("Failed to capture the frame.")
        break

    captured_image = cv2.GaussianBlur(frame, ksize=(3, 3), sigmaX=2)
    face_locations = face_recognition.face_locations(captured_image)

    for (top, right, bottom, left) in face_locations:
        # Draw a rectangle around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 255), 2)

        # Extract the face region
        face = captured_image[top:bottom, left:right]
        gray_image = cv2.cvtColor(face, cv2.COLOR_RGB2GRAY)
        height, width = gray_image.shape
        blank_image = np.zeros((height, width, 3), dtype=np.uint8)
        blank_image[:, :, 0] = gray_image
        blank_image[:, :, 1] = gray_image
        blank_image[:, :, 2] = gray_image
        
        ###################### Image preprocessing ####################
        target_size = (62,63,3)

        cropped_image=cv2.resize(blank_image,(target_size[1],target_size[0]))

        pixel_data = np.array([cropped_image])

        datagen = ImageDataGenerator(horizontal_flip=True,zoom_range=0.2,shear_range=0.2,featurewise_center=True,
                                    featurewise_std_normalization=True)

        datagen.fit(pixel_data)

        generator = datagen.flow(x=pixel_data,batch_size=len([cropped_image]),shuffle=False)

        preprocessed_data = next(generator)

        ########### Emotion prediction using the built model ######################
        time.sleep(3)
        predictions=model.predict(preprocessed_data)

        max_index = np.argmax(predictions) # ==> Index with maximum facial expression probability value

        emotions = {0: "angry", 1: "contempt", 2: "disgust", 3: "fear", 4: "happy", 5: "neutral", 6: "sad", 7: "surprise"}

        detected_expression = emotions[max_index]

        cv2.putText(frame, detected_expression, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

        

    cv2.imshow('Live Video', frame)
        
    # Check for the 'q' key to exit the loop
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Release the video capture and close the window
video_capture.release()
cv2.destroyAllWindows()
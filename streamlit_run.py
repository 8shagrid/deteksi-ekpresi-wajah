import streamlit as st
import cv2 as cv
import numpy as np
import joblib
from keras.models import load_model
from streamlit_webrtc import webrtc_streamer
import av

# Load the model for facial expression detection
# loadModel = load_model('model/model.keras')
loadModel = joblib.load("model/model.pkl")


# Function to predict facial expression from an image
def predict_expression(image):
    image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(image_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        face_roi = image_gray[y:y + h, x:x + w]
        face_roi = cv.resize(face_roi, (48, 48))
        face_roi = np.reshape(face_roi, (1, 48, 48, 1))
        face_roi = face_roi / 255.0

        prediction = loadModel.predict(face_roi)
        label = np.argmax(prediction)
        prob = np.max(prediction)

        label_map = {'angry': 0, 'disgust': 1, 'fear': 2, 'happy': 3, 'neutral': 4, 'sad': 5, 'surprise': 6}
        label_map_inverse = {v: k for k, v in label_map.items()}

        expression = label_map_inverse[label]

        text = f'{expression}: {prob:.2f}'
        cv.putText(image, text, (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Draw bounding box around the detected face
        cv.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

    return image


# Video Frame Callback Function for Streamlit WebRTC
def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")

    # Flip the frame horizontally (mirror effect)
    flipped_frame = cv.flip(img, 1)

    # Perform facial expression detection on the flipped frame
    output_frame = predict_expression(flipped_frame)

    return av.VideoFrame.from_ndarray(output_frame, format="bgr24")


# Streamlit App
def main():
    st.title("Real-time Facial Expression Detection with WebRTC")

    # Create a Streamlit WebRTC component for video streaming
    webrtc_streamer(
        key="example",
        video_frame_callback=video_frame_callback,
        async_processing=True,
    )


if __name__ == "__main__":
    main()

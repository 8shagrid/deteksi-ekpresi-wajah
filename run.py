import cv2 as cv
import numpy as np
from keras.models import load_model

# Memuat model
loadModel = load_model('model/model100Epoch.keras')

# Fungsi untuk mendeteksi dan memprediksi ekspresi wajah dari gambar
def predict_expression(image):
    image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(image_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        face_roi = image_gray[y:y + h, x:x + w]
        face_roi = cv.resize(face_roi, (48, 48))
        face_roi = np.reshape(face_roi, (1, 48, 48, 1))
        face_roi = face_roi / 255.0

        # Menampilkan kotak di sekitar wajah
        cv.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Melakukan prediksi menggunakan model
        prediction = loadModel.predict(face_roi)
        label = np.argmax(prediction)
        prob = np.max(prediction)

        # Membuat dictionary invers untuk memetakan indeks label ke nama kelas
        label_map = {'angry': 0, 'disgust': 1, 'fear': 2, 'happy': 3, 'neutral': 4, 'sad': 5, 'surprise': 6}
        label_map_inverse = {v: k for k, v in label_map.items()}

        # Menampilkan nama kelas
        expression = label_map_inverse[label]

        # Menampilkan hasil prediksi pada gambar
        text = f'{expression}: {prob:.2f}'
        cv.putText(image, text, (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    return image

# Mengakses kamera dan mendeteksi ekspresi wajah secara langsung
cap = cv.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Membuat mirror (flip horizontal) dari frame
    frame_mirror = cv.flip(frame, 1)

    # Deteksi dan prediksi ekspresi wajah
    output_frame = predict_expression(frame_mirror)

    # Menampilkan hasil deteksi di jendela
    cv.imshow('Facial Expression Detection', output_frame)

    # Menghentikan program dengan menekan tombol 'q'
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# Menutup kamera, menghancurkan jendela, dan menghentikan mode interaktif matplotlib
cap.release()
cv.destroyAllWindows()

### Bahasa Indonesia:
# Judul Proyek: Deteksi Ekspresi Wajah Menggunakan Klasifikasi Berbasis Convolutional Neural Network (CNN)

Deskripsi:
Proyek ini bertujuan untuk mengembangkan sebuah model deteksi ekspresi wajah yang mampu mengenali dan mengklasifikasikan ekspresi wajah manusia, seperti senang, sedih, marah, kaget, dll. Proyek ini menggunakan teknik pemrosesan gambar dan pembelajaran mendalam dengan memanfaatkan Convolutional Neural Network (CNN) sebagai arsitektur modelnya. Dataset yang digunakan bersumber dari kaggle yang di-upload oleh JONATHAN OHEIX.

https://www.kaggle.com/datasets/jonathanoheix/face-expression-recognition-dataset.

Langkah-langkah Proyek:
1. Pra-pemrosesan Data: Gambar-gambar ekspresi wajah diolah dengan mengubahnya menjadi citra grayscale dan membaginya menjadi dataset pelatihan dan validasi.
2. Pembuatan Arsitektur Model: Model CNN dibangun dengan lapisan konvolusi, max pooling, flatten, serta lapisan dense dengan fungsi aktivasi yang sesuai.
3. Pelatihan Model: Model CNN dikenakan pada dataset pelatihan untuk melakukan pelatihan. Nilai loss dan akurasi dievaluasi pada dataset validasi selama pelatihan.
4. Evaluasi Model: Model diuji pada dataset uji untuk mengukur kinerjanya dalam mengklasifikasikan ekspresi wajah.
5. Deteksi Wajah Langsung dari Kamera: Model yang telah dilatih digunakan untuk mendeteksi ekspresi wajah langsung dari kamera, dengan menampilkan nama kelas ekspresi dan probabilitasnya pada gambar.

Teknologi yang Digunakan:
- Python
- TensorFlow dan Keras
- OpenCV
- dll.

Cara Menjalankan Proyek:
1. Pastikan Anda memiliki Python diinstal di komputer Anda.
2. Unduh atau klon proyek ini dari repository GitHub.
3. Buka terminal atau command prompt dan pindah ke direktori proyek.
4. Buat virtual environment untuk proyek ini:
```
py -m venv .venv
```
atau
```
python -m venv .venv
```
5. Aktifkan virtual environmentnya:
```
.venv\Scripts\activate
```
6. Install semua dependensi dengan menjalankan perintah:
```
pip install -r requirements.txt
```
7. Setelah dependensi terinstal, Anda dapat menjalankan skrip proyek melalui jupyter notebook.

Proyek ini diharapkan dapat memberikan kontribusi dalam pengembangan teknologi pengenalan ekspresi wajah dan menjadi sumber pembelajaran bagi mereka yang tertarik dalam pemrosesan gambar dan pembelajaran mesin.


### English:
# Project Title: Facial Expression Detection using Convolutional Neural Network (CNN) based Classification

Description:
This project aims to develop a facial expression detection model that can recognize and classify human facial expressions such as happiness, sadness, anger, surprise, etc. The project employs image processing and deep learning techniques utilizing Convolutional Neural Networks (CNN) as the model architecture. The dataset used is sourced from kaggle which was uploaded by JONATHAN OHEIX.

https://www.kaggle.com/datasets/jonathanoheix/face-expression-recognition-dataset.

Project Steps:
1. Data Preprocessing: Facial expression images are preprocessed by converting them to grayscale images and splitting them into training and validation datasets.
2. Model Architecture: A CNN model is constructed using convolutional, max-pooling, flatten, and dense layers with appropriate activation functions.
3. Model Training: The CNN model is trained on the training dataset. Loss and accuracy values are evaluated on the validation dataset during training.
4. Model Evaluation: The model is tested on a separate test dataset to measure its performance in classifying facial expressions.
5. Real-time Facial Expression Detection: The trained model is used to detect facial expressions in real-time from a camera, displaying the class name and its probability on the image.

Technologies Used:
- Python
- TensorFlow and Keras
- OpenCV
- etc.

How to Run the Project:
1. Ensure you have Python installed on your computer.
2. Download or clone this project from the GitHub repository.
3. Open a terminal or command prompt and navigate to the project directory.
4. Create a virtual environment for this project:
```
py -m venv .venv
```
or
```
python -m venv .venv
```
5. Activate the virtual environment:
```
.venv\Scripts\activate
```
6. Install all dependencies by running the command:
```
pip install -r requirements.txt
```
7. Once the dependencies are installed, you can run the project script via jupyter notebook.

This project is expected to contribute to the advancement of facial expression recognition technology and serve as a learning resource for individuals interested in image processing and machine learning.

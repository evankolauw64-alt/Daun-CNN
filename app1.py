import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Membersihkan sesi Keras untuk mencegah memory leak
tf.keras.backend.clear_session()

# Membaca model menggunakan cache agar tidak dimuat ulang setiap kali layar direfresh
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('leaf_detection_classifier_tf', compile=False)

model = load_model()

# Streamlit App
st.title("Sistem Pengenalan Jenis Buah Berdasarkan Daun")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# Menggunakan Dictionary untuk menyimpan teks rekomendasi agar kode lebih rapi
# Anda tinggal mengganti teks di dalam tanda kutip dengan informasi yang sebenarnya
rekomendasi_daun = {
    'Alpukat': "Ini adalah daun Alpukat.",
    'Gersen': "Ini adalah daun Gersen.",
    'Jambu': "Ini adalah daun Jambu.",
    'Jeruk_Nipis': "Ini adalah daun Jeruk Nipis.",
    'Kelengkeng': "Ini adalah daun Kelengkeng.",
    'Mangga': "Ini adalah daun Mangga.",
    'Mulbery': "Ini adalah daun Mulbery.",
    'Nangka': "Ini adalah daun Nangka.",
    'Pepaya': "Ini adalah daun Pepaya.",
    'Pisang': "Ini adalah daun Pisang.",
    'Rambutan': "Ini adalah daun Rambutan.",
    'Semangka': "Ini adalah daun Semangka.",
    'Singkong': "Ini adalah daun Singkong.",
    'Sirih': "Ini adalah daun Sirih."
}

if uploaded_file is not None:
    try:
        # 1. Load the image
        image = Image.open(uploaded_file)
        
        # Konversi gambar ke RGB untuk menghindari error dimensi jika formatnya PNG (RGBA)
        image = image.convert('RGB')
        
        # Display the uploaded image
        st.image(image, caption='Uploaded Image', use_column_width=True)
        st.write("")
        st.write("Classifying...")

        # 2. Preprocess the image for prediction
        img = image.resize((150, 150))  # Resize the image to the desired dimensions
        img_array = np.array(img)  # Convert the image to an array
        img_array = np.expand_dims(img_array, axis=0)  # Add an extra dimension for batch
        img_array = img_array / 255.0  # Normalize the image data

        # 3. Make prediction
        prediction = model.predict(img_array)
        max_prob = np.max(prediction)  # Get the highest probability
        predicted_class_index = np.argmax(prediction)
        
        # Pastikan list ini urutannya persis sama dengan output dari training model
        class_list = ['Alpukat', 'Gersen', 'Jambu', 'Jeruk_Nipis', 'Kelengkeng', 'Mangga', 'Mulbery', 'Nangka', 'Pepaya', 'Pisang', 'Rambutan', 'Semangka', 'Singkong', 'Sirih']
        predicted_class = class_list[predicted_class_index]

        # 4. Check if the prediction is confident enough
        threshold = 0.8  # Define a confidence threshold (adjustable)  
        
        if max_prob < threshold:
            st.warning("Gambar yang diunggah bukan gambar yang terdeteksi secara akurat oleh sistem. Harap unggah gambar daun yang relevan dan lebih jelas.")
        else:
            # Mengganti underscore dengan spasi khusus untuk tampilan di layar agar lebih rapi (misal: Jeruk_Nipis -> Jeruk Nipis)
            display_name = predicted_class.replace('_', ' ')
            
            # Display the predicted class
            st.success(f"**Predicted class:** {display_name}")
            st.info(f"Tingkat Keyakinan: {max_prob * 100:.2f}%")

            # Menampilkan rekomendasi dengan memanggil dictionary
            # st.subheader("Rekomendasi Penanganan")
            teks_rekomendasi = rekomendasi_daun.get(predicted_class, "Rekomendasi belum tersedia untuk daun ini.")
            st.write(teks_rekomendasi)

    except Exception as e:
        # Menampilkan pesan error asli untuk mempermudah perbaikan (debugging)
        st.error(f"Terjadi kesalahan saat memproses gambar: {e}")
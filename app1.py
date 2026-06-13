import os
print("Memuat TensorFlow (ini mungkin memakan waktu beberapa detik)...")
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import random
from shutil import copyfile, rmtree

# =============================
# KONFIGURASI DATASET
# =============================
base_dir = r'D:\Tugas evan unima\semester 9-10 evan\Pengenalan Jenis Daun CNN\dataset - Copy'
bahan_dir = os.path.join(base_dir, 'bahan')
train_dir = os.path.join(base_dir, 'latih')
validation_dir = os.path.join(base_dir, 'validasi')

classes = ['Alpukat', 'Gersen', 'Jambu', 'Jeruk_Nipis', 'Kelengkeng', 'Mangga', 'Mulbery', 'Nangka', 'Pepaya', 'Pisang', 'Rambutan', 'Semangka', 'Singkong', 'Sirih']

# =============================
# BERSIHKAN FOLDER TRAIN & VAL (ANTI DUPLIKASI)
# =============================
print("\nMembersihkan folder latih dan validasi lama jika ada...")
for folder in [train_dir, validation_dir]:
    for cls in classes:
        cls_path = os.path.join(folder, cls)
        if os.path.exists(cls_path):
            rmtree(cls_path)
        os.makedirs(cls_path)

# =============================
# FUNGSI SPLIT DATA
# =============================
def train_val_split(source, train, val, train_ratio=0.8):
    files = os.listdir(source)
    random.shuffle(files)
    train_size = int(train_ratio * len(files))

    for f in files[:train_size]:
        copyfile(os.path.join(source, f), os.path.join(train, f))
    for f in files[train_size:]:
        copyfile(os.path.join(source, f), os.path.join(val, f))

# =============================
# SPLIT DATASET
# =============================
print("Memulai pembagian dan penyalinan gambar dataset (Mohon tunggu)...")
for cls in classes:
    print(f" -> Memproses daun: {cls}") # Menambahkan indikator agar terminal tidak terlihat mati
    
    # Pastikan folder bahan untuk kelas ini ada agar tidak error
    source_path = os.path.join(bahan_dir, cls)
    if not os.path.exists(source_path):
        print(f"    PERINGATAN: Folder {source_path} tidak ditemukan!")
        continue
        
    train_val_split(
        source_path,
        os.path.join(train_dir, cls),
        os.path.join(validation_dir, cls),
        train_ratio=0.8
    )

# =============================
# DATA GENERATOR
# =============================
print("\nMenyiapkan Data Generator...")
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    horizontal_flip=True,
    shear_range=0.3,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.1,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=10,
    class_mode='categorical',
    classes=classes
)

val_generator = val_datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=10,
    class_mode='categorical',
    classes=classes
)

print("\nLabel kelas:", train_generator.class_indices)

# =============================
# MODEL CNN (FINAL)
# =============================
# PERBAIKAN: Mengambil jumlah kelas secara otomatis dari list 'classes'
num_classes = len(classes) 
print(f"\nMembangun arsitektur model untuk {num_classes} kelas...")

model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(150, 150, 3)),
    tf.keras.layers.Conv2D(16, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(200, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(500, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(num_classes, activation='softmax') # Diperbaiki agar outputnya 14, bukan 4
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# =============================
# TRAINING
# =============================
print("\nMemulai proses training (pelatihan)...")
history = model.fit(
    train_generator,
    epochs=25,
    validation_data=val_generator,
    verbose=1
)

# =============================
# SIMPAN MODEL (DEPLOY AMAN)
# =============================
model.save("leaf_detection_classifier_tf")
print("\nModel berhasil dilatih dan disimpan sebagai folder 'leaf_detection_classifier_tf'")

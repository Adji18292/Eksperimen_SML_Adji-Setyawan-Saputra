import os
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import mlflow
import mlflow.keras

os.environ["MLFLOW_TRACKING_USERNAME"] = "Adji18292"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "d47068cae7c375ebefd29ab5a22552675d8e093c"

TRACKING_URI = "https://dagshub.com/Adji18292/MSML-Fruits-Classification.mlflow"
mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment("Eksperimen_CNN_Fruits")

dataset_path = "./fruits_preprocessing"
print("Memuat dataset dari:", dataset_path)
dataset = tf.data.Dataset.load(dataset_path)

DATASET_SIZE = len(dataset)
train_size = int(0.8 * DATASET_SIZE)
train_dataset = dataset.take(train_size)
val_dataset = dataset.skip(train_size)

model = models.Sequential([
    layers.InputLayer(input_shape=(100, 100, 3)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(131, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

with mlflow.start_run(run_name="Run_Adji_Setyawan"):
    print("Memulai proses training...")
    
    mlflow.keras.autolog()
    
    history = model.fit(train_dataset, validation_data=val_dataset, epochs=3)
    
    akurasi_akhir = history.history['accuracy'][-1]
    val_akurasi_akhir = history.history['val_accuracy'][-1]
    mlflow.log_metric("selisih_akurasi_train_val", abs(akurasi_akhir - val_akurasi_akhir))
    
    with open("model_summary_Adji.txt", "w", encoding="utf-8") as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))
    mlflow.log_artifact("model_summary_Adji.txt")
    
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Grafik Loss Model')
    plt.legend()
    plt.savefig("grafik_loss.png")
    mlflow.log_artifact("grafik_loss.png")

print("Training selesai! Silakan cek DagsHub Anda.")
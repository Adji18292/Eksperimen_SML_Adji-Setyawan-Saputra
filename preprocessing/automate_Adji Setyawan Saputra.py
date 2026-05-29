import os
import argparse
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras import layers

def process_and_save_data(input_dir, output_dir, batch_size=32, img_size=(100, 100), classes=None):
    """
    Fungsi untuk memuat gambar, melakukan preprocessing, dan menyimpan dataset.
    """
    print(f"Mulai memproses data dari: {input_dir}")
    
    dataset = image_dataset_from_directory(
        input_dir,
        shuffle=True,
        batch_size=batch_size,
        image_size=img_size,
        class_names=classes
    )
    
    normalization_layer = layers.Rescaling(1./255)
    
    AUTOTUNE = tf.data.AUTOTUNE
    processed_dataset = dataset.map(
        lambda x, y: (normalization_layer(x), y), 
        num_parallel_calls=AUTOTUNE
    ).cache().prefetch(buffer_size=AUTOTUNE)
    
    os.makedirs(output_dir, exist_ok=True)
    tf.data.Dataset.save(processed_dataset, output_dir)
    print(f"Dataset berhasil diproses dan disimpan di: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Otomatisasi Preprocessing Dataset Fruits-360")
    parser.add_argument('--input', type=str, required=True, help="Path ke folder dataset mentah (misal: fruits-360_raw/Training)")
    parser.add_argument('--output', type=str, required=True, help="Path untuk menyimpan dataset hasil preprocessing")
    parser.add_argument('--classes', type=str, nargs='+', help="Daftar kelas spesifik yang ingin diproses (contoh: 'Apple 10' 'Banana 1')")
    
    args = parser.parse_args()
    
    process_and_save_data(args.input, args.output, classes=args.classes)

if __name__ == "__main__":
    main()
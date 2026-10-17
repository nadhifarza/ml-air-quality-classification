import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

def main():
    print("[*] Memuat dataset kualitas udara...")
    dataset_path = "data/air_quality_dataset.csv"
    df = pd.read_csv(dataset_path)

    # Menentukan Fitur (X) dan Target/Label (y)
    X = df[['temperature', 'humidity', 'pm25']]
    y = df['status_kualitas']

    # Membagi data menjadi Data Latih (Training) dan Data Uji (Testing)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    print("[*] Melatih model Machine Learning (K-Nearest Neighbors)...")
    model = KNeighborsClassifier(n_neighbors=1)
    model.fit(X_train, y_train)

    print("[*] Melakukan prediksi pada data uji...")
    y_pred = model.predict(X_test)

    # Evaluasi Performa Model
    acc = accuracy_score(y_test, y_pred)
    print(f"\n[+] Akurasi Model: {acc * 100:.2f}%")
    print("\n[+] Laporan Klasifikasi:")
    print(classification_report(y_test, y_pred))

    # Simulasi Prediksi untuk Data Sensor Baru
    print("\n[*] Menguji prediksi untuk data sensor baru (Suhu: 29.5, Kelembapan: 78, PM2.5: 40.0):")
    sample_data = [[29.5, 78.0, 40.0]]
    prediction = model.predict(sample_data)
    print(f"    -> Hasil Prediksi Status Udara: {prediction[0]}")

if __name__ == "__main__":
    main()
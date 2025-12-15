import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib


def preprocess_diabetes_data(input_path, output_dir):
    """
    Fungsi preprocessing otomatis untuk dataset diabetes.
    Menghasilkan data train dan test yang siap digunakan
    untuk pelatihan model machine learning.
    """

    # Load data
    df = pd.read_csv(input_path)

    # Handle missing values (nilai 0 tidak valid)
    cols_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    for col in cols_with_zeros:
        df[col] = df[col].replace(0, np.nan)
        df[col] = df[col].fillna(df[col].median())

    # Split feature dan target
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns)

    # Gabungkan kembali dengan target
    train_final = X_train_scaled.copy()
    train_final['Outcome'] = y_train.values

    test_final = X_test_scaled.copy()
    test_final['Outcome'] = y_test.values

    # Save hasil
    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(scaler, f"{output_dir}/scaler.pkl")
    train_final.to_csv(f"{output_dir}/diabetes_train.csv", index=False)
    test_final.to_csv(f"{output_dir}/diabetes_test.csv", index=False)
    print("Preprocessing selesai, file berhasil disimpan.")

    return train_final, test_final


if __name__ == "__main__":
    preprocess_diabetes_data(
        input_path="diabetes_raw.csv",
        output_dir="../diabetes_preprocessing"
    )
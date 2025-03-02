import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from scipy.stats import skew, kurtosis, entropy
from glob import glob
from Utils.benford_analysis import benford_deviation as bd
from Utils.zipf_analysis import zipf_correlation as zc

# ---- Feature Extraction Function ---- #
def extract_features(df):
    """Extracts statistical features from any dataset."""
    features = {}

    # Numerical Features
    numeric_df = df.select_dtypes(include=['int64', 'float64'])
    if not numeric_df.empty:
        features['num_mean'] = numeric_df.mean().mean()
        features['num_std'] = numeric_df.std().mean()
        features['num_min'] = numeric_df.min().mean()
        features['num_max'] = numeric_df.max().mean()
        features['num_skew'] = np.clip(skew(numeric_df, nan_policy='omit').mean(), -10, 10)
        features['num_kurtosis'] = np.clip(kurtosis(numeric_df, nan_policy='omit').mean(), -10, 10)
        features['benford_mae'] = bd(numeric_df.stack())
    else:
        # Default values when there are no numerical columns
        features['num_mean'] = 0
        features['num_std'] = 0
        features['num_min'] = 0
        features['num_max'] = 0
        features['num_skew'] = 0
        features['num_kurtosis'] = 0
        features['benford_mae'] = 0

    # Categorical Features
    categorical_df = df.select_dtypes(include=['object', 'category'])
    if not categorical_df.empty:
        features['num_categorical'] = len(categorical_df.columns)
        features['cat_unique_ratio'] = categorical_df.nunique().mean()
        features['cat_mode_freq'] = categorical_df.mode().iloc[0].value_counts().mean()
        features['cat_entropy'] = entropy(categorical_df.apply(lambda x: x.value_counts(normalize=True), axis=0), nan_policy='omit').mean()
        features['zipf_corr'] = zc(categorical_df.stack())
    else:
        # Default values when there are no categorical columns
        features['num_categorical'] = 0
        features['cat_unique_ratio'] = 0
        features['cat_mode_freq'] = 0
        features['cat_entropy'] = 0
        features['zipf_corr'] = 0

    return pd.DataFrame([features])

# ---- Train Random Forest with Real and Fake Data ---- #
def train_random_forest(real_data_folder, fake_data_folder):
    """Trains a Random Forest on multiple real and fake datasets."""
    feature_list = []

    # Load & Process Real Datasets
    real_files = glob(os.path.join(real_data_folder, "*.csv"))
    for file in real_files:
        df = pd.read_csv(file)
        if len(df) > 1000:
            df = df.sample(n=1000, random_state=42)  # Randomly sample 1000 rows
        real_features = extract_features(df)
        real_features["label"] = 1  # Real
        feature_list.append(real_features)

    # Load & Process Fake Datasets
    fake_files = glob(os.path.join(fake_data_folder, "*.csv"))
    for file in fake_files:
        df = pd.read_csv(file)
        if len(df) > 1000:
            df = df.sample(n=1000, random_state=42)  # Randomly sample 1000 rows
        fake_features = extract_features(df)
        fake_features["label"] = 0  # Fake
        feature_list.append(fake_features)

    # Combine all extracted features
    feature_df = pd.concat(feature_list, ignore_index=True)

    # Prepare input features and labels
    X = feature_df.drop(columns=["label"])
    y = feature_df["label"]

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train Random Forest Classifier
    rf_classifier = RandomForestClassifier(n_estimators=250, max_depth=6, random_state=42)
    rf_classifier.fit(X_scaled, y)

    # Save Model and Scaler
    os.makedirs("models/randomForest", exist_ok=True)
    joblib.dump(rf_classifier, "models/randomForest/random_forest_original.pkl")
    joblib.dump(scaler, "models/randomForest/scaler_original.pkl")

    print(f"✅ Random Forest trained on {len(real_files)} real datasets + {len(fake_files)} fake datasets!")

# ---- Classify Completely New Dataset ---- #
def classify_new_dataset(file_path):
    """Classifies a new dataset as Real or Fake based on extracted features."""
    try:
        new_df = pd.read_csv(file_path)
        if new_df.empty:
            print("⚠️ The dataset is empty and cannot be classified.")
            return
        
        # Extract features
        new_features = extract_features(new_df)

        # Load trained model and scaler
        rf_classifier = joblib.load("models/randomForest/random_forest_original.pkl")
        scaler = joblib.load("models/randomForest/scaler_original.pkl")

        # Standardize features
        new_X_scaled = scaler.transform(new_features)

        # Predict
        prediction = rf_classifier.predict(new_X_scaled)
        label = "Real" if prediction[0] == 1 else "Fake"

        print(f"✅ Classification Result for original model: {label}")

    except Exception as e:
        print(f"⚠️ Error during classification for original model: {e}")


if __name__ == "__main__":
    real_data_folder = "TrainingData/realData"  
    fake_data_folder = "TrainingData/fakeData"
    
    print("Training Random Forest Model...")
    train_random_forest(real_data_folder, fake_data_folder)
    train_random_forest(real_data_folder, fake_data_folder)
    print("Training Completed!")
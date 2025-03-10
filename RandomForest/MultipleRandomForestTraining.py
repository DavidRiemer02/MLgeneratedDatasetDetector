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

def extract_features(df):
    """Extracts statistical features from any dataset."""
    features = {}

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
        features.update({
            'num_mean': 0, 'num_std': 0, 'num_min': 0, 'num_max': 0,
            'num_skew': 0, 'num_kurtosis': 0, 'benford_mae': 0
        })

    categorical_df = df.select_dtypes(include=['object', 'category'])
    if not categorical_df.empty:
        features['num_categorical'] = len(categorical_df.columns)
        features['cat_unique_ratio'] = categorical_df.nunique().mean()
        features['cat_mode_freq'] = categorical_df.mode().iloc[0].value_counts().mean()
        features['cat_entropy'] = entropy(categorical_df.apply(lambda x: x.value_counts(normalize=True), axis=0), nan_policy='omit').mean()
        features['zipf_corr'] = zc(categorical_df.stack())
    else:
        features.update({'num_categorical': 0, 'cat_unique_ratio': 0, 'cat_mode_freq': 0, 'cat_entropy': 0, 'zipf_corr': 0})

    return pd.DataFrame([features])

def train_multiple_models(real_data_folder, fake_data_folder, sample_size, n_estimators, max_depth):
    """Trains a Random Forest model with specific hyperparameters."""
    feature_list = []
    
    real_files = glob(os.path.join(real_data_folder, "*.csv"))
    for file in real_files:
        df = pd.read_csv(file)
        if len(df) > sample_size:
            df = df.sample(n=sample_size, random_state=42)
        real_features = extract_features(df)
        real_features["label"] = 1  # Real
        feature_list.append(real_features)
    
    fake_files = glob(os.path.join(fake_data_folder, "*.csv"))
    for file in fake_files:
        df = pd.read_csv(file)
        if len(df) > sample_size:
            df = df.sample(n=sample_size, random_state=42)
        fake_features = extract_features(df)
        fake_features["label"] = 0  # Fake
        feature_list.append(fake_features)
    
    feature_df = pd.concat(feature_list, ignore_index=True)
    X = feature_df.drop(columns=["label"])
    y = feature_df["label"]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    rf_classifier = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    rf_classifier.fit(X_scaled, y)
    
    os.makedirs("models/randomForest", exist_ok=True)
    model_filename = f"random_forest_s{sample_size}_n{n_estimators}_d{max_depth}.pkl"
    scaler_filename = f"scaler_s{sample_size}_n{n_estimators}_d{max_depth}.pkl"
    
    joblib.dump(rf_classifier, os.path.join("models/randomForest", model_filename))
    joblib.dump(scaler, os.path.join("models/randomForest", scaler_filename))
    
    print(f"✅ Random Forest trained with Sample Size={sample_size}, n_estimators={n_estimators}, max_depth={max_depth}!")

def classify_new_dataset_multiple_models(file_path):
    """Classifies a dataset using all trained models in the models folder."""
    try:
        new_df = pd.read_csv(file_path)
        if new_df.empty:
            print("⚠️ The dataset is empty and cannot be classified.")
            return
        
        new_features = extract_features(new_df)
        model_files = glob("models/randomForest/random_forest_*.pkl")
        
        results = {}
        for model_file in model_files:
            model_name = os.path.basename(model_file).replace(".pkl", "")
            scaler_file = model_file.replace("random_forest", "scaler")
            
            rf_classifier = joblib.load(model_file)
            scaler = joblib.load(scaler_file)
            new_X_scaled = scaler.transform(new_features)
            prediction = rf_classifier.predict(new_X_scaled)
            label = "Real" if prediction[0] == 1 else "Fake"
            results[model_name] = label
        
        print("✅ Classification Results:")
        for model, label in results.items():
            print(f"{model}: {label}")
    
    except Exception as e:
        print(f"⚠️ Error during classification: {e}")

if __name__ == "__main__":
    real_data_folder = "TrainingData/realData"  
    fake_data_folder = "TrainingData/fakeData"
    
    print("Training Multiple Random Forest Models...")
    train_multiple_models(real_data_folder, fake_data_folder, 500, 100, 20)
    train_multiple_models(real_data_folder, fake_data_folder, 1000, 250, 5)
    train_multiple_models(real_data_folder, fake_data_folder, 2000, 1000, 5)

    print("Training Completed!")

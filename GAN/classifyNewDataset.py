import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

# Load the trained Discriminator model
class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.model(x)

# Load dataset (replace with new dataset path)
new_data_path = "data/Artificial_Data.csv"
new_df = pd.read_csv(new_data_path)

# Load pre-fitted scalers from training data
original_data_path = "data/CD.csv"
real_df = pd.read_csv(original_data_path)

# Identify columns
categorical_columns = real_df.select_dtypes(include=['object', 'category']).columns.tolist()
numerical_columns = real_df.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Fit scalers based on original training dataset
num_scaler = MinMaxScaler()
cat_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

num_scaled = num_scaler.fit_transform(real_df[numerical_columns])
cat_encoded = cat_encoder.fit_transform(real_df[categorical_columns])

# Process the new dataset using the same transformations
new_num_scaled = num_scaler.transform(new_df[numerical_columns])
new_cat_encoded = cat_encoder.transform(new_df[categorical_columns])
new_processed = np.hstack((new_num_scaled, new_cat_encoded))
new_data_tensor = torch.tensor(new_processed, dtype=torch.float32)

# Load trained discriminator
input_dim = new_data_tensor.shape[1]
discriminator = Discriminator(input_dim)
discriminator.load_state_dict(torch.load("models/discriminator.pth"))
discriminator.eval()

# Function to classify rows
def classify_row(row_tensor):
    with torch.no_grad():
        prediction = discriminator(row_tensor.unsqueeze(0))  # Add batch dimension
        confidence = prediction.item()
        return "Real" if confidence > 0.5 else "Generated", confidence

# Classify new dataset
print("\n🔍 **Discriminator Results for New Data** 🔍")
for i, row in enumerate(new_data_tensor[:5]):  # Check first 5 rows
    label, confidence = classify_row(row)
    print(f"New Data Row {i + 1}: {label} (Confidence: {confidence:.4f})")

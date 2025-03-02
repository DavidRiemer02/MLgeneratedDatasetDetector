import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Load CSV dataset
csv_file = "TrainingData/realData/car_price_dataset.csv"
df = pd.read_csv(csv_file)

# Identify column types
categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Preprocessing: Encoding categorical and normalizing numerical features
num_scaler = MinMaxScaler()
cat_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

# Extract numerical and categorical columns separately
df_num = df[numerical_columns]
df_cat = df[categorical_columns]

# Fit and transform numerical data
num_scaled = num_scaler.fit_transform(df_num)

# Fit and transform categorical data
cat_encoded = cat_encoder.fit_transform(df_cat)

processed_data = np.hstack((num_scaled, cat_encoded))


preprocessor = ColumnTransformer([
    ("num", num_scaler, numerical_columns),
    ("cat", cat_encoder, categorical_columns)
])

processed_data = preprocessor.fit_transform(df)
num_features = processed_data.shape[1]  # Total number of transformed features

# Convert to PyTorch tensor
real_data = torch.tensor(processed_data, dtype=torch.float32)

# Define the Generator
class Generator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
            nn.Tanh()
        )

    def forward(self, z):
        return self.model(z)

# Define the Discriminator
class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# Hyperparameters
z_dim = 50  # Size of random noise vector
lr = 0.0002  # Learning rate
num_epochs = 8000
batch_size = 32

# Initialize models and optimizers
generator = Generator(z_dim, num_features)
discriminator = Discriminator(num_features)

g_optimizer = optim.Adam(generator.parameters(), lr=lr)
d_optimizer = optim.Adam(discriminator.parameters(), lr=lr)
loss_fn = nn.BCELoss()

# Training the GAN
for epoch in range(num_epochs):
    for _ in range(5):
        # Train Discriminator
        real_samples = real_data[torch.randint(0, real_data.shape[0], (batch_size,))]

        z = torch.randn(batch_size, z_dim)
        fake_samples = generator(z)

         # 🛠️ **Add Gaussian noise to real & fake samples**
        real_samples += 0.05 * torch.randn_like(real_samples)  # Perturb real data
        fake_samples += 0.05 * torch.randn_like(fake_samples)

        d_real = discriminator(real_samples)
        d_fake = discriminator(fake_samples.detach())

        d_loss_real = loss_fn(d_real, torch.ones_like(d_real))
        d_loss_fake = loss_fn(d_fake, torch.zeros_like(d_fake))
        d_loss = (d_loss_real + d_loss_fake) / 2

        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()

    # Train Generator
    z = torch.randn(batch_size, z_dim) * 0.5 + torch.rand(batch_size, z_dim) * 0.5
    fake_samples = generator(z)
    d_fake = discriminator(fake_samples)

    g_loss = loss_fn(d_fake, torch.ones_like(d_fake))  # Fool the discriminator

    g_optimizer.zero_grad()
    g_loss.backward()
    g_optimizer.step()

    if epoch % 1000 == 0:
        print(f"Epoch {epoch}: D Loss: {d_loss.item()}, G Loss: {g_loss.item()}")

# Save the trained discriminator model
torch.save(discriminator.state_dict(), "models/gan/discriminator.pth")
print("✅ Discriminator model saved as 'discriminator.pth' under models/gan folder")


# Generate new synthetic CSV rows
z = torch.randn(10, z_dim)  # Generate 10 synthetic rows
generated_data = generator(z).detach().numpy()

# **Manual Inverse Transform**
# 1️⃣ Reverse numerical scaling
# Ensure the correct subset is passed for inverse transformation
generated_data[:, :len(numerical_columns)] = num_scaler.inverse_transform(
    generated_data[:, :len(numerical_columns)]
)
# 2️⃣ Convert one-hot encoded categorical features back to labels
cat_start = len(numerical_columns)
cat_end = num_features
onehot_encoded = generated_data[:, cat_start:cat_end]

decoded_categorical = []
for i, cat_col in enumerate(categorical_columns):
    categories = cat_encoder.categories_[i]
    indices = np.argmax(onehot_encoded[:, sum(len(c) for c in cat_encoder.categories_[:i]):sum(len(c) for c in cat_encoder.categories_[:i + 1])], axis=1)
    decoded_categorical.append([categories[idx] for idx in indices])

# Convert list to DataFrame
decoded_categorical = np.array(decoded_categorical).T

# Combine numerical and categorical data
final_generated_data = np.column_stack((generated_data[:, :len(numerical_columns)], decoded_categorical))

# Create DataFrame with correct column names
new_df = pd.DataFrame(final_generated_data, columns=numerical_columns + categorical_columns)

# Save to CSV
new_df.to_csv("TrainingData/fakedata/generated_data.csv", index=False)
print("Generated data saved to 'TrainingData/fakedata/generated_data.csv'")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import os

class GAN:
    def __init__(self, data_path, z_dim=50, lr=0.0002, num_epochs=5000, batch_size=32):
        self.data_path = data_path
        self.z_dim = z_dim
        self.lr = lr
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.df = pd.read_csv(data_path)
        self.base_name = os.path.basename(data_path).split('.')[0]  # Extract dataset name
        self._preprocess()
        self._init_models()
    
    def _preprocess(self):
        categorical_columns = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
        numerical_columns = self.df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
        self.num_scaler = MinMaxScaler()
        self.cat_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

        # Fit transformers
        num_scaled = self.num_scaler.fit_transform(self.df[numerical_columns]) if numerical_columns else np.array([])
        cat_encoded = self.cat_encoder.fit_transform(self.df[categorical_columns]) if categorical_columns else np.array([])

        # Combine transformed numerical and categorical data
        if num_scaled.size and cat_encoded.size:
            self.processed_data = np.hstack((num_scaled, cat_encoded))
        elif num_scaled.size:
            self.processed_data = num_scaled
        else:
            self.processed_data = cat_encoded

        self.num_features = self.processed_data.shape[1]
        self.real_data = torch.tensor(self.processed_data, dtype=torch.float32)
        self.numerical_columns = numerical_columns
        self.categorical_columns = categorical_columns

    
    class Generator(nn.Module):
        def __init__(self, input_dim, output_dim):
            super().__init__()
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
    
    class Discriminator(nn.Module):
        def __init__(self, input_dim):
            super().__init__()
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
    
    def _init_models(self):
        self.generator = self.Generator(self.z_dim, self.num_features)
        self.discriminator = self.Discriminator(self.num_features)
        
        self.g_optimizer = optim.Adam(self.generator.parameters(), lr=self.lr)
        self.d_optimizer = optim.Adam(self.discriminator.parameters(), lr=self.lr)
        self.loss_fn = nn.BCELoss()
    
    def train(self):
        for epoch in range(self.num_epochs):
            for _ in range(5):
                real_samples = self.real_data[torch.randint(0, self.real_data.shape[0], (self.batch_size,))]
                z = torch.randn(self.batch_size, self.z_dim)
                fake_samples = self.generator(z)
                
                real_samples += 0.05 * torch.randn_like(real_samples)
                fake_samples += 0.05 * torch.randn_like(fake_samples)
                
                d_real = self.discriminator(real_samples)
                d_fake = self.discriminator(fake_samples.detach())
                
                d_loss_real = self.loss_fn(d_real, torch.ones_like(d_real))
                d_loss_fake = self.loss_fn(d_fake, torch.zeros_like(d_fake))
                d_loss = (d_loss_real + d_loss_fake) / 2
                
                self.d_optimizer.zero_grad()
                d_loss.backward()
                self.d_optimizer.step()
                
            z = torch.randn(self.batch_size, self.z_dim)
            fake_samples = self.generator(z)
            d_fake = self.discriminator(fake_samples)
            
            g_loss = self.loss_fn(d_fake, torch.ones_like(d_fake))
            
            self.g_optimizer.zero_grad()
            g_loss.backward()
            self.g_optimizer.step()
            
            if epoch % 1000 == 0:
                print(f"Epoch {epoch}: D Loss: {d_loss.item()}, G Loss: {g_loss.item()}")
        
        # Save the trained discriminator model with dataset name
        os.makedirs("models/gan", exist_ok=True)
        model_path = f"models/gan/discriminator_{self.base_name}.pth"
        torch.save(self.discriminator.state_dict(), model_path)
        print(f"✅ Discriminator model saved as '{model_path}'")
    
    def generate_data(self, num_samples=10):
        z = torch.randn(num_samples, self.z_dim)
        generated_data = self.generator(z).detach().numpy()
        
        generated_data[:, :len(self.numerical_columns)] = self.num_scaler.inverse_transform(
            generated_data[:, :len(self.numerical_columns)]
        )
        
        cat_start = len(self.numerical_columns)
        cat_end = self.num_features
        onehot_encoded = generated_data[:, cat_start:cat_end]
        
        decoded_categorical = []
        for i, cat_col in enumerate(self.categorical_columns):
            categories = self.cat_encoder.categories_[i]
            indices = np.argmax(onehot_encoded[:, sum(len(c) for c in self.cat_encoder.categories_[:i]):sum(len(c) for c in self.cat_encoder.categories_[:i + 1])], axis=1)
            decoded_categorical.append([categories[idx] for idx in indices])
        
        decoded_categorical = np.array(decoded_categorical).T
        final_generated_data = np.column_stack((generated_data[:, :len(self.numerical_columns)], decoded_categorical))
        new_df = pd.DataFrame(final_generated_data, columns=self.numerical_columns + self.categorical_columns)
        
        save_path = f"TrainingData/fakedata/{self.base_name}_generated.csv"
        new_df.to_csv(save_path, index=False)
        print(f"Generated data saved to '{save_path}'")

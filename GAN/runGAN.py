from ganTraining import GAN

# Path to your real dataset
data_path = "TrainingData/realData/car_price_dataset.csv"

# Initialize and Train the GAN
gan_model = GAN(data_path)
# Train the GAN model
#gan_model.train()
  

# Generate Synthetic Data
gan_model.generate_data(num_samples=1000)  # Change num_samples as needed
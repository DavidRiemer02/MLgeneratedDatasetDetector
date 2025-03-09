import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def benford_deviation(series):
    """Computes deviation from Benford's Law using Mean Absolute Error (MAE)."""
    series = series.dropna().astype(str)  # Remove NaN values and convert to string
    
    # Extract first valid digit from positive numbers only
    first_digits = series.str.extract(r'(\d)')[0].dropna().astype(int)
    
    # Expected Benford's distribution
    benford_dist = np.array([np.log10(1 + 1/d) for d in range(1, 10)])
    observed_dist, _ = np.histogram(first_digits, bins=np.arange(1, 11), density=True)
    
    # Compute Mean Absolute Error (MAE)
    deviation = np.abs(observed_dist - benford_dist).mean()
    
    return deviation

def visualize_benford_law(csv_file):
    """Reads a CSV file and visualizes the compliance of each numerical column with Benford's Law one by one, only if MAE is not NaN."""
    df = pd.read_csv(csv_file)
    
    num_columns = df.select_dtypes(include=[np.number]).columns
    
    benford_dist = np.array([np.log10(1 + 1/d) for d in range(1, 10)])
    digits = np.arange(1, 10)
    
    for col in num_columns:
        deviation = benford_deviation(df[col])  # Get only the deviation

        # Skip plotting if deviation is NaN
        if np.isnan(deviation):
            print(f"Skipping column '{col}' due to NaN MAE.")
            continue

        # Extract first digits manually
        first_digits = df[col].dropna().astype(str).str.extract(r'(\d)')[0].dropna().astype(int)
        
        # Compute observed distribution
        observed_counts = first_digits.value_counts(normalize=True).sort_index()
        observed_dist = np.array([observed_counts.get(d, 0) for d in digits])

        # Create and show individual plot
        plt.figure(figsize=(8, 5))
        plt.bar(digits, observed_dist, alpha=0.7, color='lightblue', label='Observed')
        plt.plot(digits, benford_dist, marker='o', linestyle='-', color='pink', label='Benford Expected')
        
        plt.xticks(digits)
        plt.xlabel("First Digit")
        plt.ylabel("Frequency")
        plt.title(f"Benford's Law Compliance: {col} (MAE: {deviation:.4f})")
        plt.legend()
        
        plt.show()

csv_file = "TrainingData/fakeData/chatGPT_Benford_s_Law_Dataset.csv"
visualize_benford_law(csv_file)
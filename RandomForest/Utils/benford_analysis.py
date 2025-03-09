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
    """Reads a CSV file and visualizes the compliance of each numerical column with Benford's Law."""
    df = pd.read_csv(csv_file)
    
    num_columns = df.select_dtypes(include=[np.number]).columns
    fig, axes = plt.subplots(len(num_columns), 1, figsize=(8, 5 * len(num_columns)))
    
    if len(num_columns) == 1:
        axes = [axes]  # Ensure we have an iterable list of axes
    
    benford_dist = np.array([np.log10(1 + 1/d) for d in range(1, 10)])
    digits = np.arange(1, 10)
    
    for ax, col in zip(axes, num_columns):
        deviation, first_digits = benford_deviation(df[col])
        
        observed_counts = first_digits.value_counts(normalize=True).sort_index()
        
        ax.bar(digits, observed_counts.reindex(digits, fill_value=0), alpha=0.7, label='Observed')
        ax.plot(digits, benford_dist, marker='o', linestyle='-', color='r', label='Benford Expected')
        
        ax.set_xticks(digits)
        ax.set_xlabel("First Digit")
        ax.set_ylabel("Frequency")
        ax.set_title(f"Benford's Law Compliance: {col} (MAE: {deviation:.4f})")
        ax.legend()
    
    plt.tight_layout()
    plt.show()

csv_file = "/mnt/data/your_file.csv"  # Replace with the actual file path
visualize_benford_law(csv_file)
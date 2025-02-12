import numpy as np
import pandas as pd
from scipy.stats import linregress

def zipf_correlation(series):
    """Computes correlation with expected Zipfian distribution."""
    counts = series.value_counts().sort_values(ascending=False).values
    if len(counts) < 2:
        return np.nan  # Not enough data for correlation

    ranks = np.arange(1, len(counts) + 1)
    log_ranks, log_counts = np.log(ranks), np.log(counts)

    # Compute Pearson correlation coefficient
    correlation = linregress(log_ranks, log_counts).rvalue  

    return correlation
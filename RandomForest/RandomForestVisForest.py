import joblib
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
import os

# Load the trained Random Forest model
model_path = "models/randomForest/random_forest_multi.pkl"
rf_classifier = joblib.load(model_path)

# Select a few trees to visualize
num_trees_to_plot = 3  # Adjust as needed (e.g., 3 trees)
fig, axes = plt.subplots(nrows=num_trees_to_plot, ncols=1, figsize=(12, 6 * num_trees_to_plot))

# Plot selected decision trees from the Random Forest
for i in range(num_trees_to_plot):
    plot_tree(rf_classifier.estimators_[i], filled=True, feature_names = [
    'Mean of all numerical values', 'Standard Deviation of numerical values', 'Minimum value in numerical columns', 'Maximum value in numerical columns', 'Skewness of numerical values', 'Kurtosis of numerical values', 'deviation from Benfords Law using MAE',
    'Number of categorical columns in the dataset', 'Average number of unique values per categorical column', 'Most frequent categorys percentage in each categorical column', 'Entropy of categorical distributions', 'Correlation with expected Zipfian distribution'
], ax=axes[i] if num_trees_to_plot > 1 else axes)

plt.tight_layout()
plt.show()
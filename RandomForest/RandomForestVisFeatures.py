import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import plot_tree
import os

# Load the trained Random Forest model
model_path = "models/randomForest/random_forest_multi.pkl"
rf_classifier = joblib.load(model_path)

# Define feature names based on the extracted statistical features used during training
feature_names = [
    'Mean of all numerical values', 'Standard Deviation of numerical values', 'Minimum value in numerical columns', 'Maximum value in numerical columns', 'Skewness of numerical values', 'Kurtosis of numerical values', 'deviation from Benfords Law using MAE',
    'Number of categorical columns in the dataset', 'Average number of unique values per categorical column', 'Most frequent categorys percentage in each categorical column', 'Entropy of categorical distributions', 'Correlation with expected Zipfian distribution'
]

# Extract feature importances from the trained model
feature_importances = rf_classifier.feature_importances_

# Create a DataFrame for visualization
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Plot Feature Importance
plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'], importance_df['Importance'], color='pink')
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.title("Feature Importance in Random Forest Classifier")
plt.gca().invert_yaxis()  # Highest importance at the top
plt.show()

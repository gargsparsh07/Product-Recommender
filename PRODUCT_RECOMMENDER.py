# Step 0: Uploading the Dataset
from google.colab import files
uploaded = files.upload()


import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_squared_error




# Step 1: Load dataset
df = pd.read_csv("ratings_Beauty.csv")




# Step 2: Limit to 1000 users and 1000 items
top_users = df['UserId'].value_counts().index[:1000]
top_items = df['ProductId'].value_counts().index[:1000]

df = df[df['UserId'].isin(top_users) & df['ProductId'].isin(top_items)]




# Step 3: Encode user and item IDs
user_enc = LabelEncoder()
item_enc = LabelEncoder()
df['user'] = user_enc.fit_transform(df['UserId'])
df['item'] = item_enc.fit_transform(df['ProductId'])




# Step 4: Create user-item matrix
n_users = df['user'].nunique()
n_items = df['item'].nunique()

matrix = np.zeros((n_users, n_items))
for row in df.itertuples():
    matrix[row.user, row.item] = row.Rating




# Step 5: Apply SVD
n_components = 15
svd = TruncatedSVD(n_components=n_components, random_state=42)
reduced_matrix = svd.fit_transform(matrix)




# Step 6: Predict and evaluate
predicted_matrix = np.dot(reduced_matrix, svd.components_)

actuals = []
preds = []

for row in df.itertuples():
    actuals.append(row.Rating)
    preds.append(predicted_matrix[row.user, row.item])

rmse = np.sqrt(mean_squared_error(actuals, preds))
print(f" RMSE: {rmse:.4f}")




# Step 7: Visualization
import seaborn as sns
import matplotlib.pyplot as plt

# Plot a small section for clarity (e.g., first 20 users and 20 items)
subset = matrix[:20, :20]

plt.figure(figsize=(12, 8))
sns.heatmap(subset, cmap="YlGnBu", annot=False, cbar=True)
plt.title("User-Item Rating Matrix Heatmap (First 20 Users x 20 Items)")
plt.xlabel("Item ID")
plt.ylabel("User ID")
plt.show()

# Heatmap of predicted ratings
pred_subset = predicted_matrix[:20, :20]

plt.figure(figsize=(12, 8))
sns.heatmap(pred_subset, cmap="YlGnBu", annot=False, cbar=True)
plt.title("Predicted Ratings Heatmap (First 20 Users x 20 Items)")
plt.xlabel("Item ID")
plt.ylabel("User ID")
plt.show()




# Step 8: Precision of our Model
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# RMSE (Root Mean Squared Error)
rmse = np.sqrt(mean_squared_error(actuals, preds))

# MAE (Mean Absolute Error)
mae = mean_absolute_error(actuals, preds)

# R² Score (explains how much variance is captured)
r2 = r2_score(actuals, preds)

print(f" RMSE (Root Mean Squared Error): {rmse:.4f}")
print(f" MAE (Mean Absolute Error): {mae:.4f}")
print(f" R² Score (Explained Variance): {r2:.4f}")
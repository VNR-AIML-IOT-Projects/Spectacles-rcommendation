import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('normalized_file.csv')
features = df.drop('face_shape', axis=1)
target = df['face_shape']

# Standardize the features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Apply PCA
pca = PCA(n_components=2)
data_pca = pca.fit_transform(features_scaled)

print(pca.explained_variance_)
print(pca.explained_variance_ratio_)

face_shape_colors = {
    'Round_face_shape': 'red',
    'Oval_face_shape': 'green',
    'Square_face_shape': 'blue',
    'Heart_face_shape': 'purple',
    'Oblong_face_shape': 'orange'
}

# Check if all face shapes are mapped to a color
# print(df['face_shape'].unique())  # This will show all unique face shapes
# print(face_shape_colors.keys())   # This will show all keys in your mapping dictionary


# Map each face shape to the corresponding color
colors = df['face_shape'].map(face_shape_colors)

# print(colors.head())


plt.figure(figsize=(8, 6))
plt.scatter(data_pca[:, 0], data_pca[:, 1], c = colors)
plt.xlabel("PC 1")
plt.ylabel("PC 2")
plt.show()

import pandas as pd
import umap
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

# Load your dataset
df = pd.read_csv('normalized_file.csv')

# Assuming the last column is the target label 'face_shape'
features = df.iloc[:, :-1].values
labels = df.iloc[:, -1].values

# Initialize UMAP to reduce to 3 components for 3D plotting
reducer = umap.UMAP(n_components=3, random_state=42)

# Fit and transform the data
embedding = reducer.fit_transform(features)

# Create a DataFrame for the embedding with labels for plotting
embedding_df = pd.DataFrame(embedding, columns=['UMAP1', 'UMAP2', 'UMAP3'])
embedding_df['face_shape'] = labels

# Define a color palette with explicit mapping to face shapes
palette = {
    'Heart_face_shape': 'red',    # Heart shape will be red
    'Oblong_face_shape': 'blue',  # Oblong shape will be blue
    'Oval_face_shape': 'green',   # Oval shape will be green
    'Square_face_shape': 'orange',# Square shape will be orange
    'Round_face_shape': 'purple'  # Round shape will be purple
}

# Plot the result with the specified color palette
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(
    embedding_df['UMAP1'],
    embedding_df['UMAP2'],
    embedding_df['UMAP3'],
    c=embedding_df['face_shape'].map(palette),
    s=50
)
legend1 = ax.legend(*scatter.legend_elements(), title="Face Shapes")
ax.add_artist(legend1)
plt.title('3D UMAP projection of the Face Shapes dataset')
plt.show()

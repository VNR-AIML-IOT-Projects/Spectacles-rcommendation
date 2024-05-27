import pandas as pd
import umap
import matplotlib.pyplot as plt
import seaborn as sns

# Load your dataset
df = pd.read_csv('normalized_file.csv')

# Assuming the last column is the target label 'face_shape'
features = df.iloc[:, :-1].values
labels = df.iloc[:, -1].values

# Initialize UMAP
reducer = umap.UMAP(n_components=2,)

# Fit and transform the data
embedding = reducer.fit_transform(features)

# Create a DataFrame for the embedding with labels for plotting
embedding_df = pd.DataFrame(embedding, columns=['UMAP1', 'UMAP2'])
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
sns.scatterplot(x='UMAP1', y='UMAP2', hue='face_shape', data=embedding_df, palette=palette)
plt.title('UMAP projection of the Face Shapes dataset')
plt.show()

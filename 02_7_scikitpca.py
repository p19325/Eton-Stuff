import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Generating 'squished' data
np.random.seed()
X = (np.random.rand(2, 2) @ np.random.randn(2, 200)).T
X[:, 1] *= 0.5

# Centering the data
X_centered = X - np.mean(X, axis=0)

# Applying PCA using sklearn
pca = PCA(n_components=2)
pca.fit(X_centered)

# Transforming the data using PCA
X_pca = pca.transform(X_centered)

# Plotting the original and PCA-transformed data
plt.figure(figsize=(14, 7))

plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], alpha=0.5)
plt.title('Original Data')
plt.xlabel('X coordinate')
plt.ylabel('Y coordinate')

plt.subplot(1, 2, 2)
plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.5)
plt.title('PCA Transformed Data')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')

plt.tight_layout()
plt.show()

# Display the amount of variance each principal component is responsible for
print("Explained variance ratio:", pca.explained_variance_ratio_)

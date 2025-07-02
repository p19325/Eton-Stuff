# Importing the libraries
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from sklearn.metrics import confusion_matrix, accuracy_score

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Importing the dataset
dataset = pd.read_csv('./datasets/Social_Network_Ads.csv').to_numpy()
X = dataset[:, :-1]
y = dataset[:, -1]

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Feature Scaling
sc = StandardScaler()
X_train_scaled = sc.fit_transform(X_train)
X_test_scaled = sc.transform(X_test)

# Train the model

classifiers = []

classifiers.append(LogisticRegression(random_state=0))
classifiers.append(KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2))
classifiers.append(SVC(kernel='rbf', random_state=0))

titles = ['Logistic Regression', 'K-Nearest Neighbors', 'Optimal Hyperplane (SVM)']

fig, axes = plt.subplots(1, 3, figsize=(18, 5))  # 1 row, 3 columns

for i, classifier in enumerate(classifiers):

    classifier.fit(X_train_scaled, y_train)

    # Predicting the test set results
    y_pred = classifier.predict(X_test_scaled)

    # Making the confusion matrix and checking accuracy
    print(titles[i])
    print(confusion_matrix(y_test, y_pred))
    print(accuracy_score(y_test, y_pred))
    print()

    x_min, x_max = X_train[:, 0].min() - 10,    X_train[:, 0].max() + 10
    y_min, y_max = X_train[:, 1].min() - 10000, X_train[:, 1].max() + 10000

    x_range = np.arange(start=x_min, stop=x_max, step=0.1)
    y_range = np.arange(start=y_min, stop=y_max, step=1000)
    axis_1, axis_2 = np.meshgrid(x_range, y_range)

    grid_points = sc.transform(np.array([axis_1.ravel(), axis_2.ravel()]).T)
    predictions = classifier.predict(grid_points).reshape(axis_1.shape)

    ax = axes[i]
    ax.contourf(axis_1, axis_2, predictions, alpha=0.75, cmap=ListedColormap(['white', 'grey']))
    ax.set_xlim(axis_1.min(), axis_1.max())
    ax.set_ylim(axis_2.min(), axis_2.max())

    ax.scatter(X_train[y_train == 0, 0], X_train[y_train == 0, 1], color='red', label='Training, class 0')
    ax.scatter( X_train[y_train == 1, 0], X_train[y_train == 1, 1], color='green', label='Training, class 1')
    ax.scatter(X_test[y_test == 0, 0], X_test[y_test == 0, 1], color='yellow', label='Test, class 0')
    ax.scatter(X_test[y_test == 1, 0], X_test[y_test == 1, 1], color='blue', label='Test, class 1')

    ax.set_title(titles[i])
    ax.set_xlabel('Age')    
    ax.set_ylabel('Estimated Salary')
    ax.legend()

plt.tight_layout()
plt.show()
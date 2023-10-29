import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

# X data
X = np.genfromtxt('ex2_x_data.csv', delimiter=',', dtype=float)

# Y data
Y = np.genfromtxt('ex2_y_data.csv', dtype=float)

print('X shape:', X.shape)
print('Y shape:', Y.shape)

# Shuffle the data
random_state = np.random.RandomState(42)
shuffle = random_state.permutation(X.shape[0])
X = X[shuffle]
Y = Y[shuffle]

# Divide into 2/3 for training and 1/3 for testing
split = int(2 * X.shape[0] / 3)
X_train, X_test = X[:split], X[split:]
y_train, y_test = Y[:split], Y[split:]

print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)
print('y_train shape:', y_train.shape)
print('y_test shape:', y_test.shape)

#  logistic regression
classifier = LogisticRegression(max_iter=9000)

classifier.fit(X_train, y_train)

# Evaluate the classifier on the test data
y_pred = classifier.predict(X_test)

# confusion matrix
matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(matrix)


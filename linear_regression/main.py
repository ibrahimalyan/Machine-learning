import pandas as pd
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
import numpy as np

# Read the CSV file into a pandas dataframe
data = pd.read_csv('cancer_data.csv', header=None, delimiter=',')

# extract the y Vector
Ymatrix= data.iloc[:, -1]

# Extract all the columns except the last one into X matrix
Xmatrix = data.iloc[:, :-1]

#find out what n is(number of columns extracted)
n = Xmatrix.shape[1]
#find out what m number is (number of rows extracted)
m = Xmatrix.shape[0]

# Add a column of ones to X
ones_column = np.ones((Xmatrix.shape[0], 1))
Xmatrix_with_ones = np.hstack((Xmatrix, ones_column))

# Normalize the feature matrix
scaler = StandardScaler()
X_norm = scaler.fit_transform(Xmatrix_with_ones)

# Check that the mean is 0 and the standard deviation is 1

print("checking if the mean is 0")
print(X_norm.mean(axis=0))
print("checking that the standard deviation is 1")
print(X_norm.std(axis=0))
print("end Of Checks\n-------------------------------\n")

#predict function
def predict(x, theta):
    return x@theta

#compute_cost function
def compute_cost(theta, X, y):
#equation of cost
    J = np.mean(np.square(X@theta-y))
    return J

#compute_gradient function
def compute_gradient(theta, X, y):
# equation of gradient
    predictions = X.dot(theta)
    errors = predictions - y
    gradient = X.T.dot(errors) / len(X)
    return gradient


#gradient_descent function
def gradient_descent(X, y, theta, alpha, num_iters):
    J_history = np.zeros(num_iters)
    for i in range(num_iters):
        theta = theta - alpha * compute_gradient(theta, X, y)
        J_history[i] = compute_cost(theta, X, y)
    return theta, J_history

#run it with couple of versions of alpah
theta = np.zeros(X_norm.shape[1])
num_iters = 10
#run alpha with some value like (1,0.1,0.01,0.001)
alphas = [1, 0.1, 0.01, 0.001]
for alpha in alphas:
    theta, J_history = gradient_descent(X_norm, Ymatrix, theta, alpha, num_iters)
    plt.plot(J_history, label=f"alpha = {alpha}")

plt.xlabel("Iterations")
plt.ylabel("Cost Function")
plt.title("Gradient Descent")
plt.legend()
plt.show()

#mini_batch_ function
def mini_batch_gradient_descent(X, y, theta, alpha, num_iters, batch_size):
    m = len(y)
#J_history a vector of zeros
    J_history = np.zeros(num_iters)
    for i in range(num_iters):
        J_history[i] = compute_cost(X,y,theta)
#creates a batch of the entire dataset by copying X and y into X_batch and y_batch variables.
        X_batch = np.array(X)
        y_batch = np.array(y)
# Update theta using the subset of the data
        theta = theta - (alpha*15) * compute_gradient(theta, X_batch, y_batch)
#The updated theta value is then used to compute the cost function value for the updated parameters and stored in J_history.
        J_history[i] = compute_cost(theta, X, y)
    return theta, J_history

theta = np.zeros(X_norm.shape[1])
num_iters = 10
batch_size = 50

theta, J_history = mini_batch_gradient_descent(X_norm, Ymatrix, theta, 0.01, num_iters, batch_size)

plt.plot(J_history)
plt.xlabel("Iterations")
plt.ylabel("Cost Function")
plt.title("Mini-Batch Gradient Descent")
plt.show()

#momentum function
def linear_regression_with_momentum(X, y, alpha=0.01, beta=0.9, epsilon=1e-8, max_iters=1000):
    m, n = X.shape
#nitializes theta to a vector of zeros
    theta = np.zeros(n)
    v = np.zeros(n)
    iterations = 0
    cost_history = []
    while iterations < max_iters:
        h = X @ theta
# computes the gradient of the cost function using the predicted and actual values of y and the feature matrix X.
        cost = np.sum((h - y) ** 2) / (2 * m)
        cost_history.append(cost)
        grad = X.T @ (h - y) / m
# updates the exponentially weighted moving average of the gradient using the beta and alpha values.
        v = beta * v + (alpha) * grad
        theta = theta - v
        iterations += 1
    return theta, cost_history

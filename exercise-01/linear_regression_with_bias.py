import numpy as np


def predict(X, w, b):
    return X * w + b


def loss(X, Y, w, b):
    return np.average((predict(X, w, b) - Y) ** 2)


def train(X, Y, iterations, lr):
    w = b = 0
    for i in range(iterations):
        current_loss = loss(X, Y, w, b)
        print("Iteration %4d => Loss: %.6f" % (i, current_loss))

        if loss(X, Y, w + lr, b) < current_loss:
            w += lr
        elif loss(X, Y, w - lr, b) < current_loss:
            w -= lr
        elif loss(X, Y, w, b + lr) < current_loss:
            b += lr
        elif loss(X, Y, w, b - lr) < current_loss:
            b -= lr
        else:
            return w, b

    raise Exception("Couldn't converge within %d iterations" % iterations)


# Import the dataset
X, Y = np.loadtxt("pizza.txt", skiprows=1, unpack=True)

# Train the system
w, b = train(X, Y, iterations=10000, lr=0.01)
print("\nw=%.3f, b=%.3f" % (w, b))

# Predict the number of pizzas
print("Prediction: x=%d => y=%.2f" % (20, predict(20, w, b)))

# Plot the chart
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()
plt.plot(X, Y, "bo")
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xlabel("Reservations", fontsize=30)
plt.ylabel("Pizzas", fontsize=30)
x_edge, y_edge = 50, 50
plt.axis([0, x_edge, 0, y_edge])
plt.plot([0, x_edge], [b, predict(x_edge, w, b)], linewidth=1.0, color="g")
plt.show()

# The original function you provided
def predict(X, w, b=0):
    return X * w + b

# Your labeled training data
reservations = [13, 2, 14]
pizzas_sold = [33, 16, 32]

# Our initial guess for the weight (w)
w_guess = 2

# We will store our predictions here
predictions = []

# Loop through each data point to make a prediction
for x_value in reservations:
    y_hat = predict(x_value, w_guess)
    predictions.append(y_hat)

print("Reservations (X):", reservations)
print("Actual Pizzas (y):", pizzas_sold)
print("Predicted Pizzas (y_hat):", predictions)

# Now, let's calculate the error for each prediction
errors = []
total_error = 0

# A simple loop to calculate the error
for i in range(len(reservations)):
    error = pizzas_sold[i] - predictions[i]
    errors.append(error)
    total_error += abs(error) # We use the absolute value to sum up the magnitude of the errors

print("Errors (y - y_hat):", errors)
print("Total Absolute Error:", total_error)

# The example you asked for:
print("\nExample from your prompt:")
print(f"Predicting with X=13 and a guessed weight (w) of 2 gives a prediction of {predict(13, 2)}")
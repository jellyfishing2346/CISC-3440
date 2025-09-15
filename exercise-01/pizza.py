import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import seaborn as sns

# Path to your data file
file_path = 'pizza.txt'

x_values = []
y_values = []
with open(file_path, 'r') as file:
    # Skip the header row
    next(file)
    for line in file:
        # Split the line by whitespace
        try:
            x, y = map(float, line.strip().split())
            x_values.append(x)
            y_values.append(y)
        except ValueError:
            # Skip any blank or invalid lines
            continue

# Ensure there is data to train on
if not x_values:
    print("Error: No valid data found in the file after skipping header.")
else:
    # Convert lists to NumPy arrays and reshape for scikit-learn
    X = np.array(x_values).reshape(-1, 1)
    y = np.array(y_values)

    # Create and train the linear regression model
    model = LinearRegression()
    model.fit(X, y)

    # Get the coefficients (w) and intercept (b)
    w = model.coef_[0]
    b = model.intercept_

    print(f"Model trained successfully!")

    # Print the linear regression equation in the format y = wx + b
    print(f"The linear regression equation is: y = {w:.2f}x + {b:.2f}")

    # The predict function from the image
    def predict(X, w, b):
        return X * w + b

    # You can use your new function to make a prediction
    # For example, to predict the number of pizzas for 15 reservations:
    new_reservations = 15
    predicted_pizzas = predict(new_reservations, w, b)
    print(f"Prediction for {new_reservations} reservations: {predicted_pizzas:.2f} pizzas.")

    # Generate points for the regression line
    x_line = np.linspace(min(x_values), max(x_values), 100).reshape(-1, 1)
    y_line = model.predict(x_line)

    # --- Graphing the data ---
    sns.set() # activate seaborn for better styling
    
    plt.axis([0, 50, 0, 50]) # scale axes 0 to 50
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)

    plt.xlabel("Reservations", fontsize=30)
    plt.ylabel("Pizzas", fontsize=30)
    
    # Plot the original data points
    plt.plot(X, y, 'bo', label='Data Points') # 'bo' means blue circles
    
    # Plot the regression line
    plt.plot(x_line, y_line, 'r-', label='Linear Regression') # 'r-' means red line
    
    plt.legend() # Show the legend
    plt.show() # display the graph

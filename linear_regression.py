import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =========================
# Load Dataset
# =========================
df = pd.read_csv("data.csv")

x = df["Crabs"].values
y = df["Clams"].values

# =========================
# Initialize Parameters
# =========================
m = 0.0
c = 0.0

learning_rate = 0.001
epochs = 50000

n = len(x)
cost_history = []

# =========================
# Training
# =========================
for epoch in range(epochs):

    # Predictions
    y_pred = m * x + c

    # Cost Function (MSE/2)
    cost = (1 / (2 * n)) * np.sum((y_pred - y) ** 2)
    cost_history.append(cost)

    # Gradients
    dm = (1 / n) * np.sum((y_pred - y) * x)
    dc = (1 / n) * np.sum(y_pred - y)

    # Update Parameters
    m = m - learning_rate * dm
    c = c - learning_rate * dc

    # Print Progress
    if epoch % 500 == 0:
        print(f"Epoch: {epoch}")
        print(f"Cost : {cost:.4f}")
        print(f"m    : {m:.4f}")
        print(f"c    : {c:.4f}")
        print("-" * 30)

# =========================
# Final Model
# =========================
print("\nTraining Complete")
print(f"Final Slope (m): {m:.4f}")
print(f"Final Intercept (c): {c:.4f}")

# =========================
# Test Prediction
# =========================
crabs = 3

prediction = m * crabs + c

print(f"\nPredicted Clams for {crabs} crabs = {prediction:.2f}")


# =========================
# Graph 1: Data + Best Fit Line
# =========================

plt.figure(figsize=(8,5))

plt.scatter(x, y, label="Actual Data")

x_line = np.linspace(min(x), max(x), 100)
y_line = m * x_line + c

plt.plot(x_line, y_line, label="Best Fit Line")

plt.xlabel("Crabs")
plt.ylabel("Clams")
plt.title("Linear Regression")
plt.legend()
plt.grid(True)

plt.show()

# Predictions
y_pred = m * x + c

# R² Score
ss_res = np.sum((y - y_pred) ** 2)

ss_tot = np.sum((y - np.mean(y)) ** 2)

r2 = 1 - (ss_res / ss_tot)

print(f"R² Score = {r2:.4f}")
print(cost_history)


# =========================
# Graph 2: Cost vs Epoch
# =========================

plt.figure(figsize=(8,5))

plt.plot(cost_history)

plt.xlabel("Epoch")
plt.ylabel("Cost (J)")
plt.title("Cost Function Convergence")
plt.grid(True)

plt.show()


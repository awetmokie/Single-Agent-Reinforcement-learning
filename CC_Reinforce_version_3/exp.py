import numpy as np

# Define the number of random variables
n = 10

# Generate a set of random variables
X = np.random.normal(size=n)

# Compute the cumulative sum
S = np.cumsum(X)

# Compute the expected value of each random variable
E = np.mean(X)

# Compute the expected value of the cumulative sum
E_S = n * E

# Print the results
print("Random variables:", X)
print("Cumulative sum:", S)
print("Expected value of each random variable:", E)
print("Expected value of the cumulative sum:", E_S)

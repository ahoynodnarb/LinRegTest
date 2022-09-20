from matplotlib import pyplot as plt
import numpy as np


def h(a, b, x):
    return a + b * x


def cost(predictions, expectations, n):
    total = 0
    for expected, predicted in zip(expectations, predictions):
        total += (expected - predicted) ** 2
    return total / n


def d_slope(predictions, expectations, inputs, n):
    return -(2 / n) * sum(
        [inputs[i] * (expectations[i] - predictions[i]) for i in range(n)]
    )


def d_intercept(predictions, expectations, n):
    return -(2 / n) * sum([expectations[i] - predictions[i] for i in range(n)])


def gradient_descent(inputs, expectations):
    a = b = 0
    iterations = 5000
    n = len(inputs)
    learning_rate = 0.0001
    prev_cost = -1
    for _ in range(iterations):
        predictions = [h(a, b, x) for x in inputs]
        c = cost(predictions, expectations, n)
        if prev_cost == -1 or c < prev_cost:
            a_d = d_intercept(predictions, expectations, n)
            b_d = d_slope(predictions, expectations, inputs, n)
            a = a - learning_rate * a_d
            b = b - learning_rate * b_d
    return (a, b)


data = np.genfromtxt("data.csv", delimiter=",", dtype=np.int32)
x = [d[0] for d in data[1:]]
y = [d[1] for d in data[1:]]
output = gradient_descent(x, y)
print(f"a = {output[0]} b = {output[1]}")
reg = [h(output[0], output[1], x[i]) for i in range(len(x))]
plt.plot(x, y, color="blue")
plt.plot(x, reg, color="red")
plt.show()

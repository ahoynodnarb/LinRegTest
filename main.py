# import numpy as np
import csv


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
    iterations = 50000
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


with open("data.csv", "r") as fin:
    reader = csv.reader(fin, quoting=csv.QUOTE_NONNUMERIC, delimiter=",")
    x = []
    y = []
    next(reader, None)
    for row in reader:
        x.append(row[0])
        y.append(row[1])

    output = gradient_descent(x, y)
    print(f"h = {output[0]}x + {output[1]}")

import numpy as np
import matplotlib.pyplot as plt
import sys
import math
from scipy.stats import beta


def combination(n, m):
    return math.factorial(n) / (math.factorial(m) * math.factorial(n-m))


def binomial(p, n, m):
    return combination(n, m) * (p ** m) * ((1-p) ** (n-m))


def online_learning(inputs, a, b):
    # prior = 0
    for i in range(len(inputs)):
        n = len(inputs[i]) - 1
        m = 0
        for j in range(n):
            if inputs[i][j] == '1':
                m += 1
        p = m / n
        print("Binomial likelihood by MLE: %d/%d: " % (m, n))
        a = m + a
        b = n - m + b
        draw_beta_function(a, b)
        print("Parameters of beta distribution:")
        print("a: ", a)
        print("b: ", b)
        print("mean: ", a/(a+b))
        print("variance: ", (a * b) / (np.power(a+b, 2) * (a + b + 1)))
        print("------------------------------------")


def draw_beta_function(a, b):
    fig, ax = plt.subplots(1, 1)
    x = np.linspace(0, 1, 1000)
    ax.plot(x, beta.pdf(x, a, b), 'r-', lw=5, alpha=0.6, label='beta pdf')
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: ", sys.argv[0], "<a> <b>")
        print("Use default a and b value: a=2, b=2")
        a = 2
        b = 2
    else:
        a = float(sys.argv[1])
        b = float(sys.argv[2])

    file = open('/Users/islab/PycharmProjects/ml/hw2/binary_outcomes.txt', 'r')
    inputs = file.readlines()

    draw_beta_function(a, b)
    online_learning(inputs, a, b)


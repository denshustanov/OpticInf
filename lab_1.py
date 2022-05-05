import numpy as np
import math
from matplotlib import pyplot as plt
import matplotlib

matplotlib.use('TkAgg')


def f(x, b):
    return np.cos(b * x) + 1j * np.sin(b * x)


def lagguerre_polynomial(x, n):
    return sum(
        [(-1 ** k) / math.factorial(k) * math.factorial(n) / (math.factorial(k) * math.factorial(n - k)) * (x ** k) for
         k in range(n + 1)])


def k(xi, x, alpha):
    return np.exp(-alpha * x * xi) * lagguerre_polynomial(x * xi, 3)


def plot_f(b_s):
    fig, ax = plt.subplots(2, len(b_s))
    x = np.linspace(0, 100, 1000)
    ax[0, 0].set(ylabel='Amplitude')
    ax[1, 0].set(ylabel='Phase')
    for i in range(len(b_s)):
        print(i)
        v = f(x, b_s[i])
        ax[0, i].set_title(r'$\beta$' + ' = ' + str(b_s[i]))
        ax[0, i].plot(x, np.abs(v))
        ax[1, i].plot(x, np.angle(v), color='orange')
        ax[1, i].set(xlabel='x')
    plt.show()


def integral_operator(c, p, q, alpha, beta, n, m):
    h_x = 2 * c / n
    x = [-c + i * h_x for i in range(n)]
    fx = [f(xk, beta) for xk in x]
    h_xi = (q - p) / m
    xis = [p + i * h_xi for i in range(m)]
    vals = []
    for xi in xis:
        vals.append(sum([
            k(xi, x[i], alpha) * fx[i] * h_x for i in range(n)
        ]))
    return vals


def plot_integral_output():
    c = 3
    p = -3
    q = 3
    alpha = 0.5
    beta = 0.1
    n = 100
    m = 1000

    p_s = [1, 2, 3]

    fig, ax = plt.subplots(2, len(p_s))
    ax[0, 0].set(ylabel='Amplitude')
    ax[1, 0].set(ylabel='Phase')

    for i in range(len(p_s)):
        v = integral_operator(3, -p_s[i], p_s[i], alpha, beta, n, m)
        xi = np.linspace(-p_s[i], p_s[i], m)
        ax[0, i].plot(xi, np.abs(v))
        ax[1, i].plot(xi, np.angle(v), color='orange')
        ax[1, i].set(xlabel='x')
        ax[0, i].set_title('p = ' + str(-p_s[i]) + ', q = ' + str(p_s[i]))
    plt.show()


def plot_integral_alpha():
    c = 3
    p = -3
    q = 3
    alpha = 0.5
    beta = 0.1
    n = 100
    m = 1000

    a_s = [0.1, 0.2, 0.5, 1]

    fig, ax = plt.subplots(2, len(a_s))
    ax[0, 0].set(ylabel='Amplitude')
    ax[1, 0].set(ylabel='Phase')

    for i in range(len(a_s)):
        v = integral_operator(c, p, q, a_s[i], beta, n, m)
        xi = np.linspace(p, q, m)
        ax[0, i].plot(xi, np.abs(v))
        ax[1, i].plot(xi, np.angle(v), color='orange')
        ax[1, i].set(xlabel=r'$\xi$')
        ax[0, i].set_title(r'$\alpha$' + ' = ' + str(a_s[i]))
    plt.show()


def plot_integral_region():
    p = -3
    q = 3
    alpha = 0.5
    beta = 0.1
    n = 100
    m = 1000

    c_s = [1, 2, 3, 4]

    fig, ax = plt.subplots(2, len(c_s))
    ax[0, 0].set(ylabel='Amplitude')
    ax[1, 0].set(ylabel='Phase')

    for i in range(len(c_s)):
        v = integral_operator(c_s[i], p, q, alpha, beta, n, m)
        xi = np.linspace(p, q, m)
        ax[0, i].plot(xi, np.abs(v))
        ax[1, i].plot(xi, np.angle(v), color='orange')
        ax[1, i].set(xlabel=r'$\xi$')
        ax[0, i].set_title('c = ' + str(c_s[i]))
    plt.show()

plot_f([0.1, 0.5, 1, -0.1, -0.5])
plot_integral_output()
plot_integral_alpha()
plot_integral_region()

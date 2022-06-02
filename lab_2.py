import numpy as np
import math
from scipy.fft import fft, fftshift
from matplotlib import pyplot as plt
import matplotlib

matplotlib.use('TkAgg')


def gaussian_beam(x):
    return np.exp(-np.power(x, 2))


def rect(x):
    return np.where(np.abs((x + 2) / 2) > 1 / 2, 0, 1)


def gaussian_beam_2d(x, y):
    return np.exp(-np.power(x, 2) - np.power(y, 2))


def rect_2d(x, y):
    return np.where(np.abs((x + 2) / 2) > 1 / 2, 0, 1) * np.where(np.abs((y + 2) / 2) > 1 / 2, 0, 1)


def analytic_transform(x):
    return np.exp(-2 * 1j * x) * np.sin(x) / x


def single_dim_finite_fourier_with_fft(n, m, a, field):
    x = np.linspace(-a, a, n)
    f = field(x)
    k = (m - n) // 2
    f = np.append(f, np.zeros(k))
    f = np.insert(f, 0, np.zeros(k))
    f = fftshift(f)
    f = fft(f) * (2.0 * a / n)
    f = fftshift(f)
    f = f[f.size // 2 - n // 2:f.size // 2 + n // 2]
    b = n ** 2 / (4 * a * m)
    x = np.linspace(-b, b, f.size)
    return f, x


def single_dim_finite_fourier(u, a, n, field):
    hx = 2 * a / n
    x = np.linspace(-a, a, n)
    f = field(x)
    v = np.sum(np.multiply(f, np.exp(-2 * np.pi * u * x * 1j)))
    return v * hx


def double_dim_fourier(a, n, m, field):
   x = np.linspace(-a, a, n)
   f = np.array([field(x, i) for i in x])
   ft = np.zeros([m, m])
   a = (m - n) // 2
   b = (m + n) // 2
   ft[a:b, a:b] += f
   ft = np.fft.ifftshift(ft)
   ft = np.fft.fft2(ft)
   ft = np.fft.ifftshift(ft)
   ft = ft[a:b, a:b]
   b = n ** 2 / (4 * a * m)
   x = np.linspace(-b, b, n)
   return ft, x


def plot_2d(a, n, field):
    x = np.linspace(-a, a, n)
    f = np.array([field(x, i) for i in x])
    plt.pcolor(x, x, f)
    plt.show()


r, x = double_dim_fourier(5, 500, 5000, gaussian_beam_2d)
plt.set_cmap('plasma')
fig, ax = plt.subplots(2)

ax[0].pcolor(x, x, np.abs(r))
ax[1].pcolor(x, x, np.angle(r))
ax[0].set(ylabel='Amplitude')
ax[1].set(ylabel='Phase')

plt.show()

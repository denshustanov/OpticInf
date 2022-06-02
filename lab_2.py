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
    return np.exp(4 * np.pi * 1j * x) * np.sin(2 * np.pi * x) / (x * np.pi)


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


def single_dim_ff_from_v(v, m, a):
    n = len(v)
    k = (m - n) // 2
    f = np.append(v, np.zeros(k))
    f = np.insert(f, 0, np.zeros(k))
    f = fftshift(f)
    f = fft(f) * (2.0 * a / n)
    f = fftshift(f)
    f = f[f.size // 2 - n // 2:f.size // 2 + n // 2]
    return f


def single_dim_finite_fourier(u, a, n, field):
    hx = 2 * a / n
    x = np.linspace(-a, a, n)
    f = field(x)
    v = np.sum(np.multiply(f, np.exp(-2 * np.pi * u * x * 1j)))
    return v * hx


def double_dim_fourier(a, n, m, field):
    x = np.linspace(-a, a, n)
    f = np.array([field(x, i) for i in x])
    for i in range(len(f)):
        f[i] = single_dim_ff_from_v(f[i], m, a)
    f = f.T
    for i in range(len(f)):
        f[i] = single_dim_ff_from_v(f[i], m, a)
    f = f.T
    b = n ** 2 / (4 * a * m)
    x = np.linspace(-b, b, n)
    return f, x


def double_dim_fourier_fft2(a, n, m, field):
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


#
#
fig, ax = plt.subplots(2, 3)

x = np.linspace(-2, 2, 100)
y_1, x_1 = single_dim_finite_fourier_with_fft(100, 128, 5, gaussian_beam)
y_2, x_2 = single_dim_finite_fourier_with_fft(100, 256, 5, gaussian_beam)
y_3, x_3 = single_dim_finite_fourier_with_fft(100, 512, 5, gaussian_beam)
# y_2 = np.array([single_dim_finite_fourier(x_i, 5, 100, gaussian_beam) for x_i in x])
ax[0, 0].plot(x_1, np.abs(y_1))
ax[1, 0].plot(x_1, np.angle(y_1), color='orange')
ax[0, 1].plot(x_2, np.abs(y_2))
ax[1, 1].plot(x_2, np.angle(y_2), color='orange')
ax[0, 2].plot(x_3, np.abs(y_3))
ax[1, 2].plot(x_3, np.angle(y_3), color='orange')
ax[0, 0].set(ylabel='Amplitude')
ax[1, 0].set(ylabel='Phase')
ax[0, 0].set_title('N=100, M=128')
ax[0, 1].set_title('N=100, M=256')
ax[0, 2].set_title('N=100, M=512')
plt.show()
# # plot_2d(5, 100, rect_2d)
# #

# plot rect fourier transform
r, x = double_dim_fourier_fft2(5, 100, 256, rect_2d)
plt.set_cmap('plasma')
fig, ax = plt.subplots(2, 2)

amp = np.abs(r)
ph = np.angle(r)
x_0 = np.linspace(-5, 5, 100)
r = [rect_2d(x_0, x_i) for x_i in x_0]
ax[0, 0].pcolor(x, x, np.abs(r))
ax[1, 0].pcolor(x, x, np.angle(r))
ax[0, 1].pcolor(x, x, amp)
ax[1, 1].pcolor(x, x, ph)
ax[0, 0].set(ylabel='Amplitude')
ax[1, 0].set(ylabel='Phase')
ax[0, 0].set_title('Rect 2D')
ax[0, 1].set_title('Fourier transform')
plt.show()


import numpy as np
import math
from scipy.fft import fft
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('TkAgg')


def gaussian_beam(x):
    return np.exp(-np.power(x, 2))


def var_field(x):
    return 0


def single_dim_finite_fourier_with_fft(n, m, a, field):
    x = np.linspace(-a, a, n)
    f = field(x)
    f = np.append(f, np.zeros((m-n)//2))
    f = np.insert(f, 0, np.zeros((m-n)//2))
    f = np.append(f[f.size//2:], f[:f.size//2])
    f = fft(f)*(2.0*a/n)
    f = np.append(f[f.size // 2:], f[:f.size // 2])
    f = f[f.size // 2 - n // 2:f.size // 2 + n // 2]
    b = n**2/(4*a*m)
    x = np.linspace(-b, b, f.size)
    return f, x


def single_dim_finite_fourier(u, a, n, field):
    hx = 2*a/n
    x = np.linspace(-a, a, n)
    f = field(x)
    v = np.sum(np.multiply(f, np.exp(-2*np.pi*u*x*1j)))
    return v*hx


fy1, fx1 = single_dim_finite_fourier_with_fft(500, 2**14, 5, gaussian_beam)
fy2 = [single_dim_finite_fourier(u, 5, 500, gaussian_beam) for u in fx1]

fig, ax = plt.subplots(2, 2)
ax[0, 0].plot(fx1, np.abs(fy1))
ax[1, 0].plot(fx1, np.angle(fy1), color='orange')
ax[0, 0].set(ylabel='Amplitude')
ax[1, 0].set(ylabel='Phase')
ax[0, 1].plot(fx1, np.abs(fy2))
ax[1, 1].plot(fx1, np.angle(fy2), color='orange')
plt.show()

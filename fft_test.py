from matplotlib import pyplot as plt
import numpy as np
import matplotlib
from scipy.fft import fft, fftfreq
matplotlib.use('TkAgg')
N = 1000
T = 1.0/800
x = np.linspace(0, N*T, N, endpoint=False)
y = np.sin(x) + 0.5*np.sin(1000*x)
f = fft(y)
yf = fft(y)
xf = fftfreq(N, T)[:N//2]
plt.plot(xf, 2.0/N * np.abs(yf[0:N//2]))
plt.show()
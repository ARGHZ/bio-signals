import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

matriz = np.genfromtxt('../data/datosProy01.csv', delimiter=',')

sig = matriz[:, 0]
sig_noise = matriz[:, 0]
corr = signal.correlate(sig, sig_noise, mode='same')

fig, (ax_orig, ax_noise, ax_corr) = plt.subplots(3, 1, sharex=True)
ax_orig.plot(sig)
ax_orig.set_title('Original signal')
ax_noise.plot(sig_noise)
ax_noise.set_title('Signal with noise')
ax_corr.plot(corr)
ax_corr.axhline(0.5, ls=':')
ax_corr.set_title('Cross-correlated')
ax_orig.margins(0, 0.1)
fig.tight_layout()
fig.show()
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fftpack import fft, fftshift

np.random.seed(1234)

def autocorrelationmatrix(target_matrix):
    m, n = matriz.shape

    iterator = np.arange(0, n)
    n_correlations = [[] for n in range(n)]
    for j in iterator:

        current_vector = matriz[:, j]
        correlation = signal.correlate(current_vector, current_vector, mode='same')

        n_correlations[j] = correlation

    return  n_correlations

fs = 10e3
N = 1e5
amp = 2*np.sqrt(2)
freq = 1234.0
noise_power = 0.001 * fs / 2
time = np.arange(N) / fs
x = amp*np.sin(2*np.pi*freq*time)
x += np.random.normal(scale=np.sqrt(noise_power), size=time.shape)

matriz = np.genfromtxt('data/datosProy01.csv', delimiter=',')

correlations = np.array(autocorrelationmatrix(matriz))
mean_of_correlations = np.mean(correlations, axis=0)
spectrum_mean_of_correlations = fftshift(fft(mean_of_correlations))

mean_of_samples = np.mean(matriz, axis=1)
mean_of_samples_correlation = signal.correlate(mean_of_samples, mean_of_samples, mode='same')
spectrum_mean_of_samples_correlation = fftshift(fft(mean_of_samples_correlation))

f, Pper_spec = signal.periodogram(mean_of_correlations, scaling='spectrum')
plt.semilogy(f, Pper_spec)
plt.xlabel('frequency [Hz]')
plt.ylabel('PSD')
plt.title('Periodograma del Promedio de Auto-correlaciones')
plt.grid()
#plt.ylim([1e11, 1e18])
plt.show()

f, Pper_spec = signal.periodogram(mean_of_samples_correlation, scaling='spectrum')
plt.semilogy(f, Pper_spec)
plt.xlabel('frequency [Hz]')
plt.ylabel('PSD')
plt.title('Periodograma del Promedio de Auto-correlaciones')
plt.grid()
#plt.ylim([1e4, 1e16])
plt.show()

vector = mean_of_samples
sm.graphics.tsa.plot_acf(vector, fft=True)
plt.show()


print 'Hi there, you are in main file'
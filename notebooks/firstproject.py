import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, fftshift
from scipy.signal import correlate


def f(x_values):
    equation = x_values[0] + 0.5 * x_values[1] + 0.25 * x_values[2]
    return equation


def arithmeticmeanofmatrix(target_matrix):

    new_matrix = np.mean(target_matrix, 1)

    return new_matrix


def autocorrelationmatrix(target_matrix):
    m, n = matriz.shape

    iterator = np.arange(0, n)
    n_correlations = [[] for n in range(n)]
    for j in iterator:

        current_vector = matriz[:, j]
        correlation = correlate(current_vector, current_vector, mode='same')

        n_correlations[j] = correlation

    return  n_correlations


matriz = np.genfromtxt('../data/datosProy01.csv', delimiter=',')

N = matriz.shape[0]
lag = 10

M = 2*N - 1
start = -(N / 2)  # -pi+pi/M
step = 1  # 2*pi/M
stop = (N / 2)  # pi
freq_vector = np.arange(start, stop, step) # w = start:step:stop


correlations = np.array(autocorrelationmatrix(matriz))
mean_of_correlations = np.mean(correlations, axis=0)
spectrum_mean_of_correlations = abs(fftshift(fft(mean_of_correlations)))

mean_of_samples = np.mean(matriz, axis=1)
mean_of_samples_correlation = correlate(mean_of_samples, mean_of_samples, mode='same')
spectrum_mean_of_samples_correlation = abs(fftshift(fft(mean_of_samples_correlation)))

# Select five samples randomly
random_idx = np.arange(0, matriz.shape[1] + 1)
np.random.shuffle(random_idx)
random_idx = [33, 5, 4, 42, 31]
#random_idx = random_idx[:5]
selected_samples = np.array(correlations)
selected_samples = selected_samples[:][random_idx].reshape(368, 5)
# Select five samples randomly

x_limits = matriz.shape[0] / 2

plt.plot(selected_samples)
plt.title('Cinco Auto-correlaciones')
plt.savefig('../data/cinco_muestras.png')
plt.show()
plt.clf()

plt.plot(mean_of_correlations)
plt.ylabel('Magnitude')
plt.xlabel('Lag')
plt.title('Funciones de auto-correlaciones promediadas')
plt.savefig('../data/promedio_funciones_autocorrelacion.png')
plt.show()
plt.clf()

plt.plot(freq_vector, spectrum_mean_of_correlations)
plt.title('Espectro del promedio de funciones')
plt.ylabel('Magnitude')
plt.xlabel('Frecuency')
#plt.xlim([-x_limits, x_limits])
plt.savefig('../data/espectro_promedio_funciones_autocorrelacion.png')
plt.show()
plt.clf()

plt.plot(mean_of_samples_correlation)
plt.title('Auto-correlacion del promedio de las muestras')
plt.ylabel('Magnitude')
plt.xlabel('Lag')
plt.savefig('../data/autocorrelacion_del_promedio.png')
plt.show()
plt.clf()

plt.plot(freq_vector, spectrum_mean_of_samples_correlation)
plt.title('Espectro de auto-correlacion del promedio de muestras')
plt.ylabel('Magnitude')
plt.xlabel('Frecuency')
plt.savefig('../data/espectro_autocorrelacion_del_promedio.png')
plt.show()

print 'estimation process has ended'
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
from math import pi
from scipy.fftpack import fft, fftshift
from scipy.signal import convolve


def f(x_values):
    equation = x_values[0] + 0.5 * x_values[1] + 0.25 * x_values[2]
    return equation


def arithmeticmean(target = np.array([])):
    return target.mean()


def averagequareddeviation(target = np.array([])):
    return target.var(target,ddof=1)


def prebuildvector(start, finish, steps):
    return np.arange(start, finish, steps)

def buildmatrix(n_rows):
    grid = []
    for i in range(n_rows):
        grid.append([])

    return grid


N = 1000
xvar = 4
lag = 10

x = sqrt(xvar) * np.random.randn(1, N+2)
x = x.flatten()

matriz = np.genfromtxt('../data/datosProy01.csv', delimiter=',')

xn_idx = [np.arange(2, N + 2), np.arange(1, N + 1), np.arange(0, N)]
xn_idx = np.array(xn_idx)

xn = x[xn_idx[0]] # x(3:N+2); % x(n)
xn1 = x[xn_idx[1]] # x(2:N+1); % x(n-1)
xn2 = x[xn_idx[2]] # xn1(1:N); % x(n-2)

y = np.array(xn + 0.5 * xn1 + 0.25 * xn2)

Ryy = np.zeros((1, 2 * N)).flatten()

first_iter_start = -N + 1
first_iter_end = N
first_iterator = np.arange(first_iter_start, first_iter_end, 1)

for k in first_iterator:
    left_idx = (1, k + 1)
    right_idx = (N + k, N)
    ndx_1 = np.arange(max(left_idx), min(right_idx))
    #print "{} - {}".format(left_idx, right_idx)
    left_idx = (1, 1 - k);
    right_idx = (N - k, N);
    ndx_2 = np.arange(max(left_idx), min(right_idx))
    print "{} - {}".format(left_idx, right_idx)
    y_ndx_1 = y[ndx_1]
    y_ndx_2 = y[ndx_2]

    equation = sum(y_ndx_1 * y_ndx_2) / N

    if  k < 0:
        y_idx = (N + k) - 1
    else:
        y_idx = (N + k)

    Ryy[y_idx] = equation
print 'First iterator has finished'
Rtrue = np.multiply(xvar, [0.0, 0.0, 1.0/4, 5.0/8, 21.0/16, 5.0/8, 1.0/4, 0.0, 0.0]) # true value of Ryy
M = 2 * N - 1

start = -pi + pi / M
step = 2 * pi / M
stop = pi
w = np.arange(start, stop, step) # frequency vector
Strue = xvar / 16 * (21 + 20 * np.cos(w) + 8 * np.cos(2 * w)) # true value of Syy
Syy_noisy = abs(fftshift(fft(Ryy))) # power spectral estimate

stp = 100 # number of points to average (stp must be evenly divisible by M+1)
till = (M + 1) / stp
Syy = [[] for i in range(till)]

second_iterator = np.arange(0, M, stp)
for i in second_iterator:

    criterion = (M + 1)  / stp
    Syy_idx = (i - 1) / stp + 1

    if i < criterion:
        # Syy((i - 1) / stp + 1) = mean(Syy_noisy(i:i + stp - 1));
        Syy_noisy_idxs = (i, i + stp - 1)
    else:
        # Syy((i - 1) / stp + 1) = mean(Syy_noisy(i - 1:i + stp - 2));
        Syy_noisy_idxs = (i - 1, i + stp - 2)
    vector = Syy_noisy[Syy_noisy_idxs[0]:Syy_noisy_idxs[1]]
    Syy[Syy_idx] = vector

x_range = [-10, -3, -2, -1, 0, 1, 2, 3, 10]
stem_x = np.arange(-10, 10 + 1)
Ryy_idx = np.arange(N - lag, N + lag + 1);
Ryy_selected = Ryy[Ryy_idx]

plt.plot(x_range, Rtrue, 'k')
plt.stem(stem_x, Ryy_selected, 'k')
plt.xlabel('Lag')
plt.ylabel('Magnitude')
plt.title('Autocorre1ation')
plt.show()

plt.figure()
Syy_noisy_idx = np.arange(1, M + 1)
print 'before prev-last chart'
plt.plot(w, Syy_noisy[Syy_noisy_idx] , 'k')
plt.xlabel( 'Normalized Frequency (rad)')
plt.ylabel( 'Magnitude')
plt.title( ' Power Spectrum')
#plt.show()

plt.figure()
w_idx = np.arange(stp/2, len(w), stp) # round(stp/2):stp:length(w)
print 'before last chart'
plt.plot(w[w_idx], Syy, '--k', w, Strue, 'k')
plt.xlabel( 'Normalized Frequency (rad)')
plt.ylabel('Magnitude')
plt.title('Power Spectrum')
#plt.show()

print 'estimation process has ended'
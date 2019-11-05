import matplotlib.pyplot as plt
import numpy as np
from scipy import io as io
from scipy import signal as spsignal


def displaycorrelationgraphs(first_vector, second_vector):
    sig = first_vector
    sig_noise = second_vector
    corr = spsignal.correlate(sig, sig_noise, mode='same')

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


def searchformaxcoefcorr(template, (start, finish), signal):
    subset = signal[start:finish]
    template_max_value_idx = np.argmax(template)
    subset_max_value_idx = np.argmax(subset)

    delay = -abs(template_max_value_idx - subset_max_value_idx)
    init, end = start - delay, finish - delay

    relocated_subset = data[init:end]

    shape_diff = abs(template.shape[0] - relocated_subset.shape[0])
    if shape_diff > 0:
        filling_arr = np.zeros(shape_diff)
        relocated_subset = np.append(relocated_subset, filling_arr)
    new_coef_corr = np.corrcoef(template, relocated_subset)
    coef_corr = round(new_coef_corr[0, 1], 4)
    merged_vectors = np.array([template, relocated_subset])
    merged_vectors = np.mean(merged_vectors, axis=0)

    plt.plot(template, 'k', relocated_subset, 'c', merged_vectors, 'r--')
    title = 'PQRST with CoefCorr {}'.format(coef_corr)
    plt.title(title)
    plt.margins(0, 0.1)
    plt.tight_layout()
    #plt.show()
    plt.clf()

    return {'coef_corr': coef_corr, 'vector': merged_vectors}


def selectqrsfromsignal(data):
    signal_length = len(data)
    init_idx, end_idx = 0, 700 + 1
    subset_template = data[init_idx:end_idx]
    step_length = abs(init_idx - end_idx)
    step_iter = np.arange(0, signal_length, step_length)
    step_iter = step_iter[1:]

    averaged_signal = []
    for idx in step_iter:
        start, finish = idx - step_length, idx
        #print '[{}, {}]'.format(start, finish)
        coef_corr = searchformaxcoefcorr(subset_template, (start, finish), data)
        averaged_PQRST = coef_corr['vector']
        averaged_signal.extend(averaged_PQRST)
        #print "({} - {}) === coef corr: {}".format(start, finish, coef_corr['coef_corr'])

    corr = spsignal.correlate(data, averaged_signal, 'same')

    fig, (ax_orig, ax_noise, ax_corr) = plt.subplots(3, 1, sharex=True)
    ax_orig.plot(data)
    ax_orig.set_title('Original signal')
    ax_noise.plot(averaged_signal, 'r')
    ax_noise.set_title('Signal Averaged')
    ax_corr.plot(corr, 'g')
    #ax_corr.axhline(0.5, ls=':')
    ax_corr.set_title('Cross-correlated')
    #ax_orig.margins(0, 0.1)
    #fig.tight_layout()
    fig.show()


def gridsearchbutterworthfilter(dataset, scattered_params, btype='low'):
    init_idx, end_idx = 0, 700
    subset_template = dataset[init_idx:end_idx]
    signal = subset_template

    fs = 1000.0
    for (order, freq) in scattered_params:
        n, w_n = order, freq / (fs / 2)
        title = 'Butterworth {} n = {}  w_n = {}'.format(btype, n, w_n)
        b, a = spsignal.butter(n, w_n, btype)
        filtered_signal = spsignal.lfilter(b, a, signal)
        plt.title(title)
        plt.plot(signal, 'c-', filtered_signal, 'r+')
        file_path_figure = '../data/second_project/{}.png'.format(title)
        plt.savefig(file_path_figure)
        #plt.show()
        plt.clf()


if __name__ == '__main__':
    #mat = io.loadmat('file.mat')
    data = np.loadtxt('../data/ecg_hfn.dat')

    selectqrsfromsignal(data)

    params = ((2, 10), (8, 20), (8, 40), (8, 70))
    # gridsearchbutterworthfilter(data, params, 'low')

    order_set = np.arange(2, 8 + 1, 1)
    freq_set = np.arange(0.5, 5 + 1, 0.5)

    params = []
    for order in order_set:
        for freq in freq_set:
            params.append((order, freq))
    #gridsearchbutterworthfilter(data, params, 'high')

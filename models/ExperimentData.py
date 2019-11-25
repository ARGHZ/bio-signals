# coding=utf-8
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import io
from statsmodels.tsa.ar_model import AR


class ExperimentData():

    def __init__(self, matditc):
        self.test = matditc['test']
        self.class_1 = np.array(matditc['clase1']).T
        self.class_2 = np.array(matditc['clase2']).T
        self.class_3 = np.array(matditc['clase3']).T

        self.class_1_mean = np.mean(self.class_1, axis=0)
        self.class_2_mean = np.mean(self.class_2, axis=0)
        self.class_3_mean = np.mean(self.class_3, axis=0)
        self.matrix = []

    def buildmatrixwithclasses(self, shuffle=False):
        shapes = (self.class_1.shape[1], self.class_2.shape[1], self.class_3.shape[1])

        fill_zeros = np.zeros((self.class_1.shape[0], abs(self.class_1.shape[1] - self.class_2.shape[1])))
        self.class_2 = np.concatenate((self.class_2, fill_zeros), axis=1)

        fill_zeros = np.zeros((self.class_1.shape[0], abs(self.class_1.shape[1] - self.class_3.shape[1])))
        self.class_3 = np.concatenate((self.class_3, fill_zeros), axis=1)

        tags = np.repeat(1, self.class_1.shape[0])
        tags = np.reshape(tags, (tags.shape[0], 1))
        self.class_1 = np.concatenate((self.class_1, tags), axis=1)

        tags = np.repeat(2, self.class_2.shape[0])
        tags = np.reshape(tags, (tags.shape[0], 1))
        self.class_2 = np.concatenate((self.class_2, tags), axis=1)

        tags = np.repeat(3, self.class_3.shape[0])
        tags = np.reshape(tags, (tags.shape[0], 1))
        self.class_3 = np.concatenate((self.class_3, tags), axis=1)

        matrix = np.concatenate((self.class_1, self.class_2, self.class_3))

        if shuffle:
            np.random.shuffle(matrix)

        self.matrix = matrix

    def extractcharacteristicsvector(self, maxlags=(1, 2, 3, 4, 5, 6, 23, 24, 25, 28, 33)):
        dataframe = {'base_models': [], 'order_p': [], 'class': []}
        for maxlag in maxlags:
            classification = {'base_models': {}, 'train_models': []}
            tag_class = 1
            template_model_fit = autoregressionmodel(self.class_1_mean, maxlag=maxlag)
            #dataframe['base_models']['class1'] = template_model_fit.params
            for class_set in self.class_1:
                class_set_model_fit = autoregressionmodel(class_set, maxlag=maxlag)
                classification['train_models'].append(np.append(class_set_model_fit.params, tag_class))

                if template_model_fit.params.shape[0] != class_set_model_fit.params.shape[0]:
                    print 'template AR model vector size: {} | class AR model vector size: {}'. \
                        format(template_model_fit.params.shape[0], class_set_model_fit.params.shape[0])
            classification['train_models'] = np.array(classification['train_models'])
            class1_coeffs = classification['train_models']

            classification = {'base_models': {}, 'train_models': []}
            tag_class = 2
            template_model_fit = autoregressionmodel(self.class_2_mean, maxlag=maxlag)
            # dataframe['base_models']['class1'] = template_model_fit.params
            for class_set in self.class_2:
                class_set_model_fit = autoregressionmodel(class_set, maxlag=maxlag)
                classification['train_models'].append(np.append(class_set_model_fit.params, tag_class))

                if template_model_fit.params.shape[0] != class_set_model_fit.params.shape[0]:
                    print 'template AR model vector size: {} | class AR model vector size: {}'. \
                        format(template_model_fit.params.shape[0], class_set_model_fit.params.shape[0])
            classification['train_models'] = np.array(classification['train_models'])
            class2_coeffs = classification['train_models']

            classification = {'base_models': {}, 'train_models': []}
            tag_class = 3
            template_model_fit = autoregressionmodel(self.class_3_mean, maxlag=maxlag)
            # dataframe['base_models']['class1'] = template_model_fit.params
            for class_set in self.class_3:
                class_set_model_fit = autoregressionmodel(class_set, maxlag=maxlag)
                classification['train_models'].append(np.append(class_set_model_fit.params, tag_class))

                if template_model_fit.params.shape[0] != class_set_model_fit.params.shape[0]:
                    print 'template AR model vector size: {} | class AR model vector size: {}'. \
                        format(template_model_fit.params.shape[0], class_set_model_fit.params.shape[0])
            classification['train_models'] = np.array(classification['train_models'])
            class3_coeffs = classification['train_models']

            all_characteristiccs_matrix = np.concatenate((class1_coeffs, class2_coeffs, class3_coeffs), axis=0)
            file_name = 'coeffs_order_{}_{}_{}.csv'.format(maxlag, 'aic', 'nc')
            np.savetxt('../data/third_project/{}'.format(file_name), all_characteristiccs_matrix, delimiter=',')

        return dataframe


def gridsearchautoregressionmodel(data):
    class_tags = ('clase1', 'clase2', 'clase3')

    for clss_name in class_tags:
        subset = data[clss_name].T
        subset = np.mean(subset, axis=0)
        model = AR(subset)
        maxlag = 40
        order = model.select_order(maxlag=maxlag, ic='bic', trend='nc')
        print '{} = {}'.format(clss_name, order)


def autoregressionmodel(dataset, maxlag, debug=False):
    model = AR(dataset)
    model_fit = model.fit(maxlag=maxlag, trend='nc')

    if model_fit.k_ar != len(model_fit.params) and model_fit.k_ar != maxlag:
        debug = True

    if debug:
        print('Lag: %s' % model_fit.k_ar)
        print('Coefficients: %s' % model_fit.params)
        print 'Number of Coefficients: {}'.format(len(model_fit.params))

    return  model_fit


if __name__ == '__main__':

    mat = io.loadmat('../data/third_project/datosProy3_2019.mat')
    gridsearchautoregressionmodel(mat)
    xperiment_data = ExperimentData(mat)
    xperiment_data.buildmatrixwithclasses()
    data = xperiment_data.extractcharacteristicsvector()
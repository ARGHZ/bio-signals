# coding=utf-8
import cProfile, pstats, StringIO
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import io
from scipy.spatial.distance import euclidean

from utiles import leerarchivo, guardararchivo, guardar_csv, contenido_csv, binarizearray
from auto_regressive import autoregressionmodel
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn import svm
from sklearn.model_selection import KFold

__author__ = 'Juan David Carrillo López'

pr = cProfile.Profile()


def learningtoclassify(t_dataset, i_iter='', data_set=[], specific_clf=[]):
    features_space = data_set

    np.random.shuffle(features_space)

    c, gamma, cache_size = 1.0, 0.1, 300

    classifiers = {'Poly-2 Kernel': svm.SVC(kernel='poly', degree=2, C=c, cache_size=cache_size),
                   'AdaBoost': AdaBoostClassifier(
                       base_estimator=DecisionTreeClassifier(max_depth=1, min_samples_leaf=1), learning_rate=0.5,
                       n_estimators=100, algorithm='SAMME'),
                   'GradientBoosting': GradientBoostingClassifier(n_estimators=100, learning_rate=0.5,
                                                                  max_depth=1, random_state=0)}

    if len(specific_clf) >= 1:
        type_classifier = {selected_clf.split('_')[1]: None for selected_clf in specific_clf}
    else:
        type_classifier = {'multi': None}

    x = features_space[:, :features_space.shape[1] - 1]

    kf_total = KFold(n_splits=10)
    for type_clf in type_classifier.keys():
        if len(specific_clf) == 0:
            general_metrics = {'Poly-2 Kernel': [[], [], [], []], 'AdaBoost': [[], [], [], []],
                               'GradientBoosting': [[], [], [], []]}
            for i in ('binary', 'multi'):
                for j in classifiers.keys():
                    specific_clf.append('{}_{}_kfolds_{}'.format(t_dataset, i, j))
        else:
            general_metrics = {selected_clf.split('_')[3]: [[], [], [], []] for selected_clf in specific_clf}
        if type_clf == 'binary':
            y = np.array(binarizearray(features_space[:, features_space.shape[1] - 1:].ravel()))
        else:
            y = features_space[:, features_space.shape[1] - 1:].ravel()

        for train_ind, test_ind in kf_total.split(x):
            scaled_test_set = x[test_ind]
            for i_clf, (clf_name, clf) in enumerate(classifiers.items()):
                actual_clf = '{}_{}_kfolds_{}'.format(t_dataset, type_clf, clf_name)
                print actual_clf
                try:
                    specific_clf.index(actual_clf)
                except ValueError:
                    pass
                else:
                    pr.enable()
                    inst_clf = clf.fit(x[train_ind], y[train_ind])
                    pr.disable()
                    s = StringIO.StringIO()
                    sortby = 'cumulative'
                    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
                    tt = round(ps.total_tt, 6)
                    #  print '------------------------------------>>>>   {} \n{}\n'.format(clf_name)
                    y_pred = clf.predict(scaled_test_set)
                    # y_true = y[test_ind]
                    if type_clf == 'multi':
                        y_true = np.random.random_integers(1, 3, test_ind.shape[0])
                    else:
                        y_true = np.random.random_integers(0, 1, test_ind.shape[0])
                    ind_score = inst_clf.score(x[test_ind], y_true)
                    accuracy = accuracy_score(y_true, y_pred)
                    print accuracy
                    if accuracy >= 0.5:
                        print classification_report(y_true, y_pred)
                    general_metrics[clf_name][0].append(ind_score)
                    last_metric = '-'.join([str(elem) for elem in confusion_matrix(y_true, y_pred).ravel()])
                    general_metrics[clf_name][2].append(tt)
                    general_metrics[clf_name][3].append(last_metric)


def preprocessingdata(data):

    defined_classes = ('clase1', 'clase2', 'clase3')

    new_vector = []

    for current_class in defined_classes:
        selected_set = data[current_class].T
        averaged_vector = np.mean(selected_set, axis=0)

        pair_sets = {'averaged': averaged_vector, 'dataset': selected_set, 'tag': current_class}
        new_vector.append(pair_sets)
    return new_vector


def processunknownclassdata(test_data):
    characteristics_matrix = []
    for class_set in test_data:
        class_set_model_fit = autoregressionmodel(class_set)
        characteristics_matrix.append(class_set_model_fit.params)

    return characteristics_matrix


def extractcharacteristicsvector(data):
    classification = {'base_models': {}, 'train_models': []}
    for pair_set in data:
        averaged, dataset, tag = pair_set['averaged'], pair_set['dataset'], pair_set['tag']
        tag_class = int(tag[len(tag) - 1])
        template_model_fit = autoregressionmodel(averaged)
        classification['base_models'][tag] = template_model_fit.params
        for class_set in dataset:
            class_set_model_fit = autoregressionmodel(class_set)
            classification['train_models'].append(np.append(class_set_model_fit.params, tag_class))

            if template_model_fit.params.shape[0] != class_set_model_fit.params.shape[0]:
                print 'template AR model vector size: {} | class AR model vector size: {}'. \
                    format(template_model_fit.params.shape[0], class_set_model_fit.params.shape[0])
    classification['train_models'] = np.array(classification['train_models'])
    return classification


def armodelclassificatorpredict(characteristic_vector, base_models):
    euc_distances = []

    for class_name, coefficients in base_models.items():
        tag = class_name[len(class_name) - 1]
        euc = euclidean(characteristic_vector, coefficients)
        euc_distances.append((euc, tag))

    sorted_euc_distances = sorted(euc_distances)
    return sorted_euc_distances[0][1]


def makepredictions(ar_models, testing_data=[], report_quality=False):
    base_models, train_models = ar_models['base_models'], ar_models['train_models']

    defined_classes = ('clase1', 'clase2', 'clase3')
    y = []
    if len(testing_data) < 1:
        for training_model in train_models:
            x = training_model[:len(training_model) - 1]
            predicted_class = armodelclassificatorpredict(x, base_models)
            correct_class = training_model[len(training_model) - 1]
            y.append((int(correct_class), int(predicted_class)))
    else:
        for testing_vector in testing_data:
            x = testing_vector
            predicted_class = armodelclassificatorpredict(x, base_models)
            y.append(int(predicted_class))
    y = np.array(y)
    print report_quality
    if report_quality:
        y_true, y_pred = y[:, 0], y[:, 1]
        accuracy = accuracy_score(y_true, y_pred)
        print accuracy
        print classification_report(y_true, y_pred)
    return  y


def savedatainmatfile(file_name, data):
    saved_data =io.savemat(file_name, data)
    return saved_data


if __name__ == '__main__':
    #mat = io.loadmat('file.mat')
    mat = io.loadmat('../data/third_project/datosProy3_2019.mat')
    data = preprocessingdata(mat)
    arm_classifiers = extractcharacteristicsvector(data)
    predictions = makepredictions(arm_classifiers, report_quality=True)
    '''
    unclass_data =  np.array(processunknownclassdata(mat['test']))
    arm_unclass = extractcharacteristicsvector(unclass_data)
    predictions = makepredictions(arm_classifiers, testing_data=unclass_data)
    saving_data = {'predictions': predictions}
    savedatainmatfile('../data/third_project/predictions_i.mat', saving_data)
    predictions = io.loadmat('../data/third_project/predictions_i.mat')
    '''

    learningtoclassify('biosignals', 1, np.array(arm_classifiers['train_models']))

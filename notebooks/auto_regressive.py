from pandas import read_csv
from matplotlib import pyplot
from statsmodels.tsa.ar_model import AR
from sklearn.metrics import mean_squared_error
from scipy import io

def autoregressionmodel(dataset, debug=False):
    model = AR(dataset)
    model_fit = model.fit(maxlag=38, trend='nc')

    if debug:
        print('Lag: %s' % model_fit.k_ar)
        print('Coefficients: %s' % model_fit.params)
        print 'Number of Coefficients: {}'.format(len(model_fit.params))

    return  model_fit


def gridsearchautoregressionmodel(dataset, debug=False):
    model = AR(dataset)
    model_fit = model.fit(ic='bic', trend='nc')

    if debug:
        print('Lag: %s' % model_fit.k_ar)
        print('Coefficients: %s' % model_fit.params)
        print 'Number of Coefficients: {}'.format(len(model_fit.params))

    return  model_fit


if __name__ == '__main__':
    series = io.loadmat('../data/third_project/datosProy3_2019.mat')
    # split dataset
    X = series['clase3'].T
    train, test = X[0, :], X[1, :]

    # train autoregression
    model_fit = autoregressionmodel(train)

    gridsearchautoregressionmodel(train, True)

    # make predictions
    predictions = model_fit.predict(start=len(train), end=len(train)+len(test)-1, dynamic=False)

    error = mean_squared_error(test, predictions)
    print('Test MSE: %.3f' % error)
    # plot results
    pyplot.plot(test)
    pyplot.plot(predictions, color='red')
    # pyplot.show()
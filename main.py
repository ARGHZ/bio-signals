from pandas import read_csv
from matplotlib import pyplot
from statsmodels.tsa.ar_model import AR
from sklearn.metrics import mean_squared_error
from scipy import io

series = io.loadmat('data/third_project/datosProy3_2019.mat')
# split dataset
X = series['clase1'].T
train, test = X[0, :], X[1, :]
# train autoregression
model = AR(train)
model_fit = model.fit(ic='aic')
print('Lag: %s' % model_fit.k_ar)
print('Coefficients: %s' % model_fit.params)
print 'Number of Coefficients: {}'.format(len(model_fit.params))
# make predictions
predictions = model_fit.predict(start=len(train), end=len(train)+len(test)-1, dynamic=False)

error = mean_squared_error(test, predictions)
print('Test MSE: %.3f' % error)
# plot results
pyplot.plot(test)
pyplot.plot(predictions, color='red')
pyplot.show()
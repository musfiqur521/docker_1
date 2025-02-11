import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

a3= pd.read_csv('abc.csv')
a3
a3.columns = ['Date', 'Temperature', 'Anomaly']
a3.Date = a3.Date.floordiv(100)
#pd.set_option('precision', 2)

X_train, X_test, y_train, y_test = train_test_split(a3.Date.values.reshape(-1, 1), a3.Temperature.values,random_state=11)
X_train.shape
X_test.shape
linear_regression = LinearRegression()
linear_regression.fit(X=X_train, y=y_train)

linear_regression.coef_
predict = (lambda x: linear_regression.coef_ * x + linear_regression.intercept_)
axes = sns.scatterplot(data=a3, x='Date', y='Temperature', hue='Temperature', palette='winter', legend=False)
x = np.array([min(a3.Date.values), max(a3.Date.values)])
y = predict(x)
line = plt.plot(x, y)
print ("Coefficient: ", linear_regression.coef_)
print ("Y Intercept", linear_regression.intercept_)
a3
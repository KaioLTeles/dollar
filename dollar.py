import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score


#Loading the dataset from the csv
dollar = pd.read_csv("dollarDataset.csv")


# Number of Instances: 1761
#   EIXO X - dollar.data:
#       - Date
#       - Price of the dollar (Field used on this study)

#   EIXO Y - diabetes.target

dollar_X = dollar[['valor']]


# First 1700 values ​​for training
dollar_X_training = dollar_X[:-61]
# Last 61 values for test
dollar_X_test = dollar_X[-61:]

print(dollar_X_training.size)
print("===================")
print(dollar_X_test.size)

# First 1700 values ​​for training
dollar_Y_training = dollar.target[:-61]
# Last 61 values for test
dollar_Y_test = dollar.target[-61:]

# Linear Regression model
regr = linear_model.LinearRegression()

regr.fit(dollar_X_training, dollar_Y_training)

dollar_Y_pred = regr.predict(dollar_X_test)

print('Coeficientes: \n', regr.coef_)

print("Erro quadrado médio: %.2f"
      % mean_squared_error(dollar_Y_test, dollar_Y_pred))

# score
print('Variancia score: %.2f' % r2_score(dollar_Y_test, dollar_Y_pred))



import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score


data = np.loadtxt('data',delimiter=',').transpose()
X_na = data[0].reshape(-1,1)
X_cp = data[1].reshape(-1,1)
X_both = data[:2,].transpose()
y = data[2]

reg_cp = LinearRegression().fit(X_cp, y)
reg_cp.score(X_cp, y)
mean_absolute_error(reg_cp.predict(X_cp), y)

reg_na = LinearRegression().fit(X_na, y)
reg_na.score(X_na, y)
mean_absolute_error(reg_na.predict(X_na), y)

reg_both = LinearRegression().fit(X_both, y)
reg_both.score(X_both, y)
mean_absolute_error(reg_both.predict(X_both), y)


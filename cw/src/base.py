import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import  GradientBoostingRegressor

data = pd.read_csv('students.csv')
le = LabelEncoder()
for col in data.columns:
    data[col] = le.fit_transform(data[col])
X = data.drop(['G3', 'G2', 'G1', 'school'], axis=1)
y = data['G3']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = GradientBoostingRegressor(learning_rate = 0.01, max_depth=9, n_estimators = 500, subsample = 0.5)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
s = 0
cnt = 0
y_test = y_test.values
y_pred = y_pred
for i in range(len(y_test)):
    if y_test[i] > 0.1:
        s += abs(y_test[i] - y_pred[i]) / y_test[i]
        cnt += 1
print(s / cnt * 100)


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import  GradientBoostingRegressor
import matplotlib.pyplot as plt
data = pd.read_csv('students.csv')
le = LabelEncoder()
for col in data.columns:
    data[col] = le.fit_transform(data[col])
X = data.drop(['G3', 'G2', 'G1', 'school'], axis=1)
y = data['G3']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
def research_all_params():
    lrs = [0.0001, 0.01, 0.1]
    max_depths = [5, 9, 12] # 9 
    n_estimators = [10, 500, 1000] # 500
    subsamples = [0.05, 0.5, 0.8] # 0.5

    for subsample in subsamples:
        for n_estimator in n_estimators:
            for max_depth in max_depths:
                for lr in lrs:
                

                    model = GradientBoostingRegressor(learning_rate = lr,
                                                      max_depth=max_depth,
                                                      n_estimators = n_estimator,
                                                      subsample = subsample)
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    s = 0
                    cnt = 0
                    y_testt = y_test.values
                    y_predd = y_pred
                    for i in range(len(y_testt)):
                        if y_testt[i] > 0.1:
                            s += abs(y_testt[i] - y_predd[i]) / y_testt[i]
                            cnt += 1
                    err = s / cnt * 100
                    print("\hline {:.2f} & {} & {} & {:.4f} & {:.1f} \\\\".format(
                                                                                 subsample,
                                                                                 n_estimator,
                                                                                 max_depth,
                                                                                 lr,
                                                                                 err))
def calc_err(y, y_pred):
    s = 0
    cnt = 0
    y = y.values
    for i in range(len(y)):
        if y[i] > 0.1:
            s += abs(y[i] - y_pred[i]) / y[i]
            cnt += 1
    err = s / cnt * 100
    return err
def research(models, x, xlab):
    train_errors = []
    control_errors = []
    
    for model in models:
        
        model.fit(X_train, y_train)
        y_pred_test = model.predict(X_test)
        control_errors.append(calc_err(y_test, y_pred_test))

        y_pred_train = model.predict(X_train)
        train_errors.append(calc_err(y_train, y_pred_train))
    
    plt.plot(x, train_errors, label='Обучающая выборка')
    plt.plot(x, control_errors, label='Контрольная выборка')
    plt.xlabel(xlab)
    plt.ylabel('Значение ошибки')
    plt.title('Зависимость значения ошибки от ' + xlab)
    plt.legend()
    plt.savefig("research_" + xlab + ".png")
    plt.clf()


d = {"learning_rate" : np.linspace(0.001, 0.1, num = 50),
     "max_depth" : np.linspace(2, 21, num = 20),
     "n_estimators" : np.linspace(10, 1000, num = 50),
     "subsample" : np.linspace(0.01, 0.9, num = 50)}

for k in d:
    models = []
    for val in d[k]:
        lr = 0.01
        max_depth = 9
        n_estimator = 500
        subsample = 0.8
        if k == "learning_rate":
            lr = val
        elif k == "max_depth":
            max_depth = val
        elif k == "n_estimators":
            n_estimator = val
        else:
            subsample = val
        models.append(GradientBoostingRegressor(learning_rate = lr,
                                          max_depth=int(max_depth),
                                          n_estimators = int(n_estimator),
                                          subsample = subsample))
    research(models, d[k], k)
    

import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import warnings

warnings.filterwarnings("ignore")
le = LabelEncoder()
con = ["Оценка благополучия","Оценка социальной поддержки","Ожидаемая продолжительность здоровой жизни","Свобода принимать важные решения","Индекс Щедрости","Индекс отношения к коррупции","Оценка риска безработицы","Индекс кредитного оптимизма","Индекс страха социальных конфликтов","Индекс семьи","Индекс продовольственной безопасности","Чувство технологического прогресса","Чувство неравенства доходов в обществе"]
ind = ["Среднегодовой доход","V алкоголя в год","Количество членов семьи","Количество лет образования","Доля дохода семьи на продовольствие"]
soc = ["Коэффициент Джини сообщества","Издержки сообщества на окружающую среду","Охват беспроводной связи в сообществе","Количество смертей от заболеваний в сообществе","Волатильность цен в сообществе"]

names = ["линейная", "k-соседей", "случайный лес"]

def prep():
    df = pd.read_csv('data.csv')
    df = df.drop(['Респондент'], axis=1)
    
    df["Сообщество"] = le.fit_transform(df["Сообщество"]) 

    df_unknown = df.loc[df["Ощущаемое счастье"] == 'Неизвестно']
    df_known = df.loc[df["Ощущаемое счастье"] != 'Неизвестно']

    df_known["Ощущаемое счастье"] = le.fit_transform(df_known["Ощущаемое счастье"])

    df_known.to_csv('data_known.csv', index=False)
    df_unknown.to_csv('data_unknown.csv', index=False)
    return (df_known, df_unknown)
def draw_corr(df):
    corr=df.corr()
    plt.figure(figsize=(30,20))
    sns.heatmap(corr, annot=True, cmap="Reds")
    plt.title('Correlation Heatmap', fontsize=20)
    plt.savefig("correlation.png")
    plt.clf()
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
def research_models(X, y, models, names, target, model_type, cut = ""):
    plt.figure(figsize=(10,8))
    i = 0
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    i_best = -1
    best = -1
    for model in models:
        name = names[i]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        if model_type == "clf":
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')

            conf_matrix = confusion_matrix(y_test, y_pred)
            if target == "Ощущаемое счастье":
                sns.heatmap(conf_matrix, annot=True, xticklabels=le.classes_, yticklabels=le.classes_)
            else:
                sns.heatmap(conf_matrix, annot=True)
            plt.title(f'acc={accuracy:.2f}, prec={precision:.2f}, Recall={recall:.2f}, f1={f1:.2f}')
            plt.savefig(f"matrix_errors_{target}_{name}{cut}.png")
            plt.clf()

            print("\hline {} & {} & {:.3f} & {:.3f} & {:.3f} & {:.3f} \\\\".format(name, target,
                                                                                       accuracy,
                                                                                       precision,
                                                                                       recall,
                                                                                       f1 ))
            if best == -1 or f1 > best:
                best = f1
                best_i = i
        elif model_type == "reg":
            mape = calc_err(y_test, y_pred)
            print("\hline {} & {} & {:.3f} \\\\".format(name, target, mape))
            if best == -1 or mape < best:
                best = mape
                best_i = i
        i += 1
    print()
    return best_i

        
def research_features(X, y, model, target, name, trashhold = 0.1):
    imp_features = []
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    importances = []
    if name == 'случайный лес':
        importances=model.feature_importances_
    elif name == 'линейная':
        coef_abs = np.abs(model.coef_)
        scaler = MinMaxScaler()
        importances = scaler.fit_transform(coef_abs.reshape(-1, 1)).flatten()
    feat_importances = {}
    colors = []
    for j,features in zip(importances,X_train.columns):
        feat_importances[features] = j
        if j > trashhold:
            imp_features.append(features)
        if features in con:
            colors.append("red")
        elif features in ind:
            colors.append("green")
        else:
            colors.append("blue")

    plt.figure(figsize=(20,20))

    plt.bar(range(len(feat_importances)), list(feat_importances.values()), align='edge', color=colors)
    plt.xticks(range(len(feat_importances)), list(feat_importances.keys()),  rotation=80, fontsize = 8)

    plt.title("важность признаков")
    plt.savefig(f"importancies_{target}_{name}.png")
    plt.clf()
    return imp_features

prep()
df_known, df_unknown = pd.read_csv('data_known.csv'), pd.read_csv('data_unknown.csv')
#draw_corr(df_known)


def task1(target = "Ощущаемое счастье"):
    clfs = [LogisticRegression(),
          KNeighborsClassifier(),
          RandomForestClassifier()]
    X = df_known[con]
    y = df_known[target]
    print("точность моделей: ")
    idx_best = research_models(X, y, clfs, names, target, "clf")
    print("лучшая модель по точности: " + names[idx_best])
    imp_features = research_features(X, y, clfs[idx_best], target, names[idx_best])
    print("выбранные состояния: \n", imp_features)
    print("доля выбранных состояний: ", len(imp_features) / len(ind + soc))
    X = df_known[imp_features]
    y = df_known[target]

    print("ошибка выбранной модели при наличии только выбранных состояний: ")
    research_models(X, y, [clfs[idx_best]], [names[idx_best]], target, "clf", cut = "_cut")
    
    return (idx_best, imp_features)
    
def task2(states):
    best_regs = {'idx' : [],
                 'reasons' : []}
    for s in states:
        print("-------состояние-------")
        print(s)
        
        regrs = [LinearRegression(),
                 KNeighborsRegressor(),
                 RandomForestRegressor()]
        X = df_known[ind + soc]
        y = df_known[s]
        print("ошибки моделей: ")
        idx_best = research_models(X, y, regrs, names, s, "reg")
        print("лучшая модель по ошибкам: " + names[idx_best])
        imp_features = research_features(X, y, regrs[idx_best], s, names[idx_best], trashhold = 0.001)
        print("выбранные причины: \n", imp_features)
        print("доля выбранных причин: ", len(imp_features) / len(ind + soc))
        best_regs['idx'].append(idx_best)
        best_regs['reasons'].append(imp_features)

        X = df_known[imp_features]
        y = df_known[s]
        
        print("ошибка выбранной модели при наличии только выбранных состояний: ")
        research_models(X, y, [regrs[idx_best]], [names[idx_best]], s, "reg", cut = "_cut")
    return best_regs
def replace_values(arr):
    new_arr = []
    for val in arr:
        if val in ['Prospering', 'Thriving', 'Blooming']:
            new_arr.append('Thriving')
        elif val in ['Doing well', 'Just ok', 'Coping', 'Struggling']:
            new_arr.append('Struggling')
        elif val in ['Suffering', 'Depressed', 'Hopeless']:
            new_arr.append('Suffering')
        else:
            new_arr.append(val)
    return new_arr


def task_3(clf_idx, states, best_regs):
    df_known_train, df_known_test = train_test_split(df_known, test_size=0.1, random_state=42)

    regrs = [LinearRegression(),
         KNeighborsRegressor(),
         RandomForestRegressor()]
    clfs = [LogisticRegression(),
          KNeighborsClassifier(),
          RandomForestClassifier()]
    known_predictions = {}
    unknown_predictions = {}
    for i in range(len(best_regs['idx'])):
        X = df_known_train[best_regs['reasons'][i]]
        y = df_known_train[states[i]]

        
        regrs[best_regs['idx'][i]].fit(X, y)
        y_pred_known = regrs[best_regs['idx'][i]].predict(df_known_test[best_regs['reasons'][i]])
        y_pred_unknown = regrs[best_regs['idx'][i]].predict(df_unknown[best_regs['reasons'][i]])
        
        known_predictions[states[i]] = y_pred_known
        unknown_predictions[states[i]] = y_pred_unknown

    known_predictions_df = pd.DataFrame(known_predictions)
    unknown_predictions_df = pd.DataFrame(unknown_predictions)
    X = df_known_train[states]
    y = df_known_train["Ощущаемое счастье"]
    clfs[idx].fit(X, y)

    y_pred_known = clfs[idx].predict(known_predictions_df)
    
    accuracy = accuracy_score(df_known_test["Ощущаемое счастье"], y_pred_known)
    precision = precision_score(df_known_test["Ощущаемое счастье"], y_pred_known, average='weighted')
    recall = recall_score(df_known_test["Ощущаемое счастье"], y_pred_known, average='weighted')
    f1 = f1_score(df_known_test["Ощущаемое счастье"], y_pred_known, average='weighted')

    conf_matrix = confusion_matrix(df_known_test["Ощущаемое счастье"], y_pred_known)

    sns.heatmap(conf_matrix, annot=True, xticklabels=le.classes_, yticklabels=le.classes_)

    plt.title(f'acc={accuracy:.2f}, prec={precision:.2f}, Recall={recall:.2f}, f1={f1:.2f}')
    plt.savefig(f"matrix_errors_итог.png")
    plt.clf()

    y_pred_unknown = clfs[idx].predict(unknown_predictions_df)
    df_unknown['res_full'] = le.inverse_transform(y_pred_unknown)
    df_unknown['res'] = replace_values(le.inverse_transform(y_pred_unknown))
    df_unknown.to_csv('data_unknown_res.csv', index=False)

    
print("-----классификация счастья по состояниям--------")
idx, states = task1(target = "Ощущаемое счастье")
best_regs = task2(states)

print("-----применение выбранных моделей--------")

task_3(idx, states, best_regs)


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import imageio
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras import utils
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
EPOCHS_COUNT = 50
def custom_activation(alpha=1.0):
    def activation(x):
        return 1 / (1 + tf.exp(-alpha * x))
    return activation
data = pd.read_csv('students.csv')
le = LabelEncoder()
for col in data.columns:
    data[col] = le.fit_transform(data[col])
    
print(data.head())
print(data.describe())
##plt.figure(figsize=(30, 20))
##
##df = pd.melt(data, data.columns[-1], data.columns[:-1])
##for i, col in enumerate(df.columns[:-1]):
##    plt.subplot(6, 6, i+1)
##    sns.kdeplot(data=df, col, hue='G3', palette='viridis')
##plt.tight_layout()
##plt.savefig("hists.png")
##plt.clf()

X = data.drop(['G3', 'G2', 'G1', 'school'], axis=1)
y = data['G3']

data["G3"].hist(bins=20)
plt.title('G3 - Number of students')
plt.savefig("G3_hist.png")
plt.clf()

corr=data.corr()
plt.figure(figsize=(30,20))
sns.heatmap(corr, annot=True, cmap="Reds")
plt.title('Correlation Heatmap', fontsize=20)
plt.savefig("correlation.png")
plt.clf()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


models = [LinearRegression(),
          GradientBoostingRegressor(learning_rate = 0.01, max_depth=9, n_estimators = 500, subsample = 0.8),
          KNeighborsRegressor(),
          RandomForestRegressor(),
          tf.keras.models.Sequential([
              tf.keras.layers.Dense(64, activation=custom_activation(alpha=1.0), input_dim=29),
              tf.keras.layers.Dense(20, activation='softmax')])]
names = ["линейная", "GradientBoostingRegressor", "k-соседей", "случайный лес", "MLP"]
models[4].compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
plt.figure(figsize=(10,8))
i = 0
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
for model in models:
    name = names[i]
    #print(name)
    if names[i] == "MLP":
        model.fit(X_train, y_train, epochs=EPOCHS_COUNT, validation_data=(X_test, y_test))
    else:
        model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    if names[i] == "MLP":
        y_pred = np.argmax(y_pred, axis=1)
    plt.xlim([0, 20])
    plt.ylim([0, 20])
    plt.scatter(y_test, y_pred)
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    err = calc_err(y_test, y_pred)
##    rsq = r2_score(y_test, y_pred)
##    mse = mean_squared_error(y_test, y_pred)
    plt.title(f'MAPE={err:.2f}')
    plt.savefig(f"errors_{name}.png")
    plt.clf()
    print("\hline {} & {:.3f} \\\\".format(name, err))
                                                              
                    
    if names[i] == "GradientBoostingRegressor" and names[i] == "случайный лес":
        importances=model.feature_importances_
        feat_importances = {}

        for j,features in zip(importances,X_train.columns):
            feat_importances[features] = j

        plt.figure(figsize=(20,20))

        plt.bar(range(len(feat_importances)), list(feat_importances.values()), align='center')
        plt.xticks(range(len(feat_importances)), list(feat_importances.keys()),  rotation=60, fontsize = 12)

        plt.title("Feature Importance")
        plt.savefig(f"importancies_{name}.png")
    i += 1


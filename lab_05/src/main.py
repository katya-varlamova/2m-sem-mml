import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


data = pd.read_csv('data.csv')

print(data.describe())
sns.pairplot(data, hue='iris_type', palette='husl')
plt.savefig("pairs.png")
plt.clf()

correlation_matrix = data.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Матрица корреляции для всех данных')
plt.savefig("whole_matrix.png")
plt.clf()

for iris_class in data['iris_type'].unique():
    subset = data[data['iris_type'] == iris_class]
    correlation_matrix = subset.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.title(f'матрица корреляций для {iris_class}')
    plt.savefig(f"matrix_{iris_class}.png")
    plt.clf()


X = data.drop('iris_type', axis=1)
y = data['iris_type']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = GaussianNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
conf_matrix = confusion_matrix(y_test, y_pred)



sns.heatmap(conf_matrix, annot=True)
plt.title(f'acc={accuracy:.2f}, prec={precision:.2f}, Recall={recall:.2f}, f1={f1:.2f}')
plt.savefig(f"matrix_errors_2.png")
plt.clf()
print(conf_matrix)

print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')
print('Confusion Matrix:')

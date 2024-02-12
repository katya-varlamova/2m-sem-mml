import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Задаем параметры theta
theta_1 = 2
theta_2 = 1
theta_3 = 0.5

# Генерируем данные для обучающей выборки
np.random.seed(0)
X_train = np.linspace(0, 10, 100)  # значения x от 0 до 10
y_train = theta_1 * X_train + theta_2 * np.sin(X_train) + theta_3 + np.random.normal(0, 0.5, 100)  # значения y(x)с шумом

min_mse = float('inf')
best_degree = 0
degrees = list(range(1, 50))
errors = []
for degree in degrees:  # Перебираем степени полиномов от 1 до 100
    poly_features = PolynomialFeatures(degree=degree)
    X_poly = poly_features.fit_transform(X_train.reshape(-1, 1))


    model = LinearRegression()
    model.fit(X_poly, y_train)

    y_pred = model.predict(X_poly)

    mse = mean_squared_error(y_train, y_pred)
    errors.append(mse)
    if mse < min_mse:
        min_mse = mse
        best_degree = degree

print(f"Оптимальная степень полинома: {best_degree}")

best_poly_features = PolynomialFeatures(degree=best_degree)
X_poly_best = best_poly_features.fit_transform(X_train.reshape(-1, 1))

best_model = LinearRegression()
best_model.fit(X_poly_best, y_train)

plt.plot(degrees, errors)
plt.xlabel('Степень полинома')
plt.ylabel('Значение ошибки')
plt.title('Зависимость значения ошибки от степени полинома')
plt.show()

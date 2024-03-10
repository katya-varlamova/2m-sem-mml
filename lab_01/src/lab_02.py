import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def true_function(x):
    return 1 / (1 + 25 * x**2)

l = 21
X_train = np.array([4 * (i - 1) / (l - 1) - 2 for i in range(1, l + 1)]).reshape(-1, 1)
X_control = np.array([4 * (i - 0.5) / (l - 1) - 2 for i in range(1, l)]).reshape(-1, 1)
y_train = true_function(X_train)
y_control = true_function(X_control)

def fit_polynomial_regression(X, y, degree):
    poly_features = PolynomialFeatures(degree=degree)
    X_poly = poly_features.fit_transform(X)
    model = LinearRegression()
    model.fit(X_poly, y)
    return model, poly_features


def calculate_error(model, poly_features, X, y):
    X_poly = poly_features.transform(X)
    y_pred = model.predict(X_poly)
    return mean_squared_error(y, y_pred), y_pred


degrees = np.arange(1, 21)
train_errors = []
control_errors = []

for degree in degrees:
    model, poly_features = fit_polynomial_regression(X_train, y_train, degree)
    train_error, y_p_t = calculate_error(model, poly_features, X_train, y_train)
    control_error, y_p_c = calculate_error(model, poly_features, X_control, y_control)
    train_errors.append(train_error)
    control_errors.append(control_error)
    if degree == 16 or degree == 20 or degree == 13 or degree == 10 or degree == 5:
        plt.plot(X_train, y_train, label = "train")
        plt.plot(X_control, y_control, label = "control")
        plt.plot(X_train, y_p_t, label = "predicted train")
        plt.plot(X_control, y_p_c, label = "predicted control")
        plt.legend()
        plt.title('полином')
        plt.savefig(str(degree) + ".png")
        plt.clf()

plt.plot(degrees, train_errors, label='Обучающая выборка')
plt.plot(degrees, control_errors, label='Контрольная выборка')
plt.xlabel('Степень полинома')
plt.ylabel('Значение ошибки')
plt.title('Зависимость значения ошибки от степени полинома')
plt.legend()
plt.show()


optimal_degree = degrees[np.argmin(control_errors)]
print(f'Optimal polynomial degree for approximation: {optimal_degree}')

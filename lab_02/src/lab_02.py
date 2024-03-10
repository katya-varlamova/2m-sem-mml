import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from matplotlib.widgets import Slider
import warnings
warnings.filterwarnings("ignore")
def true_function(x):
    return 1 / (1 + 25 * x**2)

l = 21
X_train = np.array([4 * (i - 1) / (l - 1) - 2 for i in range(1, l + 1)]).reshape(-1, 1)
X_control = np.array([4 * (i - 0.5) / (l - 1) - 2 for i in range(1, l)]).reshape(-1, 1)
y_train = true_function(X_train)
y_control = true_function(X_control)
degrees = [10, 13, 16]
def fit_polynomial_regression(X, y, degree, model):
    poly_features = PolynomialFeatures(degree=degree)
    X_poly = poly_features.fit_transform(X)
    model.fit(X_poly, y)
    return poly_features


def calculate_error(model, poly_features, X, y):
    X_poly = poly_features.transform(X)
    y_pred = model.predict(X_poly)
    return mean_squared_error(y, y_pred), y_pred
def get_errs(model, X_train, y_train, X_control, y_control):
    train_errors = []
    control_errors = []
    for degree in degrees:
        poly_features = fit_polynomial_regression(X_train, y_train, degree, model)
        train_error, y_p_t = calculate_error(model, poly_features, X_train, y_train)
        control_error, y_p_c = calculate_error(model, poly_features, X_control, y_control)
        train_errors.append(train_error)
        control_errors.append(control_error)
    return train_errors,control_errors

def update_ridge(val):
    train_errors_ridge, control_errors_ridge = get_errs(Ridge(alpha=val), X_train, y_train, X_control, y_control)
    ridge_plot_train.set_ydata(train_errors_ridge)
    ridge_plot_control.set_ydata(control_errors_ridge)
    fig.canvas.draw_idle()
    
def update_lasso(val):
    train_errors_lasso, control_errors_lasso = get_errs(Lasso(alpha=val), X_train, y_train, X_control, y_control)
    lasso_plot_train.set_ydata(train_errors_lasso)
    lasso_plot_control.set_ydata(control_errors_lasso)
    fig.canvas.draw_idle()

def output_errors():
    models = [LinearRegression(),
              Lasso(alpha=0.1),
              Lasso(alpha=10),
              Lasso(alpha=100),
              Lasso(alpha=1000),
              Ridge(alpha=0.1),
              Ridge(alpha=10),
              Ridge(alpha=100),
              Ridge(alpha=1000)]
    names = ["Линейная",
             "Лассо, alpha = 0.1",
             "Лассо, alpha = 10",
             "Лассо, alpha = 100",
             "Лассо, alpha = 1000",
             "Ридж, alpha = 0.1",
             "Ридж, alpha = 10",
             "Ридж, alpha = 100",
             "Ридж, alpha = 1000"]
    degs = [5, 10, 20]
    i = 0
    for model in models:
        for degree in degs:
            poly_features = fit_polynomial_regression(X_train, y_train, degree, model)
            train_error, y_p_t = calculate_error(model, poly_features, X_train, y_train)
            control_error, y_p_c = calculate_error(model, poly_features, X_control, y_control)
            print("\hline {} & {} & {:.3f} & {:.3f} & {:.3f} & {:.3f} \\\\".format(names[i],
                                                                 degree,
                                                                 np.max(np.abs(y_p_t)),
                                                                 np.max(np.abs(y_p_c)),
                                                                 train_error,
                                                                 control_error))
        i += 1
output_errors()
fig, ax = plt.subplots()

ax_slider_ridge = plt.axes([0.1, 0.01, 0.8, 0.03])
ax_slider_lasso = plt.axes([0.1, 0.9, 0.8, 0.03])

train_errors_lin, control_errors_lin = get_errs(LinearRegression(), X_train, y_train, X_control, y_control)
ax.plot(degrees, train_errors_lin, label='Обучающая выборка, линейная')
ax.plot(degrees, control_errors_lin, label='Контрольная выборка, линейная')

train_errors_lasso, control_errors_lasso = get_errs(Lasso(alpha=0.1), X_train, y_train, X_control, y_control)
lasso_plot_train,  = ax.plot(degrees, train_errors_lasso, label='Обучающая выборка, лассо')
lasso_plot_control,  = ax.plot(degrees, control_errors_lasso, label='Контрольная выборка, лассо')

train_errors_ridge, control_errors_ridge = get_errs(Ridge(alpha=0.1), X_train, y_train, X_control, y_control)
ridge_plot_train,  = ax.plot(degrees, train_errors_ridge, label='Обучающая выборка, ридже')
ridge_plot_control,  = ax.plot(degrees, control_errors_ridge, label='Контрольная выборка, ридже')

slider_ridge = Slider(ax_slider_ridge, 'Ridge', valmin=0, valmax=1000, valinit=0.1)
slider_ridge.on_changed(update_ridge)

slider_lasso = Slider(ax_slider_lasso, 'Lasso', valmin=0, valmax=1000, valinit=0.1)
slider_lasso.on_changed(update_lasso)
ax.legend()

plt.xlabel('Степень полинома')
plt.ylabel('Значение ошибки')
plt.title('Зависимость значения ошибки от степени полинома')
plt.legend()
plt.show()


optimal_degree = degrees[np.argmin(control_errors)]
print(f'Optimal polynomial degree for approximation: {optimal_degree}')

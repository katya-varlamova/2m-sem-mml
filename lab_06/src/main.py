import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import imageio
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
from sklearn.metrics import mean_squared_error
from tensorflow.keras import utils
import pandas as pd
EPOCHS_COUNT = 50
def draw_line(
        x_data: np.ndarray,
        w_1: float,
        w_2: float,
        w_c: float):
    y_arr = -(w_1 * x_data + w_c) / w_2
    plt.plot(x_data, y_arr, linestyle='-')
    return 
def show_lines(x_min: float, x_max: float, neurons: np.ndarray, bias: np.ndarray):
    line_data_x = np.arange(x_min, x_max, (x_min + x_max) / 4)
    # w_1*x + w_2*y + w_c = 0
    # y = -(w_1*x + w_c) / w_2
    for w_1, w_2, w_c in zip(neurons[0], neurons[1], bias):
        draw_line(line_data_x, w_1, w_2, w_c)
    return 
def gen_data():
    n_samples = 300
    n_features = 2
    n_classes = 4
    X, y = make_blobs(n_samples=[300, 250, 200, 150], n_features=n_features, centers=None, cluster_std=1, random_state=42)
    X_additional, y_additional = make_blobs(n_samples=[150, 300], n_features=n_features, centers=None, cluster_std=1, random_state=45)
    X = np.vstack([X, X_additional])
    y = np.hstack([y, y_additional])
    return X, y
def draw_data(X, y):
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolors='k')
    plt.title('Сгенерированные данные')
    plt.xlabel('Признак 1')
    plt.ylabel('Признак 2')
    plt.savefig("data_viz.png")
    
def custom_activation(alpha=1.0):
    def activation(x):
        return 1 / (1 + tf.exp(-alpha * x))
    return activation
def plot_layer(model, X, y, name_graph):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    
    Z = model.predict(grid_points)
    Z = np.argmax(Z, axis=1).reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolors='k')
    plt.title('Разделяющие поверхности промежуточного слоя')
    plt.xlabel('Признак 1')
    plt.ylabel('Признак 2')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    show_lines(x_min, x_max, model.layers[0].get_weights()[0], model.layers[0].get_weights()[1])

    plt.savefig(name_graph)
    plt.clf()
    
X, y = gen_data()

draw_data(X, y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model_names = ['relu_1', "relu_100", "sigmoid_1", "sigmoid_5"]

models = [
    tf.keras.models.Sequential([
        tf.keras.layers.Dense(6, activation=tf.keras.layers.LeakyReLU(alpha=1), input_dim=2),
        tf.keras.layers.Dense(4, activation='softmax')]), #]#,#]#,
    tf.keras.models.Sequential([
        tf.keras.layers.Dense(6, activation=tf.keras.layers.LeakyReLU(alpha=100), input_dim=2),
        tf.keras.layers.Dense(4, activation='softmax')]),
    tf.keras.models.Sequential([
        tf.keras.layers.Dense(6, activation=custom_activation(alpha=1.0), input_dim=2),
        tf.keras.layers.Dense(4, activation='softmax')]),
    tf.keras.models.Sequential([
        tf.keras.layers.Dense(6, activation=custom_activation(alpha=5), input_dim=2),
        tf.keras.layers.Dense(4, activation='softmax')])    ]
model_num = 0
for model in models:
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    name = model_names[model_num]
    epochs_ctr = 0

    images = []
    errs_train = []
    errs_test = []
    layers = np.arange(1, EPOCHS_COUNT + 1)
    def plot_intermediate_layer(model, X):
        global epochs_ctr
        epochs_ctr += 1
        plot_layer(model, X, y_test, f"img_{name}_{epochs_ctr}.png")
        y_train_pred = model.predict(X_train)
        y_train_pred = np.argmax(y_train_pred, axis=1)
        y_test_pred = model.predict(X_test)
        y_test_pred = np.argmax(y_test_pred, axis=1)
        errs_train.append(mean_squared_error(y_train, y_train_pred))
        errs_test.append(mean_squared_error(y_test, y_test_pred))

    history = model.fit(X_train, y_train, batch_size=5, epochs=EPOCHS_COUNT, validation_data=(X_test, y_test), callbacks=[tf.keras.callbacks.LambdaCallback(on_epoch_end=lambda epoch, logs: plot_intermediate_layer(model, X_test))])

    plt.plot(layers, errs_train, label='Обучающая выборка')
    plt.plot(layers, errs_test, label='Контрольная выборка')
    plt.legend()
    plt.title('СКО')
    plt.savefig(f"std_{name}.png")
    plt.clf()
    
    for i in range(1, EPOCHS_COUNT+1):
        images.append(imageio.imread(f"img_{name}_{i}.png"))
    imageio.mimsave(f'{name}_training_history.gif', images, duration=1000)

    y_pred = model.predict(X_test)
    y_pred = np.argmax(y_pred, axis=1)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    conf_matrix = confusion_matrix(y_test, y_pred)

    sns.heatmap(conf_matrix, annot=True)
    plt.title(f'acc={accuracy:.2f}, prec={precision:.2f}, Recall={recall:.2f}, f1={f1:.2f}')

    plt.savefig(f"matrix_errors_{name}.png")
    plt.clf()
    model_num += 1

    plot_layer(model, X_test, y_test, f"before_{name}.png")
    want = "0"
    u = 0
    while want == "1":
        print("Хотите ввести точку? (1 -- да)")
        want = input()
        if want == "1":
            x1 = float(input("x1: "))
            x2 = float(input("x2: "))
            cluster = int(input("cluster: (0, 1, 2, 3): "))
            
            
            plot_layer(model, np.vstack([X_test, [x1, x2]]), np.append(y_test, cluster), f"after_{name}_{u}.png")
            u += 1
            
    

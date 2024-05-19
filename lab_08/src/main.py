import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pygad
from sklearn.model_selection import train_test_split
import imageio.v2 as imageio
import os
def true_function(x):
    return 1 / (1 + 25 * x**2)

l = 21
X_train = np.array([4 * (i - 1) / (l - 1) - 2 for i in range(1, l + 1)]).reshape(-1, 1)
X_control = np.array([4 * (i - 0.5) / (l - 1) - 2 for i in range(1, l)]).reshape(-1, 1)
y_train = true_function(X_train)
y_control = true_function(X_control)

def sbx_crossover(parents, offspring_size, ga_instance):
    offspring = np.empty(offspring_size)
    pc = 0.5 
    eta_c = 0.5 

    for i in range(0, offspring_size[0], 2):
        parent1_idx = i % parents.shape[0]
        parent2_idx = (i + 1) % parents.shape[0]

        parent1 = parents[parent1_idx]
        parent2 = parents[parent2_idx]

        child1 = np.empty(len(parent1))
        child2 = np.empty(len(parent2))

        for j in range(len(parent1)):
            if np.random.rand() <= pc:
                u = np.random.rand()
                if u <= 0.5:
                    beta = (2*u)**(1/(eta_c+1))
                else:
                    beta = (1/(2*(1-u)))**(1/(eta_c+1))

                child1[j] = 0.5*((1+beta)*parent1[j] + (1-beta)*parent2[j])
                child2[j] = 0.5*((1-beta)*parent1[j] + (1+beta)*parent2[j])
            else:
                child1[j] = parent1[j]
                child2[j] = parent2[j]

        offspring[i, :] = child1
        offspring[i + 1, :] = child2

    return offspring

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


def find_best_deg():
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

def fitness_func(ga_instance, solution, solution_idx):
    y_pred = np.polyval(solution, X_train)
    return 1 / mean_squared_error(y_train, y_pred)
def GA(deg):
    ga_instance = pygad.GA(num_generations=400, num_parents_mating=10,
                           fitness_func=fitness_func, sol_per_pop=21,
                           save_best_solutions=True,
                           num_genes=deg + 1, gene_type=float,
                           crossover_type = sbx_crossover,
                           mutation_type = "random",
                           mutation_percent_genes = 5,
                           parent_selection_type = "sss",
                           keep_parents = 1)
    ga_instance.run()
    errs_train = []
    errs_test = []
    gens = []
    images = []
    for generation in range(400):
        gens.append(generation)
        best_solution = ga_instance.best_solutions[generation]

        plt.plot(X_train, true_function(X_train), label="Оригинальная функция (обучающая)")
        plt.plot(X_train, np.polyval(best_solution, X_train),  label="Аппроксимация к-тов ГА (обучающая)")

        plt.plot(X_control, true_function(X_control), label="Оригинальная функция (тестовая)")
        plt.plot(X_control, np.polyval(best_solution, X_control), label="Аппроксимация к-тов ГА (тестовая)")

        plt.legend()
        errs_train.append(mean_squared_error(y_train, np.polyval(best_solution, X_train)))
        errs_test.append(mean_squared_error(y_control, np.polyval(best_solution, X_control)))

        plt.title(f"эпоха {generation}")
        plt.savefig("img_{}.png".format(generation))
        plt.clf()
        
        images.append(imageio.imread("img_{}.png".format(generation)))
        os.remove("img_{}.png".format(generation))
        
    imageio.mimsave(f'training_history_{deg}.gif', images, duration=50)

    plt.plot(gens, errs_train,  label="Ошибки на обучающей")
    plt.plot(gens, errs_test, label="Ошибки на тестовой")
    plt.xlabel("эпоха")
    plt.ylabel("СКО")
    plt.title("ошибки")
    plt.legend()
    plt.savefig(f"errs_{deg}.png")
    plt.clf()
    return errs_train[-1], errs_test[-1]
def comp():
    errs_train = []
    errs_test = []
    degs = []
    for i in [7]:
        e1, e2 = GA(i)
        errs_train.append(e1)
        errs_test.append(e2)
        degs.append(i)
        
##    plt.plot(degs, errs_train,  label="Ошибки на обучающей")
##    plt.plot(degs, errs_test, label="Ошибки на тестовой")
##    plt.xlabel("степень полинома")
##    plt.ylabel("СКО")
##    plt.title("ошибки")
##    plt.legend()
##    plt.savefig(f"errs_degs.png")
##    plt.clf()
    

def best_comp():
    ga_instance = pygad.GA(num_generations=400, num_parents_mating=10,
                           fitness_func=fitness_func, sol_per_pop=21,
                           save_best_solutions=True,
                           num_genes=8, gene_type=float,
                           crossover_type = sbx_crossover,
                           mutation_type = "random",
                           mutation_percent_genes = 20,
                           parent_selection_type = "sss",
                           keep_parents = 1)
    ga_instance.run()
    best_solution = ga_instance.best_solutions[-1]

    model, poly_features = fit_polynomial_regression(X_train, y_train, 10)
    
    train_error_poly, _ = calculate_error(model, poly_features, X_train, y_train)
    control_error_poly, _ = calculate_error(model, poly_features, X_control, y_control)
    train_error_ga = mean_squared_error(y_train, np.polyval(best_solution, X_train))
    control_error_ga = mean_squared_error(y_control, np.polyval(best_solution, X_control))
    print("\hline обучающая & {:10.3f} & {:10.3f} \\\\".format(train_error_poly, train_error_ga))
    print("\hline  тестовая & {:10.3f} & {:10.3f} \\\\".format(control_error_poly, control_error_ga))
    
best_comp()

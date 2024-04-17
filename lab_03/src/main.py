import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sklearn.utils import resample

def gen_normal_data(a, sigma, size):
    smth = np.random.normal(loc=a, scale=sigma, size=size)
    _, p_value_normality_smth = stats.normaltest(smth)
    while p_value_normality_smth < 0.99:
        smth = np.random.normal(loc=a, scale=sigma, size=size)
        _, p_value_normality_smth = stats.normaltest(smth)
    return smth
def print_menu():
    print("0 - сгенерировать и вывести выборки")
    print("1 - проверить соответствие выборок нормальному закону")
    print("2 - Осуществить проверку гипотезы H_0:a_1= a_2 против альтернативы H_1:a_1≠a_2")
    print("3 - Производить сдвиг вправо математического ожидания второй выборки a_2 на величину ∆=0.01 (a_2=a_2+∆) и осуществлять проверку гипотезы H_0:a_1= a_2 до тех пор, пока гипотеза H_0 не будет отвергнута. Рассчитать 95% доверительные интервалы для математических ожиданий двух выборок в момент, когда гипотеза H_0 была отвергнута")
    print("4 - Для второй выборки назначить a_2 равным середине пройденного отрезка из пункта 3. Постепенно увеличивать число элементов в выборках и осуществлять проверку гипотезы H_0:a_1= a_2 до тех пор, пока гипотеза H_0 не будет отвергнута. Рассчитать 95% доверительные интервалы для математических ожиданий двух выборок в момент, когда гипотеза H_0 была отвергнута")

print_menu()
value = int(input())
x = []
y = []
n = 30
m = 30
a_1 = 2
a_2 = 2
sigma_1 = 0.5
sigma_2 = 0.5
interval = 0
while value != -1:
    if value == 0:
        plt.clf()
        p_value_equal_means = 0
        while p_value_equal_means < 0.9:
            x = gen_normal_data(a_1, sigma_1, n)
            y = gen_normal_data(a_2, sigma_2, m)
            _, p_value_equal_means = stats.ttest_ind(x, y, equal_var=True)
        plt.hist(x, alpha=0.5, label='выборка X')
        plt.hist(y, alpha=0.5, label='выборка Y')
        plt.legend()
        plt.savefig("samples.png")
        plt.clf()
    elif value == 1:
        _, p_value_normality_x = stats.normaltest(x)
        _, p_value_normality_y = stats.normaltest(y)
        print("p-value для x = ", p_value_normality_x)
        print("p-value для y = ", p_value_normality_y)
    elif value == 2:
        _, p_value_equal_means = stats.ttest_ind(x, y, equal_var=True)
        print("p-value для a1=a2", p_value_equal_means)
    elif value == 3:
        plt.clf()
        fig, ax = plt.subplots()
        
        delta = 0.01
        interval = 0
        y_s = [y]
        _, p_value_equal_means = stats.ttest_ind(x, y_s[-1], equal_var=True)
        means_diff = [np.mean(x) - np.mean(y_s[-1])]
        values = [p_value_equal_means]
        ax.hist(x, alpha=0.5, label='выборка X')
        _, _, h = ax.hist(y_s[-1], alpha=0.5, label='выборка Y', color = 'red')

        while p_value_equal_means >= 0.05:
            interval += delta
            y_s.append(np.array(y) + interval)
            _, p_value_equal_means = stats.ttest_ind(x, y_s[-1], equal_var=True)
            values.append(p_value_equal_means)
            means_diff.append(np.mean(x) - np.mean(y_s[-1]))

        def update(iternum):
            global h
            h.remove()
            plt.title("p_value = {:.3f}, interval = {}".format(values[iternum], delta * iternum))
            _, _, h = ax.hist(y_s[iternum], alpha=0.5, color = 'red')
            
        ani = animation.FuncAnimation(fig, update, frames=len(y_s), interval=400)
        ani.save('animated_plot_3.gif', writer='pillow')
        print("интервал = {:.3f}, p-value = {:.3f}".format(interval, p_value_equal_means))
        ci_x = stats.norm.interval(0.95, loc=np.mean(x), scale=stats.sem(x))
        ci_y = stats.norm.interval(0.95, loc=np.mean(y_s[-1]), scale=stats.sem(y_s[-1]))
        print("интервал выборки X = {};\nинтервал выборки Y = {}".format(ci_x, ci_y))
        
        plt.clf()
        plt.plot([i * delta for i in range(len(y_s))], means_diff, label='разница средних')
        plt.plot([i * delta for i in range(len(y_s))], values, label='P-value')
        plt.legend()
        plt.savefig("diffs_3.png")
    elif value == 4:
        plt.clf()
        fig, ax = plt.subplots()
        y_s = [np.array(y) + interval / 2]
        x_s = [x]
        n_ = n
        m_ = m
        _, p_value_equal_means = stats.ttest_ind(x_s[-1], y_s[-1], equal_var=True)
        _, _, h1 = ax.hist(x_s[-1], alpha=0.5, color = 'blue')
        _, _, h2 = ax.hist(y_s[-1], alpha=0.5, color = 'red')
        values = [p_value_equal_means]
        sizes = [n_]
        means_diff = [np.mean(x) - np.mean(y)]
        print(p_value_equal_means)
        while p_value_equal_means >= 0.05:
            n_ += 200
            m_ += 200
            x_s.append(gen_normal_data(a_1, sigma_1, n_))#(resample(x, replace=True, n_samples=n_))
            y_s.append(gen_normal_data(a_2 + interval / 2, sigma_2, m_))# (resample(np.array(y) + interval / 2, replace=True, n_samples=m_)) # 
            _, p_value_equal_means = stats.ttest_ind(x_s[-1], y_s[-1], equal_var=True)
            values.append(p_value_equal_means)
            sizes.append(n_)
            means_diff.append(np.mean(x_s[-1]) - np.mean(y_s[-1]))
            print(p_value_equal_means)
        def update(iternum):
            global h1
            global h2
            h1.remove()
            h2.remove()
            plt.title("p_value = {:.3f}, size = {}".format(values[iternum], sizes[iternum]))
            _, _, h1 = ax.hist(x_s[iternum], alpha=0.5, color = 'blue')
            _, _, h2 = ax.hist(y_s[iternum], alpha=0.5, color = 'red')

        ani = animation.FuncAnimation(fig, update, frames=len(y_s), interval=800)
        ani.save('animated_plot_4.gif', writer='pillow')
        ci_x = stats.norm.interval(0.95, loc=np.mean(x_s[-1]), scale=stats.sem(x_s[-1]))
        ci_y = stats.norm.interval(0.95, loc=np.mean(y_s[-1]), scale=stats.sem(y_s[-1]))
        print("интервал выборки X = {};\nинтервал выборки Y = {}".format(ci_x, ci_y))
        plt.clf()
        plt.plot(sizes, means_diff, label='разница средних')
        plt.plot(sizes, values, label='P-value')
        plt.legend()
        plt.savefig("diffs_4.png")
    print()
    value = int(input())



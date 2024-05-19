import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import umap
import seaborn as sns
import imageio
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, adjusted_rand_score
n_cluster = 4
cols = [ "sepal_length","sepal_width","petal_length","petal_width"]
def draw_description(data):
    confidence_level = 0.95  

    for col in data.columns:
        column_data = data[col]
        
        mean = np.mean(column_data)

        confidence_interval = stats.t.interval(0.95, len(column_data)-1, loc=mean, scale=stats.sem(column_data))

        std_dev = np.std(column_data)

        plt.hist(column_data, bins='auto', alpha=0.7, color='skyblue', edgecolor='black', linewidth=1.2)
        plt.axvline(mean, color='r', linestyle='dashed', linewidth=2, label=f'Мат. ожидание: {mean:.2f}')
        plt.axvline(mean - std_dev, color='g', linestyle='dashed', linewidth=2, label=f'Мат. ожидание - std: {mean - std_dev:.2f}')
        plt.axvline(mean + std_dev, color='g', linestyle='dashed', linewidth=2, label=f'Мат. ожидание + std: {mean + std_dev:.2f}')
        plt.axvspan(confidence_interval[0], confidence_interval[1], color='gray', alpha=0.3, label='Доверительный интервал (95%)')
        plt.xlabel('Значение')
        plt.ylabel('Частота')
        plt.title(f'Гистограмма данных для {col}')
        plt.legend()
        plt.savefig(f"{col}.png")
        plt.clf()
def draw_corr(data):
    corr=data.corr()
    plt.figure(figsize=(30,20))
    sns.heatmap(corr, annot=True, cmap="Reds")
    plt.title('Корреляции', fontsize=20)
    plt.savefig("correlation.png")
    plt.clf()


def read_data(fn):
    return pd.read_csv(fn)

def preprocess_data(df):
    return df[cols].dropna(how='any')

def count_elbow(df):
    inertia = []
    n = 15
    for k in range(1, n):
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(df)
        # sum of the distances of all points within a cluster from the centroid of the point.
        inertia.append(kmeans.inertia_) 

    plt.figure(figsize=(8, 6))
    plt.plot(range(1, n), inertia, marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    plt.title('Elbow Method for Optimal K')
    plt.savefig("elbow.png")
    plt.clf()
    
def count_k_means(df):
    kmeans = KMeans(n_cluster)
    kmeans.fit(df)
    return kmeans.labels_
def count_dbscan(df):
    dbscan = DBSCAN(eps=4.2, min_samples=5) 
    return dbscan.fit_predict(df)
def count_pca(df):
    pca = PCA(n_components=2)
    return pca.fit_transform(df)
def count_umap(df):
    return umap.UMAP(n_components=2).fit_transform(df)

def plot(title, labels, df_resize, dir_name = "./"):
    plt.figure(figsize=(12, 6))
    sns.scatterplot(x=df_resize[:, 0], y=df_resize[:, 1], hue=labels, palette='viridis')
    plt.title(title)
    plt.legend()
    plt.savefig(dir_name + title + ".png")
    plt.clf()
    
def visual(df, f_cluster, f_resize, title, dir_name = "./"):
    df_resize = f_resize(df)
    labels = f_cluster(df)
    
    plot(title, labels, df_resize, dir_name = "./")
    
def vary_metric_dbscan(df, answers):
    metric = ["cosine", "euclidean", "l1", "l2", "manhattan"]
    scores = []
    for m in metric:
        dbscan = DBSCAN(eps=4.2, min_samples=5, metric = m)
        labels = dbscan.fit_predict(df) 
        df_resize = count_umap(df)
        plot(f"metric_{m}", labels, df_resize, dir_name = "research_dbscan_metric/")
        scores.append(adjusted_rand_score(answers, labels))
        
    plt.figure(figsize=(12, 6))
    plt.plot(metric, scores)
    plt.title("scores")
    plt.legend()
    plt.savefig("research_dbscan_metric/scores.png")
    plt.clf()
def vary_eps_dbscan(df, answers):
    epss = np.linspace(2, 6, 21)
    scores = []
    for eps in epss:
        dbscan = DBSCAN(eps=eps, min_samples=5)
        labels = dbscan.fit_predict(df) 
        df_resize = count_umap(df)
        eps  = round(eps * 10)
        plot(f"metric_{eps}", labels, df_resize, dir_name = "research_dbscan_eps/")
        scores.append(adjusted_rand_score(answers, labels))
        
    plt.figure(figsize=(12, 6))
    plt.plot(epss, scores)
    plt.title("scores")
    plt.legend()
    plt.savefig("research_dbscan_eps/scores.png")
def vary_min_samples_dbscan(df, answers):
    scores = []
    x = []
    for s in range(5, 50, 5): # 150 всего
        dbscan = DBSCAN(eps=4.2, min_samples=s)
        labels = dbscan.fit_predict(df) 
        df_resize = count_umap(df)
        plot(f"samples_{s}", labels, df_resize, dir_name = "research_dbscan_samples/")
        scores.append(adjusted_rand_score(answers, labels))
        x.append(s)
        
    plt.figure(figsize=(12, 6))
    plt.plot(x, scores)
    plt.title("scores")
    plt.legend()
    plt.savefig("research_dbscan_samples/scores.png")
def vary_num_clusters_kmeans(df, answers):
    scores = []
    x = []
    for cluster in range(2, 11, 1):
        kmeans = KMeans(cluster)
        kmeans.fit(df)
        labels = kmeans.labels_
        df_resize = count_umap(df)
        plot(f"clusters_{cluster}", labels, df_resize, dir_name = "research_kmeans_clusters/")
        scores.append(adjusted_rand_score(answers, labels))
        x.append(cluster)
        
    plt.figure(figsize=(12, 6))
    plt.plot(x, scores)
    plt.title("scores")
    plt.legend()
    plt.savefig("research_kmeans_clusters/scores.png")
def vary_centroid_algo_kmeans(df, answers):
    scores = []
    metrics = ['random', 'k-means++']
    for metric in metrics:
        kmeans = KMeans(n_cluster, init=metric)
        kmeans.fit(df)
        labels = kmeans.labels_
        df_resize = count_umap(df)
        plot(f"algo_{metric}", labels, df_resize, dir_name = "research_kmeans_algo/")
        scores.append(adjusted_rand_score(answers, labels))
        
    plt.figure(figsize=(12, 6))
    plt.plot(metrics, scores)
    plt.title("scores")
    plt.legend()
    plt.savefig("research_kmeans_algo/scores.png")
def count_accuracy(df, answers):
    kmeans = KMeans(4, init = 'k-means++')
    kmeans.fit(df)
    labels_kmeans = kmeans.labels_
    print("kmeans", adjusted_rand_score(answers, labels_kmeans))

    dbscan = DBSCAN(eps=4.2, min_samples=5)
    labels_dbscan = dbscan.fit_predict(df)
##    for i in range(len(answers)):
##        print(labels_kmeans[i], labels_dbscan[i], answers[i])
    print("dbscan", adjusted_rand_score(answers, labels_dbscan))
    
data = read_data('data.csv')
le = LabelEncoder()
for col in data.columns:
    data[col] = le.fit_transform(data[col])
answers = data["iris_type"]
# draw_description(data)
#draw_corr(data)
##
##data = preprocess_data(data)
count_accuracy(data, answers)
##vary_num_clusters_kmeans(data, answers)
##vary_min_samples_dbscan(data, answers)
##vary_metric_dbscan(data, answers)
##vary_eps_dbscan(data, answers)
##vary_centroid_algo_kmeans(data, answers)

##count_elbow(data)
##visual(data, count_k_means, count_pca, "k_means_pca")
##visual(data, count_dbscan, count_pca, "dbscan_pca")
##visual(data, count_dbscan, count_umap, "dbscan_umap")
##visual(data, count_k_means, count_umap, "k_means_umap")



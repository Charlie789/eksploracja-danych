from helper_functions import exercise

import argparse
import matplotlib.pyplot as plt
import pandas as pd
import random

from os import path
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


@exercise
def zad1():
    data_path = 'dane/Sales_Transactions_Dataset_Weekly.csv'
    if not path.isfile(data_path):
        print(
            f'Brak pliku {data_path}, należy go pobrać z http://archive.ics.uci.edu/ml/machine-learning-databases/00396/Sales_Transactions_Dataset_Weekly.csv')
        return
    df = pd.read_csv(data_path, index_col='Product_Code')
    print(f'Wczytano {data_path}')
    # Dane zawierają kolumny użyteczne w tym zadaniu W0-W51, oraz niepotrzebne na poczet tego zadania - MIN, MAX oraz znormalizowane
    df = df.iloc[:, :52]
    print('Pozostawiono tylko kolumny W0-W51')
    # Pozostałe kolumny należy przeskalować, aby mozna było użyć algorytmu PCA
    scaler = StandardScaler()
    scaled_features = pd.DataFrame(scaler.fit_transform(df))
    scaled_features.columns = df.columns
    print('Przeskalona wartości przy pomocy StandardScaler')

    # Użycie algorytmu PCA, aby zmniejszyć wymiarowość danych
    pca = PCA(n_components=2)
    processed_data = pca.fit_transform(scaled_features)
    processed_df = pd.DataFrame(data=processed_data, columns=[f'COMP{i}' for i in range(processed_data.shape[1])])
    print('Zmniejszono wymiarowość do 2 przy użyciu PCA')
    return processed_df


@exercise
def zad2(clusters, data):
    kmeans = KMeans(n_clusters=clusters, init='k-means++', random_state=0)
    processed_df_kmeans = kmeans.fit_predict(data)

    markers = [(i, j, 0) for i in range(2, 10) for j in range(1, 3)]

    # Dodanie do wykresu poszczególnych klastrów
    for i in range(clusters):
        plt.scatter(
            data[processed_df_kmeans == i]['COMP0'],
            data[processed_df_kmeans == i]['COMP1'],
            s=20,
            c=[(random.random(), random.random(), random.random())],
            marker=markers[i],
            label=f'Skupienie {i}'
        )
    # Dodanie do wykresu centroidów
    plt.scatter(
        kmeans.cluster_centers_[:, 0],
        kmeans.cluster_centers_[:, 1],
        s=30,
        c='red',
        marker='*',
        label='Centroidy'
    )
    plt.title('K-MEANS++')
    plt.legend()
    plt.show()


def prepare_parser():
    parser = argparse.ArgumentParser(
        description='sum the integers at the command line')
    parser.add_argument(
        '--clusters', type=int, default=5,
        help='Number of clusters for k-means, default: 5')
    return parser


if __name__ == '__main__':
    parser = prepare_parser()
    args = parser.parse_args()
    data = zad1()
    zad2(args.clusters, data)

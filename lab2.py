import numpy
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA

from helper_functions import exercise


@exercise
def zad1(df):
    pd.set_option('display.max_columns', None)
    print(f'Wymiar (kolumny x indeksy): {df.shape[1]}x{df.shape[0]}')
    print(f'Liczba unikalnych wartości w wektorze "target": {len(numpy.unique(cancer.target))}')
    transform = VarianceThreshold(threshold=0.05)
    transform.fit_transform(df)
    variances = transform.variances_
    dropped_features = df.columns.values[variances < 0.05]
    print(f'Kolumny nadające się do usunięcia, których treshold jest mniejszy od 0.05: \n{dropped_features}')
    df.to_csv(sep=';', path_or_buf='dane/breast.csv')


@exercise
def zad2(df):
    n_components = 5
    pca = PCA(n_components=n_components)
    processed_data = pca.fit_transform(df)
    processed_df = pd.DataFrame(data=processed_data, columns=[f'COMP{i}' for i in range(n_components)])
    processed_df.to_csv(sep=';', path_or_buf='dane/breast_pca_5.csv')
    print(f'Wariancja po zmniejszeniu wymiarowości: \n{processed_df.var()}\n')
    print('Wariancja wyjaśniona dla poszczególnych składowych:')
    for i, value in enumerate(pca.explained_variance_ratio_):
        print(f'{processed_df.columns[i]}: {value:.5f}')


if __name__ == '__main__':
    cancer = load_breast_cancer(as_frame=True)
    zad1(cancer.frame)
    zad2(cancer.frame)

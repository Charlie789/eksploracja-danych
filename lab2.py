import numpy
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.feature_selection import VarianceThreshold


def zad1():
    print('***** ZAD 1 *****')
    cancer = load_breast_cancer(as_frame=True)
    df = cancer.frame
    pd.set_option('display.max_columns', None)
    print(f'Wymiar (kolumny x indeksy): {df.shape[1]}x{df.shape[0]}')
    print(f'Liczba unikalnych wartości w wektorze "target": {len(numpy.unique(cancer.target))}')
    transform = VarianceThreshold(threshold=0.05)
    transform.fit_transform(df)
    variances = transform.variances_
    dropped_features = df.columns.values[variances < 0.05]
    print(f'Kolumny nadające się do usunięcia, których treshold jest mniejszy od 0.05: \n{dropped_features}')
    df.to_csv(sep=';', path_or_buf='dane/breast.csv')


if __name__ == '__main__':
    zad1()

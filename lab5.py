import pandas as pd

from os import path

from helper_functions import exercise

import warnings

warnings.filterwarnings("ignore")

x_train = None
y_train = None
x_test = None
y_test = None

y_pred_knn = None
y_pred_svm = None
y_pred_tree = None
y_pred_forest = None

model_knn = None
model_svc = None
model_tree = None
model_forest = None


@exercise
def zad1():
    path_x_train = "dane/X_train.txt"
    path_y_train = "dane/y_train.txt"
    path_x_test = "dane/X_test.txt"
    path_y_test = "dane/y_test.txt"

    if not path.isfile(path_x_train) \
            or not path.isfile(path_y_train) \
            or not path.isfile(path_x_test) \
            or not path.isfile(path_y_test):
        print(
            'Brak jednego z plików, należy go pobrać z https://archive.ics.uci.edu/ml/datasets/Smartphone-Based+Recognition+of+Human+Activities+and+Postural+Transitions')
        exit(1)

    print('Trwa ładowanie danych...')
    global x_train, y_train, x_test, y_test
    x_train = pd.read_csv(path_x_train, delimiter=' ', header=None)
    y_train = pd.read_csv(path_y_train, delimiter=' ', header=None)
    x_test = pd.read_csv(path_x_test, delimiter=' ', header=None)
    y_test = pd.read_csv(path_y_test, delimiter=' ', header=None)
    print('Załadowane pomyślnie dane')


if __name__ == '__main__':
    zad1()

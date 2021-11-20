import pandas as pd

from os import path
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

from helper_functions import exercise


x_train = None
y_train = None
x_test = None
y_test = None

y_pred_knn = None


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
        print('Brak jednego z plików, należy go pobrać z https://archive.ics.uci.edu/ml/datasets/Smartphone-Based+Recognition+of+Human+Activities+and+Postural+Transitions')
        exit(1)

    print('Trwa ładowanie danych...')
    global x_train, y_train, x_test, y_test
    x_train = pd.read_csv(path_x_train, delimiter=' ')
    y_train = pd.read_csv(path_y_train, delimiter=' ')
    x_test = pd.read_csv(path_x_test, delimiter=' ')
    y_test = pd.read_csv(path_y_test, delimiter=' ')
    print('Załadowane pomyślnie dane')


@exercise
def zad2():
    global y_pred_knn
    knn = KNeighborsClassifier()
    knn_model = knn.fit(x_train, y_train.values.ravel())
    y_pred_knn = knn_model.predict(x_test)


@exercise
def zad3():
    def print_scores(predition_model):
        print(f'Accuracy Score: {accuracy_score(y_test, predition_model)}')
        print(f'Classification Report:\n{classification_report(y_test, predition_model)}')
        print(f'Confusion Matrix:\n{confusion_matrix(y_test, predition_model)}')

    print("--KNN--")
    print_scores(y_pred_knn)


if __name__ == '__main__':
    zad1()
    zad2()
    zad3()

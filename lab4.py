import pandas as pd
import numpy as np

from os import path

from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier

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
        print('Brak jednego z plików, należy go pobrać z https://archive.ics.uci.edu/ml/datasets/Smartphone-Based+Recognition+of+Human+Activities+and+Postural+Transitions')
        exit(1)

    print('Trwa ładowanie danych...')
    global x_train, y_train, x_test, y_test
    x_train = pd.read_csv(path_x_train, delimiter=' ', header=None)
    y_train = pd.read_csv(path_y_train, delimiter=' ', header=None)
    x_test = pd.read_csv(path_x_test, delimiter=' ', header=None)
    y_test = pd.read_csv(path_y_test, delimiter=' ', header=None)
    print('Załadowane pomyślnie dane')


@exercise
def zad2():
    global y_pred_knn, y_pred_svm, y_pred_tree, y_pred_forest
    global model_knn, model_svc, model_tree, model_forest

    # KNN
    print("Przygotowywanie modelu KNN...")
    knn = KNeighborsClassifier()
    model_knn = knn.fit(x_train, y_train.values.ravel())
    y_pred_knn = model_knn.predict(x_test)
    print("Przygotowano model KNN")

    # SVM
    print("Przygotowywanie modelu SVM...")
    svc = svm.SVC(C=1, kernel='linear')
    model_svc = svc.fit(x_train, y_train.values.ravel())
    y_pred_svm = model_svc.predict(x_test)
    print("Przygotowano model SVM")

    # Decision Tree
    print("Przygotowywanie modelu Decision Tree...")
    forest = DecisionTreeClassifier()
    model_tree = forest.fit(x_train, y_train.values.ravel())
    y_pred_tree = model_tree.predict(x_test)
    print("Przygotowano model Decision Tree")

    # Random Forest
    print("Przygotowywanie modelu Random Forest...")
    forest = RandomForestClassifier(n_jobs=-1)
    model_forest = forest.fit(x_train, y_train.values.ravel())
    y_pred_forest = model_forest.predict(x_test)
    print("Przygotowano model Random Forest")


@exercise
def zad3():
    def print_scores(predition_model, name):
        print(f"--{name}--")
        print(f'Accuracy Score: {accuracy_score(y_test, predition_model)}')
        print(f'Classification Report:\n{classification_report(y_test, predition_model)}')
        print(f'Confusion Matrix:\n{confusion_matrix(y_test, predition_model)}')
        print()

    print_scores(y_pred_knn, "KNN")
    print_scores(y_pred_svm, "SVM")
    print_scores(y_pred_tree, "Decision Tree")
    print_scores(y_pred_forest, "Random Forest")


@exercise
def zad4():
    def make_cross_validation(model, name):
        print(f"--{name}--")
        print("Trwa przetwarzanie kros-walidacji...")
        validation = cross_validate(model, x_train, y_train, cv=5)
        average = np.average(validation["test_score"])
        std_deviation = np.std(validation["test_score"])
        print(f"Kros-walidacja: {validation}")
        print(f"Średni wynik: {average}")
        print(f"Odchylenie standardowe: {std_deviation}")
        print()

    make_cross_validation(model_knn, "KNN")
    make_cross_validation(model_svc, "SVC")
    make_cross_validation(model_tree, "Decision Tree")
    make_cross_validation(model_forest, "Random Forest")


if __name__ == '__main__':
    zad1()
    zad2()
    zad3()
    zad4()

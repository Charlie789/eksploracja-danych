import pandas as pd
import numpy as np

from os import path

from sklearn import svm
from sklearn.ensemble import VotingClassifier, BaggingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier

from helper_functions import exercise

import warnings

warnings.filterwarnings("ignore")

x_train = None
y_train = None
x_test = None
y_test = None

comparison = {}


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


@exercise
def zad2():
    knn = KNeighborsClassifier()
    svc = svm.SVC(C=1, kernel='linear')
    lr = LogisticRegression()

    voting_ens = VotingClassifier(estimators=[('knn', knn), ('svc', svc), ('lr', lr)], voting='hard')

    _generate_excel_report([knn, lr, svc, voting_ens], 'dane/ensemble_learning.xlsx')


@exercise
def zad3():
    knn = KNeighborsClassifier()
    svc = svm.SVC(C=1, kernel='linear')
    lr = LogisticRegression()

    bagging_knn = BaggingClassifier(base_estimator=knn, n_estimators=5)
    bagging_svc = BaggingClassifier(base_estimator=svc, n_estimators=5)
    bagging_lr = BaggingClassifier(base_estimator=lr, n_estimators=5)

    _generate_excel_report(
        [bagging_knn, bagging_svc, bagging_lr],
        'dane/aggregation.xlsx',
        ['Bagging_knn', 'Bagging_svc', 'Bagging_lr']
    )


@exercise
def zad4():
    ada = AdaBoostClassifier()
    xg = XGBClassifier()

    _generate_excel_report(
        [ada, xg],
        'dane/boosting.xlsx'
    )


@exercise
def zad5():
    out_name = 'dane/comparison.xlsx'
    output_df = pd.DataFrame.from_dict(
        comparison,
        orient='index',
        columns=['cross-average', 'cross-std', 'ACC', 'Recall', 'F1-score', 'AUC']
    )
    output_df.to_excel(out_name)
    print(f'Raport wygenerowany do pliku: {out_name}')


def _generate_excel_report(clasifiers_list, out_name, clf_names=None):
    global comparison
    output = {}
    for i, clf in enumerate(clasifiers_list):
        if clf_names:
            clf_name = clf_names[i]
        else:
            clf_name = clf.__class__.__name__
        print(f'Trwa generowanie raportu dla {clf_name}...')
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)

        report = classification_report(y_test, y_pred, output_dict=True)

        validation = cross_validate(clf, x_train, y_train, cv=5)
        average = np.average(validation["test_score"])
        std_deviation = np.std(validation["test_score"])

        comparison[clf_name] = output[clf_name] = [
            average,
            std_deviation,
            accuracy_score(y_test, y_pred),
            report['macro avg']['recall'],
            report['macro avg']['f1-score'],
            report['macro avg']['precision']
        ]

    output_df = pd.DataFrame.from_dict(
        output,
        orient='index',
        columns=['cross-average', 'cross-std', 'ACC', 'Recall', 'F1-score', 'AUC']
    )
    output_df.to_excel(out_name)
    print(f'Raport wygenerowany do pliku: {out_name}')


if __name__ == '__main__':
    zad1()
    zad2()
    zad3()
    zad4()
    zad5()

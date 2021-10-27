import pandas as pd

from collections import Counter
from github import Github
from matplotlib import pyplot as plt
from os import environ


def zad1():
    df = pd.read_csv('dane/netflix_titles.csv', index_col='show_id')

    print('****** ZAD 1 ******')
    print(f'Ilość wczytanych wierszych: {len(df)}')
    print(f'Wymiar (kolumny x indeksy): {df.shape[1]}x{df.shape[0]}')
    print(f'Ilość pustych wartości:\n{df.isna().sum()}')
    print('\n')


def zad2():
    df = pd.read_csv('dane/titanic.csv')
    pd.set_option('display.max_columns', None)

    cum_sum = 0
    empty_dict = {}
    for column in df.columns:
        part_sum = df[column].isna().sum()
        cum_sum += part_sum
        empty_percentage = part_sum / len(df)
        empty_dict[column] = [part_sum, cum_sum, empty_percentage]
    empty = pd.DataFrame.from_dict(empty_dict, orient='index', columns=['Ilość pustych', 'Suma skumulowana', 'Procent pustych'])

    print('****** ZAD 2 ******')
    print(df)
    print(f'Ilość wczytanych wierszych: {len(df)}')
    print(f'Ilość pustych wartości: {df.isna().sum().sum()}')
    print(f'Ilość pustych wartości, suma skumulowana:')
    print(empty)

    columns_to_delete = [column for column in df.columns if empty_dict[column][2] > 0.3]
    print(f'Kolumny: "{columns_to_delete}" posiadają > 30% pustych rekordów, zostaną usunięte')
    df.drop(columns=columns_to_delete, inplace=True)
    print(df)

    print('Zastępuje wartości "female" na 0 i "male" na 1')
    df.replace(to_replace={'female': 0, 'male': 1}, inplace=True)
    print(df)

    print('\n')


def zad3():
    git = Github(environ['GH_ACCESS'])
    user = git.get_user('MikiKru')
    repos = user.get_repos()
    print('****** ZAD 3 ******')
    print(f'Liczba projektów: {repos.totalCount}')
    print('Trwa pobieranie języków z repozytorium...')
    try:
        langs_list = [list(repo.get_languages()) for repo in repos]
    except KeyboardInterrupt:
        return
    langs_counter = Counter(langs[0] for langs in langs_list if len(langs) > 0)
    print('Liczba projektów w danym języku:')
    for element in langs_counter.items():
        print(element)

    fig = plt.figure(figsize=(10, 7))
    plt.pie(langs_counter.values(), labels=langs_counter.keys())
    plt.show()


if __name__ == '__main__':
    zad1()
    zad2()
    zad3()

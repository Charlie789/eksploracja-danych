import pandas as pd


def zad1():
    df = pd.read_csv('dane/netflix_titles.csv', index_col='show_id')

    print('****** ZAD 1 ******')
    print(f'Ilość wczytanych wierszych: {len(df)}')
    print(f'Wymiar (kolumny x indeksy): {df.shape[1]}x{df.shape[0]}')
    print(f'Ilość pustych wartości:\n{df.isna().sum()}')
    print('\n')


def zad2():
    df = pd.read_csv('dane/titanic.csv')

    cum_sum = 0
    empty_dict = {}
    for column in df.columns:
        part_sum = df[column].isna().sum()
        cum_sum += part_sum
        empty_dict[column] = [part_sum, cum_sum]
    empty = pd.DataFrame.from_dict(empty_dict, orient='index', columns=['Ilość pustych', 'Suma skumulowana'])

    print('****** ZAD 2 ******')
    print(f'Ilość pustych wartości: {df.isna().sum().sum()}')
    print(f'Ilość pustych wartości, suma skumulowana:')
    print(empty)


def zad3():
    pass


if __name__ == '__main__':
    zad1()
    zad2()

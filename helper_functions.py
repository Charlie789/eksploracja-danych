def exercise(ex):
    def inner(*args):
        print(f'******* {ex.__name__.upper()} *******')
        ex(*args)
        print()
    return inner

def exercise(ex):
    def inner(*args):
        print(f'******* {ex.__name__.upper()} *******')
        return_value = ex(*args)
        print()
        return return_value
    return inner

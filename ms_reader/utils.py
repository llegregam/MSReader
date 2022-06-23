from pandas import option_context


def print_df(message):
    with option_context('display.max_rows', None, 'display.max_columns',
                        None):  # more options can be specified also
        print(message)


def format(x):
    if x == "<LLOQ" or x == ">ULOQ":
        return x
    elif float(x) <= 0:
        return "ND"
    else:
        return float(x)

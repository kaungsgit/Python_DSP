from collections.abc import Iterable


def flatten_list(lis):
    """ Flatten list items recursively """
    for item in lis:
        if isinstance(item, Iterable) and not isinstance(item, str):
            for x in flatten_list(item):
                yield x
        else:
            yield item

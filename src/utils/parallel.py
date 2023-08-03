import multiprocessing

from typing import Callable


def map(fn: Callable, tasks: list, workers: int) -> list:
    if workers >= 2:
        p = multiprocessing.Pool(workers)
        output = p.map(fn, tasks)
        p.close()
    else:
        output = [fn(task) for task in tasks]

    return output


def starmap(fn: Callable, tasks: list, workers: int) -> list:
    if workers >= 2:
        p = multiprocessing.Pool(workers)
        output = p.starmap(fn, tasks)
        p.close()
    else:
        output = [fn(*task) for task in tasks]

    return output

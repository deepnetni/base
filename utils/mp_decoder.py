# import multiprocessing as mp
from pathos.multiprocessing import ProcessingPool as Pool
from functools import wraps
from tqdm import tqdm

import tqdm_pathos
import time


# class mpMapRun(object):
#     def __init__(self, num=3) -> None:
#         self.num = num
#         pass

#     def __call__(self, func):
#         @wraps(func)  # return the name of `func` when calling object.__name__
#         def wrapper(*args, **kwargs):
#             out = []
#             p = Pool(self.num)

#             out = p.map(
#                 func,
#                 tqdm(
#                     args,
#                     total=len(args),
#                 ),
#             )
#             # for i in args:
#             #     func(i)

#             # p.close()
#             p.terminate()

#             return out  # a list

#         return wrapper


class mpMap(object):
    """
    Input: a single list with args, kwargs, e.g., `[a1, a2, ...], b=x, c=d`
    """

    def __init__(self, num=None) -> None:
        self.num = num

    def __call__(self, func):
        @wraps(func)  # return the name of `func` when calling object.__name__
        def wrapper(*args, **kwargs):
            out = []
            # p = Pool(self.num)
            # param = list(zip(*args))
            param = args
            # out = p.map(func, tqdm(param[0], total=len(args), ncols=80), *param[1:])

            out = tqdm_pathos.map(
                func,
                *param,
                n_cpus=self.num,
                tqdm_kwargs=dict(ncols=80),
                **kwargs,
            )

            # p.terminate()

            return list(out)

        return wrapper


class mpStarMap(object):
    """
    Input: multi-lists where each list represent a argument unit, e.g, `[a1, a2, ..], [b1, b2, ..]`.
    """

    def __init__(self, num=None) -> None:
        self.num = num

    def __call__(self, func):
        @wraps(func)  # return the name of `func` when calling object.__name__
        def wrapper(*args, **kwargs):
            out = []
            # param = list(zip(*args))
            param = args
            # out = p.map(func, tqdm(param[0], total=len(args), ncols=80), *param[1:])

            out = tqdm_pathos.starmap(
                func,
                zip(*param),
                n_cpus=self.num,
                tqdm_kwargs=dict(ncols=80),
                **kwargs,
            )

            # p.terminate()

            return list(out)

        return wrapper


class mpStarMapPair(object):
    """
    Input: a list composed of tuples, where each element represents a complete input, e.g., `[(a, b), (..)]`.
    """

    def __init__(self, num=None) -> None:
        self.num = num

    def __call__(self, func):
        @wraps(func)  # return the name of `func` when calling object.__name__
        def wrapper(*args, **kwargs):
            out = []
            # param = list(zip(*args))
            param = args
            # out = p.map(func, tqdm(param[0], total=len(args), ncols=80), *param[1:])

            out = tqdm_pathos.starmap(
                func,
                *param,
                n_cpus=self.num,
                tqdm_kwargs=dict(ncols=80),
                **kwargs,
            )

            # p.terminate()

            return list(out)

        return wrapper


@mpMap(4)
def test(i):
    # print(f"hello {i}")
    time.sleep(1)
    # print(f"hello e{i}")
    return i * 2


@mpStarMap()
def stest(i, j):
    print(f"hello {i}, {j}")
    time.sleep(1)
    # print(f"hello e{i},{j}")
    return i * 2


@mpStarMapPair()
def stest_2(i, j):
    print(f"hello {i}, {j}")
    time.sleep(1)
    # print(f"hello e{i},{j}")
    return i * 2


if __name__ == "__main__":
    # (1,10), (2,10), (3,10)
    # out = stest([1, 2, 3], j=10)
    # out = test([1, 2, 3])

    # i = [1, 2, 3, 4]
    # j = [10, 2, 3, 4]
    # out = stest(i, j)

    l = [(1, 10), (2, 2), (3, 3), (4, 4)]
    out = stest_2(l)

    print(out)

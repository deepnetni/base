#!/usr/bin/env python3
import numpy as np


class A:
    def __init__(self, a=False) -> None:
        self.a = a

    def run(self):
        return 1 if self.a is False else (1, 2)


if __name__ == "__main__":
    a = np.array([[1, 2, 5], [2, 3, 4]])
    print(a.shape)
    b = np.sum(a**2, 0)
    print(b)

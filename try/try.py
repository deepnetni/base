import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class A:
    def __init__(self) -> None:
        self.b()

    @staticmethod
    def a():
        print("a")


class B(A):
    def __init__(self) -> None:
        super().__init__()

    def b(self):
        print("b")


b = B()

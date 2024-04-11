import numpy as np


depth = 4
kernel = [2, 2, 2, 2]
stride = [1, 1, 1, 1]


def compute_dilated_receptive_field():
    df = 1
    prod_stride = stride[0]
    for i in range(depth):
        dil = 2**i
        ksize = 1 + (kernel[i] - 1) * dil
        df += (ksize - 1) * prod_stride
        print(f"{i}-th RF {df}")
        prod_stride = prod_stride * stride[i]


if __name__ == "__main__":
    compute_dilated_receptive_field()

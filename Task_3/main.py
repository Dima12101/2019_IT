import numpy as np


def get_indices(N, n_batches, split_ratio):
    size_jk = (N - 1) / (n_batches + (1 / split_ratio))
    i, j, k = 0, 0, 0
    for index_batch in range(n_batches):
        j = i + size_jk * (1 / split_ratio)
        k = j + size_jk
        yield np.array([i, j, k])
        i += size_jk


def main():
    for inds in get_indices(100, 5, 0.25):
        print(inds)
    # expected result:
    # [0, 44, 55]
    # [11, 55, 66]
    # [22, 66, 77]
    # [33, 77, 88]
    # [44, 88, 99]


if __name__ == "__main__":
    main()
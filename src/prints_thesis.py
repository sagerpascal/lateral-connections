from typing import Optional

import numpy as np
import matplotlib.pyplot as plt


def print_support_strength():
    x = np.arange(0, 30, 0.1)
    y = np.where(x < 10, x, 10 - 0.5 * (x - 10))

    plt.figure(dpi=100, figsize=(8, 4))
    plt.plot(x, y)
    plt.xlabel("Received support")
    plt.ylabel("Normalised support")
    plt.xticks([0, 10, 20, 30], [0, "ρ", "2ρ", "3ρ"])
    plt.yticks([0, 5, 10], [0, "0.5ρ", "ρ"])
    plt.xlim(0, 30)
    plt.ylim(0, 10)
    plt.tight_layout()
    plt.grid()
    plt.show()


def print_support_active_cells():
    min_support_active = [1, 4.6, 10.5, 10.6, 9.5, 9.7, 9.6, 9.8, 9.7, 9.7]
    max_support_active = [1, 9.4, 20.3, 20.2, 20.7, 20.6, 20.7, 20.5, 20.8, 20.7]
    avg_support_active = [1, 6.9, 15.4, 15.4, 15.7, 15.7, 15.7, 15.7, 15.7, 15.7]

    min_support_active_inhibition = [x if x <= 14.3 else 14.3 - (x-14.3) * 0.5 for x in min_support_active]
    max_support_active_inhibition = [min(14.3, x) for x in max_support_active]
    avg_support_active_inhibition = [x if x <= 14.3 else 14.3 - (x-14.3) * 0.5 for x in avg_support_active]

    avg_support_inactive = [0, 0.32, 0.33, 0.41, 0.37, 0.35, 0.34, 0.33, 0.33, 0.33]
    x = np.arange(1, len(avg_support_active) + 1, 1)

    fig, ax = plt.subplots(dpi=100, figsize=(8, 4))
    ax.plot(x, avg_support_active, label="avg. support active cells (without inhibition)")
    ax.fill_between(x, min_support_active, max_support_active, color='b', alpha=.15, label="min/max support (without inhibition)")

    ax.plot(x, avg_support_active_inhibition, label="avg. support active cells (with inhibition)", color='g')
    ax.fill_between(x, min_support_active_inhibition, max_support_active_inhibition, color='g', alpha=.15, label="min/max support (with inhibition)")

    ax.plot(x, [11*1.3] * len(avg_support_active), color='r', linestyle='--', label="ρ")
    ax.plot(x, avg_support_inactive, label="avg. support inactive cells")
    plt.legend()
    plt.ylabel("Support")
    plt.xlabel("Epoch")
    plt.yticks(np.arange(0, 22, 2), np.arange(0, 22, 2))
    plt.xticks(np.arange(0, 11, 1), np.arange(0, 11, 1))
    plt.xlim(1, 10)
    plt.ylim(0, 22)
    plt.tight_layout()
    plt.grid()
    plt.show()


def print_norm_factor():
    norm_factor = [1, 3.7, 3.8, 6.6, 6.3, 5.9, 5.6, 5.4, 5.3, 5.3]
    x = np.arange(1, len(norm_factor) + 1, 1)

    fig, ax = plt.subplots(dpi=100, figsize=(6, 6))
    ax.plot(x, norm_factor, label="normalisation factor")
    # plt.legend()
    plt.ylabel("Normalisation factor")
    plt.xlabel("Epoch")
    plt.yticks(np.arange(1, 7, .5), np.arange(1, 7, .5))
    plt.xticks(np.arange(0, 11, 1), np.arange(0, 11, 1))
    plt.xlim(1, 10)
    plt.ylim(1, 7)
    plt.tight_layout()
    plt.grid()
    plt.show()

if __name__ == '__main__':
    # print_support_strength()
    print_support_active_cells()
    # print_norm_factor()

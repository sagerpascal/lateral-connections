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



if __name__ == '__main__':
    print_support_strength()

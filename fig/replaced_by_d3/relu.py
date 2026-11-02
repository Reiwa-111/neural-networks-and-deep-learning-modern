"""
relu
~~~~

Plots a graph of the squashing function used by a rectified linear
unit."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR = Path(__file__).resolve().parent


def main():
    z = np.arange(-2, 2, 0.1)
    y = np.maximum(0, z)

    fig, ax = plt.subplots()
    ax.plot(z, y)
    ax.set_ylim([-2.0, 2.0])
    ax.set_xlim([-2.0, 2.0])
    ax.grid(True)
    ax.set_xlabel("z")
    ax.set_title("Rectified linear unit")

    fig.savefig(OUTPUT_DIR / "relu.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()

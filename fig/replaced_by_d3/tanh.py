"""
tanh
~~~~

Plots a graph of the tanh function."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR = Path(__file__).resolve().parent


def main():
    z = np.arange(-5, 5, 0.1)
    t = np.tanh(z)

    fig, ax = plt.subplots()
    ax.plot(z, t)
    ax.set_ylim([-1.0, 1.0])
    ax.set_xlim([-5, 5])
    ax.grid(True)
    ax.set_xlabel("z")
    ax.set_title("tanh function")

    fig.savefig(OUTPUT_DIR / "tanh.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()

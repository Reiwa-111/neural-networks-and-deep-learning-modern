"""
sigmoid
~~~~~~~

Plots a graph of the sigmoid function."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy

OUTPUT_DIR = Path(__file__).resolve().parent


def main():
    z = numpy.arange(-5, 5, 0.1)
    sigma = 1 / (1 + numpy.exp(-z))

    fig, ax = plt.subplots()
    ax.plot(z, sigma)
    ax.set_ylim([-0.5, 1.5])
    ax.set_xlim([-5, 5])
    ax.grid(True)
    ax.set_xlabel("z")
    ax.set_title("sigmoid function")

    fig.savefig(OUTPUT_DIR / "sigmoid.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()

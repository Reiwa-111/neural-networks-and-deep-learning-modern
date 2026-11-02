"""
step
~~~~~~~

Plots a graph of a step function."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy

OUTPUT_DIR = Path(__file__).resolve().parent


def main():
    z = numpy.arange(-5, 5, 0.02)
    step = numpy.where(z >= 0.0, 1.0, 0.0)

    fig, ax = plt.subplots()
    ax.plot(z, step)
    ax.set_ylim([-0.5, 1.5])
    ax.set_xlim([-5, 5])
    ax.grid(True)
    ax.set_xlabel("z")
    ax.set_title("step function")

    fig.savefig(OUTPUT_DIR / "step.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()

"""
misleading_gradient_contours
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Plots the contours of the function from misleading_gradient.py"""

from pathlib import Path

# Third party libraries
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy

OUTPUT_DIR = Path(__file__).resolve().parent


def main():
    X = numpy.arange(-1, 1, 0.02)
    Y = numpy.arange(-1, 1, 0.02)
    X, Y = numpy.meshgrid(X, Y)
    Z = X**2 + 10 * Y**2

    fig, ax = plt.subplots()
    ax.contour(
        X, Y, Z, levels=[0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    )
    ax.set_xlabel("$w_1$", fontsize=16)
    ax.set_ylabel("$w_2$", fontsize=16)

    fig.savefig(OUTPUT_DIR / "misleading_gradient_contours.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()

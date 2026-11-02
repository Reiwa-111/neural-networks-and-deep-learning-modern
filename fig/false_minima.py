"""
false_minimum
~~~~~~~~~~~~~

Plots a function of two variables with many false minima."""

from pathlib import Path

# Third party libraries
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy
from matplotlib.ticker import LinearLocator

OUTPUT_DIR = Path(__file__).resolve().parent


def main():
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    X = numpy.arange(-5, 5, 0.1)
    Y = numpy.arange(-5, 5, 0.1)
    X, Y = numpy.meshgrid(X, Y)
    Z = numpy.sin(X) * numpy.sin(Y) + 0.2 * X

    colortuple = ("w", "b")
    colors = numpy.empty(X.shape, dtype=str)
    for x in range(len(X)):
        for y in range(len(Y)):
            colors[x, y] = colortuple[(x + y) % 2]

    ax.plot_surface(
        X, Y, Z, rstride=1, cstride=1, facecolors=colors, linewidth=0
    )

    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_zlim(-2, 2)
    ax.xaxis.set_major_locator(LinearLocator(3))
    ax.yaxis.set_major_locator(LinearLocator(3))
    ax.zaxis.set_major_locator(LinearLocator(3))

    fig.savefig(OUTPUT_DIR / "false_minima.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()

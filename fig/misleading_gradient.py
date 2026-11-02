"""
misleading_gradient
~~~~~~~~~~~~~~~~~~~

Plots a function which misleads the gradient descent algorithm."""

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
    X = numpy.arange(-1, 1, 0.025)
    Y = numpy.arange(-1, 1, 0.025)
    X, Y = numpy.meshgrid(X, Y)
    Z = X**2 + 10 * Y**2

    colortuple = ("w", "b")
    colors = numpy.empty(X.shape, dtype=str)
    for x in range(len(X)):
        for y in range(len(Y)):
            colors[x, y] = colortuple[(x + y) % 2]

    ax.plot_surface(
        X, Y, Z, rstride=1, cstride=1, facecolors=colors, linewidth=0
    )

    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(0, 12)
    ax.xaxis.set_major_locator(LinearLocator(3))
    ax.yaxis.set_major_locator(LinearLocator(3))
    ax.zaxis.set_major_locator(LinearLocator(3))
    ax.text(0.05, -1.8, 0, "$w_1$", fontsize=20)
    ax.text(1.5, -0.25, 0, "$w_2$", fontsize=20)
    ax.text(1.79, 0, 9.62, "$C$", fontsize=20)

    fig.savefig(OUTPUT_DIR / "misleading_gradient.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()

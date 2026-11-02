"""
pca_limitations
~~~~~~~~~~~~~~~

Plot graphs to illustrate the limitations of PCA.
"""

from pathlib import Path

# Third-party libraries
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR = Path(__file__).resolve().parent


def main():
    # Use a fixed seed so the figures are reproducible.
    rng = np.random.default_rng(0)

    z = np.linspace(-2, 2, 20)
    theta = np.linspace(-4 * np.pi, 4 * np.pi, 20)
    x = np.sin(theta) + 0.03 * rng.standard_normal(20)
    y = np.cos(theta) + 0.03 * rng.standard_normal(20)

    # Plot just the data
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.plot(x, y, z, "ro")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_zlabel("$z$")
    fig.savefig(OUTPUT_DIR / "pca_hard_data.png", dpi=150)
    plt.close(fig)

    # Plot the data and the helix together
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    z_helix = np.linspace(-2, 2, 100)
    theta_helix = np.linspace(-4 * np.pi, 4 * np.pi, 100)
    x_helix = np.sin(theta_helix)
    y_helix = np.cos(theta_helix)
    ax.plot(x, y, z, "ro")
    ax.plot(x_helix, y_helix, z_helix, "-", color="0.5")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_zlabel("$z$")
    fig.savefig(OUTPUT_DIR / "pca_hard_data_fit.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()

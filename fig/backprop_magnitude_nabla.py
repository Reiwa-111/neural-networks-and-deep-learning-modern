"""
backprop_magnitude_nabla
~~~~~~~~~~~~~~~~~~~~~~~~

Using backprop2 I constructed a 784-30-30-30-30-30-10 network to classify
MNIST data.  I ran ten mini-batches of size 100, with eta = 0.01 and
lambda = 0.05, using:

net.SGD(otd[:1000], 1, 100, 0.01, 0.05,

I obtained the following norms for the (unregularized) nabla_w for the
respective mini-batches:

[0.90845722175923671, 2.8852730656073566, 10.696793986223632, 37.75701921183488, 157.7365422527995, 304.43990075227839]
[0.22493835119537842, 0.6555126517964851, 2.6036801277234076, 11.408825365731225, 46.882319190445472, 70.499637502698221]
[0.11935180022357521, 0.19756069137133489, 0.8152794148335869, 3.4590802543293977, 15.470507965493903, 31.032396017142556]
[0.15130005837653659, 0.39687135985664701, 1.4810006139254532, 4.392519005642268, 16.831939776937311, 34.082104455938733]
[0.11594085276308999, 0.17177668061395848, 0.72204558746599512, 3.05062409378366, 14.133001132214286, 29.776204839994385]
[0.10790389807606221, 0.20707152756018626, 0.96348134037828603, 3.9043824079499561, 15.986873430586924, 39.195258080490895]
[0.088613291101645356, 0.129173436407863, 0.4242933114455002, 1.6154682713449411, 7.5451567587160069, 20.180545544006566]
[0.086175380639289575, 0.12571016850457151, 0.44231149185805047, 1.8435833504677326, 7.61973813981073, 19.474539356281781]
[0.095372080184163904, 0.15854489503205446, 0.70244235144444678, 2.6294803575724157, 10.427062019753425, 24.309420272033819]
[0.096453131000155692, 0.13574642196947601, 0.53551377709415471, 2.0247466793066895, 9.4503978546018068, 21.73772148470092]

Note that results are listed in order of layer.  They clearly show how
the magnitude of nabla_w decreases as we go back through layers.

In this program I take min-batches 7, 8, 9 as representative and plot
them.  I omit the results from the first and final layers since they
correspond to 784 input neurons and 10 output neurons, not 30 as in
the other layers, making it difficult to compare results.

Note that I haven't attempted to preserve the whole workflow here. It
involved some minor hacking around with backprop2, which messed up
that code.  That's why I've simply put the results in by hand below.
"""

from pathlib import Path

# Third-party libraries
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTPUT_DIR = Path(__file__).resolve().parent

# Norms of the (unregularized) gradient for mini-batches 7, 8 and 9.
nw1 = [0.129173436407863, 0.4242933114455002, 1.6154682713449411, 7.5451567587160069]
nw2 = [0.12571016850457151, 0.44231149185805047, 1.8435833504677326, 7.61973813981073]
nw3 = [0.15854489503205446, 0.70244235144444678, 2.6294803575724157, 10.427062019753425]


def main():
    fig, ax = plt.subplots()
    layers = range(1, 5)
    ax.plot(layers, nw1, "ro-", label="mini-batch 7")
    ax.plot(layers, nw2, "go-", label="mini-batch 8")
    ax.plot(layers, nw3, "bo-", label="mini-batch 9")
    ax.set_xlabel("Layer $l$")
    ax.set_ylabel(r"$\Vert\nabla C^l_w\Vert$")
    ax.set_xticks([1, 2, 3, 4])
    ax.legend()
    fig.savefig(OUTPUT_DIR / "backprop_magnitude_nabla.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()

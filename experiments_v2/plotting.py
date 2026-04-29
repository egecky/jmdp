import os
import matplotlib.pyplot as plt
import numpy as np


def plot_residuals(residuals, outpath):
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    (fig, ax) = plt.subplots()
    ax.plot(residuals, linewidth=1.5)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Bellman residual")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.margins(x=0)
    fig.tight_layout(pad=0.2)
    fig.savefig(outpath, dpi=300)
    plt.close(fig)

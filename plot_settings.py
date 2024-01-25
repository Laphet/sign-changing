import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from mpl_toolkits.axes_grid1 import make_axes_locatable

rc("text", usetex=True)
# Nature suggests that fontsizes should be between 5~7pt
rc("font", **{"size": 6})
SMALL_FONT_SIZE = 5.0

# rc("legend", fontsize=8)
A4_WIDTH = 6.5
plt.style.use("seaborn-v0_8-paper")

FIGS_ROOT_PATH = "resources"

RATIO = 0.4


def plot_elem_dat(dat: np.ndarray, ax, ran=[0.0, 1.0]):
    # In our setting, dat[i, j] means the block arround (x, y) = (i*h, j*h)
    posi = ax.imshow(
        dat.T,
        aspect="equal",
        interpolation="none",
        origin="lower",
        extent=(0.0, 1.0, 0.0, 1.0),
        vmin=ran[0],
        vmax=ran[1],
    )
    ax.tick_params(axis="both", which="both", labelsize=SMALL_FONT_SIZE)
    return posi


def plot_node_dat(dat: np.ndarray, ax, ran=[0.0, 1.0]):
    # In our setting, dat[i, j] means the node at (x, y) = (j*h, i*h)
    xx = np.linspace(0.0, 1.0, dat.shape[1])
    yy = np.linspace(0.0, 1.0, dat.shape[0])
    posi = ax.pcolormesh(xx, yy, dat, shading="gouraud", vmin=ran[0], vmax=ran[1])
    ax.set_aspect("equal", "box")
    ax.tick_params(axis="both", which="both", labelsize=SMALL_FONT_SIZE)
    return posi


def append_colorbar(fig, ax, posi):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad="2%")
    cbar = fig.colorbar(posi, cax=cax)
    cbar.ax.tick_params(labelsize=SMALL_FONT_SIZE)


def plot_coeff(coeff: np.ndarray, file_name: str, l=None):
    fig = plt.figure(figsize=(A4_WIDTH * RATIO, A4_WIDTH * RATIO), layout="constrained")
    ax = fig.add_subplot()
    # In our setting, coeff[i, j] means the block arround (x, y) = (i*h, j*h),
    # which means the image contain i_max cols and j_max row.
    # xx = np.linspace(0.0, 1.0, coeff.shape[0] + 1)
    # yy = np.linspace(0.0, 1.0, coeff.shape[1] + 1)
    # posi = ax.pcolormesh(yy, xx, coeff.T, shading="flat")
    # ax.set_aspect("equal", "box")
    posi = plot_elem_dat(coeff, ax, [np.min(coeff), np.max(coeff)])

    # Make the height of colorbar equal to the ax.
    # Fancy calls from deeper matplotlib.
    append_colorbar(fig, ax, posi)

    ax.set_xticks([0.0, 0.5, 1.0], ["0.0", "0.5", "1.0"])
    # ax.set_xticks([], [])
    if l != None and 0.0 < l < 1.0:
        ax.set_yticks([0.0, l, 1.0], ["0.0", "$l$", "1.0"])
    else:
        ax.set_yticks([0.0, 1.0], ["0.0", "1.0"])
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")

    plt.savefig(
        "{0:s}/{1:s}.pdf".format(FIGS_ROOT_PATH, file_name), bbox_inches="tight"
    )


def plot_solution(u: np.ndarray, file_name: str):
    fine_grid = round(np.sqrt(u.shape[0])) + 1
    if u.shape[0] != (fine_grid - 1) ** 2:
        print(
            "The input vector has a dim={0:d}, while calculated fine_grid={1:d}".format(
                u.shape[0], fine_grid
            )
        )
        return
    # In our setting, u[j*(N-1)+i] means the u at (x, y)=(i*h, j*h)
    u_inner = u.reshape((fine_grid - 1, -1))
    u_ex = np.zeros((fine_grid + 1, fine_grid + 1))
    u_ex[1:-1, 1:-1] = u_inner

    # This method is not perfect.
    # plot_coeff(u_ex.T, file_name)
    fig = plt.figure(figsize=(A4_WIDTH * RATIO, A4_WIDTH * RATIO), layout="constrained")
    ax = fig.add_subplot()

    posi = plot_node_dat(u_ex, ax, [np.min(u), np.max(u)])

    append_colorbar(fig, ax, posi)
    ax.set_xticks([0.0, 0.5, 1.0], ["0.0", "0.5", "1.0"])
    ax.set_yticks([0.0, 0.5, 1.0], ["0.0", "0.5", "1.0"])
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")

    plt.savefig(
        "{0:s}/{1:s}.pdf".format(FIGS_ROOT_PATH, file_name), bbox_inches="tight"
    )


if __name__ == "__main__":
    from square_inclusion_cells_settings import get_test_settings as periodic_settings
    from simple_flat_interface_settings import get_test_settings as flat_settings

    fine_grid = 256

    coeff = periodic_settings(fine_grid, [-1.0, 1.0], 8)
    plot_coeff(coeff, "periodic-cells-coeff-32")

    coeff = periodic_settings(fine_grid, [-1.0, 1.0], 16)
    plot_coeff(coeff, "periodic-cells-coeff-16")

    l = 0.25
    coeff, _, u = flat_settings(fine_grid, [1.0, 1.0], l)
    plot_coeff(coeff, "flat-interface", l)

    plot_solution(u, "flat-interface-solution")

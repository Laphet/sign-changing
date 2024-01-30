import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

plt.style.use("seaborn-v0_8-paper")

# Nature suggests that fontsizes should be between 5~7pt.
DEFAULT_FONT_SIZE = 7
SMALL_FONT_SIZE = 6
# plt.rc("text", usetex=True)
plt.rc("font", size=DEFAULT_FONT_SIZE)  # controls default text sizes
plt.rc("axes", titlesize=DEFAULT_FONT_SIZE)  # fontsize of the axes title
plt.rc("axes", labelsize=DEFAULT_FONT_SIZE)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=SMALL_FONT_SIZE)  # fontsize of the tick labels
plt.rc("ytick", labelsize=SMALL_FONT_SIZE)  # fontsize of the tick labels
plt.rc("legend", fontsize=SMALL_FONT_SIZE)  # legend fontsize
plt.rc("figure", titlesize=DEFAULT_FONT_SIZE)  # fontsize of the figure title

A4_WIDTH = 6.5
# Nature suggests the width should be 180mm.
NATURE_WIDTH = 7.0866142

FIGS_ROOT_PATH = "resources"

RATIO = 0.4


def plot_elem_dat(dat: np.ndarray, ax, ran=None):
    # In our setting, dat[i, j] means the block arround (x, y) = (i*h, j*h)
    if ran == None:
        posi = ax.imshow(
            dat.T,
            aspect="equal",
            interpolation="none",
            origin="lower",
            extent=(0.0, 1.0, 0.0, 1.0),
        )
    else:
        posi = ax.imshow(
            dat.T,
            aspect="equal",
            interpolation="none",
            origin="lower",
            extent=(0.0, 1.0, 0.0, 1.0),
            vmin=ran[0],
            vmax=ran[1],
        )
    return posi


def plot_node_dat(dat: np.ndarray, ax, ran=None):
    # In our setting, dat[i, j] means the node at (x, y) = (j*h, i*h)
    xx = np.linspace(0.0, 1.0, dat.shape[1])
    yy = np.linspace(0.0, 1.0, dat.shape[0])
    if ran == None:
        posi = ax.pcolormesh(xx, yy, dat, shading="gouraud")
    else:
        posi = ax.pcolormesh(xx, yy, dat, shading="gouraud", vmin=ran[0], vmax=ran[1])
    ax.set_aspect("equal", "box")
    return posi


def append_colorbar(fig, ax, posi):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("top", size="5%", pad="2%")
    cbar = fig.colorbar(posi, cax=cax, orientation="horizontal")
    cbar.ax.tick_params(labelsize=SMALL_FONT_SIZE, rotation=15)
    cax.xaxis.set_ticks_position("top")
    return cbar


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
    from simple_flat_interface_settings import get_test_settings
    from itertools import product

    fine_grid = 400
    sigma_pm_list = [[1.1, 1.0], [1.0, 1.1], [1.01, 1.0], [1.0, 1.01]]
    l_list = [0.5, 0.5 - 1.0 / 128]

    fig = plt.figure(layout="constrained")
    axs = fig.subplots(len(l_list), len(sigma_pm_list))

    for l_ind, sigma_ind in product(range(len(l_list)), range(len(sigma_pm_list))):
        ax = axs[l_ind, sigma_ind]
        l, sigma = l_list[l_ind], sigma_pm_list[sigma_ind]
        (
            coeff,
            _,
            _,
        ) = get_test_settings(fine_grid, sigma, l)
        posi = plot_elem_dat(coeff, ax)
        append_colorbar(fig, ax, posi)
        ax.set_xticks([], [])
        ax.set_yticks([], [])

    plt.savefig(
        "{0:s}/{1:s}.pdf".format(FIGS_ROOT_PATH, "flat-interface-settings"),
        bbox_inches="tight",
    )

import numpy as np
from itertools import product


def get_test_settings(fine_grids: int, sigma_im, cell_len=10):
    coeff = np.zeros((fine_grids, fine_grids))

    for elem_ind_x, elem_ind_y in product(range(fine_grids), range(fine_grids)):
        sub_elem_ind_x = elem_ind_x % cell_len
        sub_elem_ind_y = elem_ind_y % cell_len
        if (
            cell_len // 2 - cell_len // 10
            <= sub_elem_ind_x
            < cell_len // 2 + cell_len // 10
            or cell_len // 2 - cell_len // 10
            <= sub_elem_ind_y
            < cell_len // 2 + cell_len // 10
        ):
            coeff[elem_ind_x, elem_ind_y] = sigma_im[0]
        else:
            coeff[elem_ind_x, elem_ind_y] = sigma_im[1]
    return coeff


if __name__ == "__main__":
    import plot_settings
    import matplotlib.patches as patches
    from cem_gmsfem import CemGmsfem

    fine_grid = 400
    coarse_grid = 10
    sub_grid = fine_grid // coarse_grid
    H = 1.0 / coarse_grid
    eigen_num = 3

    sigma_im = [-0.1, 1.0]
    coeff = get_test_settings(fine_grid, sigma_im, sub_grid)

    coarse_elem_ind_x, coarse_elem_ind_y = 1, 2
    coarse_elem_ind = coarse_elem_ind_y * coarse_grid + coarse_elem_ind_x

    fig = plot_settings.plt.figure(
        figsize=(plot_settings.A4_WIDTH, plot_settings.A4_WIDTH), layout="constrained"
    )
    axs = fig.subplots(4, 4)

    # Put a rectangle to indicate which coarse element.
    ax = axs[0, 0]
    posi = plot_settings.plot_elem_dat(coeff, ax, sigma_im)
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_xticks([0.0, 0.5, 1.0], ["0.0", "0.5", "1.0"])
    ax.set_yticks([0.0, 0.5, 1.0], ["0.0", "0.5", "1.0"])
    rect = patches.Rectangle(
        (coarse_elem_ind_x * H, coarse_elem_ind_y * H),
        H,
        H,
        linewidth=1,
        edgecolor="r",
        facecolor="none",
    )
    ax.add_patch(rect)
    plot_settings.append_colorbar(fig, ax, posi)

    solver = CemGmsfem(coarse_grid, eigen_num, 0, coeff)
    solver.get_eigen_pair()
    this_eigen_vec = solver.eigen_vec[
        :, coarse_elem_ind * solver.eigen_num : (coarse_elem_ind + 1) * solver.eigen_num
    ]

    # Guessed bound for the range of eigenfunctions.
    eigen_vec_ran = [-0.5, 0.5]

    for i in [0, 1, 2]:
        dat = this_eigen_vec[:, i].reshape((solver.sub_grid + 1, -1))
        ax = axs[0, i + 1]
        posi = plot_settings.plot_node_dat(dat, ax, eigen_vec_ran)
        plot_settings.append_colorbar(fig, ax, posi)
        ax.set_xticks([], [])
        ax.set_yticks([], [])

    ms_basis_dat = np.zeros((3 * eigen_num, solver.tot_node))

    for os_ly in [1, 2, 3]:
        solver = CemGmsfem(coarse_grid, eigen_num, os_ly, coeff)
        solver.get_eigen_pair()
        solver.get_ind_map()
        solver.get_ms_basis_on_coarse_elem(coarse_elem_ind)
        ms_basis_dat[3 * os_ly - 3, :] = solver.get_glb_vec(
            coarse_elem_ind, (solver.basis_list[coarse_elem_ind])[:, 0]
        )
        ms_basis_dat[3 * os_ly - 2, :] = solver.get_glb_vec(
            coarse_elem_ind, (solver.basis_list[coarse_elem_ind])[:, 1]
        )
        ms_basis_dat[3 * os_ly - 1, :] = solver.get_glb_vec(
            coarse_elem_ind, (solver.basis_list[coarse_elem_ind])[:, 2]
        )

    for os_ly in [1, 2, 3]:
        for i in range(eigen_num):
            ax = axs[os_ly, i]
            dat = ms_basis_dat[3 * (os_ly - 1) + i].reshape((fine_grid + 1, -1))
            print(
                "os_ly={0:d}, i={1:d}, min={2:.1f}, max={3:.1f}".format(
                    os_ly, i, np.min(dat), np.max(dat)
                )
            )
            posi = plot_settings.plot_node_dat(dat, ax, eigen_vec_ran)
            plot_settings.append_colorbar(fig, ax, posi)
            ax.set_xlabel("$x_1$")
            ax.set_ylabel("$x_2$")
            ax.set_xticks([0.0, 0.5, 1.0], ["0.0", "0.5", "1.0"])
            ax.set_yticks([0.0, 0.5, 1.0], ["0.0", "0.5", "1.0"])

    plot_settings.plt.savefig(
        "{0:s}/{1:s}.pdf".format(
            plot_settings.FIGS_ROOT_PATH,
            "cross-inclusion-coeff-{0:d}x{0:d}".format(coarse_grid),
        ),
        bbox_inches="tight",
    )

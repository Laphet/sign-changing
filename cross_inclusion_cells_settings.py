import numpy as np
from itertools import product

DAT_ROOT_PATH = "resources"


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
    from cem_gmsfem import CemGmsfem

    fine_grid = 400
    coarse_grid = 10
    sub_grid = fine_grid // coarse_grid
    H = 1.0 / coarse_grid
    eigen_num = 3
    sigma_im = [-0.1, 1.0]
    coarse_elem_ind_x, coarse_elem_ind_y = 1, 2
    coarse_elem_ind = coarse_elem_ind_y * coarse_grid + coarse_elem_ind_x
    osly_list = [1, 2, 3, 4, 5, 6, 7, 8]

    coeff = np.zeros((fine_grid, fine_grid))
    eigen_vec_dat = np.zeros((eigen_num, (sub_grid + 1) ** 2))
    ms_basis_dat = np.zeros((len(osly_list), eigen_num, (fine_grid + 1) ** 2))
    errors_dat = np.zeros((2, eigen_num, len(osly_list) - 1))

    prepare_dat = True
    if prepare_dat:
        coeff = get_test_settings(fine_grid, sigma_im, sub_grid)
        for i, osly in enumerate(osly_list):
            solver = CemGmsfem(coarse_grid, eigen_num, osly, coeff)
            solver.get_eigen_pair()
            if i == 0:
                eigen_vec_dat = solver.eigen_vec[
                    :,
                    coarse_elem_ind
                    * solver.eigen_num : (coarse_elem_ind + 1)
                    * solver.eigen_num,
                ].T
            solver.get_ind_map()
            solver.get_ms_basis_on_coarse_elem(coarse_elem_ind)
            for j in range(eigen_num):
                ms_basis_dat[i, j, :] = solver.get_glb_vec(
                    coarse_elem_ind, (solver.basis_list[coarse_elem_ind])[:, j]
                )
            if i == len(osly_list) - 1:
                solver.get_glb_A()
                rela = np.zeros((2, eigen_num))
                for k in range(eigen_num):
                    u = ms_basis_dat[i, k, :]
                    rela[0, k] = np.sqrt(np.dot(solver.glb_A.dot(u), u))
                    rela[1, k] = np.linalg.norm(u)
                for k1, k2 in product(range(len(osly_list) - 1), range(eigen_num)):
                    delta_u = ms_basis_dat[k1, k2, :] - ms_basis_dat[i, k2, :]
                    delta_u_h1 = np.sqrt(np.dot(solver.glb_A.dot(delta_u), delta_u))
                    delta_u_l2 = np.linalg.norm(delta_u)
                    errors_dat[0, k2, k1] = delta_u_h1 / rela[0, k2]
                    errors_dat[1, k2, k1] = delta_u_l2 / rela[1, k2]
            print("Finish osly={:d}".format(osly))
        np.save("{0:s}/{1:s}".format(DAT_ROOT_PATH, "coeff.npy"), coeff)
        np.save("{0:s}/{1:s}".format(DAT_ROOT_PATH, "eigen-vec.npy"), eigen_vec_dat)
        np.save("{0:s}/{1:s}".format(DAT_ROOT_PATH, "ms-basis.npy"), ms_basis_dat)
        np.save("{0:s}/{1:s}".format(DAT_ROOT_PATH, "errors.npy"), errors_dat)
        print("Save all data!")
    else:
        coeff = np.load("{0:s}/{1:s}".format(DAT_ROOT_PATH, "coeff.npy"))
        eigen_vec_dat = np.load("{0:s}/{1:s}".format(DAT_ROOT_PATH, "eigen-vec.npy"))
        ms_basis_dat = np.load("{0:s}/{1:s}".format(DAT_ROOT_PATH, "ms-basis.npy"))
        errors_dat = np.load("{0:s}/{1:s}".format(DAT_ROOT_PATH, "errors.npy"))
        print("Load all data!")

    plot_fig = False
    if plot_fig:
        import plot_settings
        import matplotlib.patches as patches

        # Begin to plot
        fig = plot_settings.plt.figure(
            figsize=(plot_settings.A4_WIDTH, plot_settings.A4_WIDTH),
            layout="constrained",
        )
        axs = fig.subplots(4, 4)

        ax = axs[0, 0]
        posi = plot_settings.plot_elem_dat(coeff, ax, sigma_im)
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")
        ax.set_xticks([0.0, 0.5, 1.0], ["0.0", "0.5", "1.0"])
        ax.set_yticks([0.0, 0.5, 1.0], ["0.0", "0.5", "1.0"])
        # Put a rectangle to indicate which coarse element.
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

        for i in range(eigen_num):
            dat = eigen_vec_dat[i, :].reshape((solver.sub_grid + 1, -1))
            ax = axs[0, i + 1]
            posi = plot_settings.plot_node_dat(dat, ax, [-0.5, 0.5])
            plot_settings.append_colorbar(fig, ax, posi)
            ax.set_xticks([], [])
            ax.set_yticks([], [])

        for i in range(eigen_num):
            for osly_ind in [0, 1, 2]:
                ax = axs[i + 1, osly_ind]
                dat = ms_basis_dat[osly_ind, i, :].reshape((fine_grid + 1, -1))
                print(
                    "os_ly={0:d}, eigen_ind={1:d}, min={2:.1f}, max={3:.1f}".format(
                        osly_list[osly_ind], i, np.min(dat), np.max(dat)
                    )
                )
                posi = plot_settings.plot_node_dat(dat, ax)
                plot_settings.append_colorbar(fig, ax, posi)
                if osly_ind == 0:
                    ax.set_xlabel("$x_1$")
                    ax.set_ylabel("$x_2$")
                    ax.set_xticks([0.0, 0.5, 1.0], ["0.0", "0.5", "1.0"])
                    ax.set_yticks([0.0, 0.5, 1.0], ["0.0", "0.5", "1.0"])

        for i in range(eigen_num):
            ax = axs[i + 1, 3]
            ax.plot(osly_list[:-1], errors_dat[0, i, :], label="in energy norm")
            ax.plot(osly_list[:-1], errors_dat[1, i, :], label="in $L^2$ norm")
            ax.set_yscale("log")
            ax.legend()

        plot_settings.plt.savefig(
            "{0:s}/{1:s}.png".format(
                plot_settings.FIGS_ROOT_PATH,
                "cross-inclusion-coeff-{0:d}x{0:d}".format(coarse_grid),
            ),
            bbox_inches="tight",
            dpi=450,
        )

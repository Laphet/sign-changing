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

    fine_grid = 100
    coarse_grid = 10
    sub_grid = fine_grid // coarse_grid
    sigma_im = [-0.05, 1.0]
    coeff = get_test_settings(fine_grid, sigma_im, sub_grid)
    eigen_num = 3
    coarse_elem_ind_x, coarse_elem_ind_y = 1, 2
    coarse_elem_ind = coarse_elem_ind_y * coarse_grid + coarse_elem_ind_x

    fixed_solver = CemGmsfem(coarse_grid, eigen_num, 0, coeff)
    fixed_solver.get_eigen_pair()

    eigen_vec_dat = np.zeros((eigen_num, (sub_grid + 1) ** 2))
    eigen_vec_dat = fixed_solver.eigen_vec[
        :,
        coarse_elem_ind
        * fixed_solver.eigen_num : (coarse_elem_ind + 1)
        * fixed_solver.eigen_num,
    ].T

    osly_list = [1, 2, 3, 4, 5, 6, 7, 8]

    ms_basis_dat = np.zeros((len(osly_list), eigen_num, (fine_grid + 1) ** 2))

    prepare_dat = False
    if prepare_dat:
        coeff = get_test_settings(fine_grid, sigma_im, sub_grid)
        for i, osly in enumerate(osly_list):
            solver = CemGmsfem(coarse_grid, eigen_num, osly, coeff)
            solver.copy_eigen_space(fixed_solver)
            solver.get_ind_map()
            solver.get_ms_basis_on_coarse_elem(coarse_elem_ind)
            for j in range(eigen_num):
                ms_basis_dat[i, j, :] = solver.get_glb_vec(
                    coarse_elem_ind, (solver.basis_list[coarse_elem_ind])[:, j]
                )
            print("Finish osly={:d}".format(osly))
        np.save("{0:s}/{1:s}".format(DAT_ROOT_PATH, "ms-basis.npy"), ms_basis_dat)
        print("Save all data!")
    else:
        ms_basis_dat = np.load("{0:s}/{1:s}".format(DAT_ROOT_PATH, "ms-basis.npy"))
        print("Load all data!")

    errors_dat = np.zeros((2, eigen_num, len(osly_list) - 1))
    tmp_solver = CemGmsfem(coarse_grid, eigen_num, 0, np.abs(coeff))
    tmp_solver.get_glb_A()
    rela = np.zeros((2, eigen_num))
    for k in range(eigen_num):
        u = ms_basis_dat[-1, k, :]
        rela[0, k] = np.sqrt(np.dot(tmp_solver.glb_A.dot(u), u))
        rela[1, k] = np.linalg.norm(u)
    for k1, k2 in product(range(len(osly_list) - 1), range(eigen_num)):
        delta_u = ms_basis_dat[k1, k2, :] - ms_basis_dat[-1, k2, :]
        delta_u_h1 = np.sqrt(np.dot(tmp_solver.glb_A.dot(delta_u), delta_u))
        delta_u_l2 = np.linalg.norm(delta_u)
        errors_dat[0, k2, k1] = delta_u_h1 / rela[0, k2]
        errors_dat[1, k2, k1] = delta_u_l2 / rela[1, k2]

    plot_fig = True
    if plot_fig:
        import plot_settings
        import matplotlib.patches as patches

        # Begin to plot
        fig1 = plot_settings.plt.figure(
            figsize=(plot_settings.A4_WIDTH, plot_settings.A4_WIDTH * 0.3),
            layout="constrained",
        )
        axs1 = fig1.subplots(1, 4)

        fig2 = plot_settings.plt.figure(
            figsize=(plot_settings.A4_WIDTH, plot_settings.A4_WIDTH * 0.85),
            layout="constrained",
        )
        axs2 = fig2.subplots(3, 4)

        ax = axs1[0]
        posi = plot_settings.plot_elem_dat(coeff, ax, sigma_im)
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")
        ax.set_xticks([0.0, 0.5, 1.0], ["0.0", "0.5", "1.0"])
        ax.set_yticks([0.0, 0.5, 1.0], ["0.0", "0.5", "1.0"])
        # Put a rectangle to indicate which coarse element.
        H = 1.0 / coarse_grid
        rect = patches.Rectangle(
            (coarse_elem_ind_x * H, coarse_elem_ind_y * H),
            H,
            H,
            linewidth=1,
            edgecolor="r",
            facecolor="none",
        )
        ax.add_patch(rect)
        plot_settings.append_colorbar(fig1, ax, posi)

        for i in range(eigen_num):
            dat = eigen_vec_dat[i, :].reshape((fixed_solver.sub_grid + 1, -1))
            ax = axs1[i + 1]
            posi = plot_settings.plot_node_dat(dat, ax, [-0.5, 0.5])
            plot_settings.append_colorbar(fig1, ax, posi)
            ax.set_xlabel("$x_1$")
            ax.set_ylabel("$x_2$")

            ax.set_xticks(
                [0.0, 0.5, 1.0],
                [
                    "{:.2f}".format(coarse_elem_ind_x * H),
                    "{:.2f}".format(coarse_elem_ind_x * H + H / 2),
                    "{:.2f}".format(coarse_elem_ind_x * H + H),
                ],
            )
            ax.set_yticks(
                [0.0, 0.5, 1.0],
                [
                    "{:.2f}".format(coarse_elem_ind_y * H),
                    "{:.2f}".format(coarse_elem_ind_y * H + H / 2),
                    "{:.2f}".format(coarse_elem_ind_y * H + H),
                ],
            )

        for i in range(eigen_num):
            for osly_ind in [0, 1, 2]:
                ax = axs2[i, osly_ind]
                dat = ms_basis_dat[osly_ind, i, :].reshape((fine_grid + 1, -1))
                print(
                    "os_ly={0:d}, eigen_ind={1:d}, min={2:.6f}, max={3:.6f}".format(
                        osly_list[osly_ind], i, np.min(dat), np.max(dat)
                    )
                )
                posi = plot_settings.plot_node_dat(dat, ax)
                plot_settings.append_colorbar(fig2, ax, posi)
                ax.set_xlabel("$x_1$")
                ax.set_ylabel("$x_2$")
                ax.set_xticks([0.0, 0.5, 1.0], ["0.0", "0.5", "1.0"])
                ax.set_yticks([0.0, 0.5, 1.0], ["0.0", "0.5", "1.0"])
                # if osly_ind == 0:
                #     ax.set_xlabel("$x_1$")
                #     ax.set_ylabel("$x_2$")
                #     ax.set_xticks([0.0, 0.5, 1.0], ["0.0", "0.5", "1.0"])
                #     ax.set_yticks([0.0, 0.5, 1.0], ["0.0", "0.5", "1.0"])
                # else:
                #     ax.set_xticks([], [])
                #     ax.set_yticks([], [])

        for i in range(eigen_num):
            ax = axs2[i, 3]
            ax.plot(osly_list[:-1], errors_dat[0, i, :], label="in energy norm")
            ax.plot(osly_list[:-1], errors_dat[1, i, :], label="in $L^2$ norm")
            ax.set_yscale("log")
            ax.tick_params(
                axis="both", which="both", labelsize=plot_settings.DEFAULT_FONT_SIZE
            )
            ax.set_xlabel("$m$")
            ax.legend(loc=1, fontsize=plot_settings.DEFAULT_FONT_SIZE)
            # ax.yaxis.tick_right()

        fig1.savefig(
            "{0:s}/{1:s}.png".format(
                plot_settings.FIGS_ROOT_PATH,
                "cross-inclusion-eigen",
            ),
            bbox_inches="tight",
            dpi=450,
        )
        fig2.savefig(
            "{0:s}/{1:s}.png".format(
                plot_settings.FIGS_ROOT_PATH,
                "cross-inclusion-ms",
            ),
            bbox_inches="tight",
            dpi=450,
        )

import numpy as np
from itertools import product

DAT_ROOT_PATH = "resources"
VARIANCE = 0.01


def get_test_settings(fine_grids: int, sigma_im, cell_len=8):
    coeff = np.zeros((fine_grids, fine_grids))
    source = np.zeros(coeff.shape)
    for elem_ind_x, elem_ind_y in product(range(fine_grids), range(fine_grids)):
        sub_elem_ind_x = elem_ind_x % cell_len
        sub_elem_ind_y = elem_ind_y % cell_len
        if (
            cell_len // 2 - cell_len // 4
            <= sub_elem_ind_x
            < cell_len // 2 + cell_len // 4
            and cell_len // 2 - cell_len // 4
            <= sub_elem_ind_y
            < cell_len // 2 + cell_len // 4
        ):
            coeff[elem_ind_x, elem_ind_y] = sigma_im[0]
        else:
            coeff[elem_ind_x, elem_ind_y] = sigma_im[1]

        x = (elem_ind_x + 0.5) / source.shape[0]
        y = (elem_ind_y + 0.5) / source.shape[1]
        r = np.sqrt((x - 0.25) ** 2 + (y - 0.25) ** 2)
        source[elem_ind_x, elem_ind_y] += (
            np.exp(-(r**2) / 2.0 / VARIANCE) / VARIANCE / (2.0 * np.pi)
        )
        r = np.sqrt((x - 0.25) ** 2 + (y - 0.75) ** 2)
        source[elem_ind_x, elem_ind_y] += (
            np.exp(-(r**2) / 2.0 / VARIANCE) / VARIANCE / (2.0 * np.pi)
        )
        r = np.sqrt((x - 0.75) ** 2 + (y - 0.25) ** 2)
        source[elem_ind_x, elem_ind_y] += (
            np.exp(-(r**2) / 2.0 / VARIANCE) / VARIANCE / (2.0 * np.pi)
        )
        r = np.sqrt((x - 0.75) ** 2 + (y - 0.75) ** 2)
        source[elem_ind_x, elem_ind_y] += (
            np.exp(-(r**2) / 2.0 / VARIANCE) / VARIANCE / (2.0 * np.pi)
        )
    return coeff, source


if __name__ == "__main__":
    from cem_gmsfem import CemGmsfem

    fine_grid = 400
    coarse_grid = 10
    sub_grid = fine_grid // coarse_grid
    sigma_im = [-0.1, 1.0]
    # sigma_im = [-1.0, 10.0]
    token = "-0d1+1d0"
    coeff, _ = get_test_settings(fine_grid, sigma_im, sub_grid)
    eigen_num = 3
    coarse_elem_ind_x, coarse_elem_ind_y = 1, 2
    coarse_elem_ind = coarse_elem_ind_y * coarse_grid + coarse_elem_ind_x

    print("=" * 80)
    print("Start")
    print("fine_grid={0:d}, sigma_im=({1:4e},{2:4e})".format(fine_grid, *sigma_im))

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
        np.save(
            "{0:s}/{1:s}".format(DAT_ROOT_PATH, "ms-basis{0:s}.npy".format(token)),
            ms_basis_dat,
        )
        print("Save all data!")
    else:
        ms_basis_dat = np.load(
            # "{0:s}/{1:s}".format(DAT_ROOT_PATH, "ms-basis-0d1+1d0.npy")
            "{0:s}/{1:s}".format(DAT_ROOT_PATH, "ms-basis{0:s}.npy".format(token))
        )
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
        ax.set_xticks([0.0, 1.0], ["0.0", "1.0"])
        ax.xaxis.set_label_coords(0.5, -0.1)
        ax.set_yticks([0.0, 1.0], ["0.0", "1.0"])
        ax.yaxis.set_label_coords(-0.1, 0.5)
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
            posi = plot_settings.plot_node_dat(
                dat, ax, [np.min(eigen_vec_dat), np.max(eigen_vec_dat)]
            )
            plot_settings.append_colorbar(fig1, ax, posi)
            ax.set_xlabel("$x_1$")
            ax.set_ylabel("$x_2$")

            ax.set_xticks(
                [0.0, 1.0],
                [
                    "{:.2f}".format(coarse_elem_ind_x * H),
                    "{:.2f}".format(coarse_elem_ind_x * H + H),
                ],
            )
            ax.set_yticks(
                [0.0, 1.0],
                [
                    "{:.2f}".format(coarse_elem_ind_y * H),
                    "{:.2f}".format(coarse_elem_ind_y * H + H),
                ],
            )

            ax.xaxis.set_label_coords(0.5, -0.1)
            ax.yaxis.set_label_coords(-0.1, 0.5)

        ran = [None, None, None]
        for i in range(eigen_num):
            ran[i] = [np.min(ms_basis_dat[:3, i, :]), np.max(ms_basis_dat[:3, i, :])]

        for i in range(eigen_num):
            for osly_ind in [0, 1, 2]:
                ax = axs2[i, osly_ind]
                dat = ms_basis_dat[osly_ind, i, :].reshape((fine_grid + 1, -1))
                posi = plot_settings.plot_node_dat(dat, ax, ran[i])
                plot_settings.append_colorbar(fig2, ax, posi)
                ax.set_xlabel("$x_1$")
                ax.set_ylabel("$x_2$")
                ax.set_xticks([0.0, 1.0], ["0.0", "1.0"])
                ax.xaxis.set_label_coords(0.5, -0.1)
                ax.set_yticks([0.0, 1.0], ["0.0", "1.0"])
                ax.yaxis.set_label_coords(-0.1, 0.5)

        for i in range(eigen_num):
            ax = axs2[i, 3]
            ax.plot(osly_list[:-1], errors_dat[0, i, :], label="in energy norm")
            ax.plot(osly_list[:-1], errors_dat[1, i, :], label="in $L^2$ norm")
            ax.set_yscale("log")
            ax.set_xlabel("$m$")
            ax.legend(loc=1)
            # ax.yaxis.tick_right()
        fig1.savefig(
            "{0:s}/{1:s}.pdf".format(
                plot_settings.FIGS_ROOT_PATH,
                "square-inclusion-eigen{0:s}".format(token),
            ),
            bbox_inches="tight",
            # dpi=450,
        )
        fig2.savefig(
            "{0:s}/{1:s}.png".format(
                plot_settings.FIGS_ROOT_PATH,
                "square-inclusion-ms{0:s}".format(token),
            ),
            bbox_inches="tight",
            dpi=plot_settings.DPI,
        )


if __name__ == "__main_prepare_dat__":
    from cem_gmsfem import CemGmsfem
    from fem import get_fem_mat, get_mass_mat, get_fem_rhs
    import argparse
    import logging
    from logging import config
    from scipy.sparse.linalg import spsolve

    fine_grid = 100
    coarse_grid_list = [10, 20, 40, 80]
    osly_list = [0, 1, 2, 3, 4]
    cell_len_list = [20, 40, 10]

    eigen_num = 3
    sigma_im = [-0.1, 1.0]

    parse = argparse.ArgumentParser()
    parse.add_argument("--cell", default=0, type=int)
    args = parse.parse_args()
    cell_len = cell_len_list[args.cell]

    coeff, source = get_test_settings(fine_grid, sigma_im, cell_len)

    fem_mat_abs = get_fem_mat(np.abs(coeff))
    mass_mat = get_mass_mat(np.ones(coeff.shape))
    mat_fem = get_fem_mat(coeff)
    rhs = get_fem_rhs(source)
    u_fem = spsolve(mat_fem, rhs)
    u_h1 = np.sqrt(np.dot(fem_mat_abs.dot(u_fem), u_fem))
    u_l2 = np.sqrt(np.dot(mass_mat.dot(u_fem), u_fem))

    u_ext = np.zeros((fine_grid + 1, fine_grid + 1))
    u_ext[1:-1, 1:-1] = u_fem.reshape((fine_grid - 1, -1))
    np.save(
        "{0:s}/{1:s}".format(
            DAT_ROOT_PATH, "square-inclusion-cell{:d}.npy".format(cell_len)
        ),
        u_ext,
    )

    config.fileConfig(
        "log.conf",
        defaults={"logfilename": "logs/square-inclusion-cell{:d}.log".format(cell_len)},
    )

    logging.info("=" * 80)
    logging.info("Start")
    logging.info(
        "In the medium, sigma_inc={0:.4e}, sigma_mat={1:.4e}, cell_len={2:d}".format(
            *sigma_im, cell_len
        )
    )

    rela_errors_h1 = np.zeros((len(coarse_grid_list), len(osly_list)))
    rela_errors_l2 = np.zeros(rela_errors_h1.shape)

    for coarse_grid_ind, osly_ind in product(
        range(len(coarse_grid_list)), range(len(osly_list))
    ):
        coarse_grid = coarse_grid_list[coarse_grid_ind]
        osly = osly_list[osly_ind]

        solver = CemGmsfem(coarse_grid, eigen_num, osly, coeff)
        solver.setup()
        u_cem = None
        if osly == 0:
            u_cem = solver.solve_by_coarse_bilinear(source)
        else:
            u_cem = solver.solve(source)

        delta_u = u_fem - u_cem

        delta_u_h1 = np.sqrt(np.dot(fem_mat_abs.dot(delta_u), delta_u))
        rela_error_h1 = delta_u_h1 / u_h1
        rela_errors_h1[coarse_grid_ind, osly_ind] = rela_error_h1

        delta_u_l2 = np.sqrt(np.dot(mass_mat.dot(delta_u), delta_u))
        rela_error_l2 = delta_u_l2 / u_l2
        rela_errors_l2[coarse_grid_ind, osly_ind] = rela_error_l2

        logging.info(
            "relative energy error={0:6e}, plain-L2 error={1:6e}.".format(
                rela_error_h1, rela_error_l2
            )
        )
    print(rela_errors_h1)
    print(rela_errors_l2)


if __name__ == "__main_a__":
    import plot_settings

    u_cell20 = np.load(
        "{0:s}/{1:s}".format(DAT_ROOT_PATH, "square-inclusion-cell20.npy")
    )
    u_cell40 = np.load(
        "{0:s}/{1:s}".format(DAT_ROOT_PATH, "square-inclusion-cell40.npy")
    )
    sigma_im = [-0.1, 1.0]
    fine_grid = 400
    _, source = get_test_settings(fine_grid, sigma_im, 20)

    fig = plot_settings.plt.figure(
        figsize=(plot_settings.A4_WIDTH * 0.75, plot_settings.A4_WIDTH * 0.3),
        layout="constrained",
    )
    axs = fig.subplots(1, 3)

    ax = axs[0]
    posi = plot_settings.plot_elem_dat(source, ax)
    plot_settings.append_colorbar(fig, ax, posi)
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_xticks([0.0, 1.0], ["0.0", "1.0"])
    ax.xaxis.set_label_coords(0.5, -0.1)
    ax.set_yticks([0.0, 1.0], ["0.0", "1.0"])
    ax.yaxis.set_label_coords(-0.1, 0.5)

    ax = axs[1]
    posi = plot_settings.plot_node_dat(u_cell40, ax)
    plot_settings.append_colorbar(fig, ax, posi)
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_xticks([0.0, 1.0], ["0.0", "1.0"])
    ax.xaxis.set_label_coords(0.5, -0.1)
    ax.set_yticks([0.0, 1.0], ["0.0", "1.0"])
    ax.yaxis.set_label_coords(-0.1, 0.5)

    ax = axs[2]
    posi = plot_settings.plot_node_dat(u_cell20, ax)
    plot_settings.append_colorbar(fig, ax, posi)
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_xticks([0.0, 1.0], ["0.0", "1.0"])
    ax.xaxis.set_label_coords(0.5, -0.1)
    ax.set_yticks([0.0, 1.0], ["0.0", "1.0"])
    ax.yaxis.set_label_coords(-0.1, 0.5)

    fig.savefig(
        "{0:s}/{1:s}.png".format(
            plot_settings.FIGS_ROOT_PATH,
            "square-inclusion-source-u",
        ),
        bbox_inches="tight",
        dpi=plot_settings.DPI,
    )

    error_cell40 = np.array(
        [
            [
                6.08118593e-01,
                2.43294994e-01,
                5.16230471e-02,
                5.22453303e-02,
                5.23150551e-02,
            ],
            [
                6.02143921e-01,
                3.78548060e-01,
                4.96049422e-02,
                5.58312503e-02,
                5.66200143e-02,
            ],
            [
                2.77296490e-01,
                6.97828262e-01,
                4.01113407e-02,
                1.75301146e-03,
                1.37559146e-04,
            ],
            [
                1.73068366e-01,
                8.83205301e-01,
                8.89513534e-02,
                3.85498737e-03,
                1.94060015e-04,
            ],
            [
                3.25267870e-01,
                8.93104743e-02,
                5.74053343e-03,
                5.78539821e-03,
                5.78605100e-03,
            ],
            [
                3.18135323e-01,
                1.92327279e-01,
                4.98052381e-03,
                5.92211567e-03,
                6.02412334e-03,
            ],
            [
                7.44368266e-02,
                5.83254996e-01,
                2.29192819e-03,
                3.29267333e-05,
                3.07513565e-06,
            ],
            [
                2.99430683e-02,
                8.61005613e-01,
                1.06422320e-02,
                3.99088642e-05,
                1.92089883e-06,
            ],
        ]
    )
    error_cell20 = np.array(
        [
            [
                6.02796244e-01,
                5.57109192e-01,
                6.37880532e-02,
                2.65295757e-02,
                2.63681750e-02,
            ],
            [
                5.94271902e-01,
                4.53056222e-01,
                3.19473253e-02,
                2.59799175e-02,
                2.61569851e-02,
            ],
            [
                5.92705482e-01,
                6.30557027e-01,
                4.54435662e-02,
                2.58890187e-02,
                2.78185608e-02,
            ],
            [
                2.59740590e-01,
                8.80342870e-01,
                8.79519604e-02,
                4.30321316e-03,
                2.86229803e-04,
            ],
            [
                3.21252037e-01,
                3.66295288e-01,
                6.15865959e-03,
                1.49209297e-03,
                1.45477850e-03,
            ],
            [
                3.15625869e-01,
                2.88381613e-01,
                1.49814013e-03,
                1.44116445e-03,
                1.44219182e-03,
            ],
            [
                3.13784260e-01,
                4.93840781e-01,
                2.61284038e-03,
                1.36066228e-03,
                1.48492252e-03,
            ],
            [
                6.81371711e-02,
                8.49909813e-01,
                1.05737186e-02,
                4.98389194e-05,
                3.11467804e-06,
            ],
        ]
    )

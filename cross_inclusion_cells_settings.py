import numpy as np
from itertools import product

DAT_ROOT_PATH = "resources"
VARIANCE = 0.01


def get_test_settings(fine_grids: int, sigma_im, cell_len=10):
    coeff = np.zeros((fine_grids, fine_grids))
    source = np.zeros(coeff.shape)
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
    sigma_im_list = [[-0.001, 1.0], [-1000.0, 1.0]]

    parse = argparse.ArgumentParser()
    parse.add_argument("--cell", default=0, type=int)
    parse.add_argument("--sigma", default=0, type=int)
    args = parse.parse_args()
    cell_len = cell_len_list[args.cell]
    sigma_im = sigma_im_list[args.sigma]

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
            DAT_ROOT_PATH,
            "cross-inclusion-cell{0:d}-sigma{1:d}.npy".format(cell_len, args.sigma),
        ),
        u_ext,
    )

    config.fileConfig(
        "log.conf",
        defaults={
            "logfilename": "logs/cross-inclusion-cell{0:d}-sigma{1:d}.log".format(
                cell_len, args.sigma
            )
        },
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

if __name__ == "__main__":
    import plot_settings

    u_cell20_sigma0 = np.load(
        "{0:s}/{1:s}".format(DAT_ROOT_PATH, "cross-inclusion-cell20-sigma0.npy")
    )
    u_cell40_sigma0 = np.load(
        "{0:s}/{1:s}".format(DAT_ROOT_PATH, "cross-inclusion-cell40-sigma0.npy")
    )

    u_cell20_sigma1 = np.load(
        "{0:s}/{1:s}".format(DAT_ROOT_PATH, "cross-inclusion-cell20-sigma1.npy")
    )
    u_cell40_sigma1 = np.load(
        "{0:s}/{1:s}".format(DAT_ROOT_PATH, "cross-inclusion-cell40-sigma1.npy")
    )

    fine_grid = 400
    sigma_im_list = [[-0.001, 1.0], [-1000.0, 1.0]]
    cell_len_list = [20, 40]

    coeff_cell20_sigma0, _ = get_test_settings(
        fine_grid, sigma_im_list[0], cell_len_list[0]
    )
    coeff_cell40_sigma0, _ = get_test_settings(
        fine_grid, sigma_im_list[0], cell_len_list[1]
    )

    coeff_cell20_sigma1, _ = get_test_settings(
        fine_grid, sigma_im_list[1], cell_len_list[0]
    )
    coeff_cell40_sigma1, _ = get_test_settings(
        fine_grid, sigma_im_list[1], cell_len_list[1]
    )

    error_cell20_sigma0 = np.array(
        [
            [
                9.99426812e-01,
                9.97061393e-01,
                1.57271584e00,
                1.48487147e00,
                1.70198681e00,
            ],
            [
                9.99417856e-01,
                6.43400746e00,
                8.47388845e00,
                4.38152469e01,
                4.94774820e00,
            ],
            [
                9.99405250e-01,
                2.32559202e-01,
                2.49756032e-02,
                2.49947058e-02,
                2.49946031e-02,
            ],
            [
                1.00019306e00,
                1.00868025e00,
                5.15105122e-01,
                1.74370706e-02,
                5.09581194e-03,
            ],
            [
                1.00737528e00,
                9.94551869e-01,
                1.00034037e00,
                1.01945610e00,
                1.02096087e00,
            ],
            [1.00743470e00, 1.15569864e00, 1.25800289e00, 6.17475805e00, 1.68998821e00],
            [
                1.00745510e00,
                6.35925662e-02,
                9.10898606e-04,
                9.11853156e-04,
                9.11847439e-04,
            ],
            [
                1.00795920e00,
                1.00313574e00,
                1.58938518e-01,
                5.09355166e-04,
                1.18935417e-04,
            ],
        ]
    )
    error_cell40_sigma0 = np.array(
        [
            [0.99944493, 3.06305447, 0.49260854, 0.06844358, 0.93830347],
            [0.9993957, 0.11892138, 0.05141211, 0.05142528, 0.05142523],
            [1.00018417, 1.03621683, 0.21462232, 0.01354774, 0.01016451],
            [1.00889666, 1.00328517, 3.46778694, 0.04259675, 0.00653445],
            [
                1.00730273e00,
                1.33515222e00,
                7.86267305e-02,
                5.15288585e-03,
                1.46504490e-01,
            ],
            [
                1.00738179e00,
                1.74800599e-02,
                3.81301860e-03,
                3.81425455e-03,
                3.81424841e-03,
            ],
            [
                1.00789734e00,
                1.01324877e00,
                3.32044636e-02,
                6.37058694e-04,
                4.45814821e-04,
            ],
            [
                1.01219712e00,
                1.00050226e00,
                2.25930682e00,
                7.26577286e-04,
                1.31936123e-04,
            ],
        ]
    )

    error_cell20_sigma1 = np.array(
        [
            [0.74462135, 0.47985363, 0.47495082, 0.47496062, 0.47496012],
            [0.73990599, 0.82637366, 0.49186014, 0.475478, 0.47498903],
            [0.73836482, 0.8084483, 0.25896581, 0.11363937, 0.10728617],
            [0.56486625, 0.85376654, 0.29558223, 0.03088685, 0.03170778],
            [0.42550053, 0.37832994, 0.38344796, 0.38347426, 0.38347376],
            [0.4244774, 0.64782291, 0.3754701, 0.38352627, 0.38348944],
            [0.42426049, 1.0026728, 0.09735936, 0.03589979, 0.0364908],
            [0.28422366, 1.08660053, 0.09895763, 0.00347909, 0.00342085],
        ]
    )

    error_cell40_sigma1 = np.array(
        [
            [0.86014253, 0.81652545, 0.7346366, 0.73267297, 0.73257415],
            [0.85717152, 0.54101822, 0.18564663, 0.16951939, 0.16970769],
            [0.74815942, 0.62694302, 0.11561481, 0.06864644, 0.07143912],
            [0.39518474, 0.61839716, 0.06741064, 0.0132029, 0.01285546],
            [1.0013486, 1.04695486, 1.12764205, 1.12819604, 1.12812656],
            [1.00284351, 0.53084901, 0.10459479, 0.10974393, 0.1097837],
            [0.93065999, 0.70280245, 0.02311093, 0.015106, 0.01593673],
            [0.24178904, 0.71303883, 0.00927127, 0.00158098, 0.00157519],
        ]
    )

    fig_sigma0_a = plot_settings.plt.figure(
        figsize=(plot_settings.A4_WIDTH, plot_settings.A4_WIDTH * 0.3),
        layout="constrained",
    )
    axs_sigma0_a = fig_sigma0_a.subplots(1, 4)

    fig_sigma0_b = plot_settings.plt.figure(
        figsize=(plot_settings.A4_WIDTH, plot_settings.A4_WIDTH * 0.3),
        layout="constrained",
    )
    axs_sigma0_b = fig_sigma0_b.subplots(1, 4)

    fig_sigma1_a = plot_settings.plt.figure(
        figsize=(plot_settings.A4_WIDTH, plot_settings.A4_WIDTH * 0.3),
        layout="constrained",
    )
    axs_sigma1_a = fig_sigma1_a.subplots(1, 4)

    fig_sigma1_b = plot_settings.plt.figure(
        figsize=(plot_settings.A4_WIDTH, plot_settings.A4_WIDTH * 0.3),
        layout="constrained",
    )
    axs_sigma1_b = fig_sigma1_b.subplots(1, 4)

    xaxis_ticks = [1, 2, 3, 4]
    xaxis_labels = ["$1/10$", "$1/20$", "$1/40$", "$1/80$"]

    for ax in axs_sigma0_a:
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")
        ax.set_xticks([0.0, 1.0], ["0.0", "1.0"])
        ax.xaxis.set_label_coords(0.5, -0.1)
        ax.set_yticks([0.0, 1.0], ["0.0", "1.0"])
        ax.yaxis.set_label_coords(-0.1, 0.5)

    ax = axs_sigma0_a[0]
    posi = plot_settings.plot_elem_dat(coeff_cell40_sigma0, ax)
    plot_settings.append_colorbar(fig_sigma0_a, ax, posi)

    ax = axs_sigma0_a[1]
    posi = plot_settings.plot_elem_dat(coeff_cell20_sigma0, ax)
    plot_settings.append_colorbar(fig_sigma0_a, ax, posi)

    ax = axs_sigma0_a[2]
    posi = plot_settings.plot_node_dat(u_cell40_sigma0, ax)
    plot_settings.append_colorbar(fig_sigma0_a, ax, posi)

    ax = axs_sigma0_a[3]
    posi = plot_settings.plot_node_dat(u_cell20_sigma0, ax)
    plot_settings.append_colorbar(fig_sigma0_a, ax, posi)

    for ax in axs_sigma0_b:
        ax.set_yscale("log")
        ax.set_xlabel("$H$")
        ax.set_xticks(xaxis_ticks, xaxis_labels)

    ax = axs_sigma0_b[0]
    ax.plot(xaxis_ticks, error_cell40_sigma0[:4, 0], "*-", label="$Q_1$")
    ax.plot(xaxis_ticks, error_cell40_sigma0[:4, 1], "*-", label="$m=1$")
    ax.plot(xaxis_ticks, error_cell40_sigma0[:4, 2], "*-", label="$m=2$")
    ax.plot(xaxis_ticks, error_cell40_sigma0[:4, 3], "*-", label="$m=3$")
    ax.plot(xaxis_ticks, error_cell40_sigma0[:4, 4], "*-", label="$m=4$")
    ax.set_ylabel("in energy norm")
    ax.set_title("$(\sigma_*^+, \sigma_*^-, M)$=(1.0, 1.0e-3, 10)")

    ax = axs_sigma0_b[1]
    ax.plot(xaxis_ticks, error_cell40_sigma0[4:, 0], "*-", label="$Q_1$")
    ax.plot(xaxis_ticks, error_cell40_sigma0[4:, 1], "*-", label="$m=1$")
    ax.plot(xaxis_ticks, error_cell40_sigma0[4:, 2], "*-", label="$m=2$")
    ax.plot(xaxis_ticks, error_cell40_sigma0[4:, 3], "*-", label="$m=3$")
    ax.plot(xaxis_ticks, error_cell40_sigma0[4:, 4], "*-", label="$m=4$")
    ax.set_ylabel("in $L^2$ norm")

    ax = axs_sigma0_b[2]
    ax.plot(xaxis_ticks, error_cell20_sigma0[:4, 0], "*-", label="$Q_1$")
    ax.plot(xaxis_ticks, error_cell20_sigma0[:4, 1], "*-", label="$m=1$")
    ax.plot(xaxis_ticks, error_cell20_sigma0[:4, 2], "*-", label="$m=2$")
    ax.plot(xaxis_ticks, error_cell20_sigma0[:4, 3], "*-", label="$m=3$")
    ax.plot(xaxis_ticks, error_cell20_sigma0[:4, 4], "*-", label="$m=4$")
    ax.set_ylabel("in energy norm")
    ax.set_title("$(\sigma_*^+, \sigma_*^-, M)$=(1.0, 1.0e-3, 20)")

    ax = axs_sigma0_b[3]
    ax.plot(xaxis_ticks, error_cell20_sigma0[4:, 0], "*-", label="$Q_1$")
    ax.plot(xaxis_ticks, error_cell20_sigma0[4:, 1], "*-", label="$m=1$")
    ax.plot(xaxis_ticks, error_cell20_sigma0[4:, 2], "*-", label="$m=2$")
    ax.plot(xaxis_ticks, error_cell20_sigma0[4:, 3], "*-", label="$m=3$")
    ax.plot(xaxis_ticks, error_cell20_sigma0[4:, 4], "*-", label="$m=4$")
    ax.set_ylabel("in $L^2$ norm")

    handles, labels = axs_sigma0_b[0].get_legend_handles_labels()
    fig_sigma0_b.legend(
        handles=handles,
        labels=labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.1),
        ncol=5,
        fancybox=True,
        shadow=True,
    )

    for ax in axs_sigma1_a:
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")
        ax.set_xticks([0.0, 1.0], ["0.0", "1.0"])
        ax.xaxis.set_label_coords(0.5, -0.1)
        ax.set_yticks([0.0, 1.0], ["0.0", "1.0"])
        ax.yaxis.set_label_coords(-0.1, 0.5)

    ax = axs_sigma1_a[0]
    posi = plot_settings.plot_elem_dat(coeff_cell40_sigma1, ax)
    plot_settings.append_colorbar(fig_sigma1_a, ax, posi)

    ax = axs_sigma1_a[1]
    posi = plot_settings.plot_elem_dat(coeff_cell20_sigma1, ax)
    plot_settings.append_colorbar(fig_sigma1_a, ax, posi)

    ax = axs_sigma1_a[2]
    posi = plot_settings.plot_node_dat(u_cell40_sigma1, ax)
    plot_settings.append_colorbar(fig_sigma1_a, ax, posi)

    ax = axs_sigma1_a[3]
    posi = plot_settings.plot_node_dat(u_cell20_sigma1, ax)
    plot_settings.append_colorbar(fig_sigma1_a, ax, posi)

    for ax in axs_sigma1_b:
        ax.set_yscale("log")
        ax.set_xlabel("$H$")
        ax.set_xticks(xaxis_ticks, xaxis_labels)

    ax = axs_sigma1_b[0]
    ax.plot(xaxis_ticks, error_cell40_sigma1[:4, 0], "*-", label="$Q_1$")
    ax.plot(xaxis_ticks, error_cell40_sigma1[:4, 1], "*-", label="$m=1$")
    ax.plot(xaxis_ticks, error_cell40_sigma1[:4, 2], "*-", label="$m=2$")
    ax.plot(xaxis_ticks, error_cell40_sigma1[:4, 3], "*-", label="$m=3$")
    ax.plot(xaxis_ticks, error_cell40_sigma1[:4, 4], "*-", label="$m=4$")
    ax.set_ylabel("in energy norm")
    ax.set_title("$(\sigma_*^+, \sigma_*^-, M)$=(1.0, 1.0e+3, 10)")

    ax = axs_sigma1_b[1]
    ax.plot(xaxis_ticks, error_cell40_sigma1[4:, 0], "*-", label="$Q_1$")
    ax.plot(xaxis_ticks, error_cell40_sigma1[4:, 1], "*-", label="$m=1$")
    ax.plot(xaxis_ticks, error_cell40_sigma1[4:, 2], "*-", label="$m=2$")
    ax.plot(xaxis_ticks, error_cell40_sigma1[4:, 3], "*-", label="$m=3$")
    ax.plot(xaxis_ticks, error_cell40_sigma1[4:, 4], "*-", label="$m=4$")
    ax.set_ylabel("in $L^2$ norm")

    ax = axs_sigma1_b[2]
    ax.plot(xaxis_ticks, error_cell20_sigma1[:4, 0], "*-", label="$Q_1$")
    ax.plot(xaxis_ticks, error_cell20_sigma1[:4, 1], "*-", label="$m=1$")
    ax.plot(xaxis_ticks, error_cell20_sigma1[:4, 2], "*-", label="$m=2$")
    ax.plot(xaxis_ticks, error_cell20_sigma1[:4, 3], "*-", label="$m=3$")
    ax.plot(xaxis_ticks, error_cell20_sigma1[:4, 4], "*-", label="$m=4$")
    ax.set_ylabel("in energy norm")
    ax.set_title("$(\sigma_*^+, \sigma_*^-, M)$=(1.0, 1.0e+3, 20)")

    ax = axs_sigma1_b[3]
    ax.plot(xaxis_ticks, error_cell20_sigma1[4:, 0], "*-", label="$Q_1$")
    ax.plot(xaxis_ticks, error_cell20_sigma1[4:, 1], "*-", label="$m=1$")
    ax.plot(xaxis_ticks, error_cell20_sigma1[4:, 2], "*-", label="$m=2$")
    ax.plot(xaxis_ticks, error_cell20_sigma1[4:, 3], "*-", label="$m=3$")
    ax.plot(xaxis_ticks, error_cell20_sigma1[4:, 4], "*-", label="$m=4$")
    ax.set_ylabel("in $L^2$ norm")

    handles, labels = axs_sigma1_b[0].get_legend_handles_labels()
    fig_sigma1_b.legend(
        handles=handles,
        labels=labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.1),
        ncol=5,
        fancybox=True,
        shadow=True,
    )

    fig_sigma0_a.savefig(
        "{0:s}/{1:s}.png".format(
            plot_settings.FIGS_ROOT_PATH,
            "cross-inclusion-sigma0-a",
        ),
        bbox_inches="tight",
        dpi=plot_settings.DPI,
    )
    fig_sigma0_b.savefig(
        "{0:s}/{1:s}.png".format(
            plot_settings.FIGS_ROOT_PATH,
            "cross-inclusion-sigma0-b",
        ),
        bbox_inches="tight",
        dpi=plot_settings.DPI,
    )
    fig_sigma1_a.savefig(
        "{0:s}/{1:s}.png".format(
            plot_settings.FIGS_ROOT_PATH,
            "cross-inclusion-sigma1-a",
        ),
        bbox_inches="tight",
        dpi=plot_settings.DPI,
    )
    fig_sigma1_b.savefig(
        "{0:s}/{1:s}.png".format(
            plot_settings.FIGS_ROOT_PATH,
            "cross-inclusion-sigma1-b",
        ),
        bbox_inches="tight",
        dpi=plot_settings.DPI,
    )

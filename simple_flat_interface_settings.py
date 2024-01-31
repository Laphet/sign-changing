import numpy as np
from itertools import product

DAT_ROOT_PATH = "resources"


def get_test_settings(fine_grids: int, sigma_pm, l=0.25):
    coeff = np.zeros((fine_grids, fine_grids))
    source = np.zeros((fine_grids, fine_grids))
    tot_dof = (fine_grids - 1) ** 2
    u = np.zeros((tot_dof,))
    h = 1.0 / fine_grids
    for elem_ind_x, elem_ind_y in product(range(fine_grids), range(fine_grids)):
        x, y = (elem_ind_x + 0.5) * h, (elem_ind_y + 0.5) * h
        if y >= l:
            coeff[elem_ind_x, elem_ind_y] = sigma_pm[0]
        else:
            coeff[elem_ind_x, elem_ind_y] = -sigma_pm[1]
        source[elem_ind_x, elem_ind_y] = (
            sigma_pm[0]
            * sigma_pm[1]
            * (
                2.0 * y * (y - 1.0) * (y - l)
                + x * (x - 1.0) * (6.0 * y - 2.0 * (l + 1.0))
            )
        )
    for dof_ind_y, dof_ind_x in product(range(fine_grids - 1), range(fine_grids - 1)):
        x, y = (dof_ind_x + 1) * h, (dof_ind_y + 1) * h
        temp = x * (x - 1) * y * (y - 1) * (y - l)
        if y >= l:
            u[dof_ind_y * (fine_grids - 1) + dof_ind_x] = -temp * sigma_pm[1]
        else:
            u[dof_ind_y * (fine_grids - 1) + dof_ind_x] = temp * sigma_pm[0]

    return coeff, source, u


if __name__ == "__main__":
    from cem_gmsfem import CemGmsfem
    from fem import get_fem_mat, get_mass_mat
    import argparse
    import logging
    from logging import config

    # fine_grid = 256
    # coarse_grid_list = [8, 16, 32, 64]
    fine_grid = 400
    # coarse_grid_list = [10, 20, 40, 80]
    # osly_list = [0, 1, 2, 3, 4]
    # sigma_pm_list = [
    #     [1.1, 1.0],
    #     [1.0, 1.1],
    #     [1.01, 1.0],
    #     [1.0, 1.01],
    #     [10.0, 1.0],
    #     [1.0, 0.1],
    # ]
    # l_list = [0.5, 0.5 - 1.0 / 100]
    coarse_grid_list = [80]
    osly_list = [0, 3]
    sigma_pm_list = [[1.0, 1.01]]
    l_list = [0.5 - 1.0 / 100]

    parse = argparse.ArgumentParser()
    parse.add_argument("--en", default=3, type=int)
    parse.add_argument("--sigma", default=0, type=int)
    parse.add_argument("--posi", default=0, type=int)
    args = parse.parse_args()
    config.fileConfig(
        "log.conf",
        defaults={
            "logfilename": "logs/flat-interface-en{0:d}-sigma{1:d}-l{2:d}.log".format(
                args.en, args.sigma, args.posi
            )
        },
    )

    eigen_num = args.en
    sigma_pm = sigma_pm_list[args.sigma]
    l = l_list[args.posi]

    coeff, source, u = get_test_settings(fine_grid, sigma_pm, l)

    logging.info("=" * 80)
    logging.info("Start")
    logging.info(
        "In the medium, sigma+={0:.4e}, sigma-={1:.4e}, l={2:.4e}".format(*sigma_pm, l)
    )

    prepare_dat = False
    if prepare_dat:
        save_solution = True
        rela_errors_h1 = np.zeros((len(coarse_grid_list), len(osly_list)))
        rela_errors_l2 = np.zeros(rela_errors_h1.shape)

        for coarse_grid_ind, osly_ind in product(
            range(len(coarse_grid_list)), range(len(osly_list))
        ):
            coarse_grid = coarse_grid_list[coarse_grid_ind]
            osly = osly_list[osly_ind]
            cem_gmsfem = CemGmsfem(coarse_grid, eigen_num, osly, coeff)
            cem_gmsfem.setup()
            u_cem = None
            if osly == 0:
                u_cem = cem_gmsfem.solve_by_coarse_bilinear(source)
            else:
                u_cem = cem_gmsfem.solve(source)

            fem_mat_abs = get_fem_mat(cem_gmsfem.coeff_abs)
            mass_mat = get_mass_mat(np.ones(coeff.shape))

            delta_u = u - u_cem

            u_h1 = np.sqrt(np.dot(fem_mat_abs.dot(u), u))
            delta_u_h1 = np.sqrt(np.dot(fem_mat_abs.dot(delta_u), delta_u))
            rela_error_h1 = delta_u_h1 / u_h1
            rela_errors_h1[coarse_grid_ind, osly_ind] = rela_error_h1

            u_l2 = np.sqrt(np.dot(mass_mat.dot(u), u))
            delta_u_l2 = np.sqrt(np.dot(mass_mat.dot(delta_u), delta_u))
            rela_error_l2 = delta_u_l2 / u_l2
            rela_errors_l2[coarse_grid_ind, osly_ind] = rela_error_l2

            logging.info(
                "relative energy error={0:6e}, plain-L2 error={1:6e}.".format(
                    rela_error_h1, rela_error_l2
                )
            )

            if save_solution:
                np.save(
                    "{0:s}/{1:s}".format(
                        DAT_ROOT_PATH, "delta-u-osly{0:d}-1d01+1d0.npy".format(osly)
                    ),
                    cem_gmsfem.get_plot_format(delta_u),
                )

        print(rela_errors_h1)
        print(rela_errors_l2)

    plot_fig = True
    if plot_fig:
        en3_sigma2_posi0_dat = np.array(
            [
                [0.20555968, 1.30120133, 0.46207559, 0.71495813, 0.06527177],
                [0.10270882, 0.76334009, 0.21074758, 0.28448917, 0.04719334],
                [0.05116397, 0.79518043, 0.11724455, 0.1395783, 0.03308739],
                [0.02519176, 0.82960904, 0.09533627, 0.06906429, 0.01994061],
                [
                    4.22793077e-02,
                    1.09831223e-01,
                    5.49477210e-01,
                    8.20420912e-01,
                    6.83492066e-03,
                ],
                [
                    1.05619731e-02,
                    1.08804650e-01,
                    2.31819853e-01,
                    3.41021179e-01,
                    4.18941088e-03,
                ],
                [
                    2.62306961e-03,
                    3.28390911e-01,
                    1.06210624e-01,
                    1.82460800e-01,
                    4.69588010e-03,
                ],
                [
                    6.37770852e-04,
                    6.74455849e-01,
                    5.20244819e-02,
                    9.37721639e-02,
                    4.66852571e-03,
                ],
            ]
        )
        en3_sigma2_posi1_dat = np.array(
            [
                [
                    8.31604946e-01,
                    2.34365906e-01,
                    2.74526779e-01,
                    2.17502211e-02,
                    2.70265286e-03,
                ],
                [
                    8.00481821e-01,
                    7.58853074e-01,
                    1.22366650e-01,
                    1.15081867e-01,
                    4.10209049e-03,
                ],
                [
                    7.83234054e-01,
                    5.83325866e-01,
                    4.80635827e-02,
                    6.46394378e-02,
                    5.18916780e-03,
                ],
                [
                    7.58327654e-01,
                    8.04442516e-01,
                    7.59745295e-02,
                    1.11677340e-01,
                    5.63420523e-04,
                ],
                [
                    1.10791314e00,
                    2.17222602e-01,
                    3.73999354e-01,
                    1.88814093e-02,
                    2.15786093e-04,
                ],
                [
                    1.09646179e00,
                    1.09491866e-01,
                    1.63029030e-01,
                    1.49288769e-01,
                    2.79459484e-03,
                ],
                [
                    1.08080497e00,
                    3.71081282e-01,
                    5.28207548e-02,
                    8.77535244e-02,
                    5.93263982e-03,
                ],
                [
                    1.04836310e00,
                    6.77103268e-01,
                    6.71861303e-02,
                    1.53929414e-01,
                    2.35719592e-04,
                ],
            ]
        )
        en3_sigma3_posi0_dat = np.array(
            [
                [0.20555968, 1.30120133, 0.46207559, 0.71495814, 0.06527177],
                [0.10270882, 0.76334009, 0.21074758, 0.28448917, 0.04719334],
                [0.05116397, 0.79518043, 0.11724455, 0.1395783, 0.03308739],
                [0.02519176, 0.82957019, 0.09533627, 0.06906428, 0.01994061],
                [
                    4.22793077e-02,
                    1.09831223e-01,
                    5.49477210e-01,
                    8.20420910e-01,
                    6.83492028e-03,
                ],
                [
                    1.05619731e-02,
                    1.08804650e-01,
                    2.31819853e-01,
                    3.41021179e-01,
                    4.18941104e-03,
                ],
                [
                    2.62306961e-03,
                    3.28390911e-01,
                    1.06210624e-01,
                    1.82460800e-01,
                    4.69588014e-03,
                ],
                [
                    6.37770852e-04,
                    6.74454357e-01,
                    5.20244819e-02,
                    9.37721696e-02,
                    4.66854141e-03,
                ],
            ]
        )
        en3_sigma3_posi1_dat = np.array(
            [
                [
                    9.69344311e-01,
                    3.24820438e-01,
                    4.93050960e-02,
                    2.51541028e-02,
                    1.51468460e-03,
                ],
                [
                    9.06781422e-01,
                    4.89904426e-01,
                    3.87363228e-02,
                    4.80304366e-03,
                    5.45083430e-04,
                ],
                [
                    8.26528390e-01,
                    2.06337297e01,
                    1.87278599e-01,
                    3.77396182e-03,
                    3.28189592e-04,
                ],
                [
                    5.32491202e-01,
                    8.18371802e-01,
                    6.34861041e-02,
                    1.22224091e-02,
                    3.24127787e-03,
                ],
                [
                    1.30436808e00,
                    2.99145603e-01,
                    3.09348583e-03,
                    6.73567396e-04,
                    9.60597683e-05,
                ],
                [
                    1.24371860e00,
                    8.55535424e-02,
                    3.95137540e-03,
                    2.14130497e-04,
                    2.13810006e-05,
                ],
                [
                    1.13901739e00,
                    4.27398082e00,
                    1.77414416e-01,
                    1.04846237e-04,
                    1.13070597e-05,
                ],
                [
                    7.32301960e-01,
                    6.82626850e-01,
                    2.81452928e-02,
                    6.20809650e-03,
                    3.92624240e-04,
                ],
            ]
        )

        dat1 = [en3_sigma2_posi0_dat, en3_sigma3_posi0_dat]
        dat2 = [en3_sigma2_posi1_dat, en3_sigma3_posi1_dat]

        delta_u_osly0 = np.load(
            "{0:s}/{1:s}".format(
                DAT_ROOT_PATH, "delta-u-osly{0:d}-1d01+1d0.npy".format(0)
            )
        )
        delta_u_osly3 = np.load(
            "{0:s}/{1:s}".format(
                DAT_ROOT_PATH, "delta-u-osly{0:d}-1d01+1d0.npy".format(3)
            )
        )

        import plot_settings

        fig1 = plot_settings.plt.figure(
            figsize=(plot_settings.A4_WIDTH, plot_settings.A4_WIDTH * 0.3),
            layout="constrained",
        )
        axs1 = fig1.subplots(1, 4)

        fig2 = plot_settings.plt.figure(
            figsize=(plot_settings.A4_WIDTH, plot_settings.A4_WIDTH * 0.3),
            layout="constrained",
        )
        axs2 = fig2.subplots(1, 4)

        xaxis_ticks = [1, 2, 3, 4]
        xaxis_labels = ["$1/10$", "$1/20$", "$1/40$", "$1/80$"]

        for i in range(4):
            ax = axs1[i]
            ax.set_yscale("log")
            ax.set_xlabel("$H$")
            ax.set_xticks(xaxis_ticks, xaxis_labels)

        for i in [0, 1]:
            ax = axs1[2 * i]
            ax.plot(xaxis_ticks, dat1[i][:4, 0], "*-", label="$Q_1$")
            ax.plot(xaxis_ticks, dat1[i][:4, 1], "*-", label="$m=1$")
            ax.plot(xaxis_ticks, dat1[i][:4, 2], "*-", label="$m=2$")
            ax.plot(xaxis_ticks, dat1[i][:4, 3], "*-", label="$m=3$")
            ax.plot(xaxis_ticks, dat1[i][:4, 4], "*-", label="$m=4$")
            ax.set_ylabel("in energy norm")
            ax = axs1[2 * i + 1]
            ax.plot(xaxis_ticks, dat1[i][4:, 0], "*-", label="$Q_1$")
            ax.plot(xaxis_ticks, dat1[i][4:, 1], "*-", label="$m=1$")
            ax.plot(xaxis_ticks, dat1[i][4:, 2], "*-", label="$m=2$")
            ax.plot(xaxis_ticks, dat1[i][4:, 3], "*-", label="$m=3$")
            ax.plot(xaxis_ticks, dat1[i][4:, 4], "*-", label="$m=4$")
            ax.set_ylabel("in $L^2$ norm")

        axs1[0].set_title("$(\sigma_*^+, \sigma_*^-, l)$=(1.01, 1.0, 0.50)")
        # axs1[1].set_title("$(\sigma_*^+, \sigma_*^-, l)=(1.01, 1.0, 0.50)$")
        axs1[2].set_title("$(\sigma_*^+, \sigma_*^-, l)$=(1.0, 1.01, 0.50)")
        # axs1[3].set_title("$(\sigma_*^+, \sigma_*^-, l)=(1.0, 1.01, 0.50)$")

        # Insert the common legends.
        handles, labels = axs1[0].get_legend_handles_labels()
        fig1.legend(
            handles=handles,
            labels=labels,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.1),
            ncol=5,
            fancybox=True,
            shadow=True,
        )

        ax = axs2[0]
        ax.plot(xaxis_ticks, dat2[1][:4, 0], "*-", label="$Q_1$")
        ax.plot(xaxis_ticks, dat2[1][:4, 1], "*-", label="$m=1$")
        ax.plot(xaxis_ticks, dat2[1][:4, 2], "*-", label="$m=2$")
        ax.plot(xaxis_ticks, dat2[1][:4, 3], "*-", label="$m=3$")
        ax.plot(xaxis_ticks, dat2[1][:4, 4], "*-", label="$m=4$")
        ax.set_ylabel("in energy norm")
        ax.set_yscale("log")
        ax.set_xlabel("$H$")
        ax.set_xticks(xaxis_ticks, xaxis_labels)
        ax.set_title("$(\sigma_*^+, \sigma_*^-, l)$=(1.01, 1.0, 0.49)")

        ax = axs2[1]
        ax.plot(xaxis_ticks, dat2[1][4:, 0], "*-", label="$Q_1$")
        ax.plot(xaxis_ticks, dat2[1][4:, 1], "*-", label="$m=1$")
        ax.plot(xaxis_ticks, dat2[1][4:, 2], "*-", label="$m=2$")
        ax.plot(xaxis_ticks, dat2[1][4:, 3], "*-", label="$m=3$")
        ax.plot(xaxis_ticks, dat2[1][4:, 4], "*-", label="$m=4$")
        #
        ax.set_ylabel("in $L^2$ norm")
        ax.set_yscale("log")
        ax.set_xlabel("$H$")
        ax.set_xticks(xaxis_ticks, xaxis_labels)

        handles, labels = axs2[0].get_legend_handles_labels()
        fig2.legend(
            handles=handles,
            labels=labels,
            loc="lower center",
            bbox_to_anchor=(0.3, -0.10),
            ncol=5,
            fancybox=True,
            shadow=True,
            alignment="center",
        )

        ax = axs2[2]
        posi = plot_settings.plot_node_dat(delta_u_osly0, ax)
        cbar = plot_settings.append_colorbar(fig2, ax, posi)
        cbar.formatter.set_powerlimits((0, 0))
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")
        ax.set_xticks([0.0, 0.5, 1.0], ["0.0", "0.5", "1.0"])
        ax.set_yticks([0.0, 0.5, 1.0], ["0.0", "0.5", "1.0"])
        # ax.set_title("Error by $Q_1$ and $H=1/80$", y=-0.5)

        ax = axs2[3]
        posi = plot_settings.plot_node_dat(delta_u_osly3, ax)
        cbar = plot_settings.append_colorbar(fig2, ax, posi)
        cbar.formatter.set_powerlimits((0, 0))
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")
        ax.set_xticks([0.0, 0.5, 1.0], ["0.0", "0.5", "1.0"])
        ax.set_yticks([0.0, 0.5, 1.0], ["0.0", "0.5", "1.0"])
        # ax.set_title("Error by $m=3$ and $H=1/80$", y=-0.5)
        # fig2.suptitle("$(\sigma_*^+, \sigma_*^-, l)$=(1.01, 1.0, 0.49)")

        fig1.savefig(
            "{0:s}/{1:s}.pdf".format(
                plot_settings.FIGS_ROOT_PATH,
                "flat-interface-error-{0:s}".format("l0"),
            ),
            bbox_inches="tight",
        )
        fig2.savefig(
            "{0:s}/{1:s}.png".format(
                plot_settings.FIGS_ROOT_PATH,
                "flat-interface-error-{0:s}".format("l1"),
            ),
            bbox_inches="tight",
            dpi=plot_settings.DPI,
        )

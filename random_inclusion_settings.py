DAT_ROOT_PATH = "resources"

import numpy as np
from itertools import product

VARIANCE = 0.01


def get_test_settings(bin_dat, sigma_im, fine_grid):
    if fine_grid == None:
        coeff = np.zeros(bin_dat.shape)
        source = np.zeros(bin_dat.shape)
    else:
        coeff = np.zeros((fine_grid, fine_grid))
        source = np.zeros((fine_grid, fine_grid))
    for i, j in product(range(coeff.shape[0]), range(coeff.shape[1])):
        if bin_dat[i, j] <= 0.0:
            coeff[i, j] = sigma_im[0]
        else:
            coeff[i, j] = sigma_im[1]
        x, y = (i + 0.5) / source.shape[0], (j + 0.5) / source.shape[1]
        r = np.sqrt((x - 0.25) ** 2 + (y - 0.25) ** 2)
        source[i, j] += np.exp(-(r**2) / 2.0 / VARIANCE) / VARIANCE / (2.0 * np.pi)
        r = np.sqrt((x - 0.25) ** 2 + (y - 0.75) ** 2)
        source[i, j] += np.exp(-(r**2) / 2.0 / VARIANCE) / VARIANCE / (2.0 * np.pi)
        r = np.sqrt((x - 0.75) ** 2 + (y - 0.25) ** 2)
        source[i, j] += np.exp(-(r**2) / 2.0 / VARIANCE) / VARIANCE / (2.0 * np.pi)
        r = np.sqrt((x - 0.75) ** 2 + (y - 0.75) ** 2)
        source[i, j] += np.exp(-(r**2) / 2.0 / VARIANCE) / VARIANCE / (2.0 * np.pi)

    return coeff, source


if __name__ == "__main__":
    from cem_gmsfem import CemGmsfem
    from fem import get_fem_mat, get_mass_mat, get_fem_rhs
    import argparse
    import logging
    from logging import config
    from scipy.sparse.linalg import spsolve

    fine_grid = 400
    coarse_grid_list = [10, 20, 40, 80]
    # fine_grid = 200
    # coarse_grid_list = [10, 20, 40]
    eigen_num_list = [1, 2, 3, 4]
    bin_dat = np.load("{0:s}/{1:s}".format(DAT_ROOT_PATH, "random-inclusion.npy"))
    sigma_im_list = [[-0.001, 1.0], [1000.0, 1.0]]
    sigma_im_token_list = ["-0d001+1d0", "-1000d0+1d0"]

    parse = argparse.ArgumentParser()
    parse.add_argument("--osly", default=3, type=int)
    parse.add_argument("--sigma", default=0, type=int)
    args = parse.parse_args()
    config.fileConfig(
        "log.conf",
        defaults={
            "logfilename": "logs/random-inclusion-osly{0:d}-sigma{1:d}.log".format(
                args.osly, args.sigma
            )
        },
    )
    osly = args.osly
    sigma_im = sigma_im_list[args.sigma]
    token = sigma_im_token_list[args.sigma]

    coeff, source = get_test_settings(bin_dat, sigma_im, fine_grid)

    logging.info("=" * 80)
    logging.info("Start")
    logging.info(
        "In the medium, sigma-inc={0:.4e}, sigma-mat={1:.4e}".format(*sigma_im)
    )

    prepare_dat = False
    if prepare_dat:
        rela_errors_h1 = np.zeros((len(coarse_grid_list), len(eigen_num_list)))
        rela_errors_l2 = np.zeros(rela_errors_h1.shape)

        mat_fem = get_fem_mat(coeff)
        rhs = get_fem_rhs(source)
        u_fem = spsolve(mat_fem, rhs)
        np.save(
            "{0:s}/{1:s}".format(
                DAT_ROOT_PATH, "random-inclusion{:s}.npy".format(token)
            ),
            u_fem,
        )

        fem_mat_abs = get_fem_mat(np.abs(coeff))
        mass_mat = get_mass_mat(np.ones(coeff.shape))
        u_h1 = np.sqrt(np.dot(fem_mat_abs.dot(u_fem), u_fem))
        u_l2 = np.sqrt(np.dot(mass_mat.dot(u_fem), u_fem))

        for coarse_grid_ind, eigen_num_ind in product(
            range(len(coarse_grid_list)), range(len(eigen_num_list))
        ):
            coarse_grid = coarse_grid_list[coarse_grid_ind]
            eigen_num = eigen_num_list[eigen_num_ind]
            solver = CemGmsfem(coarse_grid, eigen_num, osly, coeff)
            solver.setup()
            u_cem = solver.solve(source)

            delta_u = u_cem - u_fem

            delta_u_h1 = np.sqrt(np.dot(fem_mat_abs.dot(delta_u), delta_u))
            rela_error_h1 = delta_u_h1 / u_h1
            rela_errors_h1[coarse_grid_ind, eigen_num_ind] = rela_error_h1
            delta_u_l2 = np.sqrt(np.dot(mass_mat.dot(delta_u), delta_u))
            rela_error_l2 = delta_u_l2 / u_l2
            rela_errors_l2[coarse_grid_ind, eigen_num_ind] = rela_error_l2

            logging.info(
                "relative energy error={0:6e}, plain-L2 error={1:6e}.".format(
                    rela_error_h1, rela_error_l2
                )
            )
        print(rela_errors_h1)
        print(rela_errors_l2)

    plot_fig = True
    if plot_fig:
        u_fem_sigma0_read = np.load(
            "{0:s}/{1:s}".format(
                DAT_ROOT_PATH, "random-inclusion{:s}.npy".format(sigma_im_token_list[0])
            ),
        )
        u_fem_sigma0 = np.zeros((fine_grid + 1, fine_grid + 1))
        u_fem_sigma0[1:-1, 1:-1] = u_fem_sigma0_read.reshape((fine_grid - 1, -1))

        u_fem_sigma1_read = np.load(
            "{0:s}/{1:s}".format(
                DAT_ROOT_PATH, "random-inclusion{:s}.npy".format(sigma_im_token_list[1])
            ),
        )
        u_fem_sigma1 = np.zeros((fine_grid + 1, fine_grid + 1))
        u_fem_sigma1[1:-1, 1:-1] = u_fem_sigma1_read.reshape((fine_grid - 1, -1))

        error_sigma0 = np.array(
            [
                [0.62320341, 0.61856975, 0.61844031, 0.61843374],
                [0.61962888, 0.19063506, 0.1650624, 0.14943125],
                [0.22343074, 0.14197714, 0.06720541, 0.04819243],
                [0.21255413, 0.04789757, 0.0177218, 0.01737096],
                [0.9063096, 0.90850539, 0.90860292, 0.90865772],
                [0.90846216, 0.15281023, 0.11540363, 0.07666486],
                [0.1706398, 0.09658452, 0.0268677, 0.01605306],
                [0.06760207, 0.0108938, 0.00331354, 0.00328897],
            ]
        )
        error_sigma1 = np.array(
            [
                [0.12680334, 0.07017825, 0.02834831, 0.02040072],
                [0.25405868, 0.10421821, 0.01329208, 0.01012308],
                [0.54454387, 0.13211681, 0.01255638, 0.01110845],
                [0.84563089, 0.28414627, 0.03000388, 0.02938047],
                [1.78020215e-02, 8.43342246e-03, 1.94902667e-03, 1.06095925e-03],
                [6.78457300e-02, 1.42764959e-02, 5.94996417e-04, 3.50902302e-04],
                [3.21875430e-01, 1.93104686e-02, 2.01668279e-04, 1.43653295e-04],
                [7.55743539e-01, 8.71930861e-02, 9.70916273e-04, 9.30905964e-04],
            ]
        )
        eigen_sigma0 = np.array(
            [
                [
                    [2.0872e-14, 2.4425e-14],
                    [6.9394e-04, 4.1144e-01],
                    [3.0109e-01, 4.1144e-01],
                    [4.1214e-01, 8.2289e-01],
                ],
                [
                    [1.9984e-15, 5.9952e-15],
                    [7.8060e-04, 4.1208e-01],
                    [3.0660e-01, 8.4044e-01],
                    [4.1286e-01, 1.2781e00],
                ],
                [
                    [-1.3323e-15, 8.8818e-16],
                    [1.0076e-03, 4.1463e-01],
                    [2.6166e-01, 7.5792e-01],
                    [5.0911e-01, 1.2914e00],
                ],
                [
                    [-6.6613e-16, 1.1102e-15],
                    [1.0515e-01, 6.7548e-01],
                    [3.2552e-01, 8.4987e-01],
                    [5.2282e-01, 1.3504e00],
                ],
            ]
        )
        eigen_sigma0_mean = (eigen_sigma0[:, :, 0] + eigen_sigma0[:, :, 1]) * 0.5
        eigen_sigma1 = np.array(
            [
                [
                    [1.2657e-14, 2.3981e-14],
                    [2.6850e-01, 4.3350e-01],
                    [3.7073e-01, 1.1414e00],
                    [6.6938e-01, 1.5529e00],
                ],
                [
                    [1.9984e-15, 5.5511e-15],
                    [6.8562e-02, 4.1208e-01],
                    [3.2543e-01, 8.6961e-01],
                    [5.7334e-01, 1.2637e00],
                ],
                [
                    [-4.4409e-16, 2.4425e-15],
                    [1.3648e-01, 6.5015e-01],
                    [3.2921e-01, 7.8659e-01],
                    [5.7628e-01, 1.2995e00],
                ],
                [
                    [-6.6613e-16, 1.3323e-15],
                    [1.0515e-01, 6.7548e-01],
                    [3.2552e-01, 8.4987e-01],
                    [5.2282e-01, 1.3504e00],
                ],
            ]
        )
        eigen_sigma1_mean = (eigen_sigma1[:, :, 0] + eigen_sigma1[:, :, 1]) * 0.5

        import plot_settings

        fig1 = plot_settings.plt.figure(
            figsize=(plot_settings.A4_WIDTH * 0.75, plot_settings.A4_WIDTH * 0.3),
            layout="constrained",
        )
        axs1 = fig1.subplots(1, 3)

        fig2 = plot_settings.plt.figure(
            figsize=(plot_settings.A4_WIDTH * 0.75, plot_settings.A4_WIDTH * 0.3),
            layout="constrained",
        )
        axs2 = fig2.subplots(1, 3)

        fig3 = plot_settings.plt.figure(
            figsize=(plot_settings.A4_WIDTH * 0.75, plot_settings.A4_WIDTH * 0.3),
            layout="constrained",
        )
        axs3 = fig3.subplots(1, 3)

        ax = axs1[0]
        posi = plot_settings.plot_elem_dat(bin_dat, ax)
        plot_settings.append_colorbar(fig1, ax, posi)
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")
        ax.set_xticks([0.0, 0.5, 1.0], ["0.0", "0.5", "1.0"])
        ax.set_yticks([0.0, 0.5, 1.0], ["0.0", "0.5", "1.0"])
        # ax.set_title("Medium configuration")

        ax = axs1[1]
        posi = plot_settings.plot_node_dat(u_fem_sigma0, ax)
        plot_settings.append_colorbar(fig1, ax, posi)
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")
        ax.set_xticks([0.0, 0.5, 1.0], ["0.0", "0.5", "1.0"])
        ax.set_yticks([0.0, 0.5, 1.0], ["0.0", "0.5", "1.0"])
        # ax.set_title("($\sigma_*^+$, $\sigma_*^-$)=(1.0, 1.0e-3)")

        ax = axs1[2]
        posi = plot_settings.plot_node_dat(u_fem_sigma1, ax)
        plot_settings.append_colorbar(fig1, ax, posi)
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")
        ax.set_xticks([0.0, 0.5, 1.0], ["0.0", "0.5", "1.0"])
        ax.set_yticks([0.0, 0.5, 1.0], ["0.0", "0.5", "1.0"])
        # ax.set_title("($\sigma_*^+$, $\sigma_*^-$)=(1.0, 1.0e+3)")

        xaxis_ticks = [1, 2, 3, 4]
        xaxis_labels = ["$1$", "$2$", "$3$", "$4$"]

        ax = axs2[0]
        ax.plot(xaxis_ticks, error_sigma0[0, :], "*-", label="$H=1/10$")
        ax.plot(xaxis_ticks, error_sigma0[1, :], "*-", label="$H=1/20$")
        ax.plot(xaxis_ticks, error_sigma0[2, :], "*-", label="$H=1/40$")
        ax.plot(xaxis_ticks, error_sigma0[3, :], "*-", label="$H=1/80$")
        ax.set_ylabel("in energy norm")
        ax.set_yscale("log")
        ax.set_xlabel("$k$")
        ax.set_xticks(xaxis_ticks, xaxis_labels)

        ax = axs2[1]
        ax.plot(xaxis_ticks, error_sigma0[4, :], "*-", label="$H=1/10$")
        ax.plot(xaxis_ticks, error_sigma0[5, :], "*-", label="$H=1/20$")
        ax.plot(xaxis_ticks, error_sigma0[6, :], "*-", label="$H=1/40$")
        ax.plot(xaxis_ticks, error_sigma0[7, :], "*-", label="$H=1/80$")
        ax.set_ylabel("in $L^2$ norm")
        ax.set_yscale("log")
        ax.set_xlabel("$k$")
        ax.set_xticks(xaxis_ticks, xaxis_labels)

        ax = axs2[2]
        line = ax.plot(xaxis_ticks, eigen_sigma0_mean[0, :], "*-", label="$H=1/10$")
        ax.fill_between(
            xaxis_ticks,
            eigen_sigma0[0, :, 0],
            eigen_sigma0[0, :, 1],
            alpha=0.1,
            color=line[-1].get_color(),
        )
        line = ax.plot(xaxis_ticks, eigen_sigma0_mean[1, :], "*-", label="$H=1/10$")
        ax.fill_between(
            xaxis_ticks,
            eigen_sigma0[1, :, 0],
            eigen_sigma0[1, :, 1],
            alpha=0.1,
            color=line[-1].get_color(),
        )
        line = ax.plot(xaxis_ticks, eigen_sigma0_mean[2, :], "*-", label="$H=1/10$")
        ax.fill_between(
            xaxis_ticks,
            eigen_sigma0[2, :, 0],
            eigen_sigma0[2, :, 1],
            alpha=0.1,
            color=line[-1].get_color(),
        )
        line = ax.plot(xaxis_ticks, eigen_sigma0_mean[3, :], "*-", label="$H=1/10$")
        ax.fill_between(
            xaxis_ticks,
            eigen_sigma0[3, :, 0],
            eigen_sigma0[3, :, 1],
            alpha=0.1,
            color=line[-1].get_color(),
        )
        ax.set_ylabel("eigenvalues")
        # ax.set_xlabel("$k$")
        ax.set_xticks(
            xaxis_ticks, ["$\lambda_0$", "$\lambda_1$", "$\lambda_2$", "$\lambda_3$"]
        )
        fig2.suptitle("$(\sigma_*^+, \sigma_*^-)$=(1.0, 1.0e-3)")
        handles, labels = axs2[0].get_legend_handles_labels()
        fig2.legend(
            handles=handles,
            labels=labels,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.1),
            ncol=4,
            fancybox=True,
            shadow=True,
        )

        ax = axs3[0]
        ax.plot(xaxis_ticks, error_sigma1[0, :], "*-", label="$H=1/10$")
        ax.plot(xaxis_ticks, error_sigma1[1, :], "*-", label="$H=1/20$")
        ax.plot(xaxis_ticks, error_sigma1[2, :], "*-", label="$H=1/40$")
        ax.plot(xaxis_ticks, error_sigma1[3, :], "*-", label="$H=1/80$")
        ax.set_ylabel("in energy norm")
        ax.set_yscale("log")
        ax.set_xlabel("$k$")
        ax.set_xticks(xaxis_ticks, xaxis_labels)

        ax = axs3[1]
        ax.plot(xaxis_ticks, error_sigma1[4, :], "*-", label="$H=1/10$")
        ax.plot(xaxis_ticks, error_sigma1[5, :], "*-", label="$H=1/20$")
        ax.plot(xaxis_ticks, error_sigma1[6, :], "*-", label="$H=1/40$")
        ax.plot(xaxis_ticks, error_sigma1[7, :], "*-", label="$H=1/80$")
        ax.set_ylabel("in $L^2$ norm")
        ax.set_yscale("log")
        ax.set_xlabel("$k$")
        ax.set_xticks(xaxis_ticks, xaxis_labels)

        ax = axs3[2]
        line = ax.plot(xaxis_ticks, eigen_sigma1_mean[0, :], "*-", label="$H=1/10$")
        ax.fill_between(
            xaxis_ticks,
            eigen_sigma1[0, :, 0],
            eigen_sigma1[0, :, 1],
            alpha=0.1,
            color=line[-1].get_color(),
        )
        line = ax.plot(xaxis_ticks, eigen_sigma1_mean[1, :], "*-", label="$H=1/10$")
        ax.fill_between(
            xaxis_ticks,
            eigen_sigma1[1, :, 0],
            eigen_sigma1[1, :, 1],
            alpha=0.1,
            color=line[-1].get_color(),
        )
        line = ax.plot(xaxis_ticks, eigen_sigma1_mean[2, :], "*-", label="$H=1/10$")
        ax.fill_between(
            xaxis_ticks,
            eigen_sigma1[2, :, 0],
            eigen_sigma1[2, :, 1],
            alpha=0.1,
            color=line[-1].get_color(),
        )
        line = ax.plot(xaxis_ticks, eigen_sigma1_mean[3, :], "*-", label="$H=1/10$")
        ax.fill_between(
            xaxis_ticks,
            eigen_sigma1[3, :, 0],
            eigen_sigma1[3, :, 1],
            alpha=0.1,
            color=line[-1].get_color(),
        )
        ax.set_ylabel("eigenvalues")
        # ax.set_xlabel("$k$")
        ax.set_xticks(
            xaxis_ticks, ["$\lambda_0$", "$\lambda_1$", "$\lambda_2$", "$\lambda_3$"]
        )
        fig3.suptitle("$(\sigma_*^+, \sigma_*^-)$=(1.0, 1.0e+3)")
        handles, labels = axs3[0].get_legend_handles_labels()
        fig3.legend(
            handles=handles,
            labels=labels,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.1),
            ncol=4,
            fancybox=True,
            shadow=True,
        )

        fig1.savefig(
            "{0:s}/{1:s}.png".format(
                plot_settings.FIGS_ROOT_PATH,
                "random-inclusion-main",
            ),
            bbox_inches="tight",
            dpi=plot_settings.DPI,
        )

        fig2.savefig(
            "{0:s}/{1:s}.pdf".format(
                plot_settings.FIGS_ROOT_PATH,
                "random-inclusion{:s}".format(sigma_im_token_list[0]),
            ),
            bbox_inches="tight",
            # dpi=450,
        )

        fig3.savefig(
            "{0:s}/{1:s}.pdf".format(
                plot_settings.FIGS_ROOT_PATH,
                "random-inclusion{:s}".format(sigma_im_token_list[1]),
            ),
            bbox_inches="tight",
            # dpi=450,
        )

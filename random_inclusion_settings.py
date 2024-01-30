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
        r = np.sqrt((x - 0.5) ** 2 + (y - 0.5) ** 2)
        source[i, j] = np.exp(-(r**2) / 2.0 / VARIANCE) / VARIANCE / (2.0 * np.pi)
    return coeff, source


if __name__ == "__main__":
    from cem_gmsfem import CemGmsfem
    from fem import get_fem_mat, get_mass_mat, get_fem_rhs
    import argparse
    import logging
    from logging import config
    from scipy.sparse.linalg import spsolve

    # fine_grid = 400
    # coarse_grid_list = [10, 20, 40, 80]
    fine_grid = 200
    coarse_grid_list = [10, 20, 40]
    eigen_num_list = [1, 2, 3, 4]
    bin_dat = np.load("{0:s}/{1:s}".format(DAT_ROOT_PATH, "random-inclusion.npy"))
    sigma_im_list = [[-0.001, 1.0], [1000.0, 1.0]]
    sigma_im_token_list = ["-0d001+1d0", "-1000d0+1d0"]

    parse = argparse.ArgumentParser()
    parse.add_argument("--osly", default=3, type=int)
    parse.add_argument("--sigma", default=0, type=int)
    args = parse.parse_args()
    config.fileConfig(
        "settings/log.conf",
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

    prepare_dat = True
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

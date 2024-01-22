import numpy as np
from itertools import product


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

    fine_grid = 256
    coarse_grid_list = [8, 16, 32, 64]
    osly_list = [0, 1, 2, 3, 4]
    sigma_pm_list = [[1.1, 1.0], [1.0, 1.1], [1.01, 1.0], [1.0, 1.01]]
    l_list = [5.0, 5.0 - 1.0 / 128]

    parse = argparse.ArgumentParser()
    parse.add_argument("--en", default=3, type=int)
    parse.add_argument("--sigma", default=0, type=int)
    parse.add_argument("--posi", default=0, type=int)
    args = parse.parse_args()
    config.fileConfig(
        "settings/log.conf",
        defaults={
            "logfilename": "logs/flat-interface-en{0:d}-sigma{1:d}-l{2:d}.log".format(
                args.en, args.sigma, args.posi
            )
        },
    )
    eigen_num = args.en
    sigma_pm = sigma_pm_list[args.sigma]
    l = l_list[args.posi]

    logging.info("=" * 80)
    logging.info("Start")
    logging.info(
        "In the medium, sigma+={0:.4e}, sigma-={1:.4e}, l={2:.4e}".format(*sigma_pm, l)
    )

    rela_errors_h1 = np.zeros((len(coarse_grid_list), len(osly_list)))
    rela_errors_l2 = np.zeros(rela_errors_h1.shape)

    for coarse_grid_ind, osly_ind in product(
        range(len(coarse_grid_list)), range(len(osly_list))
    ):
        coeff, source, u = get_test_settings(fine_grid, sigma_pm)

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

    print(rela_errors_h1)
    print(rela_errors_l2)

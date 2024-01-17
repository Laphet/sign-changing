import numpy as np
from itertools import product


def get_test_settings(fine_grids: int, sigma_pm):
    coeff = np.zeros((fine_grids, fine_grids))
    source = np.zeros((fine_grids, fine_grids))
    tot_dof = (fine_grids - 1) ** 2
    u = np.zeros((tot_dof,))
    h = 1.0 / fine_grids
    l = 0.5 - 1.0 / 32
    for elem_ind_x, elem_ind_y in product(range(fine_grids), range(fine_grids)):
        x, y = (elem_ind_x + 0.5) * h, (elem_ind_y + 0.5) * h
        if y >= l:
            coeff[elem_ind_x, elem_ind_y] = -sigma_pm[1]
        else:
            coeff[elem_ind_x, elem_ind_y] = sigma_pm[0]
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
            u[dof_ind_y * (fine_grids - 1) + dof_ind_x] = temp * sigma_pm[0]
        else:
            u[dof_ind_y * (fine_grids - 1) + dof_ind_x] = -temp * sigma_pm[1]

    return coeff, source, u

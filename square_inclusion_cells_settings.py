import numpy as np
from itertools import product


def get_test_settings(fine_grids: int, sigma_im, cell_len=8):
    coeff = np.zeros((fine_grids, fine_grids))

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
    return coeff

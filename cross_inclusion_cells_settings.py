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

    ms_basis_dat_a = np.load(
        "{0:s}/{1:s}".format(DAT_ROOT_PATH, "ms-basis-0d001+1d0.npy")
    )
    ms_basis_dat_b = np.load(
        "{0:s}/{1:s}".format(DAT_ROOT_PATH, "ms-basis-0d01+10d0.npy")
    )
    print(np.min(ms_basis_dat_a[0, 0, :]), np.max(ms_basis_dat_a[0, 0, :]))
    print(np.min(ms_basis_dat_b[0, 0, :]), np.max(ms_basis_dat_b[0, 0, :]))

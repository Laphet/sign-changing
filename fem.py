import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
import bilinear_bases as BB
from itertools import product


def get_dof_ind(elem_ind_x: int, elem_ind_y: int, loc_ind: int, sizes):
    if (
        (elem_ind_x == 0 and loc_ind in [0, 2])
        or (elem_ind_x == sizes[0] - 1 and loc_ind in [1, 3])
        or (elem_ind_y == 0 and loc_ind in [0, 1])
        or (elem_ind_y == sizes[1] - 1 and loc_ind in [2, 3])
    ):
        return -1
    else:
        loc_ind_y, loc_ind_x = divmod(loc_ind, 2)
        return (
            (elem_ind_y + loc_ind_y - 1) * (sizes[0] - 1) + elem_ind_x + loc_ind_x - 1
        )


def get_fem_mat(coeff: np.ndarray):
    max_data_len = coeff.shape[0] * coeff.shape[1] * BB.N_V**2
    II = -np.ones((max_data_len,), dtype=np.int32)
    JJ = -np.ones((max_data_len,), dtype=np.int32)
    VV = np.zeros((max_data_len,))
    marker = 0
    for elem_ind_x, elem_ind_y in product(range(coeff.shape[0]), range(coeff.shape[1])):
        elem_stiff_mat = coeff[elem_ind_x, elem_ind_y] * BB.elem_Laplace_stiff_mat
        for loc_ind_row, loc_ind_col in product(range(BB.N_V), range(BB.N_V)):
            dof_ind_row = get_dof_ind(elem_ind_x, elem_ind_y, loc_ind_row, coeff.shape)
            dof_ind_col = get_dof_ind(elem_ind_x, elem_ind_y, loc_ind_col, coeff.shape)
            if dof_ind_row >= 0 and dof_ind_col >= 0:
                II[marker] = dof_ind_row
                JJ[marker] = dof_ind_col
                VV[marker] = elem_stiff_mat[loc_ind_row, loc_ind_col]
                marker += 1
    tot_dof = (coeff.shape[0] - 1) * (coeff.shape[1] - 1)
    fem_mat_coo = coo_matrix(
        (VV[:marker], (II[:marker], JJ[:marker])), shape=(tot_dof, tot_dof)
    )
    return fem_mat_coo.tocsc()


def get_fem_rhs(source: np.ndarray):
    tot_dof = (source.shape[0] - 1) * (source.shape[1] - 1)
    h_x, h_y = 1.0 / (source.shape[0]), 1.0 / (source.shape[1])
    rhs = np.zeros(tot_dof)
    for elem_ind_x, elem_ind_y, loc_ind in product(
        range(source.shape[0]), range(source.shape[1]), range(BB.N_V)
    ):
        dof_ind = get_dof_ind(elem_ind_x, elem_ind_y, loc_ind, source.shape)
        if dof_ind >= 0:
            rhs[dof_ind] += 0.25 * h_x * h_y * source[elem_ind_x, elem_ind_y]
    return rhs


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

    coeff_abs = np.abs(coeff)
    fem_mat = get_fem_mat(coeff)
    rhs = get_fem_rhs(source)
    fem_mat_abs = get_fem_mat(coeff_abs)
    return fem_mat, rhs, u, fem_mat_abs


if __name__ == "__main__":
    sigma_pm_list = [[10.0, 1.0], [100.0, 1.0], [1000.0, 1.0], [1.0e4, 1.0]]
    fine_grids_list = [32, 64, 128, 256, 512]
    rela_errors = np.zeros((len(fine_grids_list), len(sigma_pm_list)))
    for fine_grids_ind, sigma_pm_ind in product(
        range(len(fine_grids_list)), range(len(sigma_pm_list))
    ):
        fine_grids = fine_grids_list[fine_grids_ind]
        sigma_pm = sigma_pm_list[sigma_pm_ind]
        fem_mat, rhs, u, fem_mat_abs = get_test_settings(fine_grids, sigma_pm)
        u_fem = spsolve(fem_mat, rhs)
        delta_u = u - u_fem
        u_nrm2 = np.sqrt(np.dot(fem_mat_abs.dot(u), u))
        delta_u_nrm2 = np.sqrt(np.dot(fem_mat_abs.dot(delta_u), delta_u))
        rela_errors[fine_grids_ind, sigma_pm_ind] = delta_u_nrm2 / u_nrm2
    print(rela_errors)

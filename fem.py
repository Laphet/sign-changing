import numpy as np
from scipy.sparse import csr_matrix
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
    fem_mat = csr_matrix(
        (VV[:marker], (II[:marker], JJ[:marker])), shape=(tot_dof, tot_dof)
    )
    return fem_mat


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


def get_mass_mat(weight: np.ndarray):
    h_x, h_y = 1.0 / (weight.shape[0]), 1.0 / (weight.shape[1])
    max_data_len = weight.shape[0] * weight.shape[1] * BB.N_V**2
    II = -np.ones((max_data_len,), dtype=np.int32)
    JJ = -np.ones((max_data_len,), dtype=np.int32)
    VV = np.zeros((max_data_len,))
    marker = 0
    for elem_ind_x, elem_ind_y in product(
        range(weight.shape[0]), range(weight.shape[1])
    ):
        elem_mass_mat = (
            weight[elem_ind_x, elem_ind_y] * BB.elem_bilinear_mass_mat * h_x * h_y
        )
        for loc_ind_row, loc_ind_col in product(range(BB.N_V), range(BB.N_V)):
            dof_ind_row = get_dof_ind(elem_ind_x, elem_ind_y, loc_ind_row, weight.shape)
            dof_ind_col = get_dof_ind(elem_ind_x, elem_ind_y, loc_ind_col, weight.shape)
            if dof_ind_row >= 0 and dof_ind_col >= 0:
                II[marker] = dof_ind_row
                JJ[marker] = dof_ind_col
                VV[marker] = elem_mass_mat[loc_ind_row, loc_ind_col]
                marker += 1
    tot_dof = (weight.shape[0] - 1) * (weight.shape[1] - 1)
    mass_mat = csr_matrix(
        (VV[:marker], (II[:marker], JJ[:marker])), shape=(tot_dof, tot_dof)
    )
    return mass_mat


if __name__ == "__main__":
    from simple_flat_interface_settings import get_test_settings

    sigma_pm_list = [[1.0, -1.0]]
    fine_grid_list = [8]
    rela_errors = np.zeros((len(fine_grid_list), len(sigma_pm_list)))
    for fine_grid_ind, sigma_pm_ind in product(
        range(len(fine_grid_list)), range(len(sigma_pm_list))
    ):
        fine_grid = fine_grid_list[fine_grid_ind]
        sigma_pm = sigma_pm_list[sigma_pm_ind]
        coeff, source, u = get_test_settings(fine_grid, sigma_pm)
        fem_mat = get_fem_mat(coeff)
        rhs = get_fem_rhs(source)
        fem_mat_abs = get_fem_mat(np.abs(coeff))
        u_fem = spsolve(fem_mat, rhs)
        delta_u = u - u_fem
        u_nrm2 = np.sqrt(np.dot(fem_mat_abs.dot(u), u))
        delta_u_nrm2 = np.sqrt(np.dot(fem_mat_abs.dot(delta_u), delta_u))
        rela_errors[fine_grid_ind, sigma_pm_ind] = delta_u_nrm2 / u_nrm2
    print(rela_errors)

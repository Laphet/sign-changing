import numpy as np
from scipy.sparse import csr_matrix
from itertools import product

QUAD_ORDER = 2
QUAD_CORD, QUAD_WGHT = np.polynomial.legendre.leggauss(QUAD_ORDER)
N_V = 4


def get_locbase_val(loc_ind: int, x: float, y: float):
    val = -1.0
    if loc_ind == 0:
        val = 0.25 * (1.0 - x) * (1.0 - y)
    elif loc_ind == 1:
        val = 0.25 * (1.0 + x) * (1.0 - y)
    elif loc_ind == 2:
        val = 0.25 * (1.0 - x) * (1.0 + y)
    elif loc_ind == 3:
        val = 0.25 * (1.0 + x) * (1.0 + y)
    else:
        raise ValueError("Invalid option, loc_ind={:.d}".format(loc_ind))
    return val


def get_locbase_grad_val(loc_ind: int, x: float, y: float):
    grad_val_x, grad_val_y = -1.0, -1.0
    if loc_ind == 0:
        grad_val_x, grad_val_y = -0.25 * (1.0 - y), -0.25 * (1.0 - x)
    elif loc_ind == 1:
        grad_val_x, grad_val_y = 0.25 * (1.0 - y), -0.25 * (1.0 + x)
    elif loc_ind == 2:
        grad_val_x, grad_val_y = -0.25 * (1.0 + y), 0.25 * (1.0 - x)
    elif loc_ind == 3:
        grad_val_x, grad_val_y = 0.25 * (1.0 + y), 0.25 * (1.0 + x)
    else:
        raise ValueError("Invalid option, loc_ind={:.d}".format(loc_ind))
    return grad_val_x, grad_val_y


def get_loc_stiff(loc_ind_i: int, loc_ind_j: int):
    val = 0.0
    for quad_ind_x, quad_ind_y in product(range(QUAD_ORDER), range(QUAD_ORDER)):
        quad_cord_x, quad_wght_x = QUAD_CORD[quad_ind_x], QUAD_WGHT[quad_ind_x]
        quad_cord_y, quad_wght_y = QUAD_CORD[quad_ind_y], QUAD_WGHT[quad_ind_y]
        grad_val_ix, grad_val_iy = get_locbase_grad_val(
            loc_ind_i, quad_cord_x, quad_cord_y
        )
        grad_val_jx, grad_val_jy = get_locbase_grad_val(
            loc_ind_j, quad_cord_x, quad_cord_y
        )
        val += (
            (grad_val_ix * grad_val_jx + grad_val_iy * grad_val_jy)
            * quad_wght_x
            * quad_wght_y
        )
    return val


def get_loc_mass(loc_ind_i: int, loc_ind_j: int):
    val = 0.0
    for quad_ind_x, quad_ind_y in product(range(QUAD_ORDER), range(QUAD_ORDER)):
        quad_cord_x, quad_wght_x = QUAD_CORD[quad_ind_x], QUAD_WGHT[quad_ind_x]
        quad_cord_y, quad_wght_y = QUAD_CORD[quad_ind_y], QUAD_WGHT[quad_ind_y]
        val_i = get_locbase_val(loc_ind_i, quad_cord_x, quad_cord_y)
        val_j = get_locbase_val(loc_ind_j, quad_cord_x, quad_cord_y)
        val += 0.25 * 1.0 * 1.0 * val_i * val_j * quad_wght_x * quad_wght_y
    return val


elem_Laplace_stiff_mat = np.zeros((N_V, N_V))
elem_bilinear_mass_mat = np.zeros((N_V, N_V))

for loc_ind_i, loc_ind_j in product(range(N_V), range(N_V)):
    elem_Laplace_stiff_mat[loc_ind_i, loc_ind_j] = get_loc_stiff(loc_ind_i, loc_ind_j)
    elem_bilinear_mass_mat[loc_ind_i, loc_ind_j] = get_loc_mass(loc_ind_i, loc_ind_j)


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
    max_data_len = coeff.shape[0] * coeff.shape[1] * N_V**2
    II = -np.ones((max_data_len,), dtype=np.int32)
    JJ = -np.ones((max_data_len,), dtype=np.int32)
    VV = np.zeros((max_data_len,))
    marker = 0
    for elem_ind_x, elem_ind_y in product(range(coeff.shape[0]), range(coeff.shape[1])):
        elem_stiff_mat = coeff[elem_ind_x, elem_ind_y] * elem_Laplace_stiff_mat
        for loc_ind_row, loc_ind_col in product(range(N_V), range(N_V)):
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
        range(source.shape[0]), range(source.shape[1]), range(N_V)
    ):
        dof_ind = get_dof_ind(elem_ind_x, elem_ind_y, loc_ind, source.shape)
        if dof_ind >= 0:
            rhs[dof_ind] += 0.25 * h_x * h_y * source[elem_ind_x, elem_ind_y]
    return rhs


def get_mass_mat(weight: np.ndarray):
    h_x, h_y = 1.0 / (weight.shape[0]), 1.0 / (weight.shape[1])
    max_data_len = weight.shape[0] * weight.shape[1] * N_V**2
    II = -np.ones((max_data_len,), dtype=np.int32)
    JJ = -np.ones((max_data_len,), dtype=np.int32)
    VV = np.zeros((max_data_len,))
    marker = 0
    for elem_ind_x, elem_ind_y in product(
        range(weight.shape[0]), range(weight.shape[1])
    ):
        elem_mass_mat = (
            weight[elem_ind_x, elem_ind_y] * elem_bilinear_mass_mat * h_x * h_y
        )
        for loc_ind_row, loc_ind_col in product(range(N_V), range(N_V)):
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

    # from scipy.sparse.linalg import spsolve
    # The default sparse solver in scipy is superLU, with optional interfaces to UMFPACK.
    # All those solvers are not parallelized.
    # from scipy.sparse.linalg import spsolve
    # Try to use the parallelized solver in the mkl library (Pardiso).
    import pypardiso

    spsolve = pypardiso.spsolve

    sigma_pm_list = [[10.0, 1.0]]
    fine_grid_list = [8, 16, 32, 64]
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

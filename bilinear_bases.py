import numpy as np
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
elem_Bilinear_mass_mat = np.zeros((N_V, N_V))

for loc_ind_i, loc_ind_j in product(range(N_V), range(N_V)):
    elem_Laplace_stiff_mat[loc_ind_i, loc_ind_j] = get_loc_stiff(loc_ind_i, loc_ind_j)
    elem_Bilinear_mass_mat[loc_ind_i, loc_ind_j] = get_loc_mass(loc_ind_i, loc_ind_j)

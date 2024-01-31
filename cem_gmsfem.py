import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
from scipy.sparse.linalg import eigsh
import bilinear_bases as BB
from itertools import product

import logging


class CemGmsfem:
    def __init__(
        self,
        coarse_grid: int,
        eigen_num: int,
        oversamp_layer: int,
        coeff: np.ndarray,
    ) -> None:
        if coeff.shape[0] != coeff.shape[1]:
            print("The shape {0:d} != {1:d}".format(coeff.shape[0], coeff.shape[1]))
            return
        self.fine_grid = coeff.shape[0]
        self.coarse_grid = coarse_grid
        self.sub_grid = self.fine_grid // self.coarse_grid
        self.coarse_elem = self.coarse_grid**2
        self.sub_elem = self.sub_grid**2
        self.tot_node = (self.fine_grid + 1) ** 2
        self.h = 1.0 / self.fine_grid
        self.eigen_num = eigen_num
        self.oversamp_layer = oversamp_layer
        self.tot_fd_num = self.coarse_elem * self.eigen_num
        self.coeff = coeff
        self.coeff_abs = np.abs(self.coeff)
        self.kappa = 24.0 * self.coarse_grid**2 * self.coeff
        self.kappa_abs = np.abs(self.kappa)
        logging.info(
            "CEM-GMsFEM uses fine_grid={0:d}, coarse_grid={1:d}, eigen_num={2:d}, oversamp_layer={3:d}".format(
                self.fine_grid, self.coarse_grid, self.eigen_num, self.oversamp_layer
            )
        )
        # in get_eigen_pair()
        self.eigen_vec = np.zeros(
            ((self.sub_grid + 1) ** 2, self.coarse_elem * self.eigen_num)
        )
        self.eigen_val = np.zeros((self.coarse_elem * self.eigen_num,))
        self.S_abs_mat_list = [None] * self.coarse_elem
        self.S_mat_list = [None] * self.coarse_elem
        # in get_ind_map()
        self.ind_map_list = [None] * self.coarse_elem
        self.ind_map_rev_list = [None] * self.coarse_elem
        self.loc_fd_num = np.zeros((self.coarse_elem,), dtype=np.int32)
        # in get_ms_basis()
        self.basis_list = [None] * self.coarse_elem
        # in get_glb_A()
        self.glb_A = None
        # in get_glb_basis_spmat()
        self.glb_basis_spmat = None
        self.glb_basis_spmat_T = None
        # in setup()
        self.A_ms = None

    def get_coarse_ngh_elem_lim(self, coarse_elem_ind):
        coarse_elem_ind_y, coarse_elem_ind_x = divmod(coarse_elem_ind, self.coarse_grid)
        coarse_ngh_elem_lf_lim = max(0, coarse_elem_ind_x - self.oversamp_layer)
        coarse_ngh_elem_dw_lim = max(0, coarse_elem_ind_y - self.oversamp_layer)
        coarse_ngh_elem_rg_lim = min(
            self.coarse_grid, coarse_elem_ind_x + self.oversamp_layer + 1
        )
        coarse_ngh_elem_up_lim = min(
            self.coarse_grid, coarse_elem_ind_y + self.oversamp_layer + 1
        )
        return (
            coarse_ngh_elem_lf_lim,
            coarse_ngh_elem_rg_lim,
            coarse_ngh_elem_dw_lim,
            coarse_ngh_elem_up_lim,
        )

    def get_eigen_pair(self):
        fd_num = (self.sub_grid + 1) ** 2
        for coarse_elem_ind_y, coarse_elem_ind_x in product(
            range(self.coarse_grid), range(self.coarse_grid)
        ):
            max_data_len = self.sub_elem * BB.N_V**2
            II = np.zeros((max_data_len,), dtype=np.int32)
            JJ = np.zeros((max_data_len,), dtype=np.int32)
            A_abs_val = np.zeros((max_data_len,))
            S_abs_val = np.zeros((max_data_len,))
            S_val = np.zeros((max_data_len,))
            marker = 0
            for sub_elem_ind_y, sub_elem_ind_x in product(
                range(self.sub_grid), range(self.sub_grid)
            ):
                fine_elem_ind_x = coarse_elem_ind_x * self.sub_grid + sub_elem_ind_x
                fine_elem_ind_y = coarse_elem_ind_y * self.sub_grid + sub_elem_ind_y
                loc_coeff_abs = self.coeff_abs[fine_elem_ind_x, fine_elem_ind_y]
                loc_kappa_abs = self.kappa_abs[fine_elem_ind_x, fine_elem_ind_y]
                loc_kappa = self.kappa[fine_elem_ind_x, fine_elem_ind_y]
                for loc_ind_row, loc_ind_col in product(range(BB.N_V), range(BB.N_V)):
                    iy, ix = divmod(loc_ind_row, 2)
                    jy, jx = divmod(loc_ind_col, 2)
                    II[marker] = (
                        (sub_elem_ind_y + iy) * (self.sub_grid + 1)
                        + sub_elem_ind_x
                        + ix
                    )
                    JJ[marker] = (
                        (sub_elem_ind_y + jy) * (self.sub_grid + 1)
                        + sub_elem_ind_x
                        + jx
                    )
                    A_abs_val[marker] = loc_coeff_abs * (
                        BB.elem_Laplace_stiff_mat[loc_ind_row, loc_ind_col]
                    )
                    S_abs_val[marker] = (
                        self.h**2
                        * loc_kappa_abs
                        * BB.elem_bilinear_mass_mat[loc_ind_row, loc_ind_col]
                    )
                    S_val[marker] = (
                        self.h**2
                        * loc_kappa
                        * BB.elem_bilinear_mass_mat[loc_ind_row, loc_ind_col]
                    )
                    marker += 1

            A_abs_mat = csr_matrix(
                (A_abs_val[:marker], (II[:marker], JJ[:marker])), shape=(fd_num, fd_num)
            )
            S_abs_mat = csr_matrix(
                (S_abs_val[:marker], (II[:marker], JJ[:marker])), shape=(fd_num, fd_num)
            )
            S_mat = csr_matrix(
                (S_val[:marker], (II[:marker], JJ[:marker])), shape=(fd_num, fd_num)
            )

            val, vec = eigsh(
                A_abs_mat, k=self.eigen_num, M=S_abs_mat, sigma=-1.0, which="LM"
            )

            coarse_elem_ind = coarse_elem_ind_y * self.coarse_grid + coarse_elem_ind_x
            self.eigen_val[
                coarse_elem_ind
                * self.eigen_num : (coarse_elem_ind + 1)
                * self.eigen_num
            ] = val
            self.eigen_vec[
                :,
                coarse_elem_ind
                * self.eigen_num : (coarse_elem_ind + 1)
                * self.eigen_num,
            ] = vec
            self.S_abs_mat_list[coarse_elem_ind] = S_abs_mat
            self.S_mat_list[coarse_elem_ind] = S_mat

    def get_node_ind(self, fine_elem_ind_x, fine_elem_ind_y, loc_ind):
        is_bdry_node = (
            (fine_elem_ind_x == 0 and loc_ind in [0, 2])
            or (fine_elem_ind_x == self.fine_grid - 1 and loc_ind in [1, 3])
            or (fine_elem_ind_y == 0 and loc_ind in [0, 1])
            or (fine_elem_ind_y == self.fine_grid - 1 and loc_ind in [2, 3])
        )
        loc_ind_y, loc_ind_x = divmod(loc_ind, 2)
        node_ind = (
            (fine_elem_ind_y + loc_ind_y) * (self.fine_grid + 1)
            + fine_elem_ind_x
            + loc_ind_x
        )
        return node_ind, is_bdry_node

    def get_ind_map(self):
        # A list of ind_map, ind_map[coarse_elem_ind] is the ind_map
        # and the reverse map list
        for coarse_elem_ind in range(self.coarse_elem):
            # Get the mapping ind_map_dic[glb_node_ind] = loc_fd_ind
            # and the reverse mapping ind_map_rev_dic[loc_fd_ind] = glb_node_ind
            # glb_node_ind 0 ~ (fine_grid+1)**2
            ind_map_dic = {}
            ind_map_rev_dic = {}
            fd_ind = 0
            lf_lim, rg_lim, dw_lim, up_lim = self.get_coarse_ngh_elem_lim(
                coarse_elem_ind
            )
            node_ind_x_lf_lim = lf_lim * self.sub_grid + 1
            node_ind_x_rg_lim = rg_lim * self.sub_grid
            node_ind_y_dw_lim = dw_lim * self.sub_grid + 1
            node_ind_y_up_lim = up_lim * self.sub_grid

            for node_ind_y in range(node_ind_y_dw_lim, node_ind_y_up_lim):
                for node_ind_x in range(node_ind_x_lf_lim, node_ind_x_rg_lim):
                    # node_ind 0 ~ (fine_grid + 1)**2
                    node_ind = node_ind_y * (self.fine_grid + 1) + node_ind_x
                    ind_map_dic[node_ind] = fd_ind
                    ind_map_rev_dic[fd_ind] = node_ind
                    fd_ind += 1
            self.ind_map_list[coarse_elem_ind] = ind_map_dic
            self.ind_map_rev_list[coarse_elem_ind] = ind_map_rev_dic
            self.loc_fd_num[coarse_elem_ind] = fd_ind

    def get_glb_vec(self, coarse_elem_ind, vec):
        ind_map_rev = self.ind_map_rev_list[coarse_elem_ind]
        glb_vec = np.zeros((self.tot_node,))
        for loc_fd_ind, node_ind in ind_map_rev.items():
            glb_vec[node_ind] = vec[loc_fd_ind]
        return glb_vec

    def get_ms_basis_on_coarse_elem(self, coarse_elem_ind: int):
        max_data_len = (2 * self.oversamp_layer + 1) ** 2 * (
            (self.sub_grid + 1) ** 4 + self.sub_elem * BB.N_V**2
        )
        fd_num = self.loc_fd_num[coarse_elem_ind]
        ind_map_dic = self.ind_map_list[coarse_elem_ind]
        II = -np.ones((max_data_len,), dtype=np.int32)
        JJ = -np.ones((max_data_len,), dtype=np.int32)
        VV = np.zeros((max_data_len,))
        marker = 0
        rhs_basis = np.zeros((fd_num, self.eigen_num))
        lf_lim, rg_lim, dw_lim, up_lim = self.get_coarse_ngh_elem_lim(coarse_elem_ind)
        for coarse_ngh_elem_ind_y, coarse_ngh_elem_ind_x in product(
            range(dw_lim, up_lim), range(lf_lim, rg_lim)
        ):
            coarse_ngh_elem_ind = (
                coarse_ngh_elem_ind_y * self.coarse_grid + coarse_ngh_elem_ind_x
            )
            S_mat = self.S_mat_list[coarse_ngh_elem_ind]
            S_abs_mat = self.S_abs_mat_list[coarse_ngh_elem_ind]
            eigen_vec = self.eigen_vec[
                :,
                coarse_ngh_elem_ind
                * self.eigen_num : (coarse_ngh_elem_ind + 1)
                * self.eigen_num,
            ]
            S_abs_E_mat = S_abs_mat.dot(eigen_vec)
            Et_S_E_mat = eigen_vec.T @ (S_mat.dot(eigen_vec))
            Q_mat = S_abs_E_mat @ Et_S_E_mat @ S_abs_E_mat.T
            Z_mat = Et_S_E_mat @ S_abs_E_mat.T
            node_sub_ind_list = -np.ones(((self.sub_grid + 1) ** 2,), dtype=np.int32)
            fd_ind_list = -np.ones(((self.sub_grid + 1) ** 2,), dtype=np.int32)
            marker_ = 0
            for node_sub_ind_y, node_sub_ind_x in product(
                range(self.sub_grid + 1), range(self.sub_grid + 1)
            ):
                node_sub_ind = node_sub_ind_y * (self.sub_grid + 1) + node_sub_ind_x
                node_ind_y = coarse_ngh_elem_ind_y * self.sub_grid + node_sub_ind_y
                node_ind_x = coarse_ngh_elem_ind_x * self.sub_grid + node_sub_ind_x
                node_ind = node_ind_y * (self.fine_grid + 1) + node_ind_x
                if node_ind in ind_map_dic:
                    fd_ind = ind_map_dic[node_ind]
                    node_sub_ind_list[marker_] = node_sub_ind
                    fd_ind_list[marker_] = fd_ind
                    marker_ += 1
            for ind_i in range(marker_):
                node_sub_ind_i = node_sub_ind_list[ind_i]
                fd_ind_i = fd_ind_list[ind_i]
                for ind_j in range(marker_):
                    node_sub_ind_j = node_sub_ind_list[ind_j]
                    fd_ind_j = fd_ind_list[ind_j]
                    II[marker] = fd_ind_i
                    JJ[marker] = fd_ind_j
                    VV[marker] = Q_mat[node_sub_ind_i, node_sub_ind_j]
                    marker += 1
                if coarse_ngh_elem_ind == coarse_elem_ind:
                    for eigen_ind in range(self.eigen_num):
                        rhs_basis[fd_ind_i, eigen_ind] += Z_mat[
                            eigen_ind, node_sub_ind_i
                        ]
            for sub_elem_ind_y, sub_elem_ind_x in product(
                range(self.sub_grid), range(self.sub_grid)
            ):
                fine_elem_ind_y = coarse_ngh_elem_ind_y * self.sub_grid + sub_elem_ind_y
                fine_elem_ind_x = coarse_ngh_elem_ind_x * self.sub_grid + sub_elem_ind_x
                loc_coeff = self.coeff[fine_elem_ind_x, fine_elem_ind_y]

                for loc_ind_i in range(BB.N_V):
                    node_ind_i, _ = self.get_node_ind(
                        fine_elem_ind_x, fine_elem_ind_y, loc_ind_i
                    )
                    if node_ind_i in ind_map_dic:
                        fd_ind_i = ind_map_dic[node_ind_i]
                        for loc_ind_j in range(BB.N_V):
                            node_ind_j, _ = self.get_node_ind(
                                fine_elem_ind_x, fine_elem_ind_y, loc_ind_j
                            )
                            if node_ind_j in ind_map_dic:
                                fd_ind_j = ind_map_dic[node_ind_j]
                                II[marker] = fd_ind_i
                                JJ[marker] = fd_ind_j
                                VV[marker] = loc_coeff * (
                                    BB.elem_Laplace_stiff_mat[loc_ind_i, loc_ind_j]
                                )
                                marker += 1
        Op_mat = csr_matrix(
            (VV[:marker], (II[:marker], JJ[:marker])), shape=(fd_num, fd_num)
        )
        basis_wrt_coarse_elem = np.zeros(rhs_basis.shape)
        for eigen_ind in range(self.eigen_num):
            basis = spsolve(Op_mat, rhs_basis[:, eigen_ind])
            basis_wrt_coarse_elem[:, eigen_ind] = basis
        self.basis_list[coarse_elem_ind] = basis_wrt_coarse_elem

    def get_ms_basis(self):
        # assert self.oversamp_layer > 0 and self.eigen_num > 0
        # assert len(self.ind_map_list) > 0

        prc_flag = 1
        for coarse_elem_ind in range(self.coarse_elem):
            self.get_ms_basis_on_coarse_elem(coarse_elem_ind)
            if coarse_elem_ind > prc_flag / 10 * self.coarse_elem:
                logging.info(
                    "......{0:.2f}%".format(coarse_elem_ind / self.coarse_elem * 100.0)
                )
                prc_flag += 1

    def get_glb_A(self):
        max_data_len = self.fine_grid**2 * BB.N_V**2
        II = -np.ones((max_data_len,), dtype=np.int32)
        JJ = -np.ones((max_data_len,), dtype=np.int32)
        VV = np.zeros((max_data_len,))
        marker = 0
        for fine_elem_ind_y, fine_elem_ind_x in product(
            range(self.fine_grid), range(self.fine_grid)
        ):
            coeff = self.coeff[fine_elem_ind_x, fine_elem_ind_y]
            elem_stiff_mat = coeff * BB.elem_Laplace_stiff_mat
            for loc_ind_i, loc_ind_j in product(range(BB.N_V), range(BB.N_V)):
                node_ind_i, _ = self.get_node_ind(
                    fine_elem_ind_x, fine_elem_ind_y, loc_ind_i
                )
                node_ind_j, _ = self.get_node_ind(
                    fine_elem_ind_x, fine_elem_ind_y, loc_ind_j
                )
                II[marker] = node_ind_i
                JJ[marker] = node_ind_j
                VV[marker] = elem_stiff_mat[loc_ind_i, loc_ind_j]
                marker += 1

        self.glb_A = csr_matrix(
            (VV[:marker], (II[:marker], JJ[:marker])),
            shape=(self.tot_node, self.tot_node),
        )

    def get_glb_basis_spmat(self):
        max_data_len = np.sum(self.loc_fd_num) * self.eigen_num
        II = -np.ones((max_data_len,), dtype=np.int32)
        JJ = -np.ones((max_data_len,), dtype=np.int32)
        VV = np.zeros((max_data_len,))
        marker = 0
        for coarse_elem_ind in range(self.coarse_elem):
            for eigen_ind in range(self.eigen_num):
                fd_ind = coarse_elem_ind * self.eigen_num + eigen_ind
                for loc_fd_ind, node_ind in self.ind_map_rev_list[
                    coarse_elem_ind
                ].items():
                    II[marker] = node_ind
                    JJ[marker] = fd_ind
                    VV[marker] = self.basis_list[coarse_elem_ind][loc_fd_ind, eigen_ind]
                    marker += 1
        self.glb_basis_spmat = csr_matrix(
            (VV[:marker], (II[:marker], JJ[:marker])),
            shape=(self.tot_node, self.tot_fd_num),
        )
        self.glb_basis_spmat_T = self.glb_basis_spmat.transpose().tocsr()

    def setup(self):
        self.get_glb_A()
        logging.info("Finish getting the global stiffness matrix.")
        if self.oversamp_layer >= 1:
            self.get_eigen_pair()
            logging.info("Finish getting all eigenvalue-vector pairs.")
            eigen_val_minmax = self.eigen_val.reshape(
                (self.coarse_elem, self.eigen_num)
            )
            eigen_min_range = (
                np.min(np.min(eigen_val_minmax, axis=1)),
                np.max(np.min(eigen_val_minmax, axis=1)),
            )
            eigen_max_range = (
                np.min(np.max(eigen_val_minmax, axis=1)),
                np.max(np.max(eigen_val_minmax, axis=1)),
            )
            logging.info(
                "lambda_min=({0:.4e}, {1:.4e}), lambda_max=({2:.4e}, {3:.4e})".format(
                    *eigen_min_range, *eigen_max_range
                )
            )

            self.get_ind_map()
            logging.info(
                "Finish getting maps of [global node index] to [local freedom index]."
            )
            self.get_ms_basis()
            logging.info("Finish getting the multiscale bases.")

            self.get_glb_basis_spmat()
            logging.info(
                "Finish collecting all the bases in a sparse matrix formation."
            )

            self.A_ms = self.glb_basis_spmat_T * self.glb_A * self.glb_basis_spmat
            logging.info("Finish constructing the final MS mat.")

    def get_glb_rhs(self, source):
        glb_rhs = np.zeros((self.tot_node,))
        for fine_elem_ind_x, fine_elem_ind_y, loc_ind in product(
            range(self.fine_grid), range(self.fine_grid), range(BB.N_V)
        ):
            node_ind, is_bdry_node = self.get_node_ind(
                fine_elem_ind_x, fine_elem_ind_y, loc_ind
            )
            if not is_bdry_node:
                glb_rhs[node_ind] += (
                    0.25 * self.h**2 * source[fine_elem_ind_x, fine_elem_ind_y]
                )
        return glb_rhs

    def solve(self, source):
        glb_rhs = self.get_glb_rhs(source)
        rhs = self.glb_basis_spmat_T.dot(glb_rhs)
        omega = spsolve(self.A_ms, rhs)
        u_ext = self.glb_basis_spmat.dot(omega)
        u_ext = u_ext.reshape((self.fine_grid + 1, self.fine_grid + 1))
        u = u_ext[1:-1, 1:-1]
        return u.reshape((-1,))

    def solve_by_coarse_bilinear(self, source):
        tot_dof = (self.coarse_grid - 1) ** 2
        max_data_len = tot_dof * (2 * self.sub_grid - 1) ** 2
        II = -np.ones((max_data_len,), dtype=np.int32)
        JJ = -np.ones((max_data_len,), dtype=np.int32)
        VV = np.zeros((max_data_len,))
        marker = 0
        for dof_ind_y, dof_ind_x in product(
            range(self.coarse_grid - 1), range(self.coarse_grid - 1)
        ):
            dof_ind = dof_ind_y * (self.coarse_grid - 1) + dof_ind_x
            basis_node_ind_x = (dof_ind_x + 1) * self.sub_grid
            basis_node_ind_y = (dof_ind_y + 1) * self.sub_grid
            node_ind_x_lf_lim = dof_ind_x * self.sub_grid + 1
            node_ind_x_rg_lim = (dof_ind_x + 2) * self.sub_grid
            node_ind_y_dw_lim = dof_ind_y * self.sub_grid + 1
            node_ind_y_up_lim = (dof_ind_y + 2) * self.sub_grid
            for node_ind_y, node_ind_x in product(
                range(node_ind_y_dw_lim, node_ind_y_up_lim),
                range(node_ind_x_lf_lim, node_ind_x_rg_lim),
            ):
                II[marker] = node_ind_y * (self.fine_grid + 1) + node_ind_x
                JJ[marker] = dof_ind
                VV[marker] = 1.0 - abs(node_ind_x - basis_node_ind_x) / self.sub_grid
                VV[marker] *= 1.0 - abs(node_ind_y - basis_node_ind_y) / self.sub_grid
                marker += 1
        coarse_basis_spmat = csr_matrix(
            (VV[:marker], (II[:marker], JJ[:marker])), shape=(self.tot_node, tot_dof)
        )

        coarse_basis_spmat_T = coarse_basis_spmat.transpose().tocsr()
        A_Q1 = coarse_basis_spmat_T * self.glb_A * coarse_basis_spmat

        glb_rhs = self.get_glb_rhs(source)
        rhs = coarse_basis_spmat_T.dot(glb_rhs)

        omega = spsolve(A_Q1, rhs)
        u_ext = coarse_basis_spmat.dot(omega)
        u_ext = u_ext.reshape((self.fine_grid + 1, self.fine_grid + 1))
        u = u_ext[1:-1, 1:-1]
        return u.reshape((-1,))

    def copy_eigen_space(self, fixed_solver):
        self.eigen_vec = fixed_solver.eigen_vec
        self.eigen_val = fixed_solver.eigen_val
        self.S_abs_mat_list = fixed_solver.S_abs_mat_list
        self.S_mat_list = fixed_solver.S_mat_list

    def get_plot_format(self, u):
        u_ext = np.zeros((self.fine_grid + 1, self.fine_grid + 1))
        u_ext[1:-1, 1:-1] = u.reshape((self.fine_grid - 1, -1))
        return u_ext


if __name__ == "__main__":
    from simple_flat_interface_settings import get_test_settings
    from fem import get_fem_mat, get_mass_mat

    import os
    from logging import config

    config.fileConfig("log.conf", defaults={"logfilename": "logs/test.log"})

    logging.info("=" * 80)
    logging.info("Start")

    sigma_pm = [1.0, 1.01]
    fine_grid = 400
    coarse_grid_list = [80]
    osly_list = [0]
    eigen_num = 3
    l = 0.5 - 1.0 / 100
    rela_errors_h1 = np.zeros((len(coarse_grid_list), len(osly_list)))
    rela_errors_l2 = np.zeros(rela_errors_h1.shape)

    for coarse_grid_ind, osly_ind in product(
        range(len(coarse_grid_list)), range(len(osly_list))
    ):
        coeff, source, u = get_test_settings(fine_grid, sigma_pm, l)

        coarse_grid = coarse_grid_list[coarse_grid_ind]
        osly = osly_list[osly_ind]
        cem_gmsfem = CemGmsfem(coarse_grid, eigen_num, osly, coeff)
        cem_gmsfem.setup()
        if cem_gmsfem.oversamp_layer >= 1:
            u_cem = cem_gmsfem.solve(source)
        else:
            u_cem = cem_gmsfem.solve_by_coarse_bilinear(source)

        fem_mat_abs = get_fem_mat(np.abs(coeff))
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

        DAT_ROOT_PATH = "resources"
        np.save(
            "{0:s}/{1:s}".format(
                DAT_ROOT_PATH, "delta-u-osly{0:d}-1d01+1d0.npy".format(osly)
            ),
            cem_gmsfem.get_plot_format(delta_u),
        )

        logging.info(
            "relative energy error={0:4e}, plain-L2 error={1:4e}.".format(
                rela_error_h1, rela_error_l2
            )
        )

        import plot_settings

        fig = plot_settings.plt.figure()
        ax = fig.subplots(1, 1)
        posi = plot_settings.plot_node_dat(cem_gmsfem.get_plot_format(delta_u), ax)
        plot_settings.append_colorbar(fig, ax, posi)

        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")
        ax.set_xticks([0.0, 0.5, 1.0], ["0.0", "0.5", "1.0"])
        ax.set_yticks([0.0, 0.5, 1.0], ["0.0", "0.5", "1.0"])
        plot_settings.plt.show()

    print(rela_errors_h1)
    print(rela_errors_l2)

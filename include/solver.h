/*!
 * \file solver.h
 * \author - Original code: HiFiLES Aerospace Computing Laboratory (ACL)
 *                                Aero/Astro Department. Stanford University.
 *         - Current development: Weiqi Shen
 *                                University of Florida
 *
 * High Fidelity Large Eddy Simulation (HiFiLES) Code.
 *
 * HiFiLES is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * HiFiLES is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with HiFiLES.  If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include <string>
#include "global.h"
#include "solution.h"

/*!
 * \brief Calculate the residual.
 * \param[in] FlowSol - Structure with the entire solution and mesh information.
 */
void CalcResidual(int in_file_num, int in_rk_stage, struct solution* FlowSol);

/*! get pointer to transformed discontinuous solution at a flux point */
double* get_disu_fpts_ptr(int in_ele_type, int in_ele, int in_field, int n_local_inter, int in_fpt, struct solution* FlowSol);

/*! get pointer to normal continuous transformed inviscid flux at a flux point */
double* get_norm_tconf_fpts_ptr(int in_ele_type, int in_ele, int in_field, int in_local_inter, int in_fpt, struct solution* FlowSol);

/*! get pointer to subgrid-scale flux at a flux point */
double* get_sgsf_fpts_ptr(int in_ele_type, int in_ele, int in_local_inter, int in_field, int in_dim, int in_fpt, struct solution* FlowSol);

/*! get pointer to determinant of jacobian at a flux point */
double* get_detjac_fpts_ptr(int in_ele_type, int in_ele, int in_ele_local_inter, int in_inter_local_fpt, struct solution* FlowSol);

/*! get pointer to magntiude of normal dot inverse of (determinant of jacobian multiplied by jacobian) at a solution point */
double* get_tdA_fpts_ptr(int in_ele_type, int in_ele, int in_ele_local_inter, int in_inter_local_fpt, struct solution* FlowSol);

/*! get pointer to weight at a flux point */
double* get_weight_fpts_ptr(int in_ele_type, int in_ele_local_inter, int in_inter_local_fpt, struct solution* FlowSol);

/*! get pointer to inter_detjac_inters_cubpts at a flux point */
double* get_inter_detjac_inters_cubpts_ptr(int in_ele_type, int in_ele, int in_ele_local_inter, int in_inter_local_fpt, struct solution* FlowSol);

/*! get pointer to normal at a flux point */
double* get_norm_fpts_ptr(int in_ele_type, int in_ele, int in_local_inter, int in_fpt, int in_dim, struct solution* FlowSol);

/*! get CPU pointer to coordinates at a flux point */
double* get_loc_fpts_ptr_cpu(int in_ele_type, int in_ele, int in_local_inter, int in_fpt, int in_dim, struct solution* FlowSol);

#ifdef _GPU
/*! get GPU pointer to coordinates at a flux point */
double* get_loc_fpts_ptr_gpu(int in_ele_type, int in_ele, int in_local_inter, int in_fpt, int in_dim, struct solution* FlowSol);
#endif

/*! get pointer to delta of the transformed discontinuous solution at a flux point */
double* get_delta_disu_fpts_ptr(int in_ele_type, int in_ele, int in_field, int n_local_inter, int in_fpt, struct solution* FlowSol);

/*! get pointer to gradient of the discontinuous solution at a flux point */
double* get_grad_disu_fpts_ptr(int in_ele_type, int in_ele, int in_local_inter, int in_field, int in_dim, int in_fpt, struct solution* FlowSol);

//patch solution
void patch_solution(struct solution* FlowSol);

// Initialize the solution in the mesh
void InitSolution(struct solution* FlowSol);

/*! reading a restart file */
void read_restart_ascii(int in_file_num, int in_n_files, struct solution* FlowSol);
#ifdef _HDF5
void read_restart_hdf5(int in_file_num, struct solution* FlowSol);
#endif

//calculate global time step size
void calc_time_step(struct solution* FlowSol);






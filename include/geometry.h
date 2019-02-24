/*!
 * \file geometry.h
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
#include "mesh.h"
#include "solution.h"

void SetInput(struct solution* FlowSol);

void GeoPreprocess(struct solution* FlowSol, mesh &mesh_data);

void ReadMesh(struct solution* FlowSol,mesh &mesh_data);

void CompConnectivity(mesh& mesh_data);

/*! Method that compares two cyclic faces and check if they should be matched */
void compare_cyclic_faces(hf_array<double> &xvert1, hf_array<double> &xvert2, int& num_v_per_f, int& rtag, hf_array<double> &delta_cyclic, double tol, struct solution* FlowSol);

/*! Method that checks if two cyclic faces are distance delta_cyclic apart */
bool check_cyclic(hf_array<double> &delta_cyclic, hf_array<double> &loc_center_inter_0, hf_array<double> &loc_center_inter_1, double tol, struct solution* FlowSol);

#ifdef _MPI

void match_mpifaces(hf_array<int> &in_f2v, hf_array<int> &in_f2nv, hf_array<double>& in_xv, hf_array<int>& inout_f_mpi2f, hf_array<int>& out_mpifaces_part, hf_array<double> &delta_cyclic, int n_mpi_faces, double tol, struct solution* FlowSol);

void find_rot_mpifaces(hf_array<int> &in_f2v, hf_array<int> &in_f2nv, hf_array<double>& in_xv, hf_array<int>& in_f_mpi2f, hf_array<int> &out_rot_tag_mpi, hf_array<int> &mpifaces_part, hf_array<double> delta_cyclic, int n_mpi_faces, double tol, struct solution* FlowSol);

void compare_mpi_faces(hf_array<double> &xvert1, hf_array<double> &xvert2, int& num_v_per_f, int& rtag, hf_array<double> &delta_cyclic, double tol, struct solution* FlowSol);

#endif

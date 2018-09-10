/*!
 * \file geometry.h
 * \author - Original code: SD++ developed by Patrice Castonguay, Antony Jameson,
 *                          Peter Vincent, David Williams (alphabetical by surname).
 *         - Current development: Aerospace Computing Laboratory (ACL)
 *                                Aero/Astro Department. Stanford University.
 * \version 0.1.0
 *
 * High Fidelity Large Eddy Simulation (HiFiLES) Code.
 * Copyright (C) 2014 Aerospace Computing Laboratory (ACL).
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

#include "mesh_reader.h"
#include "hf_array.h"
#include "solution.h"

#ifdef _MPI
#include "mpi.h"
#include "mpi_inters.h"
#endif

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

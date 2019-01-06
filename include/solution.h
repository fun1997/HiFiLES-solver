/*!
 * \file solution.h
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

#include "hf_array.h"
#include <string>
#include "input.h"
#include "eles.h"
#include "eles_tris.h"
#include "eles_quads.h"
#include "eles_hexas.h"
#include "eles_tets.h"
#include "eles_pris.h"
#include "int_inters.h"
#include "bdy_inters.h"

#ifdef _MPI
#include "mpi.h"
#include "mpi_inters.h"
#endif

struct solution {

  //basic parameters
  int rank;//defined in SetInput
  int nproc;//defined in SetInput
  int viscous;//defined in SetInput
  double time;//defined in InitSolution
  int n_ele_types;//defined in Initializing Elements
  int n_dims;//defined in mesh reading
  int n_steps;//defined in SetInput
  int adv_type;//defined in SetInput
  
  //restart/initialization parameter
  int restart_dump_freq;//defined in SetInput
  int ini_iter;//defined in InitSolution

  //element parameters
  hf_array<eles*> mesh_eles;
  eles_quads mesh_eles_quads;
  eles_tris mesh_eles_tris;
  eles_hexas mesh_eles_hexas;
  eles_tets mesh_eles_tets;
  eles_pris mesh_eles_pris;

  //interfaces
  int n_int_inter_types;
  int n_bdy_inter_types;
  hf_array<int_inters> mesh_int_inters;
  hf_array<bdy_inters> mesh_bdy_inters;

  //Diagnostic output quantities
  hf_array<double> body_force;
  hf_array<double> inv_force;
  hf_array<double> vis_force;
  hf_array<double> norm_residual;
  hf_array<double> integral_quantities;
  double coeff_lift;
  double coeff_drag;

  //Plotting parameters
  int plot_freq;//defined in setInput
  int write_type;//defined in SetInput

//mpi parameters
#ifdef _MPI

  int n_mpi_inter_types;
  hf_array<mpi_inters> mesh_mpi_inters;
  int n_mpi_inters;

#endif

};

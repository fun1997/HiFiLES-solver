/*!
 * \file inters.h
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


#ifdef _MPI
#include "mpi.h"
#endif
#include "hf_array.h"

class inters
{
public:

  // #### constructors ####

  // default constructor

  inters();

  // default destructor

  ~inters();

  // #### methods ####

  /*! setup inters */
  void setup_inters(int in_n_inters, int in_inter_type);

  /*! Set normal flux to be normal * f_r */
  void right_flux(hf_array<double> &f_r, hf_array<double> &norm, hf_array<double> &fn, int n_dims, int n_fields, double gamma);

  /*! Compute common inviscid flux using Rusanov flux */
  void rusanov_flux(hf_array<double> &u_l, hf_array<double> &u_r, hf_array<double> &f_l, hf_array<double> &f_r, hf_array<double> &norm, hf_array<double> &fn, int n_dims, int n_fields, double gamma);

  /*! Compute common inviscid flux using Roe flux */
  void roe_flux(hf_array<double> &u_l, hf_array<double> &u_r, hf_array<double> &norm, hf_array<double> &fn, int n_dims, int n_fields, double gamma);

  void hllc_flux(hf_array<double> &u_l, hf_array<double> &u_r, hf_array<double> &f_l, hf_array<double> &f_r, hf_array<double> &norm, hf_array<double> &fn, int n_dims, int n_fields, double gamma);

  /*! Compute common inviscid flux using Lax-Friedrich flux (works only for wave equation) */
  void lax_friedrich(hf_array<double> &u_l, hf_array<double> &u_r, hf_array<double> &norm, hf_array<double> &fn, int n_dims, int n_fields, double lambda, hf_array<double>& wave_speed);

  /*! Compute common viscous flux using LDG formulation */
  void ldg_flux(int flux_spec, hf_array<double> &u_l, hf_array<double> &u_r, hf_array<double> &f_l, hf_array<double> &f_r, hf_array<double> &norm, hf_array<double> &fn, int n_dims, int n_fields, double ldg_tau, double ldg_beta);

  /*! Compute common solution using LDG formulation */
  void ldg_solution(int flux_spec, hf_array<double> &u_l, hf_array<double> &u_r, hf_array<double> &u_c, double ldg_beta, hf_array<double>& norm);

	/*! get look up table for flux point connectivity based on rotation tag */
	void get_lut(int in_rot_tag);

	protected:

	// #### members ####

	int inters_type; // segment, quad or tri

	int order;
	int viscous;
	int LES;
	int n_inters;
	int n_fpts_per_inter;
	int n_fields;
	int n_dims;

	hf_array<double*> disu_fpts_l;
	hf_array<double*> delta_disu_fpts_l;
	hf_array<double*> norm_tconf_fpts_l;
	hf_array<double*> detjac_fpts_l;
	hf_array<double*> tdA_fpts_l;
	hf_array<double*> norm_fpts;
	hf_array<double*> pos_fpts;

  hf_array<double> pos_disu_fpts_l;
  hf_array<double*> grad_disu_fpts_l;
  hf_array<double*> normal_disu_fpts_l;

  hf_array<double> temp_u_l;
  hf_array<double> temp_u_r;

  hf_array<double> temp_grad_u_l;
  hf_array<double> temp_grad_u_r;

  hf_array<double> temp_f_l;
  hf_array<double> temp_f_r;

  hf_array<double> temp_f;
  hf_array<double> temp_loc;

	// LES and wall model quantities
	hf_array<double*> sgsf_fpts_l;
	hf_array<double*> sgsf_fpts_r;
	hf_array<double> temp_sgsf_l;
	hf_array<double> temp_sgsf_r;

  hf_array<int> lut;




};

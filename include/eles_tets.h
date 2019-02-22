/*!
 * \file eles_tets.h
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

#include "eles.h"
#include "hf_array.h"

class eles_tets: public eles
{
public:

  // #### constructors ####

  // default constructor

  eles_tets();

  // #### methods ####

  /*! set shape */
  //void set_shape(int in_s_order);

  void set_connectivity_plot();

  /*! set location of solution points */
  void set_loc_upts(void);

  /*! set location of flux points */
  void set_tloc_fpts(void);

  /*! set location and weight of interface cubature points */
  void set_inters_cubpts(void);

  /*! set location and weight of volume cubature points */
  void set_volume_cubpts(void);

  /*! set location of plot points */
  void set_loc_ppts(void);

  /*! set location of shape points */
  //void set_loc_spts(void);

  /*! set transformed normals at flux points */
  void set_tnorm_fpts(void);

  //#### helper methods ####

  void setup_ele_type_specific(void);

  /*! read restart info */
  int read_restart_info_ascii(ifstream& restart_file);
#ifdef _HDF5
  void read_restart_info_hdf5(hid_t &restart_file, int in_rest_order);
#endif

  /*! write restart info */
  void write_restart_info_ascii(ofstream& restart_file);
#ifdef _HDF5
  void write_restart_info_hdf5(hid_t &restart_file);
#endif

  /*! Compute interface jacobian determinant on face */
  double compute_inter_detjac_inters_cubpts(int in_inter, hf_array<double> d_pos);

  /*! evaluate nodal basis */
  double eval_nodal_basis(int in_index, hf_array<double> in_loc);

  /*! evaluate nodal basis */
  double eval_nodal_basis_restart(int in_index, hf_array<double> in_loc);

  /*! evaluate derivative of nodal basis */
  double eval_d_nodal_basis(int in_index, int in_cpnt, hf_array<double> in_loc);

  /*! evaluate divergence of vcjh basis */
  double eval_div_vcjh_basis(int in_index, hf_array<double>& loc);

  void fill_opp_3(hf_array<double>& opp_3);

  void get_opp_3_dg_tet(hf_array<double>& opp_3_dg);

  double eval_div_dg_tet(int in_index, hf_array<double>& loc);

  void compute_filt_matrix_tet(hf_array<double>& Filt, int vcjh_scheme_tet, double c_tet);

  /*! evaluate nodal shape basis */
  double eval_nodal_s_basis(int in_index, hf_array<double> in_loc, int in_n_spts);

  /*! evaluate derivative of nodal shape basis */
  void eval_d_nodal_s_basis(hf_array<double> &d_nodal_s_basis, hf_array<double> in_loc, int in_n_spts);

  /*! Calculate element volume */
  double calc_ele_vol(double& detjac);

  /*! Element reference length calculation */
  double calc_h_ref_specific(int in_ele);

  
  int calc_p2c(hf_array<double>& in_pos);

protected:

  // methods

  /*! set triangle Vandermonde matrix */
  void set_vandermonde();

  /*! set restart triangle Vandermonde matrix */
  void set_vandermonde_restart();

  void set_vandermonde_vol_cub(void);

  void shock_det_persson(void);
  
  /*! Compute the filter matrix for subgrid-scale models */
  void compute_filter_upts(void);

  /*! set exponential filter */
  void set_exp_filter(void);
  
  /*! set over-integration filter array */
  void set_over_int_filter(void);
  
  // members
  hf_array<double> vandermonde_rest;
  hf_array<double> inv_vandermonde_rest;

};

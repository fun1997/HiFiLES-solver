/*!
 * \file eles_pris.h
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

#include "eles.h"

class eles_pris: public eles
{
public:

  // #### constructors ####

  // default constructor

  eles_pris();

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
  void set_volume_cubpts(int in_order, hf_array<double> &out_loc_volume_cubpts, hf_array<double> &out_weight_volume_cubpts);

  /*! set location of plot points */
  void set_loc_ppts(void);

  /*! set location of shape points */
  void set_loc_spts(void);

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
#ifdef _HDF5
  void write_restart_info_hdf5(hid_t &restart_file);
#else
  void write_restart_info_ascii(ofstream& restart_file);
#endif
  /*! Compute interface jacobian determinant on face */
  double compute_inter_detjac_inters_cubpts(int in_inter, hf_array<double> d_pos);

  /*! evaluate nodal basis */
  double eval_nodal_basis(int in_index, hf_array<double> in_loc);

  /*! evaluate nodal basis for restart file*/
  double eval_nodal_basis_restart(int in_index, hf_array<double> in_loc);

  /*! evaluate derivative of nodal basis */
  double eval_d_nodal_basis(int in_index, int in_cpnt, hf_array<double> in_loc);

  /*! evaluate divergence of vcjh basis */
  //double eval_div_vcjh_basis(int in_index, hf_array<double>& loc);

  void fill_opp_3(hf_array<double>& opp_3);

  /*! evaluate nodal shape basis */
  double eval_nodal_s_basis(int in_index, hf_array<double> in_loc, int in_n_spts);

  /*! evaluate derivative of nodal shape basis */
  void eval_d_nodal_s_basis(hf_array<double> &d_nodal_s_basis, hf_array<double> in_loc, int in_n_spts);

  /*! Calculate element volume */
  double calc_ele_vol(double& detjac);

  int face0_map(int index);

  /*! Element reference length calculation */
  double calc_h_ref_specific(int in_ele);
  
  int calc_p2c(hf_array<double>& in_pos);

protected:

  //methods

  /*! set triangle Vandermonde matrix */
  void set_vandermonde_tri();

  /*! set restart triangle Vandermonde matrix */
  void set_vandermonde_tri_restart();

  void set_vandermonde3D(void);

  void calc_norm_basis(void);
  void shock_det_persson(void);
  
  /*! set exponential filter */
  void set_exp_filter(void);

  /*! set over-integration filter array */
  void set_over_int(void);

  /*! Evaluate prismatic Basis */
  double eval_pris_basis_hierarchical(int, hf_array<double>, int in_order);
  int get_pris_basis_index(int in_mode,int in_order,int &out_r,int &out_s,int &out_t);

  // members
  int n_upts_tri;
  int n_upts_1d;

  int n_upts_tri_rest;
  int n_upts_1d_rest;

  int upts_type_pri_tri;
  int upts_type_pri_1d;

  hf_array<double> loc_upts_pri_tri;
  hf_array<double> loc_upts_pri_1d;
  hf_array<double> loc_1d_fpts;

  hf_array<double> loc_upts_pri_tri_rest;
  hf_array<double> loc_upts_pri_1d_rest;

  hf_array<double> vandermonde_tri;
  hf_array<double> inv_vandermonde_tri;
  hf_array<double> vandermonde_tri_rest;
  hf_array<double> inv_vandermonde_tri_rest;

  hf_array<double> norm_basis_persson;

  /*! element edge lengths for h_ref calculation */
  hf_array<double> length;
};

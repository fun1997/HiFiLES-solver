/*!
 * \file eles_hexas.h
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

class eles_hexas: public eles
{
public:

  // #### constructors ####

  // default constructor

  eles_hexas();

  // #### methods ####

  /*! set shape */
  //void set_shape(hf_array<int> &in_n_spts_per_ele);

  void set_connectivity_plot();

  /*! set location of 1d solution points in standard interval (required for tensor product elements)*/
  void set_loc_1d_upts(void);

  /*! set location of 1d shape points in standard interval (required for tensor product elements)*/
  void set_loc_1d_spts(hf_array<double> &loc_1d_spts, int in_n_1d_spts);

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

  /*! evaluate nodal basis */
  double eval_nodal_basis_restart(int in_index, hf_array<double> in_loc);

  /*! evaluate derivative of nodal basis */
  double eval_d_nodal_basis(int in_index, int in_cpnt, hf_array<double> in_loc);

  /*! evaluate divergence of vcjh basis */
  double eval_div_vcjh_basis(int in_index, hf_array<double>& loc);

  void fill_opp_3(hf_array<double>& opp_3);

  /*! evaluate nodal shape basis */
  double eval_nodal_s_basis(int in_index, hf_array<double> in_loc, int in_n_spts);

  /*! evaluate derivative of nodal shape basis */
  void eval_d_nodal_s_basis(hf_array<double> &d_nodal_s_basis, hf_array<double> in_loc, int in_n_spts);

  /*! Compute the filter matrix for subgrid-scale models */
  void compute_filter_upts(void);

  /*! Calculate element volume */
  double calc_ele_vol(double& detjac);

  /*! Element reference length calculation */
  double calc_h_ref_specific(int in_ele);

  int calc_p2c(hf_array<double>& in_pos);

protected:

  /*! set Vandermonde matrix */
  void set_vandermonde1D(void);

  /*! set 3D Vandermonde matrix */
  void set_vandermonde3D(void);

  void set_vandermonde_vol_cub(void);

  void calc_norm_basis(void);
  void shock_det_persson(void);
  
  /*! setup the concentration hf_array required for concentration method for shock capturing */
  void set_concentration_array(void);

  /*! set filter hf_array */
  void set_exp_filter(void);

  /*! set over-integration filter array */
  void set_over_int_filter(void);

  /*! Evaluate 3D Legendre Basis */
  double eval_legendre_basis_3D_hierarchical(int, hf_array<double>, int in_order);//in basis_order for error handling
  int get_legendre_basis_3D_index(int in_mode, int in_basis_order,int &out_i,int &out_j,int &out_k);

  // members

  /*! return position of 1d solution point */
  double get_loc_1d_upt(int in_index);

  /*! location of solution points in standard interval (tensor product elements only)*/
  hf_array<double> loc_1d_upts;

  /*! location of solution points in standard interval (tensor product elements only)*/
  hf_array<double> loc_1d_upts_rest;

  hf_array<double> vandermonde1D;
  hf_array<double> inv_vandermonde1D;
  
  hf_array<double> norm_basis_persson;

  /*! Matrix of filter weights at solution points in 1D */
  hf_array<double> filter_upts_1D;
  
  /*! element edge lengths for h_ref calculation */
  hf_array<double> length;
};

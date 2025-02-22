/*!
 * \file eles.h
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

#include "global.h"
#if defined _ACCELERATE_BLAS
#include <Accelerate/Accelerate.h>
#elif defined _MKL_BLAS
#include "mkl.h"
#elif defined _STANDARD_BLAS
extern "C"
{
#include "cblas.h"
}
#else
#include <numeric>
#endif

#ifdef _HDF5
#include "hdf5.h"
#endif

#if defined _GPU
#include "cuda_runtime_api.h"
#include "cusparse_v2.h"
#endif

class eles
{
public:

  // #### constructors ####

  // default constructor

  eles();

  // default destructor

  ~eles();

  // #### methods ####

  /*! setup */
  void setup(int in_n_eles, int in_max_s_spts_per_ele);

  /*! setup initial conditions */
  void set_ics(double& time);

  /*! setup patch */
  void set_patch(void);

  /*! read data from restart file */
  void read_restart_data_ascii(ifstream& restart_file);
#ifdef _HDF5
  void read_restart_data_hdf5(hid_t &restart_file);
#endif

  /*! write data to restart file */
#ifdef _HDF5
  void write_restart_data_hdf5(hid_t &in_dataset_id);
#else
  void write_restart_data_ascii(ofstream &restart_file);
#endif

  /*! calculate the discontinuous solution at the flux points */
  void extrapolate_solution(void);

  /*! Calculate terms for some LES models */
  void calc_sgs_terms(void);

  /*! calculate transformed discontinuous inviscid flux at solution points */
  void evaluate_invFlux(void);
  void evaluate_invFlux_over_int(void);
  
  /*! calculate divergence of transformed discontinuous flux at solution points */
  void calculate_divergence(void);

  /*! calculate normal transformed discontinuous flux at flux points */
  void extrapolate_totalFlux(void);

  /*! calculate subgrid-scale flux at flux points */
  void extrapolate_sgsFlux(void);

  /*! calculate divergence of transformed continuous flux at solution points */
  void calculate_corrected_divergence(void);

  /*! calculate uncorrected transformed gradient of the discontinuous solution at the solution points */
  void calculate_gradient(void);

  /*! calculate corrected gradient of the discontinuous solution at solution points */
  void correct_gradient(void);

  /*! calculate transformed discontinuous viscous flux at solution points */
  void evaluate_viscFlux(void);

  /*! calculate source term for SA turbulence model at solution points */
  void calc_src_upts_SA(void);

  /*! advance solution using a runge-kutta scheme */
  void AdvanceSolution(int in_step, int adv_type);

  /*! Calculate element local timestep */
  double calc_dt_local(int in_ele);

  /*! get number of elements */
  int get_n_eles(void);

  // get number of ppts_per_ele
  int get_n_ppts_per_ele(void);

  // get number of peles_per_ele
  int get_n_peles_per_ele(void);

  // get number of verts_per_ele
  int get_n_verts_per_ele(void);

  /*! get number of solution points per element */
  int get_n_upts_per_ele(void);

  /*! get number of shape points per element */
  int get_n_spts_per_ele(int in_ele);

  /*! get element type */
  int get_ele_type(void);

  /*! get number of dimensions */
  int get_n_dims(void);

  /*! get number of fields */
  int get_n_fields(void);

  /*! get shape point coordinates*/
  double get_shape(int in_dim, int in_spt, int in_ele);

  /*! set shape */
  void set_shape(int in_max_n_spts_per_ele);

  /*! set shape node */
  void set_shape_node(int in_spt, int in_ele, hf_array<double>& in_pos);

  /*! set bc id */
  void set_bcid(int in_ele, int in_inter, int in_bcid);

  /*! set bc type */
  void set_bdy_ele2ele(void);

  /*! set bc type */
  void set_ele2bdy_ele(void);

  /*! set number of shape points */
  void set_n_spts(int in_ele, int in_n_spts);

  /*!  set global element number */
  void set_ele2global_ele(int in_ele, int in_global_ele);

  /*! get a pointer to the transformed discontinuous solution at a flux point */
  double* get_disu_fpts_ptr(int in_inter_local_fpt, int in_ele_local_inter, int in_field, int in_ele);

  /*! get a pointer to the normal transformed continuous flux at a flux point */
  double* get_norm_tconf_fpts_ptr(int in_inter_local_fpt, int in_ele_local_inter, int in_field, int in_ele);

  /*! get a pointer to the determinant of the jacobian at a flux point (static->computational) */
  double* get_detjac_fpts_ptr(int in_inter_local_fpt, int in_ele_local_inter, int in_ele);

  /*! get pointer to the equivalent of 'dA' (face area) at a flux point in static physical space */
  double* get_tdA_fpts_ptr(int in_inter_local_fpt, int in_ele_local_inter, int in_ele);

  /*! get pointer to the weight at a flux point  */
  double* get_weight_fpts_ptr(int in_inter_local_fpt, int in_ele_local_inter);

  /*! get pointer to the inter_detjac_inters_cubpts at a flux point  */
  double* get_inter_detjac_inters_cubpts_ptr(int in_inter_local_fpt, int in_ele_local_inter, int in_ele);

  /*! get a pointer to the normal at a flux point */
  double* get_norm_fpts_ptr(int in_inter_local_fpt, int in_ele_local_inter, int in_dim, int in_ele);

  /*! get a CPU pointer to the coordinates at a flux point */
  double* get_loc_fpts_ptr_cpu(int in_inter_local_fpt, int in_ele_local_inter, int in_dim, int in_ele);

#ifdef _GPU
  /*! get a GPU pointer to the coordinates at a flux point */
  double* get_loc_fpts_ptr_gpu(int in_inter_local_fpt, int in_ele_local_inter, int in_dim, int in_ele);
#endif

  /*! get a pointer to delta of the transformed discontinuous solution at a flux point */
  double* get_delta_disu_fpts_ptr(int in_inter_local_fpt, int in_ele_local_inter, int in_field, int in_ele);

  /*! get a pointer to gradient of discontinuous solution at a flux point */
  double* get_grad_disu_fpts_ptr(int in_inter_local_fpt, int in_ele_local_inter, int in_dim, int in_field, int in_ele);

  /*! get pointer to solution of wall model input solution point */
  double *get_wm_disu_ptr(int in_ele, int in_upt, int in_field);

  /*! calculate the farthest solution point to the interface and return the distance */
  double calc_wm_upts_dist(int in_ele,int in_local_inter,int &out_upt);

  /*! get a pointer to the subgrid-scale flux at a flux point */
  double* get_sgsf_fpts_ptr(int in_inter_local_fpt, int in_ele_local_inter, int in_field, int in_dim, int in_ele);

  /*! set opp_0 */
  void set_opp_0(int in_sparse);

  /*! set opp_1 */
  void set_opp_1(int in_sparse);

  /*! set opp_2 */
  void set_opp_2(int in_sparse);

  /*! set opp_3 */
  void set_opp_3(int in_sparse);

  /*! set opp_4 */
  void set_opp_4(int in_sparse);

  /*! set opp_5 */
  void set_opp_5(int in_sparse);

  /*! set opp_6 */
  void set_opp_6(int in_sparse);

  /*! set opp_p */
  void set_opp_p(void);

  /*! set opp_probe */
  void set_opp_probe(hf_array<double>& in_loc);

  /*! set opp_inters_cubpts */
  void set_opp_inters_cubpts(void);

  /*! set opp_volume_cubpts */
  void set_opp_volume_cubpts(hf_array<double> &in_loc_volume_cubpts,hf_array<double> &out_opp_volume_cubpts);

  /*! set opp_r */
  void set_opp_r(void);

  /*! calculate position of the plot points */
  void calc_pos_ppts(int in_ele, hf_array<double>& out_pos_ppts);

  void set_rank(int in_rank);

  hf_array<int> get_connectivity_plot();

  /*! calculate solution at the probe points */
  void calc_disu_probepoints(int in_ele, hf_array<double>& out_disu_probepoints);

  /*! calculate solution at the plot points */
  void calc_disu_ppts(int in_ele, hf_array<double>& out_disu_ppts);

  /*! calculate gradient of solution at the plot points */
  void calc_grad_disu_ppts(int in_ele, hf_array<double>& out_grad_disu_ppts);

  /*! calculate sensor at the plot points */
  void calc_sensor_ppts(int in_ele, hf_array<double>& out_sensor_ppts);

  /*! calculate time-averaged diagnostic fields at the plot points */
  void calc_time_average_ppts(int in_ele, hf_array<double>& out_disu_average_ppts);

  /*! calculate diagnostic fields at the plot points */
  void calc_diagnostic_fields_ppts(int in_ele, hf_array<double>& in_disu_ppts, hf_array<double>& in_grad_disu_ppts, hf_array<double>& in_sensor_ppts, hf_array<double>& out_diag_field_ppts, double& time);

  /*! returns position of a solution point */
  double get_loc_upt(int in_upt, int in_dim);

  /*! set transforms */
  void set_transforms(void);

  /*! set transformation at solution points*/
  void set_transforms_upts(void);

  /*! set transformation at flux points*/
  void set_transforms_fpts(void);

  /*! set transforms at the interface cubature points */
  void set_transforms_inters_cubpts(void);

  /*! set transforms at the volume cubature points */
  void set_transforms_vol_cubpts(void);
  void set_transforms_over_int_cubtps(void);
  
	/*! Calculate distance of solution points to no-slip wall */
	void calc_wall_distance(const int n_seg_noslip_inters, const int n_tri_noslip_inters, const int n_quad_noslip_inters, hf_array< hf_array<double> > &loc_noslip_bdy);

  /*! calculate position */
  void calc_pos(hf_array<double> in_loc, int in_ele, hf_array<double>& out_pos);

  /*! calculate derivative of position */
  void calc_d_pos(hf_array<double> in_loc, int in_ele, hf_array<double>& out_d_pos);

  /*! calculate derivative of position at a solution point (using pre-computed gradients) */
  void calc_d_pos_upt(int in_upt, int in_ele, hf_array<double>& out_d_pos);

  /*! calculate derivative of position at a flux point (using pre-computed gradients) */
  void calc_d_pos_fpt(int in_fpt, int in_ele, hf_array<double>& out_d_pos);

  /*! iteratively calculate location in reference domain from position in physical domain*/
  void pos_to_loc(hf_array<double>& in_pos,int in_ele,hf_array<double>& out_loc);

  /*! Calculate SGS flux */
  void calc_sgsf_upts(hf_array<double>& temp_u, hf_array<double>& temp_grad_u, double& detjac, int ele, int upt, hf_array<double>& temp_sgsf);

  /*! rotate velocity components to surface*/
  hf_array<double> calc_rotation_matrix(hf_array<double>& norm);

  /*! calculate wall shear stress using LES wall model*/
  void calc_wall_stress(double rho, hf_array<double>& urot, double ene, double mu, double Pr, double gamma, double y, hf_array<double>& tau_wall, double q_wall);

  /*! Wall function calculator for Breuer-Rodi wall model */
  double wallfn_br(double yplus, double A, double B, double E, double kappa);

  double compute_res_upts(int in_norm_type, int in_field);

  /*! calculate body forcing at solution points */
  void evaluate_body_force(int in_file_num);

  /*! Compute volume integral of diagnostic quantities */
  void CalcIntegralQuantities(int n_integral_quantities, hf_array <double>& integral_quantities);

  /*! Compute time-average diagnostic quantities */
  void CalcTimeAverageQuantities(double& time);

  void compute_wall_forces(hf_array<double>& inv_force, hf_array<double>& vis_force, double& temp_cl, double& temp_cd, ofstream& coeff_file, bool write_forces);

  hf_array<double> compute_error(int in_norm_type, double& time);

  hf_array<double> get_pointwise_error(hf_array<double>& sol, hf_array<double>& grad_sol, hf_array<double>& loc, double& time, int in_norm_type);

  //calculate cut off length scale
  double calc_inlet_length_scale();


 //------------------------
// virtual methods
//------------------------
  virtual void setup_ele_type_specific()=0;

  /*! prototype for element reference length calculation */
  virtual double calc_h_ref_specific(int in_eles) = 0;

  virtual int read_restart_info_ascii(ifstream& restart_file) = 0;

#ifdef _HDF5
  virtual void read_restart_info_hdf5(hid_t &restart_file, int in_rest_order) = 0;
#endif

#ifdef _HDF5
  virtual void write_restart_info_hdf5(hid_t &restart_file) = 0;
#else
  virtual void write_restart_info_ascii(ofstream &restart_file) = 0;
#endif

  /*! Compute interface jacobian determinant on face */
  virtual double compute_inter_detjac_inters_cubpts(int in_inter, hf_array<double> d_pos)=0;

  /*! evaluate nodal basis */
  virtual double eval_nodal_basis(int in_index, hf_array<double> in_loc)=0;

  /*! evaluate nodal basis for restart file*/
  virtual double eval_nodal_basis_restart(int in_index, hf_array<double> in_loc)=0;

  /*! evaluate derivative of nodal basis */
  virtual double eval_d_nodal_basis(int in_index, int in_cpnt, hf_array<double> in_loc)=0;

  virtual void fill_opp_3(hf_array<double>& opp_3)=0;

  /*! evaluate divergence of vcjh basis */
  //virtual double eval_div_vcjh_basis(int in_index, hf_array<double>& loc)=0;

  /*! evaluate nodal shape basis */
  virtual double eval_nodal_s_basis(int in_index, hf_array<double> in_loc, int in_n_spts)=0;

  /*! evaluate derivative of nodal shape basis */
  virtual void eval_d_nodal_s_basis(hf_array<double> &d_nodal_s_basis, hf_array<double> in_loc, int in_n_spts)=0;

  /*! Calculate element volume */
  virtual double calc_ele_vol(double& detjac)=0;

  /*! calculate which cell probe points in */
  virtual int calc_p2c(hf_array<double>& in_pos)=0;

  virtual void set_connectivity_plot()=0;

//--------------------
//GPU functions
//--------------------
#ifdef _GPU

	/*! move all to from cpu to gpu */
	void mv_all_cpu_gpu(void);

	/*! move wall distance hf_array to from cpu to gpu */
	void mv_wall_distance_cpu_gpu(void);

  /*! move wall distance magnitude hf_array to from cpu to gpu */
  void mv_wall_distance_mag_cpu_gpu(void);

	/*! copy transformed discontinuous solution at solution points to cpu */
	void cp_disu_upts_gpu_cpu(void);

	/*! copy transformed discontinuous solution at solution points to gpu */
  void cp_disu_upts_cpu_gpu(void);

  void cp_grad_disu_upts_gpu_cpu(void);

  /*! copy determinant of jacobian at solution points to cpu */
  void cp_detjac_upts_gpu_cpu(void);

  /*! copy divergence at solution points to cpu */
  void cp_div_tconf_upts_gpu_cpu(void);

  /*! copy local time stepping reference length at solution points to cpu */
  void cp_h_ref_gpu_cpu(void);

  /*! copy source term at solution points to cpu */
  void cp_src_upts_gpu_cpu(void);

  /*! copy elemental sensor values to cpu */
  void cp_sensor_gpu_cpu(void);

  /*! remove transformed discontinuous solution at solution points from cpu */
  void rm_disu_upts_cpu(void);

  /*! remove determinant of jacobian at solution points from cpu */
  void rm_detjac_upts_cpu(void);

#endif

//---------------------------------------
//Shock capturing/de-aliasing functions
//---------------------------------------

  void shock_capture(void);

  /*! element local timestep */
  hf_array<double> dt_local;
  
protected:
  // #### methods ####

  /*! methods to detect the shock*/
  virtual void shock_det_persson()=0;

  // #### members ####

  /*! viscous flag */
  int viscous;

  /*! LES flag */
  int LES;

  /*! SGS model */
  int sgs_model;

  /*! LES filter flag */
  int LES_filter;

  /*! number of elements */
  int n_eles;

  /*! number of elements that have a boundary face*/
  int n_bdy_eles;

  /*!  number of dimensions */
  int n_dims;

  /*!  number of prognostic fields */
  int n_fields;

  /*!  number of diagnostic fields */
  int n_diagnostic_fields;

  /*!  number of time averaged diagnostic fields */
  int n_average_fields;

  /*! order of solution polynomials */
  int order;

  /*! order of solution polynomials in restart file*/
  int order_rest;

  /*! number of solution points per element */
  int n_upts_per_ele;

  /*! number of solution points per element */
  int n_upts_per_ele_rest;

  /*! number of flux points per element */
  int n_fpts_per_ele;

  /*! number of vertices per element */
  int n_verts_per_ele;

  hf_array<int> connectivity_plot;

  /*! plotting resolution */
  int p_res;

  /*! solution point type */
  int upts_type;

  /*! flux point type */
  int fpts_type;

  /*! number of plot points per element */
  int n_ppts_per_ele;

  /*! number of plot elements per element */
  int n_peles_per_ele;

  /*! Global cell number of element */
  hf_array<int> ele2global_ele;

  /*! Global cell number of element */
  hf_array<int> bdy_ele2ele;

  /*! Global cell number of element */
  hf_array<int> ele2bdy_ele;

  /*! Boundary condition type of faces */
  hf_array<int> bcid;

  /*! number of shape points per element */
  hf_array<int> n_spts_per_ele;

  /*! transformed normal at flux points */
  hf_array<double> tnorm_fpts;

  /*! transformed normal at flux points */
  hf_array< hf_array<double> > tnorm_inters_cubpts;

  /*! location of solution points in standard element */
  hf_array<double> loc_upts;

  /*! location of solution points in standard element */
  hf_array<double> loc_upts_rest;

  /*! location of flux points in standard element */
  hf_array<double> tloc_fpts;

  /*! location of interface cubature points in standard element */
  hf_array< hf_array<double> > loc_inters_cubpts;

  /*! weight of interface cubature points in standard element */
  hf_array< hf_array<double> > weight_inters_cubpts;

  /*! location of volume cubature points in standard element */
  hf_array<double> loc_volume_cubpts;

  /*! weight of cubature points in standard element */
  hf_array<double> weight_volume_cubpts;

  /*! transformed normal at cubature points */
	hf_array< hf_array<double> > tnorm_cubpts;

	/*! location of plot points in standard element */
	hf_array<double> loc_ppts;

	/*! location of shape points in standard element (simplex elements only)*/
	hf_array<double> loc_spts;

	/*! number of interfaces per element */
	int n_inters_per_ele;

	/*! number of flux points per interface */
	hf_array<int> n_fpts_per_inter;

	/*! number of cubature points per interface */
	hf_array<int> n_cubpts_per_inter;

	/*! element type (0=>quad,1=>tri,2=>tet,3=>pri,4=>hex) */
	int ele_type;

	/*! order of polynomials defining shapes */
	int s_order;

  /*! maximum number of shape points used by any element */
  int max_n_spts_per_ele;

  /*! position of shape points (mesh vertices) in static-physical domain */
	hf_array<double> shape;

  /*! nodal shape basis contributions at flux points */
  hf_array<double> nodal_s_basis_fpts;

  /*! nodal shape basis contributions at solution points */
  hf_array<double> nodal_s_basis_upts;

  /*! nodal shape basis contributions at output plot points */
  hf_array<double> nodal_s_basis_ppts;

  /*! nodal shape basis contributions at output plot points */
  hf_array<double> nodal_s_basis_vol_cubpts;

  /*! nodal shape basis contributions at output plot points */
  hf_array<hf_array<double> > nodal_s_basis_inters_cubpts;

  /*! nodal shape basis derivative contributions at flux points */
  hf_array<double> d_nodal_s_basis_fpts;

  /*! nodal shape basis derivative contributions at solution points */
  hf_array<double> d_nodal_s_basis_upts;

  /*! nodal shape basis contributions at output plot points */
  hf_array<double> d_nodal_s_basis_vol_cubpts;

  /*! nodal shape basis contributions at output plot points */
  hf_array<hf_array<double> > d_nodal_s_basis_inters_cubpts;

	/*! temporary solution storage at a single solution point */
	hf_array<double> temp_u;

	/*! temporary solution gradient storage */
	hf_array<double> temp_grad_u;

	/*! Matrix of filter weights at solution points */
	hf_array<double> filter_upts;

	/*! extra arrays for similarity model: Leonard tensors, velocity/energy products */
	hf_array<double> Lu, Le, uu, ue;

	/*! temporary flux storage */
	hf_array<double> temp_f;

	/*! temporary subgrid-scale flux storage */
	hf_array<double> temp_sgsf;

	/*! storage for distance of solution points to nearest no-slip boundary */
	hf_array<double> wall_distance;
  hf_array<double> wall_distance_mag;

	hf_array<double> twall;

	/*! number of storage levels for time-integration scheme */
	int n_adv_levels;

  /*! determinant of Jacobian (transformation matrix) at solution points
   *  (J = |G|) */
	hf_array<double> detjac_upts;

  /*! determinant of Jacobian (transformation matrix) at flux points
   *  (J = |G|) */
	hf_array<double> detjac_fpts;

  /*! determinant of jacobian at volume cubature points. TODO: what is this really? */
	hf_array< hf_array<double> > vol_detjac_inters_cubpts;

	/*! determinant of volume jacobian at cubature points. TODO: what is this really? */
	hf_array< hf_array<double> > vol_detjac_vol_cubpts;

  hf_array<double> Jacobian_fpts;
  
  /*! Full vector-transform matrix from static physical->computational frame, at solution points
   *  [Determinant of Jacobian times inverse of Jacobian] [J*G^-1] */
  hf_array<double> JGinv_upts;

  /*! Full vector-transform matrix from static physical->computational frame, at flux points
   *  [Determinant of Jacobian times inverse of Jacobian] [J*G^-1] */
  hf_array<double> JGinv_fpts;

  /*! Magnitude of transformed face-area normal vector from computational -> static-physical frame
   *  [magntiude of (normal dot inverse static transformation matrix)] [ |J*(G^-1)*(n*dA)| ] */
  hf_array<double> tdA_fpts;

	/*! determinant of interface jacobian at flux points */
	hf_array< hf_array<double> > inter_detjac_inters_cubpts;

	/*! normal at flux points*/
	hf_array<double> norm_fpts;

  /*! static-physical coordinates at flux points*/
  hf_array<double> pos_fpts;

  /*! static-physical coordinates at solution points*/
  hf_array<double> pos_upts;

  /*! normal at interface cubature points*/
  hf_array< hf_array<double> > norm_inters_cubpts;

  /*!
        description: transformed discontinuous solution at the solution points
        indexing: \n
        matrix mapping:
        */
  hf_array< hf_array<double> > disu_upts;
  
	/*!
	running time-averaged diagnostic fields at solution points
	*/
	hf_array<double> disu_average_upts;


	/*!
	filtered solution at solution points for similarity and SVV LES models
	*/
	hf_array<double> disuf_upts;

  /*! position at the plot points */
  hf_array< hf_array<double> > pos_ppts;

	/*!
	description: transformed discontinuous solution at the flux points \n
	indexing: (in_fpt, in_field, in_ele) \n
	matrix mapping: (in_fpt || in_field, in_ele)
	*/
	hf_array<double> disu_fpts;

	/*!
	description: transformed discontinuous flux at the solution points \n
	indexing: (in_upt, in_dim, in_field, in_ele) \n
	matrix mapping: (in_upt, in_dim || in_field, in_ele)
	*/
	hf_array<double> tdisf_upts;

	/*!
	description: subgrid-scale flux at the solution points \n
	indexing: (in_upt, in_dim, in_field, in_ele) \n
	matrix mapping: (in_upt, in_dim || in_field, in_ele)
	*/
	hf_array<double> sgsf_upts;

	/*!
	description: subgrid-scale flux at the flux points \n
	indexing: (in_fpt, in_dim, in_field, in_ele) \n
	matrix mapping: (in_fpt, in_dim || in_field, in_ele)
	*/
	hf_array<double> sgsf_fpts;

	/*!
	normal transformed discontinuous flux at the flux points
	indexing: \n
	matrix mapping:
	*/
	hf_array<double> norm_tdisf_fpts;

	/*!
	normal transformed continuous flux at the flux points
	indexing: \n
	matrix mapping:
	*/
	hf_array<double> norm_tconf_fpts;

	/*!
	divergence of transformed continuous flux at the solution points
	indexing: \n
	matrix mapping:
	*/
	hf_array< hf_array<double> > div_tconf_upts;

	/*! delta of the transformed discontinuous solution at the flux points   */
	hf_array<double> delta_disu_fpts;

	/*! gradient of discontinuous solution at solution points */
	hf_array<double> grad_disu_upts;

	/*! gradient of discontinuous solution at flux points */
	hf_array<double> grad_disu_fpts;

  /*! source term for SA turbulence model at solution points */
  hf_array<double> src_upts;

  hf_array<double> d_nodal_s_basis;
  
#ifdef _GPU
  cusparseHandle_t handle;
#endif

  /*! operator to go from transformed discontinuous solution at the solution points to transformed discontinuous solution at the flux points */
  hf_array<double> opp_0;
  hf_array<double> opp_0_data;
  hf_array<int> opp_0_rows;
  hf_array<int> opp_0_cols;
  #if defined _MKL_BLAS
  sparse_matrix_t opp_0_mkl;
  struct matrix_descr opp_0_descr;
  #endif
  int opp_0_sparse;

#ifdef _GPU
  hf_array<double> opp_0_ell_data;
  hf_array<int> opp_0_ell_indices;
  int opp_0_nnz_per_row;
#endif

  /*! operator to go from transformed discontinuous inviscid flux at the solution points to divergence of transformed discontinuous inviscid flux at the solution points */
  hf_array<hf_array<double> > opp_1;
  hf_array<hf_array<double> > opp_1_data;
  hf_array<hf_array<int> > opp_1_rows;
  hf_array<hf_array<int> > opp_1_cols;
  #if defined _MKL_BLAS
  hf_array<sparse_matrix_t> opp_1_mkl;
  hf_array<struct matrix_descr> opp_1_descr;
#endif
  int opp_1_sparse;
#ifdef _GPU
  hf_array< hf_array<double> > opp_1_ell_data;
  hf_array< hf_array<int> > opp_1_ell_indices;
  hf_array<int> opp_1_nnz_per_row;
#endif

  /*! operator to go from transformed discontinuous inviscid flux at the solution points to normal transformed discontinuous inviscid flux at the flux points */
  hf_array< hf_array<double> > opp_2;
  hf_array< hf_array<double> > opp_2_data;
  hf_array< hf_array<int> > opp_2_rows;
  hf_array<hf_array<int> > opp_2_cols;
  #if defined _MKL_BLAS
  hf_array<sparse_matrix_t> opp_2_mkl;
  hf_array<struct matrix_descr> opp_2_descr;
#endif
  int opp_2_sparse;
#ifdef _GPU
  hf_array< hf_array<double> > opp_2_ell_data;
  hf_array< hf_array<int> > opp_2_ell_indices;
  hf_array<int> opp_2_nnz_per_row;
#endif

  /*! operator to go from normal correction inviscid flux at the flux points to divergence of correction inviscid flux at the solution points*/
  hf_array<double> opp_3;
  hf_array<double> opp_3_data;
  hf_array<int> opp_3_rows;
  hf_array<int> opp_3_cols;
#if defined _MKL_BLAS
  sparse_matrix_t opp_3_mkl;
  struct matrix_descr opp_3_descr;
#endif
  int opp_3_sparse;
#ifdef _GPU
  hf_array<double> opp_3_ell_data;
  hf_array<int> opp_3_ell_indices;
  int opp_3_nnz_per_row;
#endif

  /*! operator to go from transformed solution at solution points to transformed gradient of transformed solution at solution points */
  hf_array< hf_array<double> >  opp_4;
  hf_array< hf_array<double> >  opp_4_data;
  hf_array< hf_array<int> > opp_4_rows;
  hf_array< hf_array<int> > opp_4_cols;
#if defined _MKL_BLAS
  hf_array<sparse_matrix_t> opp_4_mkl;
  hf_array  <struct matrix_descr> opp_4_descr;
#endif
  int opp_4_sparse;
#ifdef _GPU
  hf_array< hf_array<double> > opp_4_ell_data;
  hf_array< hf_array<int> > opp_4_ell_indices;
  hf_array< int > opp_4_nnz_per_row;
#endif

  /*! operator to go from transformed solution at flux points to transformed gradient of transformed solution at solution points */
  hf_array< hf_array<double> > opp_5;
  hf_array< hf_array<double> > opp_5_data;
  hf_array< hf_array<int> > opp_5_rows;
  hf_array< hf_array<int> > opp_5_cols;
#if defined _MKL_BLAS
  hf_array<sparse_matrix_t> opp_5_mkl;
  hf_array<struct matrix_descr> opp_5_descr;
#endif
  int opp_5_sparse;
#ifdef _GPU
  hf_array< hf_array<double> > opp_5_ell_data;
  hf_array< hf_array<int> > opp_5_ell_indices;
  hf_array<int> opp_5_nnz_per_row;
#endif

  /*! operator to go from transformed solution at solution points to transformed gradient of transformed solution at flux points */
  hf_array<double> opp_6;
  hf_array<double> opp_6_data;
  hf_array<int> opp_6_rows;
  hf_array<int> opp_6_cols;
#if defined _MKL_BLAS
  sparse_matrix_t opp_6_mkl;
  struct matrix_descr opp_6_descr;
  #endif
  int opp_6_sparse;
#ifdef _GPU
  hf_array<double> opp_6_ell_data;
  hf_array<int> opp_6_ell_indices;
  int opp_6_nnz_per_row;
#endif

  /*! operator to go from discontinuous solution at the solution points to discontinuous solution at the plot points */
  hf_array<double> opp_p;
  hf_array<double> opp_probe;
  hf_array< hf_array<double> > opp_inters_cubpts;
  hf_array<double> opp_volume_cubpts;

  /*! operator to go from discontinuous solution at the restart points to discontinuous solution at the solutoin points */
  hf_array<double> opp_r;

  /*! number of fields multiplied by number of elements */
  int n_fields_mul_n_eles;

  /*! number of dimensions multiplied by number of solution points per element */
  int n_dims_mul_n_upts_per_ele;

  int rank;

  /*! mass flux through inlet */
  double mass_flux;

  /*! reference element length */
  hf_array<double> h_ref;


  /*! shock capturing/de-aliasing variables */
  hf_array<double> vandermonde;
  hf_array<double> inv_vandermonde;

  hf_array<double> exp_filter;
  hf_array<double> concentration_array;
  hf_array<double> sensor;
  hf_array<double> over_int_filter, opp_over_int_cubpts;
  hf_array<double> JGinv_over_int_cubpts;
  hf_array<double> temp_u_over_int_cubpts, temp_tdisf_over_int_cubpts;
  hf_array<double> loc_over_int_cubpts, weight_over_int_cubpts;
};

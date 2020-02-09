/*!
 * \file input.h
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

#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include "error.h"
#include "hf_array.h"
#include "bc.h"

class input
{
public:

    // #### constructors ####

    // default constructor

    input();

    ~input();

    // #### methods ####

    /*! Load input file & prepare all simulation parameters */
    void setup(char *fileNameC, int rank);

    /*! Read in parameters from file */
    void read_input_file(string fileName, int rank);

    /*! Read in boundary condition parameters from file */
    void read_boundary_param(void);

    /*! Apply non-dimensionalization and do misc. error checks */
    void setup_params(int rank);

    // #### members ####
    string fileNameS; //input file name
    /*--- basic solver parameters ---*/
    int viscous;
    int equation;
    double ldg_tau;
    double ldg_beta;
    int fix_vis;
    int order;
    int test_case;

    //parameters for time-stepping
    double time, rk_time;
    hf_array<double> RK_a, RK_b, RK_c;
    double dt;
    int dt_type;
    double CFL;
    
    int n_steps;
    string data_file_name;
    int restart_dump_freq;
    int adv_type;
    int riemann_solve_type;
    int vis_riemann_solve_type;
    int ic_form;

    //wave equation
    hf_array<double> wave_speed;
    double lambda;
    double diff_coeff;

    //gas parameter
    double gamma;
    double prandtl;
    double S_gas;
    double T_gas;
    double R_gas;
    double mu_gas;
    double c_sth;
    double mu_inf;
    double rt_inf;
    double prandtl_t;

    /* ---LES options --- */
    int LES;
    double C_s;
    int filter_type;
    double filter_ratio;
    int SGS_model;
    int wall_model;

    /* ------- restart options -------- */
    int restart_flag;
    int restart_iter;
    int n_restart_files;

    /*--- mesh parameters ---*/
    int mesh_format;
    string mesh_file;

    /* --- Shock Capturing/dealiasing options --- */
    int over_int, over_int_order;
    int shock_cap, shock_det, shock_det_field;
    double s0;
    double expf_fac;
    int expf_order,expf_cutoff;
    
    /*--- moniter and output ---*/
    int p_res;
    int write_type;
    int plot_freq;
    int n_diagnostic_fields;
    hf_array<string> diagnostic_fields;
    int n_average_fields;
    hf_array<string> average_fields;
    int n_integral_quantities;
    hf_array<string> integral_quantities;
    double spinup_time;
    int monitor_res_freq;
    int calc_force;
    int monitor_cp_freq;
    double area_ref;
    int res_norm_type; // 0:infinity norm, 1:L1 norm, 2:L2 norm
    int error_norm_type; // 0:infinity norm, 1:L1 norm, 2:L2 norm
    int probe;

    /*--- flux reconstruction parameters ---*/
    int upts_type_tri;
    int fpts_type_tri;
    int vcjh_scheme_tri;
    double c_tri;
    int sparse_tri;

    int upts_type_quad;
    int vcjh_scheme_quad;
    double eta_quad;
    double c_quad;
    int sparse_quad;

    int upts_type_hexa;
    int vcjh_scheme_hexa;
    double eta_hexa;
    int sparse_hexa;

    int upts_type_tet;
    int fpts_type_tet;
    int vcjh_scheme_tet;
    double c_tet;
    double eta_tet;
    int sparse_tet;

    int upts_type_pri_tri;
    int upts_type_pri_1d;
    int vcjh_scheme_pri_1d;
    double eta_pri;
    int sparse_pri;

    /*---- boundary_conditions ---- */
    hf_array<bc> bc_list;
    int pressure_ramp,ramp_counter;

    //cyclic interfaces
    double dx_cyclic;
    double dy_cyclic;
    double dz_cyclic;

    //public values
    double Mach_free_stream;
    double rho_free_stream;
    double L_free_stream;
    double T_free_stream;
    double u_free_stream;
    double v_free_stream;
    double w_free_stream;
    double mu_free_stream;

    /*--- reference values ---*/
    double T_ref;
    double L_ref;
    double R_ref;
    double uvw_ref;
    double rho_ref;
    double p_ref;
    double mu_ref;
    double time_ref;

    /* --- Initial Conditions ---*/
    double Mach_c_ic;
    double nx_c_ic;
    double ny_c_ic;
    double nz_c_ic;
    double Re_c_ic;
    double rho_c_ic;
    double p_c_ic;
    double T_c_ic;
    double uvw_c_ic;
    double u_c_ic;
    double v_c_ic;
    double w_c_ic;
    double mu_c_ic;
    double x_shock_ic;

    /* --- solution patch ---*/
    int patch;
    int patch_type;
    double Mv,ra,rb,xc,yc;
    double patch_x;

    /* --- SA turblence model parameters ---*/
    int RANS;
    double c_v1;
    double c_v2;
    double c_v3;
    double c_b1;
    double c_b2;
    double c_w2;
    double c_w3;
    double omega;
    double Kappa;
    double mu_tilde_c_ic;
    double mu_tilde_inf;

    double a_init, b_init;
    int bis_ind, file_lines;
    int device_num;
    int forcing;
    hf_array<double> x_coeffs;
    hf_array<double> y_coeffs;
    hf_array<double> z_coeffs;
    int perturb_ic;
};
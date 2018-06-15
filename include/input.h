/*!
 * \file input.h
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

#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include "hf_array.h"

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

    /*! Apply non-dimensionalization and do misc. error checks */
    void setup_params(int rank);

    // #### members ####

    /*--- basic solver parameters ---*/
    int viscous;
    int equation;
    double tau;
    double pen_fact;
    double fix_vis;
    double const_src;
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

    /* ---LES options --- */
    int LES;
    double C_s;
    int filter_type;
    double filter_ratio;
    int SGS_model;
    int wall_model;
    double wall_layer_t;

    /* ------- restart options -------- */
    int restart_flag;
    int restart_iter;
    int n_restart_files;
    int restart_mesh_out; // Print out separate restart file with X,Y,Z of all sol'n points?

    /*--- mesh parameters ---*/
    int mesh_format;
    string mesh_file;

    /* --- Mesh deformation options --- */
    int n_moving_bnds, motion;
    int GCL;
    int n_deform_iters;
    int mesh_output_freq;
    int mesh_output_format;
    hf_array<string> boundary_flags;
    hf_array<hf_array<double> > bound_vel_simple;
    hf_array<int> motion_type;

    /* --- Shock Capturing/dealiasing options --- */
    int over_int,N_under;
    int shock_cap,shock_det;
    double s0;
    double expf_fac,expf_order;

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
    int res_norm_type; // 0:infinity norm, 1:L1 norm, 2:L2 norm
    int error_norm_type; // 0:infinity norm, 1:L1 norm, 2:L2 norm
    int res_norm_field;
    int probe;
    string probe_file_name;

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
    double p_bound;
    hf_array<double> v_bound_sub_in_simp;
    hf_array<double> v_bound_sup_in;
    hf_array<double> v_bound_sub_in_simp2;
    hf_array<double> v_bound_sup_in2;
    hf_array<double> v_bound_sup_in3;
    hf_array<double> v_bound_far_field;

    //cyclic interfaces
    double dx_cyclic;
    double dy_cyclic;
    double dz_cyclic;

    //Sub_In_Simp
    int Sub_In_Simp;
    double Mach_sub_in_simp;
    double rho_sub_in_simp;
    double rho_bound_sub_in_simp;
    double nx_sub_in_simp;
    double ny_sub_in_simp;
    double nz_sub_in_simp;
    //Sub_In_Simp2
    int Sub_In_Simp2;
    double Mach_sub_in_simp2;
    double rho_sub_in_simp2;
    double rho_bound_sub_in_simp2;
    double nx_sub_in_simp2;
    double ny_sub_in_simp2;
    double nz_sub_in_simp2;
    //Sub_In_Char
    int Sub_In_char;
    double p_total_sub_in;
    double T_total_sub_in;
    double p_total_bound_sub_in;
    double T_total_bound_sub_in;
    double nx_sub_in_char;
    double ny_sub_in_char;
    double nz_sub_in_char;
    int pressure_ramp;
    double p_ramp_coeff;
    double T_ramp_coeff;
    double p_total_old;
    double T_total_old;
    double p_total_bound_old;
    double T_total_bound_old;
    int ramp_counter;
    //Sub_Out
    int Sub_Out;
    double p_sub_out;
    double p_bound_sub_out;
    double T_total_sub_out;
    double T_total_bound_sub_out;
    //Sup_In
    int Sup_In;
    double rho_sup_in;
    double p_sup_in;
    double Mach_sup_in;
    double rho_bound_sup_in;
    double p_bound_sup_in;
    double nx_sup_in;
    double ny_sup_in;
    double nz_sup_in;
    double T_sup_in;
    //Sup_In2
    int Sup_In2;
    double rho_sup_in2;
    double p_sup_in2;
    double Mach_sup_in2;
    double rho_bound_sup_in2;
    double p_bound_sup_in2;
    double nx_sup_in2;
    double ny_sup_in2;
    double nz_sup_in2;
    double T_sup_in2;
    //Sup_In3
    int Sup_In3;
    double rho_sup_in3;
    double p_sup_in3;
    double Mach_sup_in3;
    double rho_bound_sup_in3;
    double p_bound_sup_in3;
    double nx_sup_in3;
    double ny_sup_in3;
    double nz_sup_in3;
    double T_sup_in3;
    //Far_Field
    int Far_Field;
    double rho_far_field;
    double p_far_field;
    double Mach_far_field;
    double rho_bound_far_field;
    double p_bound_far_field;
    double nx_far_field;
    double ny_far_field;
    double nz_far_field;
    double T_far_field;

    //public values
    double Mach_free_stream;
    double rho_free_stream;
    double L_free_stream;
    double T_free_stream;
    double u_free_stream;
    double v_free_stream;
    double w_free_stream;
    double mu_free_stream;
    //wall
    double Mach_wall;
    double nx_wall;
    double ny_wall;
    double nz_wall;

    hf_array<double> v_wall;
    double uvw_wall;
    double T_wall;

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
    int patch_freq;
    double Mv,ra,rb,xc,yc;
    double patch_x;

    /* --- SA turblence model parameters ---*/
    int turb_model;
    double c_v1;
    double c_v2;
    double c_v3;
    double c_b1;
    double c_b2;
    double c_w2;
    double c_w3;
    double omega;
    double prandtl_t;
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

/*! \class fileReader
 *  \brief Simple, robust method for reading input files
 *  \author Jacob Crabill
 *  \date 4/30/2015
 */
class fileReader
{
public:
    /*! Default constructor */
    fileReader();

    fileReader(string fileName);

    /*! Default destructor */
    ~fileReader();

    /*! Set the file to be read from */
    void setFile(string fileName);

    /*! Open the file to prepare for reading simulation parameters */
    void openFile(void);

    /*! Close the file & clean up */
    void closeFile(void);

    /* === Functions to read paramters from input file === */

    /*! Read a single value from the input file; if not found, apply a default value */
    template <typename T>
    void getScalarValue(string optName, T &opt, T defaultVal);

    /*! Read a single value from the input file; if not found, throw an error and exit */
    template <typename T>
    void getScalarValue(string optName, T &opt);

    /*! Read a vector of values from the input file; if not found, apply the default value to all elements */
    template <typename T>
    void getVectorValue(string optName, vector<T> &opt, T defaultVal);

    /*! Read a vector of values from the input file; if not found, throw an error and exit */
    template <typename T>
    void getVectorValue(string optName, vector<T> &opt);

    template <typename T>
    void getVectorValue(string optName, hf_array<T> &opt);

    /*! Read a vector of values from the input file; if not found, setup vector to size 0 and continue */
    template <typename T>
    void getVectorValueOptional(string optName, vector<T> &opt);

    template <typename T>
    void getVectorValueOptional(string optName, hf_array<T> &opt);

    /*! Read in a map of type <T,U> from input file; each entry prefaced by optName */
    template <typename T, typename U>
    void getMap(string optName, map<T, U> &opt);

private:
    ifstream optFile;
    string fileName;

};

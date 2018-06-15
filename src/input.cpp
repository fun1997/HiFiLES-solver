/*!
 * \file input.cpp
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

#include <algorithm>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstring>
#include <cmath>
#include <cstdlib>
#include <string>
#include <vector>
#include <map>

#include "../include/input.h"
#include "../include/hf_array.h"
#include "../include/funcs.h"
#include "../include/global.h"

using namespace std;

// #### constructors ####

// default constructor

input::input()
{

}

input::~input()
{
}

void input::read_input_file(string fileName, int rank)
{
    fileReader opts(fileName);

    /*---initialize necessary arries to hold parameters---*/
    v_bound_sub_in_simp.setup(3);
    v_bound_sub_in_simp2.setup(3);
    v_bound_sup_in.setup(3);
    v_bound_sup_in2.setup(3);
    v_bound_sup_in3.setup(3);
    v_bound_far_field.setup(3);
    wave_speed.setup(3);
    v_wall.setup(3);

    /*
     * HiFiLES Developers - Please keep this organized!  There are
     * many parameters, so organization is the key to clarity.
     */

    /* ---- Basic Simulation Parameters ---- */

    opts.getScalarValue("equation",equation);
    opts.getScalarValue("order",order);
    opts.getScalarValue("viscous",viscous,1);
    opts.getScalarValue("mesh_file",mesh_file);
    opts.getScalarValue("ic_form",ic_form,1);
    opts.getScalarValue("test_case",test_case,0);
    opts.getScalarValue("n_steps",n_steps);
    opts.getScalarValue("restart_flag",restart_flag,0);
    if (restart_flag == 1)
    {
        opts.getScalarValue("restart_iter",restart_iter);
        opts.getScalarValue("n_restart_files",n_restart_files);
    }

    /* ---- Visualization / Monitoring / Output Parameters ---- */

    opts.getScalarValue("plot_freq",plot_freq,500);
    opts.getScalarValue("data_file_name",data_file_name,string("Mesh"));
    opts.getScalarValue("restart_dump_freq",restart_dump_freq,0);
    opts.getScalarValue("monitor_res_freq",monitor_res_freq,100);
    opts.getScalarValue("monitor_cp_freq",monitor_cp_freq,0);
    opts.getScalarValue("calc_force",calc_force,0);
    opts.getScalarValue("res_norm_type",res_norm_type,2);
    opts.getScalarValue("error_norm_type",error_norm_type,2);
    opts.getScalarValue("res_norm_field",res_norm_field,0);
    opts.getScalarValue("p_res",p_res,3);
    opts.getScalarValue("write_type",write_type,0);
    opts.getScalarValue("probe",probe,0);
    if(probe==1)
    {
        opts.getScalarValue("probe_file_name",probe_file_name);
    }
    opts.getVectorValueOptional("integral_quantities",integral_quantities);
    opts.getVectorValueOptional("diagnostic_fields",diagnostic_fields);
    opts.getVectorValueOptional("average_fields",average_fields);
    n_integral_quantities = integral_quantities.get_dim(0);
    n_diagnostic_fields = diagnostic_fields.get_dim(0);
    n_average_fields = average_fields.get_dim(0);
    //transform to lower cases
    for (int i=0; i<n_integral_quantities; i++)
    {
        std::transform(integral_quantities(i).begin(), integral_quantities(i).end(),
                       integral_quantities(i).begin(), ::tolower);
    }
    for (int i=0; i<n_diagnostic_fields; i++)
    {
        std::transform(diagnostic_fields(i).begin(), diagnostic_fields(i).end(),
                       diagnostic_fields(i).begin(), ::tolower);
    }
    for (int i=0; i<n_average_fields; i++)
    {
        std::transform(average_fields(i).begin(), average_fields(i).end(),
                       average_fields(i).begin(), ::tolower);
    }

    /* ---- Basic Solver Parameters ---- */

    opts.getScalarValue("riemann_solve_type",riemann_solve_type);
    opts.getScalarValue("vis_riemann_solve_type",vis_riemann_solve_type);
    opts.getScalarValue("adv_type",adv_type);
    opts.getScalarValue("dt_type",dt_type);
    if (dt_type == 2 && rank == 0)
    {
        cout << "!!!!!!" << endl;
        cout << "  Note: Local timestepping is still in an experimental phase,";
        cout << "  especially for viscous simulations." << endl;
        cout << "!!!!!!" << endl;
    }

    if (dt_type == 0)
    {
        opts.getScalarValue("dt",dt);
    }
    else
    {
        opts.getScalarValue("CFL",CFL);
    }

    opts.getScalarValue("tau",tau,0.);
    opts.getScalarValue("pen_fact",pen_fact,0.5);

    /* ---- Turbulence Modeling Parameters ---- */

    opts.getScalarValue("turb_model",turb_model,0);
    opts.getScalarValue("LES",LES,0);
    if (LES)
    {
        opts.getScalarValue("C_s",C_s);
        opts.getScalarValue("filter_type",filter_type);
        opts.getScalarValue("filter_ratio",filter_ratio);
        opts.getScalarValue("SGS_model",SGS_model);
        opts.getScalarValue("wall_model",wall_model);
        opts.getScalarValue("wall_layer_thickness",wall_layer_t);
    }

    /* ---- Mesh Motion Parameters ---- */

    opts.getScalarValue("motion_flag",motion,0);
    if (motion != STATIC_MESH)
    {
        opts.getScalarValue("GCL_flag",GCL,0);
        opts.getVectorValueOptional("moving_boundaries",motion_type);

        bound_vel_simple.setup(1);
        opts.getVectorValueOptional("simple_bound_velocity",bound_vel_simple(0));
        //opts.getVectorValueOptional("bound_vel_simple",bound_vel_simple);
        //      in_run_input_file >> n_moving_bnds;
        //      motion_type.setup(n_moving_bnds);
        //      bound_vel_simple.setup(n_moving_bnds);
        //      boundary_flags.setup(n_moving_bnds);
        //      for (int i=0; i<n_moving_bnds; i++) {
        //        in_run_input_file.getline(buf,BUFSIZ);
        //        in_run_input_file >> boundary_flags(i) >> motion_type(i);
        //        bound_vel_simple(i).setup(9);
        //        for (int j=0; j<9; j++) {
        //          in_run_input_file >> bound_vel_simple(i)(j);
        //          //cout << bound_vel_simple(i)(j) << " ";
        //        }
        //      }
        opts.getScalarValue("n_deform_iters",n_deform_iters);
        opts.getScalarValue("mesh_output_freq",mesh_output_freq,0);
        opts.getScalarValue("mesh_output_format",mesh_output_format,1);
        opts.getScalarValue("restart_mesh_out",restart_mesh_out,0);
    }

    /* ---- Gas Parameters ---- */

    opts.getScalarValue("gamma",gamma,1.4);
    opts.getScalarValue("prandtl",prandtl,.72);
    opts.getScalarValue("S_gas",S_gas,120.);
    opts.getScalarValue("T_gas",T_gas,291.15);
    opts.getScalarValue("R_gas",R_gas,286.9);
    opts.getScalarValue("mu_gas",mu_gas,1.827E-5);
    opts.getScalarValue("fix_vis",fix_vis,1.);

    /* ---- Boundary Conditions ---- */

    //cyclic boundary parameters
    opts.getScalarValue("dx_cyclic",dx_cyclic,(double)INFINITY);
    opts.getScalarValue("dy_cyclic",dy_cyclic,(double)INFINITY);
    opts.getScalarValue("dz_cyclic",dz_cyclic,(double)INFINITY);

    //Sub_In_Simp(use T_freestream)
    opts.getScalarValue("Sub_In_Simp",Sub_In_Simp,0);
    if (Sub_In_Simp)
    {
        opts.getScalarValue("Mach_sub_in_simp",Mach_sub_in_simp);
        opts.getScalarValue("rho_sub_in_simp",rho_sub_in_simp);
        opts.getScalarValue("nx_sub_in_simp",nx_sub_in_simp,1.);
        opts.getScalarValue("ny_sub_in_simp",ny_sub_in_simp,0.);
        opts.getScalarValue("nz_sub_in_simp",nz_sub_in_simp,0.);
    }

    //Sub_In_Simp2
    opts.getScalarValue("Sub_In_Simp2",Sub_In_Simp2,0);
    if (Sub_In_Simp2)
    {
        if (Sub_In_Simp)
        {
            opts.getScalarValue("Mach_sub_in_simp2",Mach_sub_in_simp2);
            opts.getScalarValue("rho_sub_in_simp2",rho_sub_in_simp2);
            opts.getScalarValue("nx_sub_in_simp2",nx_sub_in_simp2,1.);
            opts.getScalarValue("ny_sub_in_simp2",ny_sub_in_simp2,0.);
            opts.getScalarValue("nz_sub_in_simp2",nz_sub_in_simp2,0.);
        }
        else
        {
            FatalError("Sub_In_Simp has to be set");
        }
    }

    //Sub_In_Char
    opts.getScalarValue("Sub_In_Char", Sub_In_char,0);
    if(Sub_In_char)
    {
        opts.getScalarValue("p_total_sub_in",p_total_sub_in);
        opts.getScalarValue("T_total_sub_in",T_total_sub_in);
        opts.getScalarValue("pressure_ramp",pressure_ramp,0);
        opts.getScalarValue("nx_sub_in_char",nx_sub_in_char,1.);
        opts.getScalarValue("ny_sub_in_char",ny_sub_in_char,0.);
        opts.getScalarValue("nz_sub_in_char",nz_sub_in_char,0.);
        if (pressure_ramp)
        {
            opts.getScalarValue("p_ramp_coeff",p_ramp_coeff,0.);
            opts.getScalarValue("T_ramp_coeff",T_ramp_coeff,0.);
            opts.getScalarValue("p_total_old",p_total_old);
            opts.getScalarValue("T_total_old",T_total_old,T_free_stream);
        }
    }

    //Sub_Out
    opts.getScalarValue("Sub_Out",Sub_Out,0);
    if (Sub_Out)
    {
        opts.getScalarValue("p_sub_out",p_sub_out);
        opts.getScalarValue("T_total_sub_out",T_total_sub_out);
    }

    //Sup_In
    opts.getScalarValue("Sup_In",Sup_In,0);
    if (Sup_In)
    {
        opts.getScalarValue("p_sup_in",p_sup_in);
        opts.getScalarValue("Mach_sup_in",Mach_sup_in);
        opts.getScalarValue("nx_sup_in",nx_sup_in,1.);
        opts.getScalarValue("ny_sup_in",ny_sup_in,0.);
        opts.getScalarValue("nz_sup_in",nz_sup_in,0.);
        opts.getScalarValue("T_sup_in",T_sup_in);
    }

    //Sup_In2
    opts.getScalarValue("Sup_In2",Sup_In2,0);
    if (Sup_In2)
    {
        if (Sup_In)
        {
            opts.getScalarValue("p_sup_in2",p_sup_in2);
            opts.getScalarValue("Mach_sup_in2",Mach_sup_in2);
            opts.getScalarValue("nx_sup_in2",nx_sup_in2,1.);
            opts.getScalarValue("ny_sup_in2",ny_sup_in2,0.);
            opts.getScalarValue("nz_sup_in2",nz_sup_in2,0.);
            opts.getScalarValue("T_sup_in2",T_sup_in2);
        }
        else
        {
            FatalError("Sup_In has to be set");
        }
    }

    //Sup_In3
    opts.getScalarValue("Sup_In3",Sup_In3,0);
    if (Sup_In3)
    {
        if (Sup_In&&Sup_In2)
        {
            opts.getScalarValue("p_sup_in3",p_sup_in3);
            opts.getScalarValue("Mach_sup_in3",Mach_sup_in3);
            opts.getScalarValue("nx_sup_in3",nx_sup_in3,1.);
            opts.getScalarValue("ny_sup_in3",ny_sup_in3,0.);
            opts.getScalarValue("nz_sup_in3",nz_sup_in3,0.);
            opts.getScalarValue("T_sup_in3",T_sup_in3);
        }
        else
        {
            FatalError("Sup_In and Sup_In2 has to be set");
        }
    }

//Far_Field
    opts.getScalarValue("Far_Field",Far_Field,0);
    if (Far_Field)
    {
        opts.getScalarValue("p_far_field",p_far_field);
        opts.getScalarValue("Mach_far_field",Mach_far_field);
        opts.getScalarValue("nx_far_field",nx_far_field,1.);
        opts.getScalarValue("ny_far_field",ny_far_field,0.);
        opts.getScalarValue("nz_far_field",nz_far_field,0.);
        opts.getScalarValue("T_far_field",T_far_field);
    }

    //free_stream values are used for reference values
    opts.getScalarValue("Mach_free_stream",Mach_free_stream,1.);
    opts.getScalarValue("L_free_stream",L_free_stream,1.);
    opts.getScalarValue("T_free_stream",T_free_stream,300.);
    opts.getScalarValue("rho_free_stream",rho_free_stream,1.17723946);


    //Wall
    opts.getScalarValue("Mach_wall",Mach_wall,0.);
    opts.getScalarValue("nx_wall",nx_wall,0.);
    opts.getScalarValue("ny_wall",ny_wall,0.);
    opts.getScalarValue("nz_wall",nz_wall,0.);
    opts.getScalarValue("T_wall",T_wall,300.);

    /* ---- Initial Conditions ---- */
    if(viscous)
    {
        opts.getScalarValue("Mach_c_ic",Mach_c_ic);
        opts.getScalarValue("nx_c_ic",nx_c_ic,1.);
        opts.getScalarValue("ny_c_ic",ny_c_ic,0.);
        opts.getScalarValue("nz_c_ic",nz_c_ic,0.);
        opts.getScalarValue("T_c_ic",T_c_ic);
    }
//Inviscid
    else
    {
        opts.getScalarValue("u_c_ic",u_c_ic);
        opts.getScalarValue("v_c_ic",v_c_ic);
        opts.getScalarValue("w_c_ic",w_c_ic);
        opts.getScalarValue("p_c_ic",p_c_ic);
    }
    opts.getScalarValue("rho_c_ic",rho_c_ic);

    /* ---- solution patch ---- */
    opts.getScalarValue("patch",patch,0);
    if(patch)
    {
        opts.getScalarValue("patch_type",patch_type,0);//0: vortex
        opts.getScalarValue("patch_freq",patch_freq,0);//0: patch once
        if (patch_type==0)//vortex patch
        {
            opts.getScalarValue("Mv",Mv,0.5);
            opts.getScalarValue("ra",ra,0.075);
            opts.getScalarValue("rb",rb,0.175);
            opts.getScalarValue("xc",xc,0.25);
            opts.getScalarValue("yc",yc,0.5);
        }
        else if (patch_type==1) //uniform patch for x>patch_x with ics
        {
            opts.getScalarValue("patch_x",patch_x);
        }
    }

    /* ---- stationary shock/shock tube ic----*/
    if(ic_form==9||ic_form==10)
        opts.getScalarValue("x_shock_ic",x_shock_ic);

    /* ---- Shock Capturing / dealiasing ---- */
    opts.getScalarValue("over_int",over_int,0);
    if (over_int)
        opts.getScalarValue("N_under", N_under, order - 1);

    opts.getScalarValue("shock_cap", shock_cap, 0); //0: off 1: exponential filter 2: LFS filter
    if (shock_cap)
    {
        opts.getScalarValue("shock_det", shock_det, 0); //0: persson 1: concentration method
        opts.getScalarValue("s0", s0);                  //sensor threshold
        if (shock_cap == 1)                             //exp filter
        {
            opts.getScalarValue("expf_fac", expf_fac, 36.0);
            opts.getScalarValue("expf_order", expf_order, 4.0);
        }
    }

    /* ---- FR Element Solution Point / Correction Function Parameters ---- */
// Tris
    opts.getScalarValue("upts_type_tri",upts_type_tri,0);
    opts.getScalarValue(" fpts_type_tri",fpts_type_tri,0);
    opts.getScalarValue("vcjh_scheme_tri",vcjh_scheme_tri,0);
    opts.getScalarValue("c_tri",c_tri,0.);
    opts.getScalarValue("sparse_tri",sparse_tri,0);
// Quads
    opts.getScalarValue("upts_type_quad",upts_type_quad,0);
    opts.getScalarValue("vcjh_scheme_quad",vcjh_scheme_quad,0);
    opts.getScalarValue("eta_quad",eta_quad,0.);
    opts.getScalarValue("sparse_quad",sparse_quad,0);
// Hexs
    opts.getScalarValue("upts_type_hexa",upts_type_hexa,0);
    opts.getScalarValue("vcjh_scheme_hexa",vcjh_scheme_hexa,0);
    opts.getScalarValue("eta_hexa",eta_hexa,0.);
    opts.getScalarValue("sparse_hexa",sparse_hexa,0);
// Tets
    opts.getScalarValue("upts_type_tet",upts_type_tet,0);
    opts.getScalarValue("fpts_type_tet",fpts_type_tet,0);
    opts.getScalarValue("vcjh_scheme_tet",vcjh_scheme_tet,0);
    opts.getScalarValue("c_tet",c_tet,0.);
    opts.getScalarValue("eta_tet",eta_tet,0.);
    opts.getScalarValue("sparse_tet",sparse_tet,0);
// Prisms
    opts.getScalarValue("upts_type_pri_tri",upts_type_pri_tri,0);
    opts.getScalarValue("upts_type_pri_1d",upts_type_pri_1d,0);
    opts.getScalarValue("vcjh_scheme_pri_1d",vcjh_scheme_pri_1d,0);
    opts.getScalarValue("eta_pri",eta_pri,0.);
    opts.getScalarValue("sparse_pri",sparse_pri);

    /* ---- Advection-Diffusion Parameters ---- */
    if (equation == 1)
    {
        opts.getScalarValue("wave_speed_x",wave_speed(0));
        opts.getScalarValue("wave_speed_y",wave_speed(1),0.);
        opts.getScalarValue( "wave_speed_z",wave_speed(2),0.);
        opts.getScalarValue("diff_coeff",diff_coeff,0.);
        opts.getScalarValue("lambda",lambda);//coeff for lax-fredrich flux
    }

    /* ---- Uncategorized / Other ---- */

    opts.getScalarValue("const_src",const_src,0.);
    opts.getScalarValue("body_forcing",forcing,0);
    opts.getScalarValue("perturb_ic",perturb_ic,0);

// NOTE: the input file line must look like "x_coeffs <# coeffs> x1 x2 x3..."
    opts.getVectorValueOptional("x_coeffs",x_coeffs);
    opts.getVectorValueOptional("y_coeffs",y_coeffs);
    opts.getVectorValueOptional("z_coeffs",z_coeffs);
}

void input::setup(char* fileNameC, int rank)
{
    string fileNameS;
    ramp_counter=1;
    fileNameS.assign(fileNameC);

    /* ---- Read necessary parameters from the input file ---- */
    read_input_file(fileNameS,rank);

    /* ---- Non-Dimensionalization and other setup ---- */
    setup_params(rank);
}

void input::setup_params(int rank)
{
    // --------------------
    // ERROR CHECKING
    // --------------------

    if (monitor_res_freq == 0) monitor_res_freq = 1000;
    if (monitor_cp_freq == 0) monitor_cp_freq = 1000;

    if (!mesh_file.compare(mesh_file.size()-3,3,"neu"))
        mesh_format=0;
    else if (!mesh_file.compare(mesh_file.size()-3,3,"msh"))
        mesh_format=1;
    else
        FatalError("Mesh format not recognized");

    if (equation==0)
    {
        if (riemann_solve_type==1)//todo: maybe we can implement lax-fredrich flux for NS/Euler equation
            FatalError("Lax-Friedrich flux not supported with NS/RANS equation");
        if (ic_form==2 || ic_form==3 || ic_form==4)
            FatalError("Initial condition not supported with NS/RANS equation");
    }
    else if (equation==1)
    {
        if (riemann_solve_type==0)
            FatalError("Rusanov flux not supported with Advection-Diffusion equation");
        if (ic_form != 2 && ic_form != 3 && ic_form != 4 && ic_form != 5)
            FatalError("Initial condition not supported with Advection-Diffusion equation");
    }

    if (turb_model)
    {
        if (riemann_solve_type==2)
            FatalError("Roe flux not supported with RANS turbulent models");
        if (!viscous)
            FatalError("turbulent model not supported with inviscid flow");
        if (LES)
            FatalError("Cannot turn on RANS and LES at same time");
    }

    if (LES && !viscous)
        FatalError("LES not supported with inviscid flow");

    if (N_under > order||N_under<0)
        FatalError("Invalid under sampling order");
        // --------------------------
        // SETTING UP RK COEFFICIENTS
        // --------------------------

    #include "../data/RK_coeff.dat"

    // --------------------------
    // NON-DIMENSIONALIZATION
    // --------------------------
    if(viscous)
    {

        if (rank == 0)
            cout << endl
                 << "---------------------- Non-dimensionalization ---------------------" << endl;
                 
        if(ic_form == 0)
        {
            fix_vis  = 1.;
            R_ref     = 1.;
            c_sth     = 1.;
            rt_inf    = 1.;
            mu_inf    = 0.1;

            if (rank == 0)
            {
                cout << "Using Isentropic vortex initial condition." << endl;
                cout << "R_ref: " << R_ref << endl;
                cout << "c_sth: " << c_sth << endl;
                cout << "rt_inf: " << rt_inf << endl;
                cout << "mu_inf: " << mu_inf << endl;
            }
        }
        else     // Any other type of initial condition
        {

            // Dimensional reference quantities for temperature and length

            T_ref = T_free_stream;
            L_ref = L_free_stream;

            // Compute the reference velocity from the mach_free_stream

            uvw_ref = Mach_free_stream*sqrt(gamma*R_gas*T_free_stream);

            // set the corresponding density from the input file.

            rho_ref = rho_free_stream;
            if (Sup_In)
            {
                rho_sup_in = p_sup_in/(R_gas*T_sup_in);
            }
            if (Sup_In2)
            {
                rho_sup_in2 = p_sup_in2/(R_gas*T_sup_in2);
            }
            if (Sup_In3)
            {
                rho_sup_in3=p_sup_in3/(R_gas*T_sup_in3);
            }
            if (Far_Field)
            {
                rho_far_field = p_far_field/(R_gas*T_far_field);
            }

            // Choose the following consistent reference quantities for other variables

            p_ref     = rho_ref*uvw_ref*uvw_ref;
            mu_ref    = rho_ref*uvw_ref*L_ref;
            time_ref  = L_ref/uvw_ref;
            R_ref     = (R_gas*T_ref)/(uvw_ref*uvw_ref); // R/R_ref,non_dimensionalized R_gas

            // non-dimensionalize sutherland law parameters
            c_sth     = S_gas/T_gas;
            mu_inf    = mu_gas/mu_ref;//non-dimensionalized mu_gas for sutherland law
            rt_inf    = T_gas*R_gas/(uvw_ref*uvw_ref);

            // Set up the dimensionless conditions @ boundaries

            if(Sub_In_Simp)//HACK: only the velocity is specified and must stay subsonic when the simulation goes on
            {
                rho_bound_sub_in_simp = rho_sub_in_simp/rho_ref;
                v_bound_sub_in_simp(0) = Mach_sub_in_simp*sqrt(gamma*R_gas*T_free_stream)/uvw_ref*nx_sub_in_simp;
                v_bound_sub_in_simp(1) = Mach_sub_in_simp*sqrt(gamma*R_gas*T_free_stream)/uvw_ref*ny_sub_in_simp;
                v_bound_sub_in_simp(2) = Mach_sub_in_simp*sqrt(gamma*R_gas*T_free_stream)/uvw_ref*nz_sub_in_simp;
            }

            if(Sub_In_Simp2)//HACK: only the velocity is specified and must stay subsonic when the simulation goes on
            {
                rho_bound_sub_in_simp2 = rho_sub_in_simp2/rho_ref;
                v_bound_sub_in_simp2(0) = Mach_sub_in_simp2*sqrt(gamma*R_gas*T_free_stream)/uvw_ref*nx_sub_in_simp2;
                v_bound_sub_in_simp2(1) = Mach_sub_in_simp2*sqrt(gamma*R_gas*T_free_stream)/uvw_ref*ny_sub_in_simp2;
                v_bound_sub_in_simp2(2) = Mach_sub_in_simp2*sqrt(gamma*R_gas*T_free_stream)/uvw_ref*nz_sub_in_simp2;
            }

            if(Sub_In_char)
            {
                T_total_bound_sub_in = T_total_sub_in/T_ref;
                p_total_bound_sub_in = p_total_sub_in/p_ref;
                if (pressure_ramp)
                {
                    p_total_bound_old=p_total_old/p_ref;
                    T_total_bound_old=T_total_old/T_ref;
                }
            }

            if (Sub_Out)
            {
                p_bound_sub_out=p_sub_out/p_ref;
                T_total_bound_sub_out=T_total_sub_out/T_ref;
            }
            if (Sup_In)
            {
                rho_bound_sup_in=rho_sup_in/rho_ref;
                p_bound_sup_in=p_sup_in/p_ref;
                v_bound_sup_in(0)=Mach_sup_in*sqrt(gamma*R_gas*T_sup_in)/uvw_ref*nx_sup_in;
                v_bound_sup_in(1)=Mach_sup_in*sqrt(gamma*R_gas*T_sup_in)/uvw_ref*ny_sup_in;
                v_bound_sup_in(2)=Mach_sup_in*sqrt(gamma*R_gas*T_sup_in)/uvw_ref*nz_sup_in;
            }

            if (Sup_In2)
            {
                rho_bound_sup_in2=rho_sup_in2/rho_ref;
                p_bound_sup_in2=p_sup_in2/p_ref;
                v_bound_sup_in2(0)=Mach_sup_in2*sqrt(gamma*R_gas*T_sup_in2)/uvw_ref*nx_sup_in2;
                v_bound_sup_in2(1)=Mach_sup_in2*sqrt(gamma*R_gas*T_sup_in2)/uvw_ref*ny_sup_in2;
                v_bound_sup_in2(2)=Mach_sup_in2*sqrt(gamma*R_gas*T_sup_in2)/uvw_ref*nz_sup_in2;
            }

            if (Sup_In3)
            {
                rho_bound_sup_in3=rho_sup_in3/rho_ref;
                p_bound_sup_in3=p_sup_in3/p_ref;
                v_bound_sup_in3(0)=Mach_sup_in3*sqrt(gamma*R_gas*T_sup_in3)/uvw_ref*nx_sup_in3;
                v_bound_sup_in3(1)=Mach_sup_in3*sqrt(gamma*R_gas*T_sup_in3)/uvw_ref*ny_sup_in3;
                v_bound_sup_in3(2)=Mach_sup_in3*sqrt(gamma*R_gas*T_sup_in3)/uvw_ref*nz_sup_in3;
            }

            if (Far_Field)
            {
                rho_bound_far_field=rho_far_field/rho_ref;
                p_bound_far_field=p_far_field/p_ref;
                v_bound_far_field(0)=Mach_far_field*sqrt(gamma*R_gas*T_far_field)/uvw_ref*nx_far_field;
                v_bound_far_field(1)=Mach_far_field*sqrt(gamma*R_gas*T_far_field)/uvw_ref*ny_far_field;
                v_bound_far_field(2)=Mach_far_field*sqrt(gamma*R_gas*T_far_field)/uvw_ref*nz_far_field;
            }

            // Set up the dimensionless conditions @ moving boundaries

            uvw_wall  = Mach_wall*sqrt(gamma*R_gas*T_wall);
            v_wall(0) = (uvw_wall*nx_wall)/uvw_ref;
            v_wall(1) = (uvw_wall*ny_wall)/uvw_ref;
            v_wall(2) = (uvw_wall*nz_wall)/uvw_ref;
            T_wall    = T_wall/T_ref;

            // Set up the dimensionless initial conditions

            uvw_c_ic  = Mach_c_ic*sqrt(gamma*R_gas*T_c_ic);
            u_c_ic   = (uvw_c_ic*nx_c_ic)/uvw_ref;
            v_c_ic   = (uvw_c_ic*ny_c_ic)/uvw_ref;
            w_c_ic   = (uvw_c_ic*nz_c_ic)/uvw_ref;
            if(fix_vis)
            {
                mu_c_ic = mu_gas;
            }
            else
            {
                mu_c_ic = mu_gas*pow(T_c_ic/T_gas, 1.5)*( (T_gas + S_gas)/(T_c_ic + S_gas));
            }


            p_c_ic   = rho_c_ic*R_gas*T_c_ic;
            mu_c_ic  = mu_c_ic/mu_ref;
            rho_c_ic = rho_c_ic/rho_ref;
            p_c_ic   = p_c_ic/p_ref;
            T_c_ic   = T_c_ic/T_ref;

            // SA turblence model parameters
            prandtl_t = 0.9;
            if (turb_model == 1)
            {
                c_v1 = 7.1;
                c_v2 = 0.7;
                c_v3 = 0.9;
                c_b1 = 0.1355;
                c_b2 = 0.622;
                c_w2 = 0.3;
                c_w3 = 2.0;
                omega = 2.0/3.0;
                Kappa = 0.41;
                mu_tilde_c_ic = 5.0*mu_c_ic;
                mu_tilde_inf = 5.0*mu_inf;
            }

            // Master node outputs information about the I.C.s to the console
            if (rank==0)
            {
                cout << "Reference Values" << endl;
                cout << "uvw_ref: " << uvw_ref << " m/s" << endl;
                cout << "rho_ref: " << rho_ref << " kg/m^3" << endl;
                cout << "p_ref: " << p_ref << " Pa" << endl;
                cout << "T_ref: " << T_ref << " k" << endl;
                cout << "L_ref: " << L_ref << " m" << endl;
                cout << "time: " << time_ref << " sec" << endl;
                cout << "mu_ref: " << mu_ref << " kg/(m*s)" << endl;
                cout << "Initial Values" << endl;
                cout << "rho_c_ic=" << rho_c_ic << endl;
                cout << "u_c_ic=" << u_c_ic << endl;
                cout << "v_c_ic=" << v_c_ic << endl;
                cout << "w_c_ic=" << w_c_ic << endl;
                cout << "p_c_ic=" << p_c_ic << endl;
                cout << "T_c_ic=" << T_c_ic << endl;
                cout << "mu_c_ic=" << mu_c_ic << endl;
                cout << "Boundary Conditions" << endl;
                cout << "Sub_In_Simp: " << Sub_In_Simp << " " << Sub_In_Simp2 << endl;
                cout << "Sub_In_Char: " << Sub_In_char << endl;
                if (pressure_ramp)
                {
                    cout << "Pressure Ramp On" << endl;
                    cout << "Pressure Ramping From " << p_total_old << " Pa to " << p_total_sub_in << " Pa" << endl;
                    cout << "Pressure Ramp Rate=" << p_ramp_coeff << endl;
                    if (T_ramp_coeff == -1)
                    {
                        cout << "Isentropic Temperature" << endl;
                    }
                    else
                    {
                        cout << "Temperature Ramping From " << T_total_old << " k to " << T_total_sub_in << " k" << endl;
                        cout << "Temperature Ramp Rate=" << T_ramp_coeff << endl;
                    }
                }
                cout << "Sub_Out: " << Sub_Out << endl;
                cout << "Sup_In: " << Sup_In << Sup_In2 << Sup_In3 << endl;
                cout << "Far_Field: " << Far_Field << endl;
            }
        }
    }
}

fileReader::fileReader()
{

}

fileReader::fileReader(string fileName)
{
    this->fileName = fileName;
}

fileReader::~fileReader()
{
    if (optFile.is_open()) optFile.close();
}

void fileReader::setFile(string fileName)
{
    this->fileName = fileName;
}

void fileReader::openFile(void)
{
    optFile.open(fileName.c_str(), ifstream::in);
}

void fileReader::closeFile()
{
    optFile.close();
}

template<typename T>
void fileReader::getScalarValue(string optName, T &opt, T defaultVal)
{
    string str, optKey;

    openFile();

    if (!optFile.is_open() || !getline(optFile,str))
    {
        optFile.open(fileName.c_str());
        if (!optFile.is_open())
            FatalError("Cannont open input file for reading.");
    }

    // Rewind to the start of the file
    optFile.clear();
    optFile.seekg(0,optFile.beg);

    // Search for the given option string
    while (getline(optFile,str))
    {
        // Remove any leading whitespace & see if first word is the input option
        stringstream ss;
        ss.str(str);
        ss >> optKey;
        if (optKey.compare(optName)==0)
        {
            if (!(ss >> opt))
            {
                // This could happen if, for example, trying to assign a string to a double
                cout << "WARNING: Unable to assign value to option " << optName << endl;
                cout << "Using default value of " << defaultVal << " instead." << endl;
                opt = defaultVal;
            }

            closeFile();
            return;
        }
    }

    opt = defaultVal;
    closeFile();
}

template<typename T>
void fileReader::getScalarValue(string optName, T &opt)
{
    string str, optKey;

    openFile();

    if (!optFile.is_open())
    {
        optFile.open(fileName.c_str());
        if (!optFile.is_open())
            FatalError("Cannont open input file for reading.");
    }

    // Rewind to the start of the file
    optFile.clear();
    optFile.seekg(0,optFile.beg);

    // Search for the given option string
    while (getline(optFile,str))
    {
        // Remove any leading whitespace & see if first word is the input option
        stringstream ss;
        ss.str(str);
        ss >> optKey;

        if (optKey.compare(optName)==0)
        {
            if (!(ss >> opt))
            {
                // This could happen if, for example, trying to assign a string to a double
                cerr << "WARNING: Unable to assign value to option " << optName << endl;
                string errMsg = "Required option not set: " + optName;
                FatalError(errMsg.c_str())
            }

            closeFile();
            return;
        }
    }

    // Option was not found; throw error & exit
    string errMsg = "Required option not found: " + optName;
    FatalError(errMsg.c_str())
}

template<typename T, typename U>
void fileReader::getMap(string optName, map<T,U> &opt)
{
    string str, optKey;
    T tmpT;
    U tmpU;
    bool found;

    openFile();

    if (!optFile.is_open())
    {
        optFile.open(fileName.c_str());
        if (!optFile.is_open())
            FatalError("Cannont open input file for reading.");
    }

    // Rewind to the start of the file
    optFile.clear();
    optFile.seekg(0,optFile.beg);

    // Search for the given option string
    while (getline(optFile,str))
    {
        // Remove any leading whitespace & see if first word is the input option
        stringstream ss;
        ss.str(str);
        ss >> optKey;
        if (optKey.compare(optName)==0)
        {
            found = true;
            if (!(ss >> tmpT >> tmpU))
            {
                // This could happen if, for example, trying to assign a string to a double
                cerr << "WARNING: Unable to assign value to option " << optName << endl;
                string errMsg = "Required option not set: " + optName;
                FatalError(errMsg.c_str())
            }

            opt[tmpT] = tmpU;
            optKey = "";
        }
    }

    if (!found)
    {
        // Option was not found; throw error & exit
        string errMsg = "Required option not found: " + optName;
        FatalError(errMsg.c_str())
    }

    closeFile();
}

template<typename T>
void fileReader::getVectorValue(string optName, vector<T> &opt)
{
    string str, optKey;

    openFile();

    if (!optFile.is_open())
    {
        optFile.open(fileName.c_str());
        if (!optFile.is_open())
            FatalError("Cannont open input file for reading.");
    }

    // Rewind to the start of the file
    optFile.clear();
    optFile.seekg(0,optFile.beg);

    // Search for the given option string
    while (getline(optFile,str))
    {
        // Remove any leading whitespace & see if first word is the input option
        stringstream ss;
        ss.str(str);
        ss >> optKey;
        if (optKey.compare(optName)==0)
        {
            int nVals;
            if (!(ss >> nVals))
            {
                // This could happen if, for example, trying to assign a string to a double
                cerr << "WARNING: Unable to read number of entries for vector option " << optName << endl;
                string errMsg = "Required option not set: " + optName;
                FatalError(errMsg.c_str());
            }

            opt.resize(nVals);
            for (int i=0; i<nVals; i++)
            {
                if (!ss >> opt[i])
                {
                    cerr << "WARNING: Unable to assign all values to vector option " << optName << endl;
                    string errMsg = "Required option not set: " + optName;
                    FatalError(errMsg.c_str())
                }
            }

            closeFile();
            return;
        }
    }

    // Option was not found; throw error & exit
    string errMsg = "Required option not found: " + optName;
    FatalError(errMsg.c_str())
}

template<typename T>
void fileReader::getVectorValue(string optName, hf_array<T> &opt)
{
    string str, optKey;

    openFile();

    if (!optFile.is_open())
    {
        optFile.open(fileName.c_str());
        if (!optFile.is_open())
            FatalError("Cannont open input file for reading.");
    }

    // Rewind to the start of the file
    optFile.clear();
    optFile.seekg(0,optFile.beg);

    // Search for the given option string
    while (getline(optFile,str))
    {
        // Remove any leading whitespace & see if first word is the input option
        stringstream ss;
        ss.str(str);
        ss >> optKey;
        if (optKey.compare(optName)==0)
        {
            int nVals;
            if (!(ss >> nVals))
            {
                // This could happen if, for example, trying to assign a string to a double
                cerr << "WARNING: Unable to read number of entries for vector option " << optName << endl;
                string errMsg = "Required option not set: " + optName;
                FatalError(errMsg.c_str());
            }

            opt.setup(nVals);
            for (int i=0; i<nVals; i++)
            {
                if (!(ss >> opt(i)))
                {
                    cerr << "WARNING: Unable to assign all values to vector option " << optName << endl;
                    string errMsg = "Required option not set: " + optName;
                    FatalError(errMsg.c_str());
                }
            }

            closeFile();
            return;
        }
    }

    // Option was not found; throw error & exit
    string errMsg = "Required option not found: " + optName;
    FatalError(errMsg.c_str())
}

template<typename T>
void fileReader::getVectorValueOptional(string optName, hf_array<T> &opt)
{
    string str, optKey;

    openFile();

    if (!optFile.is_open())
    {
        optFile.open(fileName.c_str());
        if (!optFile.is_open())
            FatalError("Cannont open input file for reading.");
    }

    // Rewind to the start of the file
    optFile.clear();
    optFile.seekg(0,optFile.beg);

    // Search for the given option string
    while (getline(optFile,str))
    {
        // Remove any leading whitespace & see if first word is the input option
        stringstream ss;
        ss.str(str);
        ss >> optKey;
        if (optKey.compare(optName)==0)
        {
            int nVals;
            if (!(ss >> nVals))
            {
                // This could happen if, for example, trying to assign a string to a double
                cerr << "WARNING: Unable to read number of entries for vector option " << optName << endl;
                cerr << "Option not set: " << optName << endl;
                opt.setup(0);
                return;
            }

            opt.setup(nVals);
            for (int i=0; i<nVals; i++)
            {
                if (!(ss >> opt(i)))
                {
                    cerr << "WARNING: Unable to assign all values to vector option " << optName << endl;
                    cerr << "Option not set: " << optName << endl;
                    opt.setup(0);
                    return;
                }
            }

            closeFile();
            return;
        }
    }

    // Option was not found; setup hf_array to size 0
    opt.setup(0);
}

/*!
 * \file bdy_inters.cpp
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

#include <iostream>
#include <cmath>

#include "../include/global.h"
#include "../include/hf_array.h"
#include "../include/inters.h"
#include "../include/bdy_inters.h"
#include "../include/geometry.h"
#include "../include/solver.h"
#include "../include/output.h"
#include "../include/flux.h"
#include "../include/error.h"

#ifdef _GPU
#include "../include/cuda_kernels.h"
#endif

#ifdef _MPI
#include "mpi.h"
#endif

using namespace std;

// #### constructors ####
// default constructor

bdy_inters::bdy_inters()
{
    order=run_input.order;
    viscous=run_input.viscous;
    LES=run_input.LES;
    wall_model = run_input.wall_model;


}

bdy_inters::~bdy_inters() { }

// #### methods ####

// setup inters

void bdy_inters::setup(int in_n_inters, int in_inters_type)
{

    (*this).setup_inters(in_n_inters,in_inters_type);

    boundary_id.setup(in_n_inters);

}

void bdy_inters::set_boundary(int in_inter, int bc_id, int in_ele_type_l, int in_ele_l, int in_local_inter_l, struct solution* FlowSol)
{
    boundary_id(in_inter) = bc_id;

    for(int i=0; i<n_fields; i++)
    {
        for(int j=0; j<n_fpts_per_inter; j++)
        {
            disu_fpts_l(j,in_inter,i)=get_disu_fpts_ptr(in_ele_type_l,in_ele_l,i,in_local_inter_l,j,FlowSol);

            norm_tconf_fpts_l(j,in_inter,i)=get_norm_tconf_fpts_ptr(in_ele_type_l,in_ele_l,i,in_local_inter_l,j,FlowSol);

            if(viscous)
            {
                delta_disu_fpts_l(j,in_inter,i)=get_delta_disu_fpts_ptr(in_ele_type_l,in_ele_l,i,in_local_inter_l,j,FlowSol);
            }
        }
    }

    for(int i=0; i<n_fields; i++)
    {
        for(int j=0; j<n_fpts_per_inter; j++)
        {
            for(int k=0; k<n_dims; k++)
            {
                if(viscous)
                {
                    grad_disu_fpts_l(j,in_inter,i,k) = get_grad_disu_fpts_ptr(in_ele_type_l,in_ele_l,in_local_inter_l,i,k,j,FlowSol);
                }

                // Subgrid-scale flux
                if(LES)
                {
                    sgsf_fpts_l(j,in_inter,i,k) = get_sgsf_fpts_ptr(in_ele_type_l,in_ele_l,in_local_inter_l,i,k,j,FlowSol);
                }
            }
        }
    }

    for(int j=0; j<n_fpts_per_inter; j++)
    {
        tdA_fpts_l(j,in_inter)=get_tdA_fpts_ptr(in_ele_type_l,in_ele_l,in_local_inter_l,j,FlowSol);

        for(int k=0; k<n_dims; k++)
        {
            norm_fpts(j,in_inter,k)=get_norm_fpts_ptr(in_ele_type_l,in_ele_l,in_local_inter_l,j,k,FlowSol);

#ifdef _CPU
            pos_fpts(j,in_inter,k)=get_loc_fpts_ptr_cpu(in_ele_type_l,in_ele_l,in_local_inter_l,j,k,FlowSol);
#endif
#ifdef _GPU
            pos_fpts(j,in_inter,k)=get_loc_fpts_ptr_gpu(in_ele_type_l,in_ele_l,in_local_inter_l,j,k,FlowSol);
#endif
        }
    }

    // Get coordinates and solution at closest solution points to boundary

//      for(int j=0;j<n_fpts_per_inter;j++)
//      {

//        // flux point location

//        // get CPU ptr regardless of ifdef _CPU or _GPU
//        // - we need a CPU ptr to pass to get_normal_disu_fpts_ptr below
//        for (int k=0;k<n_dims;k++)
//          temp_loc(k) = *get_loc_fpts_ptr_cpu(in_ele_type_l,in_ele_l,in_local_inter_l,j,k,FlowSol);

//        // location of the closest solution point
//        double temp_pos[3];

//        if(viscous) {
//          for(int i=0;i<n_fields;i++)
//            normal_disu_fpts_l(j,in_inter,i) = get_normal_disu_fpts_ptr(in_ele_type_l,in_ele_l,in_local_inter_l,i,j,FlowSol, temp_loc, temp_pos);

//          for(int i=0;i<n_dims;i++)
//              pos_disu_fpts_l(j,in_inter,i) = temp_pos[i];
//        }
//      }
}

// move all from cpu to gpu

void bdy_inters::mv_all_cpu_gpu(void)
{
#ifdef _GPU

    disu_fpts_l.mv_cpu_gpu();
    norm_tconf_fpts_l.mv_cpu_gpu();
    tdA_fpts_l.mv_cpu_gpu();
    norm_fpts.mv_cpu_gpu();
    pos_fpts.mv_cpu_gpu();

    delta_disu_fpts_l.mv_cpu_gpu();

    if(viscous)
    {
        grad_disu_fpts_l.mv_cpu_gpu();
        //normal_disu_fpts_l.mv_cpu_gpu();
        //pos_disu_fpts_l.mv_cpu_gpu();
        //norm_tconvisf_fpts_l.mv_cpu_gpu();
    }
    //detjac_fpts_l.mv_cpu_gpu();

    sgsf_fpts_l.mv_cpu_gpu();

    boundary_id.mv_cpu_gpu();
    bdy_params.mv_cpu_gpu();

#endif
}

/*! Calculate normal transformed continuous inviscid flux at the flux points on the boundaries.*/

void bdy_inters::evaluate_boundaryConditions_invFlux(double time_bound)
{

#ifdef _CPU
    hf_array<double> norm(n_dims), fn(n_fields);

    //viscous
    int flux_spec;
    hf_array<double> u_c(n_fields);


    for(int i=0; i<n_inters; i++)//loop over boundary interfaces
    {
        int temp_bc_flag=run_input.bc_list(boundary_id(i)).get_bc_flag();
        for(int j=0; j<n_fpts_per_inter; j++)//loop over flux pts on that interface
        {
                for (int m=0; m<n_dims; m++)
                    norm(m) = *norm_fpts(j,i,m);

            /*! calculate discontinuous solution at flux points */
            for(int k=0; k<n_fields; k++)
                temp_u_l(k)=(*disu_fpts_l(j,i,k));

                // Get static-physical flux point location
                for (int m=0; m<n_dims; m++)
                    temp_loc(m) = *pos_fpts(j,i,m);            

            set_inv_boundary_conditions(boundary_id(i),temp_u_l.get_ptr_cpu(),temp_u_r.get_ptr_cpu(),norm.get_ptr_cpu(),temp_loc.get_ptr_cpu(),run_input.gamma,run_input.R_ref,time_bound,run_input.equation);

            /*! calculate flux from discontinuous solution at flux points */
            if(n_dims==2)
            {
                calc_invf_2d(temp_u_l,temp_f_l);
                calc_invf_2d(temp_u_r,temp_f_r);
            }
            else if(n_dims==3)
            {
                calc_invf_3d(temp_u_l,temp_f_l);
                calc_invf_3d(temp_u_r,temp_f_r);
            }
            else
                FatalError("ERROR: Invalid number of dimensions ... ");


            if (temp_bc_flag==SLIP_WALL_DUAL) // Dual consistent BC
            {
                /*! Set common numerical flux to be normal left flux*/
                right_flux(temp_f_l,norm,fn,n_dims,n_fields,run_input.gamma);
            }
            else // Call Riemann solver
            {
                /*! Calling Riemann solver */
                if (run_input.riemann_solve_type==0)   //Rusanov
                {
                    convective_flux_boundary(temp_f_l,temp_f_r,norm,fn,n_dims,n_fields);
                }
                else if (run_input.riemann_solve_type==1)   // Lax-Friedrich
                {
                    lax_friedrich(temp_u_l,temp_u_r,norm,fn,n_dims,n_fields,run_input.lambda,run_input.wave_speed);
                }
                else if (run_input.riemann_solve_type==2)   // ROE
                {
                    roe_flux(temp_u_l,temp_u_r,norm,fn,n_dims,n_fields,run_input.gamma);
                }
                else if(run_input.riemann_solve_type==3)//HLLC
                {
                    hllc_flux(temp_u_l,temp_u_r,temp_f_l,temp_f_r,norm,fn,n_dims,n_fields,run_input.gamma);
                }
                else
                    FatalError("Riemann solver not implemented");
            }

            /*! Transform back to reference space */
                for(int k=0; k<n_fields; k++)
                {
                    (*norm_tconf_fpts_l(j,i,k))=fn(k)*(*tdA_fpts_l(j,i));
                }
            
            if(viscous)
            {
                /*! boundary specification */

                if(temp_bc_flag == ADIABAT_FIX)//adiabatic
                    flux_spec = 2;
                else
                    flux_spec = 1;

                // Calling viscous riemann solver
                if (run_input.vis_riemann_solve_type==0)
                    ldg_solution(flux_spec,temp_u_l,temp_u_r,u_c,run_input.pen_fact,norm);
                else
                    FatalError("Viscous Riemann solver not implemented");

                    for(int k=0; k<n_fields; k++)
                    {
                        *delta_disu_fpts_l(j,i,k) = (u_c(k) - temp_u_l(k));
                    } 
            }

        }
    }

#endif

#ifdef _GPU
    if (n_inters!=0)
        evaluate_boundaryConditions_invFlux_gpu_kernel_wrapper(n_fpts_per_inter,n_dims,n_fields,n_inters,disu_fpts_l.get_ptr_gpu(),norm_tconf_fpts_l.get_ptr_gpu(),tdA_fpts_l.get_ptr_gpu(),ndA_dyn_fpts_l.get_ptr_gpu(),J_dyn_fpts_l.get_ptr_gpu(),norm_fpts.get_ptr_gpu(),norm_dyn_fpts.get_ptr_gpu(),pos_fpts.get_ptr_gpu(),pos_dyn_fpts.get_ptr_gpu(),grid_vel_fpts.get_ptr_gpu(),boundary_type.get_ptr_gpu(),bdy_params.get_ptr_gpu(),run_input.riemann_solve_type,delta_disu_fpts_l.get_ptr_gpu(),run_input.gamma,run_input.R_ref,viscous,motion,run_input.vis_riemann_solve_type, time_bound, run_input.wave_speed(0),run_input.wave_speed(1),run_input.wave_speed(2),run_input.lambda,run_input.equation,run_input.turb_model);
#endif
}

void bdy_inters::set_inv_boundary_conditions(int bc_id, double* u_l, double* u_r, double *norm, double *loc,double gamma, double R_ref, double time_bound, int equation)
{
    double rho_l, rho_r;
    double v_l[n_dims], v_r[n_dims];
    double e_l, e_r;
    double p_l, p_r;
    double T_l,T_r;
    double vn_l;//initialize to 0 before use
    double v_sq;//initialize to 0 before use
    double mach;
    double machn_l;

    int bc_flag=run_input.bc_list(bc_id).get_bc_flag();

    // Navier-Stokes Boundary Conditions
    if(equation==0)
    {
        // Store primitive variables for clarity
        rho_l = u_l[0];
        for (int i=0; i<n_dims; i++)
            v_l[i] = u_l[i+1]/u_l[0];
        e_l = u_l[n_dims+1];

        // Compute pressure on left side
        v_sq = 0.;
        for (int i=0; i<n_dims; i++)
            v_sq += (v_l[i]*v_l[i]);
        p_l = (gamma-1.0)*(e_l - 0.5*rho_l*v_sq);

        if(!viscous)//use dimensional gas constant
            R_ref=run_input.R_gas;

        T_l=p_l/(rho_l*R_ref);
        
        // Subsonic inflow simple (free pressure)
        if(bc_flag == SUB_IN_SIMP)
        {
            // fix density and velocity
            rho_r = run_input.bc_list(bc_id).rho;
            for (int i=0; i<n_dims; i++)
                v_r[i] =  run_input.bc_list(bc_id).velocity[i];

            // extrapolate pressure
            p_r = p_l;

            // compute energy
            v_sq = 0.;
            for (int i=0; i<n_dims; i++)
                v_sq += (v_r[i]*v_r[i]);
            e_r = (p_r/(gamma-1.0)) + 0.5*rho_r*v_sq;

            // SA model
            if (run_input.turb_model == 1)
            {
                // set turbulent eddy viscosity
                u_r[n_dims+2] = run_input.mu_tilde_inf;
            }
        }

        //outflow simple (fixed pressure)
        //Adapted implementation from FUN3D
        else if(bc_flag == SUB_OUT_SIMP)
        {

            // Compute normal velocity on left side
            vn_l = 0.;
            for (int i=0; i<n_dims; i++)
                vn_l += v_l[i]*norm[i];

            //compute local normal mach number
            machn_l = fabs(vn_l)/sqrt(gamma*p_l/rho_l);

            if(vn_l<0)//reverse flow, back pressure as total pressure
            {
                for (int i=0; i<n_dims; i++)
                    v_r[i]=vn_l*norm[i];//retain only the normal component

                v_sq = 0.;
                for (int i=0; i<n_dims; i++)
                    v_sq += (v_r[i]*v_r[i]);

                T_r=run_input.bc_list(bc_id).T_total-0.5*v_sq*(gamma-1.0)/(R_ref*gamma);//total enthalpy constant
                p_r = run_input.bc_list(bc_id).p_static*pow((1.0+0.5*(gamma-1.0)*(v_sq/(gamma*R_ref*T_r))),-gamma/(gamma-1.0));//use isentropic relation between boundary value and total value
                rho_r=p_r/(R_ref*T_r);
                // compute energy
                e_r = (p_r/(gamma-1.0)) + 0.5*rho_r*v_sq;

                // SA model
                if (run_input.turb_model == 1)
                {
                    // set turbulent eddy viscosity
                    u_r[n_dims + 2] = run_input.mu_tilde_inf;
                }
            }
            else if(vn_l>=0 && machn_l>=1)//if mach >=1 extrapolate all
            {
                rho_r = rho_l;
                for (int i=0; i<n_dims; i++)
                    v_r[i] = v_l[i];
                e_r = e_l;
            }
            else//subsonic outlet
            {
                //extrapolate velocity
                for (int i=0; i<n_dims; i++)
                    v_r[i] = v_l[i];

                // fix pressure
                p_r = run_input.bc_list(bc_id).p_static;

                // extrapolate temperature to calculate density
                rho_r = p_r/(R_ref*T_l);

                // compute energy
                v_sq = 0.;
                for (int i=0; i<n_dims; i++)
                    v_sq += (v_r[i]*v_r[i]);
                e_r = (p_r/(gamma-1.0)) + 0.5*rho_r*v_sq;

                // SA model
                if (run_input.turb_model == 1)
                {
                    // extrapolate turbulent eddy viscosity
                    u_r[n_dims+2] = u_l[n_dims+2];
                }
            }
        }

        // Subsonic inflow characteristic
        // there is one outgoing characteristic (u-c), therefore we can specify
        // all but one state variable at the inlet. The outgoing Riemann invariant
        // provides the final piece of info. Adapted from an implementation in
        // SU2.
        else if(bc_flag == SUB_IN_CHAR)
        {

            double V_r;
            double c_l, c_r_sq, c_total_sq;
            double R_plus, h_total;
            double aa, bb, cc, dd;
            double Mach_sq, alpha;
            double p_total_temp ;
            double T_total_temp;
            //Pressure/Temperature Ramp
            if (run_input.bc_list(bc_id).pressure_ramp)
            {
                if(run_input.bc_list(bc_id).p_ramp_coeff)
                {
                    p_total_temp = run_input.bc_list(bc_id).p_total_old + (run_input.bc_list(bc_id).p_total-run_input.bc_list(bc_id).p_total_old) * run_input.bc_list(bc_id).p_ramp_coeff * run_input.ramp_counter;
                    if(p_total_temp >= run_input.bc_list(bc_id).p_total)
                        p_total_temp = run_input.bc_list(bc_id).p_total;
                }
                else//coeff=0 then no ramping
                    p_total_temp = run_input.bc_list(bc_id).p_total;


                if (run_input.bc_list(bc_id).T_ramp_coeff>0) //>0 Temperature Ramp
                {
                    T_total_temp = run_input.bc_list(bc_id).T_total_old + (run_input.bc_list(bc_id).T_total-run_input.bc_list(bc_id).T_total_old) * run_input.bc_list(bc_id).T_ramp_coeff * run_input.ramp_counter;
                    if(T_total_temp >= run_input.bc_list(bc_id).T_total)
                        T_total_temp = run_input.bc_list(bc_id).T_total;
                }
                else if (run_input.bc_list(bc_id).T_ramp_coeff<0) //-1 isentropic relation across the boundary interface
                    T_total_temp = T_l*pow(p_total_temp/p_l, (gamma-1.0)/gamma);
                else
                    T_total_temp = run_input.bc_list(bc_id).T_total;
            }
            else
            {
                p_total_temp = run_input.bc_list(bc_id).p_total;
                T_total_temp = run_input.bc_list(bc_id).T_total;
            }
            // Specify Inlet conditions

            double n_free_stream[3];
            n_free_stream[0]=run_input.bc_list(bc_id).nx;
            n_free_stream[1]=run_input.bc_list(bc_id).ny;
            n_free_stream[2]=run_input.bc_list(bc_id).nz;

            // Compute normal velocity on left side
            vn_l = 0.;
            for (int i=0; i<n_dims; i++)
                vn_l += v_l[i]*norm[i];

            // Compute speed of sound
            c_l = sqrt(gamma*p_l/rho_l);

            // Extrapolate Riemann invariant
            R_plus = vn_l + 2.0*c_l/(gamma-1.0);

            // Specify total enthalpy
            h_total = gamma*R_ref/(gamma-1.0)*T_total_temp;

            // Compute total speed of sound squared
            c_total_sq = gamma*R_ref*T_total_temp;

            // Dot product of normal flow velocity
            alpha = 0.;
            for (int i=0; i<n_dims; i++)
                alpha += norm[i]*n_free_stream[i];

            // Coefficients of quadratic equation
            aa = 1.0 + 0.5*(gamma-1.0)*alpha*alpha;
            bb = -(gamma-1.0)*alpha*R_plus;
            cc = 0.5*(gamma-1.0)*R_plus*R_plus - 2.0*c_total_sq/(gamma-1.0);

            // Solve quadratic equation for velocity on right side
            // (Note: largest value will always be the positive root)
            // (Note: Will be set to zero if NaN)
            dd = bb*bb - 4.0*aa*cc;
            dd = sqrt(max(dd, 0.0));
            V_r = (-bb + dd)/(2.0*aa);
            V_r = max(V_r, 0.0);
            v_sq = V_r*V_r;

            // Compute speed of sound
            c_r_sq = c_total_sq - 0.5*(gamma-1.0)*v_sq;

            // Compute Mach number (cutoff at Mach = 1.0)
            Mach_sq = v_sq/(c_r_sq);
            Mach_sq = min(Mach_sq, 1.0);
            v_sq = Mach_sq*c_r_sq;
            V_r = sqrt(v_sq);
            c_r_sq = c_total_sq - 0.5*(gamma-1.0)*v_sq;

            // Compute velocity (based on free stream direction)
            for (int i=0; i<n_dims; i++)
                v_r[i] = V_r*n_free_stream[i];

            // Compute temperature
            T_r = c_r_sq/(gamma*R_ref);

            // Compute pressure
            p_r = p_total_temp*pow(T_r/T_total_temp, gamma/(gamma-1.0));

            // Compute density
            rho_r = p_r/(R_ref*T_r);

            // Compute energy
            e_r = (p_r/(gamma-1.0)) + 0.5*rho_r*v_sq;

            // SA model
            if (run_input.turb_model == 1)
            {
                // set turbulent eddy viscosity
                u_r[n_dims+2] = run_input.mu_tilde_inf;
            }
        }

        // Subsonic outflow characteristic(fix pressure)
        // there is one incoming characteristic, therefore one variable can be
        // specified (back pressure) and is used to update the conservative
        // variables. Compute the entropy and the acoustic Riemann variable.
        // These invariants, as well as the tangential velocity components,
        // are extrapolated. Adapted from an implementation in SU2.
        else if(bc_flag == SUB_OUT_CHAR)
        {

            double c_l, c_r;
            double R_plus, s;
            double vn_r;

            // Compute normal velocity on left side
            vn_l = 0.;
            for (int i=0; i<n_dims; i++)
                vn_l += v_l[i]*norm[i];

            // Compute speed of sound
            c_l = sqrt(gamma*p_l/rho_l);

            // Extrapolate Riemann invariant
            R_plus = vn_l + 2.0*c_l/(gamma-1.0);

            // Extrapolate entropy
            s = p_l/pow(rho_l,gamma);

            // fix pressure on the right side
            p_r = run_input.bc_list(bc_id).p_static;

            // Compute density
            rho_r = pow(p_r/s, 1.0/gamma);

            // Compute speed of sound
            c_r = sqrt(gamma*p_r/rho_r);

            // Compute normal velocity
            vn_r = R_plus - 2.0*c_r/(gamma-1.0);

            // Compute velocity and energy
            v_sq = 0.;
            for (int i=0; i<n_dims; i++)
            {
                v_r[i] = v_l[i] + (vn_r - vn_l)*norm[i];
                v_sq += (v_r[i]*v_r[i]);
            }
            e_r = (p_r/(gamma-1.0)) + 0.5*rho_r*v_sq;

            // SA model
            if (run_input.turb_model == 1)
            {
                // extrapolate turbulent eddy viscosity
                u_r[n_dims+2] = u_l[n_dims+2];
            }
        }

        // Supersonic inflow
        else if(bc_flag == SUP_IN)
        {

            // fix density and velocity
            rho_r = run_input.bc_list(bc_id).rho;

            for (int i=0; i<n_dims; i++)
                v_r[i] = run_input.bc_list(bc_id).velocity[i];

            // fix pressure
            p_r = run_input.bc_list(bc_id).p_static;

            // compute energy
            v_sq = 0.;
            for (int i=0; i<n_dims; i++)
                v_sq += (v_r[i]*v_r[i]);
            e_r = (p_r/(gamma-1.0)) + 0.5*rho_r*v_sq;
        }

        // Supersonic outflow
        else if(bc_flag == SUP_OUT)
        {
            // extrapolate density, velocity, energy
            rho_r = rho_l;
            for (int i=0; i<n_dims; i++)
                v_r[i] = v_l[i];
            e_r = e_l;
        }

        // Slip wall
        else if(bc_flag == SLIP_WALL)
        {
            // extrapolate density
            rho_r = rho_l;

            // extrapolate pressure
            p_r = p_l;

            // Compute normal velocity on left side
            vn_l = 0.;
            for (int i=0; i<n_dims; i++)
                vn_l += v_l[i]*norm[i];

            // set normal velocity
            for (int i=0; i<n_dims; i++)
                v_r[i] = v_l[i] - 2*vn_l*norm[i];

            // energy
            v_sq = 0.;
            for (int i=0; i<n_dims; i++)
                v_sq += (v_r[i]*v_r[i]);

            e_r = (p_r/(gamma-1.0)) + 0.5*rho_r*v_sq;
        }

        // Isothermal, no-slip wall (fixed)
        else if(bc_flag == ISOTHERM_FIX)
        {
            // Set state for the right side
            // extrapolate pressure
            p_r = p_l;

            // isothermal temperature
            T_r = run_input.bc_list(bc_id).T_static;

            // density
            rho_r = p_r/(R_ref*T_r);

            // no-slip
            for (int i=0; i<n_dims; i++)
                v_r[i] = -v_l[i];

            // energy
            v_sq = 0.;
            for (int i=0; i<n_dims; i++)
                v_sq += (v_r[i]*v_r[i]);

            e_r = (p_r/(gamma-1.0)) + 0.5*rho_r*v_sq;

            // SA model
            if (run_input.turb_model == 1)
            {
                // zero turbulent eddy viscosity at the wall
                u_r[n_dims+2] = 0.0;
            }
        }

        // Adiabatic, no-slip wall (fixed)
        else if(bc_flag == ADIABAT_FIX)
        {
            // extrapolate density
            rho_r = rho_l; // only useful part

            // no-slip
            for (int i=0; i<n_dims; i++)
                v_r[i] = -v_l[i];

            // energy

            e_r = e_l;

            // SA model
            if (run_input.turb_model == 1)
            {
                // zero turbulent eddy viscosity at the wall
                u_r[n_dims+2] = 0.0;
            }
        }

        // Characteristic/Riemann(far field)
        //Adapted implementation from FUN3D
        else if (bc_flag == CHAR)
        {

            double c_star;
            double vn_star;
            double vn_r;//normal velocity from outside
            double r_plus,r_minus;

            double one_over_s;
            double h_free_stream;
            double mach;


            // Compute normal velocity on left side, >0 out, <0 in
            vn_l = 0.;
            for (int i=0; i<n_dims; i++)
                vn_l += v_l[i]*norm[i];

            //compute normal velocity on right side, >0 in,<0 out
            vn_r = 0;
            for (int i=0; i<n_dims; i++)
                vn_r += run_input.bc_list(bc_id).velocity[i]*norm[i];

            r_plus  = vn_l + 2./(gamma-1.)*sqrt(gamma*p_l/rho_l);
            r_minus = vn_r - 2./(gamma-1.)*sqrt(gamma*run_input.bc_list(bc_id).p_static/run_input.bc_list(bc_id).rho);

            c_star = 0.25*(gamma-1.)*(r_plus-r_minus);
            vn_star = 0.5*(r_plus+r_minus);
            //calculate local mach number
            mach = fabs(vn_l) / sqrt((gamma * R_ref * T_l));

            // Inflow
            if (vn_l<0)
            {
                //if supersonic set the outgoing Riemann invariant to be far field value
                if (mach>1)
                {
                    r_plus  = vn_r + 2./(gamma-1.)*sqrt(gamma*run_input.bc_list(bc_id).p_static/run_input.bc_list(bc_id).rho);
                    c_star = 0.25 * (gamma - 1.) * (r_plus - r_minus);
                    vn_star = 0.5 * (r_plus + r_minus);
                }
                //free stream entropy
                one_over_s = pow(run_input.bc_list(bc_id).rho,gamma)/run_input.bc_list(bc_id).p_static;

                rho_r = pow(1./gamma*(one_over_s*c_star*c_star),1./(gamma-1.));

                // Compute velocity on the right side
                for (int i=0; i<n_dims; i++)
                    v_r[i] = vn_star*norm[i] + (run_input.bc_list(bc_id).velocity[i] - vn_r*norm[i]);

                v_sq = 0.;
                for (int i=0; i<n_dims; i++)
                    v_sq += (v_r[i]*v_r[i]);
                p_r = rho_r/gamma*c_star*c_star;
                e_r = (p_r/(gamma-1.0)) + 0.5*rho_r*v_sq;

                // SA model
                if (run_input.turb_model == 1)
                {
                    // set turbulent eddy viscosity
                    u_r[n_dims+2] = run_input.mu_tilde_inf;
                }
            }

            // Outflow
            else
            {
                //if supersonic set the incoming Riemann invariant to be local value
                if (mach>1)
                {
                    r_minus = vn_l - 2./(gamma-1.)*sqrt(gamma*p_l/rho_l);
                    c_star = 0.25 * (gamma - 1.) * (r_plus - r_minus);
                    vn_star = 0.5 * (r_plus + r_minus);
                }
                //extrapolate entropy
                one_over_s = pow(rho_l,gamma)/p_l;

                rho_r = pow(1./gamma*(one_over_s*c_star*c_star), 1./(gamma-1.));

                // Compute velocity on the right side
                for (int i=0; i<n_dims; i++)
                    v_r[i] = vn_star*norm[i] + (v_l[i] - vn_l*norm[i]);

                p_r = rho_r/gamma*c_star*c_star;
                v_sq = 0.;
                for (int i=0; i<n_dims; i++)
                    v_sq += (v_r[i]*v_r[i]);
                e_r = (p_r/(gamma-1.0)) + 0.5*rho_r*v_sq;

                // SA model
                if (run_input.turb_model == 1)
                {
                    // extrapolate turbulent eddy viscosity
                    u_r[n_dims+2] = u_l[n_dims+2];
                }
            }
        }

        // Dual consistent BC (see SD++ for more comments)
        else if (bc_flag==SLIP_WALL_DUAL)
        {
            // extrapolate density
            rho_r = rho_l;

            // Compute normal velocity on left side
            vn_l = 0.;
            for (int i=0; i<n_dims; i++)
                vn_l += v_l[i]*norm[i];

            // set u = u - (vn_l)nx
            // set v = v - (vn_l)ny
            // set w = w - (vn_l)nz
            for (int i=0; i<n_dims; i++)
                v_r[i] = v_l[i] - vn_l*norm[i];

            // extrapolate energy
            e_r = e_l;
        }

        // Boundary condition not implemented yet
        else
        {
            printf("bdy_type= %s\n",run_input.bc_list(bc_id).get_bc_type().c_str());
            printf("Boundary conditions yet to be implemented");
        }

        // Conservative variables on right side
        u_r[0] = rho_r;
        for (int i=0; i<n_dims; i++)
            u_r[i+1] = rho_r*v_r[i];
        u_r[n_dims+1] = e_r;
    }

    // Advection, Advection-Diffusion Boundary Conditions
    else if(equation==1)
    {
        // Trivial Dirichlet
        if(bc_flag==AD_WALL)
        {
            u_r[0]=0.0;
        }
    }
}


/*! Calculate normal transformed continuous viscous flux at the flux points on the boundaries. */

void bdy_inters::evaluate_boundaryConditions_viscFlux(double time_bound)
{

#ifdef _CPU
    int  flux_spec;
    hf_array<double> norm(n_dims), fn(n_fields);

    for(int i=0; i<n_inters; i++)
    {
        /*! boundary specification */
        int temp_bc_flag=run_input.bc_list(boundary_id(i)).get_bc_flag();

        if(temp_bc_flag == ADIABAT_FIX)//adiabatic
            flux_spec = 2;
        else
            flux_spec = 1;

        for(int j=0; j<n_fpts_per_inter; j++)
        {
                /*! obtain discontinuous solution at flux points */
                for(int k=0; k<n_fields; k++)
                    temp_u_l(k)=(*disu_fpts_l(j,i,k));

                /*! Get normal components and flux points location */
                for (int m=0; m<n_dims; m++)
                {
                    norm(m) = *norm_fpts(j,i,m);
                    temp_loc(m) = *pos_fpts(j,i,m);
                }            

            set_inv_boundary_conditions(boundary_id(i),temp_u_l.get_ptr_cpu(),temp_u_r.get_ptr_cpu(),norm.get_ptr_cpu(),temp_loc.get_ptr_cpu(),run_input.gamma,run_input.R_ref,time_bound,run_input.equation);

            /*! obtain physical gradient of discontinuous solution at flux points */
            for(int k=0; k<n_dims; k++)
            {
                for(int l=0; l<n_fields; l++)
                {
                    temp_grad_u_l(l,k) = *grad_disu_fpts_l(j,i,l,k);
                }
            }

            /*! Right gradient */
            if(flux_spec == 2)
            {
                // Extrapolate
                for(int k=0; k<n_dims; k++)
                {
                    for(int l=0; l<n_fields; l++)
                    {
                        temp_grad_u_r(l,k) = temp_grad_u_l(l,k);
                    }
                }

                set_vis_boundary_conditions(boundary_id(i),temp_u_l.get_ptr_cpu(),temp_u_r.get_ptr_cpu(),temp_grad_u_r.get_ptr_cpu(),norm.get_ptr_cpu(),temp_loc.get_ptr_cpu(),run_input.gamma,run_input.R_ref,time_bound,run_input.equation);
            }

            /*! calculate flux from discontinuous solution at flux points */
            if(n_dims==2)
            {

                if(flux_spec == 1)
                {
                    calc_visf_2d(temp_u_l,temp_grad_u_l,temp_f_l);
                }
                else if(flux_spec == 2)
                {
                    calc_visf_2d(temp_u_r,temp_grad_u_r,temp_f_r);
                }
                else
                    FatalError("Invalid viscous flux specification");
            }
            else if(n_dims==3)
            {

                if(flux_spec == 1)
                {
                    calc_visf_3d(temp_u_l,temp_grad_u_l,temp_f_l);
                }
                else if(flux_spec == 2)
                {
                    calc_visf_3d(temp_u_r,temp_grad_u_r,temp_f_r);
                }
                else
                    FatalError("Invalid viscous flux specification");
            }
            else
                FatalError("ERROR: Invalid number of dimensions ... ");


            // If LES (but no wall model?), get SGS flux and add to viscous flux
            if(LES)
            {

                for(int k=0; k<n_dims; k++)
                {
                    for(int l=0; l<n_fields; l++)
                    {

                        // pointer to subgrid-scale flux
                        temp_sgsf_l(l,k) = *sgsf_fpts_l(j,i,l,k);

                        // Add SGS flux to viscous flux
                        temp_f_l(l,k) += temp_sgsf_l(l,k);
                    }
                }
            }

            /*! Calling viscous riemann solver */
            if (run_input.vis_riemann_solve_type==0)
                ldg_flux(flux_spec,temp_u_l,temp_u_r,temp_f_l,temp_f_r,norm,fn,n_dims,n_fields,run_input.tau,run_input.pen_fact);
            else
                FatalError("Viscous Riemann solver not implemented");

            /*! Transform back to reference space. */
                for(int k=0; k<n_fields; k++)
                    (*norm_tconf_fpts_l(j,i,k))+=fn(k)*(*tdA_fpts_l(j,i));
        }
    }
    
#endif

#ifdef _GPU
    if (n_inters!=0)
        evaluate_boundaryConditions_viscFlux_gpu_kernel_wrapper(n_fpts_per_inter,n_dims,n_fields,n_inters,disu_fpts_l.get_ptr_gpu(),grad_disu_fpts_l.get_ptr_gpu(),norm_tconf_fpts_l.get_ptr_gpu(),tdA_fpts_l.get_ptr_gpu(),ndA_dyn_fpts_l.get_ptr_gpu(),J_dyn_fpts_l.get_ptr_gpu(),norm_fpts.get_ptr_gpu(),norm_dyn_fpts.get_ptr_gpu(),grid_vel_fpts.get_ptr_gpu(),pos_fpts.get_ptr_gpu(),pos_dyn_fpts.get_ptr_gpu(),sgsf_fpts_l.get_ptr_gpu(),boundary_type.get_ptr_gpu(),bdy_params.get_ptr_gpu(),delta_disu_fpts_l.get_ptr_gpu(),run_input.riemann_solve_type,run_input.vis_riemann_solve_type,run_input.R_ref,run_input.pen_fact,run_input.tau,run_input.gamma,run_input.prandtl,run_input.rt_inf,run_input.mu_inf,run_input.c_sth,run_input.fix_vis, time_bound, run_input.equation, run_input.diff_coeff, LES, motion, run_input.turb_model, run_input.c_v1, run_input.omega, run_input.prandtl_t);
#endif
}


void bdy_inters::set_vis_boundary_conditions(int bc_id, double* u_l, double* u_r, double* grad_u, double *norm, double *loc, double gamma, double R_ref, double time_bound, int equation)
{
    int cpu_flag;
    cpu_flag = 1;


    double v_sq;
    double inte;
    double p_l, p_r;

    double grad_vel[n_dims*n_dims];

    int bc_flag=run_input.bc_list(bc_id).get_bc_flag();

    // Adiabatic wall
    if(bc_flag == ADIABAT_FIX)
    {
        v_sq = 0.;
        for (int i=0; i<n_dims; i++)
            v_sq += (u_l[i+1]*u_l[i+1]);
        p_l   = (gamma-1.0)*( u_l[n_dims+1] - 0.5*v_sq/u_l[0]);
        p_r = p_l;

        inte = p_r/((gamma-1.0)*u_r[0]);

        if(cpu_flag) // execute always
        {
            // Velocity gradients
            for (int j=0; j<n_dims; j++)
            {
                for (int i=0; i<n_dims; i++)
                    grad_vel[j*n_dims + i] = (grad_u[i*n_fields + (j+1)] - grad_u[i*n_fields + 0]*u_r[j+1]/u_r[0])/u_r[0];
            }

            // Energy gradients (grad T = 0)
            if(n_dims == 2)
            {
                for (int i=0; i<n_dims; i++)
                    grad_u[i*n_fields + 3] = inte*grad_u[i*n_fields + 0] + 0.5*((u_r[1]*u_r[1]+u_r[2]*u_r[2])/(u_r[0]*u_r[0]))*grad_u[i*n_fields + 0] + u_r[0]*((u_r[1]/u_r[0])*grad_vel[0*n_dims + i]+(u_r[2]/u_r[0])*grad_vel[1*n_dims + i]);
            }
            else if(n_dims == 3)
            {
                for (int i=0; i<n_dims; i++)
                    grad_u[i*n_fields + 4] = inte*grad_u[i*n_fields + 0] + 0.5*((u_r[1]*u_r[1]+u_r[2]*u_r[2]+u_r[3]*u_r[3])/(u_r[0]*u_r[0]))*grad_u[i*n_fields + 0] + u_r[0]*((u_r[1]/u_r[0])*grad_vel[0*n_dims + i]+(u_r[2]/u_r[0])*grad_vel[1*n_dims + i]+(u_r[3]/u_r[0])*grad_vel[2*n_dims + i]);
            }
        }
        else
        {
            // Velocity gradients
            for (int j=0; j<n_dims; j++)
            {
                for (int i=0; i<n_dims; i++)
                    grad_vel[j*n_dims + i] = (grad_u[(j+1)*n_dims + i] - grad_u[0*n_dims + i]*u_r[j+1]/u_r[0])/u_r[0];
            }

            if(n_dims == 2)
            {
                // Total energy gradient
                for (int i=0; i<n_dims; i++)
                    grad_u[3*n_dims + i] = inte*grad_u[0*n_dims + i] + 0.5*((u_r[1]*u_r[1]+u_r[2]*u_r[2])/(u_r[0]*u_r[0]))*grad_u[0*n_dims + i] + u_r[0]*((u_r[1]/u_r[0])*grad_vel[0*n_dims + i]+(u_r[2]/u_r[0])*grad_vel[1*n_dims + i]);
            }
            else if(n_dims == 3)
            {
                for (int i=0; i<n_dims; i++)
                    grad_u[4*n_dims + i] = inte*grad_u[0*n_dims + i] + 0.5*((u_r[1]*u_r[1]+u_r[2]*u_r[2]+u_r[3]*u_r[3])/(u_r[0]*u_r[0]))*grad_u[0*n_dims + i] + u_r[0]*((u_r[1]/u_r[0])*grad_vel[0*n_dims + i]+(u_r[2]/u_r[0])*grad_vel[1*n_dims + i]+(u_r[3]/u_r[0])*grad_vel[2*n_dims + i]);
            }
        }

    }

}


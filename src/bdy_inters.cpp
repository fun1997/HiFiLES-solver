/*!
 * \file bdy_inters.cpp
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

#include <iostream>
#include <cmath>
#include <random>
#include <chrono>
#include <cstdlib>
#include <fstream>

#include <dirent.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

#include "../include/global.h"
#include "../include/bdy_inters.h"
#include "../include/solver.h"
#include "../include/flux.h"
#include "wall_model_funcs.h" 

#ifdef _GPU
#include "../include/cuda_kernels.h"
#endif

using namespace std;

// #### constructors ####
// default constructor

bdy_inters::bdy_inters()
{
}

bdy_inters::~bdy_inters() { }

// #### methods ####

// setup inters

void bdy_inters::setup(int in_n_inters, int in_inters_type)
{

    (*this).setup_inters(in_n_inters,in_inters_type);

    boundary_id.setup(in_n_inters);
    if(in_n_inters!=0){
        pos_bdr_face_vtx.setup(n_vtx,in_n_inters,n_dims);
    }

}

void bdy_inters::set_boundary(int in_inter, int bc_id, int in_ele_type_l, int in_ele_l, int in_local_inter_l, struct solution* FlowSol)
{
    boundary_id(in_inter) = bc_id;
    if(flag==0)
    {
        in_ele_type=in_ele_type_l;
        flag=1;
    }


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

        weight_fpts(j,in_inter)=get_weight_fpts_ptr(in_ele_type_l,in_local_inter_l,j,FlowSol);

        inter_detjac_inters_cubpts(j,in_inter)=get_inter_detjac_inters_cubpts_ptr(in_ele_type_l,in_ele_l,in_local_inter_l,j,FlowSol);

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

        //setup use wall model
    if (run_input.bc_list(bc_id).use_wm)
    {
        int upt_idx;
        hf_array<double *> temp_disu_ptr(n_fields);
        //find the farthest solution point to the boundary interface, calculate the distance
        wm_dist.push_back(calc_wm_upts_dist(in_ele_type_l, in_ele_l, in_local_inter_l, FlowSol, upt_idx));
        //get pointer of input point solution
        for (int i = 0; i < n_fields; i++)
        {
            temp_disu_ptr(i) = get_wm_disu_ptr(in_ele_type_l, in_ele_l, upt_idx, i, FlowSol);
        }
        wm_disu_ptr.push_back(temp_disu_ptr);
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

void bdy_inters::evaluate_boundaryConditions_invFlux(solution* FlowSol,double time_bound)
{

#ifdef _CPU
    hf_array<double> norm(n_dims), fn(n_fields);

    //viscous
    hf_array<double> u_c(n_fields);

    ofstream writefile;

    char file_name_s[256];


    sprintf(file_name_s, "add_flu_rank%d.dat", FlowSol->rank);
    writefile.open(file_name_s);

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

            //calculate inviscid boundary solution
            set_boundary_conditions(0, boundary_id(i), temp_u_l.get_ptr_cpu(), temp_u_r.get_ptr_cpu(),
                                    norm.get_ptr_cpu(), temp_loc.get_ptr_cpu(), run_input.gamma, run_input.R_ref, time_bound, run_input.equation);

            if(inlet.ibslst_inv(i)!=-1){
                writefile<<"origin  ibs= "<<inlet.ibslst_inv(i)<<" "<<"j= "<<j<<" " << setw(18)<<temp_u_r[0]<<" "<<setw(18)<<temp_u_r[1]<<" "<<setw(18)<<temp_u_r[2]<<" "<<setw(18)<<temp_u_r[3]<<" "<<endl;
                for (int m=0; m<n_dims; m++)
                    temp_u_r[m+1] += temp_u_r[0]*inlet.fluctuations(j,inlet.ibslst_inv(i),m);                
                writefile<<"after   ibs= "<<inlet.ibslst_inv(i)<<" "<<"j= "<<j<<" " << setw(18)<<temp_u_r[0]<<" "<<setw(18)<<temp_u_r[1]<<" "<<setw(18)<<temp_u_r[2]<<" "<<setw(18)<<temp_u_r[3]<<" "<<endl;
            }            

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
                    rusanov_flux(temp_u_l, temp_u_r, temp_f_l, temp_f_r, norm, fn, n_dims, n_fields, run_input.gamma);
                }
                else if (run_input.riemann_solve_type==1)   // Lax-Friedrich
                {
                    lax_friedrich(temp_u_l,temp_u_r,norm,fn,n_dims,n_fields,run_input.lambda,run_input.wave_speed);
                }
                else if (run_input.riemann_solve_type==2)   // ROEM
                {
                    roeM_flux(temp_u_l, temp_u_r, temp_f_l, temp_f_r, norm, fn, n_dims, n_fields, run_input.gamma);
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
            //calculate viscous boundary solution if is wall boundary
            if (temp_bc_flag == SLIP_WALL || temp_bc_flag == ISOTHERM_WALL || temp_bc_flag == ADIABAT_WALL || temp_bc_flag == AD_WALL || temp_bc_flag == SLIP_WALL_DUAL)
                set_boundary_conditions(1, boundary_id(i), temp_u_l.get_ptr_cpu(), temp_u_r.get_ptr_cpu(),
                                        norm.get_ptr_cpu(), temp_loc.get_ptr_cpu(), run_input.gamma, run_input.R_ref, time_bound, run_input.equation);
            // Calling viscous riemann solver
            if (run_input.vis_riemann_solve_type==0)
                ldg_solution(1,temp_u_l,temp_u_r,u_c,run_input.ldg_beta,norm);
            else
                FatalError("Viscous Riemann solver not implemented");

            for(int k=0; k<n_fields; k++)
            {
                *delta_disu_fpts_l(j,i,k) = (u_c(k) - temp_u_l(k));
            }
            }
        }
    }
    writefile.close();

#endif

#ifdef _GPU
    if (n_inters != 0)
        evaluate_boundaryConditions_invFlux_gpu_kernel_wrapper(n_fpts_per_inter, n_dims, n_fields, n_inters, disu_fpts_l.get_ptr_gpu(), norm_tconf_fpts_l.get_ptr_gpu(), tdA_fpts_l.get_ptr_gpu(), ndA_dyn_fpts_l.get_ptr_gpu(), J_dyn_fpts_l.get_ptr_gpu(), norm_fpts.get_ptr_gpu(), norm_dyn_fpts.get_ptr_gpu(), pos_fpts.get_ptr_gpu(), pos_dyn_fpts.get_ptr_gpu(), grid_vel_fpts.get_ptr_gpu(), boundary_type.get_ptr_gpu(), bdy_params.get_ptr_gpu(), run_input.riemann_solve_type, delta_disu_fpts_l.get_ptr_gpu(), run_input.gamma, run_input.R_ref, viscous, motion, run_input.vis_riemann_solve_type, time_bound, run_input.wave_speed(0), run_input.wave_speed(1), run_input.wave_speed(2), run_input.lambda, run_input.equation, run_input.RANS);
#endif
}

void bdy_inters::set_boundary_conditions(int sol_spec, int bc_id, double *u_l, double *u_r, double *norm, double *loc, double gamma, double R_ref, double time_bound, int equation)
{
    double rho_l, rho_r;
    double v_l[n_dims], v_r[n_dims];
    double e_l, e_r;
    double p_l, p_r;
    double T_l,T_r;
    double vn_l;//initialize to 0 before use
    double v_sq;//initialize to 0 before use
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
            // fix density
            rho_r = run_input.bc_list(bc_id).rho;

            // fix velocity
            for (int i = 0; i < n_dims; i++)
                v_r[i] = run_input.bc_list(bc_id).velocity[i];

            // compute energy
            v_sq = 0.;
            for (int i = 0; i < n_dims; i++)
                v_sq += (v_r[i] * v_r[i]);
            e_r = p_l / (gamma - 1.0) + 0.5 * rho_r * v_sq;

            // SA model
            if (run_input.RANS == 1)
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
                if (run_input.RANS == 1)
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

                //extrapolate density
                rho_r = rho_l;

                // fix pressure
                p_r = run_input.bc_list(bc_id).p_static;

                // compute energy
                v_sq = 0.;
                for (int i=0; i<n_dims; i++)
                    v_sq += (v_r[i]*v_r[i]);
                e_r = (p_r/(gamma-1.0)) + 0.5*rho_r*v_sq;

                // SA model
                if (run_input.RANS == 1)
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
            double R_plus;
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
            //h_total = gamma*R_ref/(gamma-1.0)*T_total_temp;

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
            if (run_input.RANS == 1)
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
            if (run_input.RANS == 1)
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

            // Compute normal velocity on left side
            vn_l = 0.;
            for (int i=0; i<n_dims; i++)
                vn_l += v_l[i]*norm[i];

            // set velocity
            if (sol_spec == 0) //inviscid solution
            {
                for (int i = 0; i < n_dims; i++)
                    v_r[i] = v_l[i] - 2 * vn_l * norm[i];
            }
            else//viscous solution
            {
                for (int i = 0; i < n_dims; i++)
                    v_r[i] = v_l[i] - vn_l * norm[i];
            }

            // energy
            v_sq = 0.;
            for (int i=0; i<n_dims; i++)
                v_sq += (v_r[i]*v_r[i]);

            e_r = p_l / (gamma - 1.0) + 0.5 * rho_r * v_sq;
        }

        // Isothermal, no-slip wall 
        else if(bc_flag == ISOTHERM_WALL)
        {
            // isothermal temperature
            T_r = run_input.bc_list(bc_id).T_static;

            // extrapolate density
            rho_r = rho_l;

            if (run_input.bc_list(bc_id).use_wm)//wall model on
            {
                if (sol_spec == 0) //inverse slip inviscid solution
                {
                    // Compute normal velocity on left side
                    vn_l = 0.;
                    for (int i = 0; i < n_dims; i++)
                        vn_l += v_l[i] * norm[i];
                   
                    //inverse normal velocity
                    for (int i = 0; i < n_dims; i++)
                        v_r[i] = v_l[i] - 2 * vn_l * norm[i];
                    
                    // energy extraploate temperature
                    v_sq = 0.;
                    for (int i = 0; i < n_dims; i++)
                        v_sq += (v_r[i] * v_r[i]);

                    e_r = p_l / (gamma - 1.0) + 0.5 * rho_r * v_sq;
                }
                else if (sol_spec == 1) //slip viscous solution
                {
                    // Compute normal velocity on left side
                    vn_l = 0.;
                    for (int i = 0; i < n_dims; i++)
                        vn_l += v_l[i] * norm[i];

                    //substract normal velocity
                    for (int i = 0; i < n_dims; i++)
                        v_r[i] = v_l[i] - vn_l * norm[i];
                    
                    // energy extrapolate temperature
                    v_sq = 0.;
                    for (int i = 0; i < n_dims; i++)
                        v_sq += (v_r[i] * v_r[i]);

                    e_r = p_l / (gamma - 1.0) + 0.5 * rho_r * v_sq;
                }
                else if (sol_spec == 2) //no-slip viscous solution
                {
                    for (int i = 0; i < n_dims; i++)
                        v_r[i] = run_input.bc_list(bc_id).velocity(i);
                    
                    // energy use wall temperature
                    v_sq = 0.;
                    for (int i = 0; i < n_dims; i++)
                        v_sq += (v_r[i] * v_r[i]);

                    e_r = rho_r * (R_ref / (gamma - 1.0) * T_r) + 0.5 * rho_r * v_sq;
                }
            }
            else //wall model off
            {
                if (sol_spec == 0) //inverse no-slip inviscid solution
                {
                    for (int i = 0; i < n_dims; i++)
                        v_r[i] = 2 * run_input.bc_list(bc_id).velocity(i) - v_l[i];
                }
                else if (sol_spec == 1) // no-slip viscous solution
                {
                    for (int i = 0; i < n_dims; i++)
                        v_r[i] = run_input.bc_list(bc_id).velocity(i);
                }
                else
                {
                    FatalError("Unrecognized flux type");
                }
                // energy use wall temperature
                v_sq = 0.;
                for (int i = 0; i < n_dims; i++)
                    v_sq += (v_r[i] * v_r[i]);

                e_r = rho_r * (R_ref / (gamma - 1.0) * T_r) + 0.5 * rho_r * v_sq;
            }

            // SA model
            if (run_input.RANS == 1)
            {
                // zero turbulent eddy viscosity at the wall
                u_r[n_dims+2] = 0.0;
            }
        }

        // Adiabatic, no-slip wall
        else if(bc_flag == ADIABAT_WALL)
        {
            // extrapolate density
            rho_r = rho_l; // only useful part

            if (run_input.bc_list(bc_id).use_wm)//wall model on
            {
                if (sol_spec == 0) //inverse slip inviscid solution
                {
                    // Compute normal velocity on left side
                    vn_l = 0.;
                    for (int i = 0; i < n_dims; i++)
                        vn_l += v_l[i] * norm[i];
                    
                    //inverse normal velocity
                    for (int i = 0; i < n_dims; i++)
                        v_r[i] = v_l[i] - 2 * vn_l * norm[i];
                }
                else if (sol_spec == 1) //slip viscous solution
                {
                    // Compute normal velocity on left side
                    vn_l = 0.;
                    for (int i = 0; i < n_dims; i++)
                        vn_l += v_l[i] * norm[i];
                        
                    //subtract normal velocity
                    for (int i = 0; i < n_dims; i++)
                        v_r[i] = v_l[i] - vn_l * norm[i];
                }
                else if (sol_spec == 2) //no-slip viscous solution
                {
                    for (int i = 0; i < n_dims; i++)
                        v_r[i] = run_input.bc_list(bc_id).velocity(i);
                }
            }
            else //wall model off
            {
                if (sol_spec == 0) //inverse no-slip inviscid solution
                {
                    for (int i = 0; i < n_dims; i++)
                        v_r[i] = 2 * run_input.bc_list(bc_id).velocity(i) - v_l[i];
                }
                else if (sol_spec == 1) // no-slip viscous solution
                {
                    for (int i = 0; i < n_dims; i++)
                        v_r[i] = run_input.bc_list(bc_id).velocity(i);
                }
                else
                {
                    FatalError("Unrecognized flux type");
                }
            }

            // energy extrapolate temperature
            v_sq = 0.;
            for (int i=0; i<n_dims; i++)
                v_sq += (v_r[i]*v_r[i]);

            e_r = p_l / (gamma - 1.0) + 0.5 * rho_r * v_sq;

            // SA model
            if (run_input.RANS == 1)
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
            double c_l,c_r;
            double one_over_s;
            double mach;


            // Compute normal velocity on left side, >0 out, <0 in
            vn_l = 0.;
            for (int i=0; i<n_dims; i++)
                vn_l += v_l[i]*norm[i];

            //compute normal velocity on right side, >0 in,<0 out
            vn_r = 0;
            for (int i=0; i<n_dims; i++)
                vn_r += run_input.bc_list(bc_id).velocity[i]*norm[i];
            
            c_l=sqrt(gamma*p_l/rho_l);
            c_r=sqrt(gamma*run_input.bc_list(bc_id).p_static/run_input.bc_list(bc_id).rho);
            mach = fabs(vn_l) / c_l;

            // Inflow
            if (vn_l<0)
            {
                //if supersonic inflow set the outgoing Riemann invariant to be far field value
                if (mach >= 1)
                {
                    r_minus = vn_r - 2. / (gamma - 1.) * c_r;
                    r_plus = vn_r + 2. / (gamma - 1.) * c_r;
                }
                else //subsonically inflow
                {
                    r_plus = vn_l + 2. / (gamma - 1.) * c_l;
                    r_minus = vn_r - 2. / (gamma - 1.) * c_r;
                }

                c_star = 0.25 * (gamma - 1.) * (r_plus - r_minus);
                vn_star = 0.5 * (r_plus + r_minus);

                //use free stream entropy to calculate density
                one_over_s = pow(run_input.bc_list(bc_id).rho,gamma)/run_input.bc_list(bc_id).p_static;
                rho_r = pow(1./gamma*(one_over_s*c_star*c_star),1./(gamma-1.));

                // Compute velocity on the right side, extrapolate tangetal right velocity
                for (int i=0; i<n_dims; i++)
                    v_r[i] = vn_star*norm[i] + (run_input.bc_list(bc_id).velocity[i] - vn_r*norm[i]);

                v_sq = 0.;
                for (int i=0; i<n_dims; i++)
                    v_sq += (v_r[i]*v_r[i]);
                p_r = rho_r/gamma*c_star*c_star;
                e_r = (p_r/(gamma-1.0)) + 0.5*rho_r*v_sq;

                // SA model
                if (run_input.RANS == 1)
                {
                    // set turbulent eddy viscosity
                    u_r[n_dims+2] = run_input.mu_tilde_inf;
                }
            }

            // Outflow
            else
            {
                //if supersonic outflow
                if (mach>=1)
                {
                    r_minus = vn_l - 2./(gamma-1.)*c_l;
                    r_plus = vn_l + 2. / (gamma - 1.) * c_l;
                }
                else //subsonically outflow
                {
                    r_plus = vn_l + 2. / (gamma - 1.) * c_l;
                    r_minus = vn_r - 2. / (gamma - 1.) * c_r;
                }

                c_star = 0.25 * (gamma - 1.) * (r_plus - r_minus);
                vn_star = 0.5 * (r_plus + r_minus);

                //extrapolate entropy
                one_over_s = pow(rho_l,gamma)/p_l;
                rho_r = pow(1./gamma*(one_over_s*c_star*c_star), 1./(gamma-1.));

                // Compute velocity on the right side, extrapolate tangental left velocity
                for (int i=0; i<n_dims; i++)
                    v_r[i] = vn_star*norm[i] + (v_l[i] - vn_l*norm[i]);
                
                v_sq = 0.;
                for (int i=0; i<n_dims; i++)
                    v_sq += (v_r[i]*v_r[i]);

                p_r = rho_r/gamma*c_star*c_star;
                e_r = (p_r/(gamma-1.0)) + 0.5*rho_r*v_sq;

                // SA model
                if (run_input.RANS == 1)
                {
                    // extrapolate turbulent eddy viscosity
                    u_r[n_dims+2] = u_l[n_dims+2];
                }
            }
        }

        // Dual consistent BC (see SD++ for more comments)
        else if (bc_flag==SLIP_WALL_DUAL)//TODO: what is this
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
                v_r[i] = v_l[i] - 2 * vn_l * norm[i];

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
    hf_array<double> norm(n_dims), fn(n_fields);
    hf_array<double> temp_u_wm(n_fields);
    int ctr = 0;
    for (int i = 0; i < n_inters; i++)
    {
        /*! boundary specification */
        int temp_bc_flag = run_input.bc_list(boundary_id(i)).get_bc_flag();

        if (temp_bc_flag != SLIP_WALL) //if not slip wall(slip wall dont need to calculate viscous flux)
        {
            if (!run_input.bc_list(boundary_id(i)).use_wm)//if not use wall model
            {
                for (int j = 0; j < n_fpts_per_inter; j++)
                {
                    /*! obtain discontinuous solution at flux points */
                    for (int k = 0; k < n_fields; k++)
                        temp_u_l(k) = (*disu_fpts_l(j, i, k));

                    /*! Get normal components and flux points location */
                    for (int m = 0; m < n_dims; m++)
                    {
                        norm(m) = *norm_fpts(j, i, m);
                        temp_loc(m) = *pos_fpts(j, i, m);
                    }

                    /*! obtain physical gradient of discontinuous solution at flux points */
                    for (int k = 0; k < n_dims; k++)
                        for (int l = 0; l < n_fields; l++)
                            temp_grad_u_l(l, k) = *grad_disu_fpts_l(j, i, l, k);

                    //calculate viscous boundary solution
                    set_boundary_conditions(1, boundary_id(i), temp_u_l.get_ptr_cpu(), temp_u_r.get_ptr_cpu(),
                                            norm.get_ptr_cpu(), temp_loc.get_ptr_cpu(), run_input.gamma, run_input.R_ref, time_bound, run_input.equation);

                    if(inlet.ibslst_inv(i)!=-1){
                
                        for (int m=0; m<n_dims; m++)
                            temp_u_r[m+1] += temp_u_r[0]*inlet.fluctuations(j,inlet.ibslst_inv(i),m);                
                        
                    }

                    set_boundary_gradients(boundary_id(i), temp_u_l, temp_u_r, temp_grad_u_l, temp_grad_u_r,
                                           norm, temp_loc, run_input.gamma, run_input.R_ref, time_bound, run_input.equation);

                    /*! calculate flux from discontinuous solution at flux points */
                    if (n_dims == 2)
                    {
                        calc_visf_2d(temp_u_r, temp_grad_u_r, temp_f_r);
                    }
                    else if (n_dims == 3)
                    {
                        calc_visf_3d(temp_u_r, temp_grad_u_r, temp_f_r);
                    }
                    else
                        FatalError("ERROR: Invalid number of dimensions ... ");

                    /*! Calling viscous riemann solver */
                    if (run_input.vis_riemann_solve_type == 0)
                        ldg_flux(1, temp_u_l, temp_u_r, temp_f_l, temp_f_r, norm, fn, n_dims, n_fields, run_input.ldg_tau, run_input.ldg_beta);
                    else
                        FatalError("Viscous Riemann solver not implemented");

                    /*! Transform back to reference space. */
                    for (int k = 0; k < n_fields; k++)
                        (*norm_tconf_fpts_l(j, i, k)) += fn(k) * (*tdA_fpts_l(j, i));
                }
            }
            else //this interface uses wall model
            {
                for (int j = 0; j < n_fpts_per_inter; j++)
                {
                    /*! obtain discontinuous solution at flux points */
                    for (int k = 0; k < n_fields; k++)
                        temp_u_l(k) = (*disu_fpts_l(j, i, k));

                    /*! Get normal components and flux points location */
                    for (int m = 0; m < n_dims; m++)
                    {
                        norm(m) = *norm_fpts(j, i, m);
                        temp_loc(m) = *pos_fpts(j, i, m);
                    }
                    //calculate viscous boundary solution
                    set_boundary_conditions(2, boundary_id(i), temp_u_l.get_ptr_cpu(), temp_u_r.get_ptr_cpu(),
                                            norm.get_ptr_cpu(), temp_loc.get_ptr_cpu(), run_input.gamma, run_input.R_ref, time_bound, run_input.equation);

                    if(inlet.ibslst_inv(i)!=-1){
                
                        for (int m=0; m<n_dims; m++)
                            temp_u_r[m+1] += temp_u_r[0]*inlet.fluctuations(j,inlet.ibslst_inv(i),m);                
                        
                    }

                    /*! obtain wall model input solutions */
                    for (int k = 0; k < n_fields; k++)
                        temp_u_wm(k) = (*wm_disu_ptr[ctr](k));

                    /*! calculate wall stress */
                    calc_wall_stress(temp_u_wm, temp_u_r, wm_dist[ctr], norm, fn);
                    /*! Transform back to reference space. */
                    for (int k = 0; k < n_fields; k++)
                        (*norm_tconf_fpts_l(j, i, k)) += fn(k) * (*tdA_fpts_l(j, i));
                }
                ctr++;
            }
        }
    }
}

void bdy_inters::set_boundary_gradients(int bc_id, hf_array<double> &u_l, hf_array<double> &u_r, hf_array<double> &grad_ul, hf_array<double> &grad_ur, hf_array<double> &norm, hf_array<double> &loc, double gamma, double R_ref, double time_bound, int equation)
{
    double v_sq;
    double inte;

    hf_array<double> grad_vel(n_dims, n_dims), grad_inte(n_dims); //component,dim

    int bc_flag = run_input.bc_list(bc_id).get_bc_flag();

    if (bc_flag == CHAR || bc_flag == SUP_IN || bc_flag == SUB_IN_SIMP || bc_flag == SUB_OUT_SIMP) //zero gradients
        grad_ur.initialize_to_zero();
    else //extrapolate gradients
        grad_ur = grad_ul;

    if (bc_flag == ADIABAT_WALL) //adiabatic wall substract norm temperature gradient from energy gradients
    {
        //right internal energy
        v_sq = 0.;
        for (int i = 0; i < n_dims; i++)
            v_sq += (u_r(i + 1) * u_r(i + 1));
        inte = (u_r(n_dims + 1) - 0.5 * v_sq / u_r(0)) / u_r(0); //inte=cvT=(rhoE-0.5*rho*u^2)/rho

        // right velocity gradients
        for (int j = 0; j < n_dims; j++)                                                             //direction
            for (int i = 0; i < n_dims; i++)                                                         //velocity component
                grad_vel(i, j) = (grad_ur(i + 1, j) - grad_ur(0, j) * u_r(i + 1) / u_r(0)) / u_r(0); //du_i/dx_j=drhou_i/dx_j-u_i*drho/dx_j)/rho

        //right internal energy gradients
        if (n_dims == 2)
        {
            for (int i = 0; i < n_dims; i++) //dinte/dx_i=drhoE/dx_i-(inte*drho/dx_i+0.5*u^2drho/dx_i+rhou*du/dx_i)
                grad_inte(i) = grad_ur(3, i) - (inte * grad_ur(0, i) + 0.5 * v_sq / (u_r[0] * u_r[0]) * grad_ur(0, i) + u_r[1] * grad_vel(0, i) + u_r[2] * grad_vel(1, i));
        }
        else
        {
            for (int i = 0; i < n_dims; i++) //dinte/dx_i=drhoE/dx_i-(inte*drho/dx_i+0.5*u^2drho/dx_i+rhou*du/dx_i)
                grad_inte(i) = grad_ur(4, i) - (inte * grad_ur(0, i) + 0.5 * v_sq / (u_r[0] * u_r[0]) * grad_ur(0, i) + u_r[1] * grad_vel(0, i) + u_r[2] * grad_vel(1, i) + u_r[3] * grad_vel(2, i));
        }

        // correct right energy gradients (set grad dT/dn = 0)
        if (n_dims == 2)
        {
            for (int i = 0; i < n_dims; i++)
                grad_ur(3, i) -= (grad_inte(0) * norm(0) + grad_inte(1) * norm(1)) * norm(i); //dinte/dn(i)=(dinte/dx_j*n_j)*n_i
        }
        else if (n_dims == 3)
        {
            for (int i = 0; i < n_dims; i++)
                grad_ur(4, i) -= (grad_inte(0) * norm(0) + grad_inte(1) * norm(1) + grad_inte(2) * norm(2)) * norm(i); //dinte/dn(i)=(dinte/dx_j*n_j)*n_i
        }
    }
}

void bdy_inters::add_les_inlet(int in_file_num,struct solution* FlowSol)
{
    if(n_inters!=0){
        
        int temp_bc_flag;
        int inlet_bc_flag;
        int id;
        int ibs;
        int count;
        int rest_info;
        int i,j,k;

        count=0;
        for(i=0; i<n_inters; i++)//loop over boundary interfaces
        {                  
            temp_bc_flag=run_input.bc_list(boundary_id(i)).get_bc_flag();
            if(temp_bc_flag==SUB_IN_SIMP||temp_bc_flag==SUB_IN_CHAR||temp_bc_flag==SUP_IN){
                count++;
            }

        }

        inlet.nbs=count;

        
    
        inlet.ibslst.setup(inlet.nbs);
        inlet.ibslst_inv.setup(n_inters);
        inlet.face_vtx_coord.setup(n_vtx,inlet.nbs,n_dims);
        inlet.v.setup(n_fpts_per_inter,inlet.nbs,n_dims);
        inlet.rou.setup(n_fpts_per_inter,inlet.nbs);
        inlet.fluctuations.setup(n_fpts_per_inter,inlet.nbs,n_dims); 
        inlet.r_ij.setup(n_fpts_per_inter,inlet.nbs,6);


        inlet.ibslst_inv.initialize_to_value(-1);
        inlet.r_ij.initialize_to_zero();
        

        count=0;

        for(i=0; i<n_inters; i++)//loop over boundary interfaces
        {
            temp_bc_flag=run_input.bc_list(boundary_id(i)).get_bc_flag();
            if(temp_bc_flag==SUB_IN_SIMP||temp_bc_flag==SUB_IN_CHAR||temp_bc_flag==SUP_IN){
                inlet.ibslst_inv(i)=count;
                inlet.ibslst(count)=i;
                count++;
            }
        }



        count=0;
        for(i=0; i<n_inters; i++)//loop over boundary interfaces
        {         
            temp_bc_flag=run_input.bc_list(boundary_id(i)).get_bc_flag();         
            if(temp_bc_flag==SUB_IN_SIMP||temp_bc_flag==SUB_IN_CHAR||temp_bc_flag==SUP_IN){    
                for(j=0; j<n_vtx; j++){
                    for (k=0; k<n_dims; k++)
                        inlet.face_vtx_coord(j,count,k) = pos_bdr_face_vtx(j,i,k);                                   
                }
                count++;
            }
           
        }
       

        
        for(i=0;i<run_input.bc_list.get_dim(0);i++){
            temp_bc_flag=run_input.bc_list(i).get_bc_flag();
            if(temp_bc_flag==SUB_IN_SIMP||temp_bc_flag==SUB_IN_CHAR||temp_bc_flag==SUP_IN){
                inlet_bc_flag=temp_bc_flag;
                id=i;
            }
        }
        inlet.id=id;


        inlet.total_area=cal_inlet_area();
        

        if(FlowSol->rank==0){
            cout<<"func1 area= "<<inlet.total_area<<endl;
            // cout<<"func2 area2= "<<area2<<endl;
        }

        if(FlowSol->rank==0){
            read_sem_restart(in_file_num, rest_info);
        }
        #ifdef _MPI
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Bcast(&rest_info, 1, MPI_INT, 0, MPI_COMM_WORLD);
        #endif

        if(rest_info == 0){

            //add_inlet_information
            inlet.type=run_input.bc_list(id).type;
            if(inlet.type==0){
                return;
            }
            inlet.mode=run_input.bc_list(id).mode;
            inlet.vis_y=run_input.bc_list(id).vis_y;            
            inlet.turb_1=run_input.bc_list(id).turb_1;
            inlet.turb_2=run_input.bc_list(id).turb_2;
            if(inlet.type==2){
                inlet.n_eddy=run_input.bc_list(id).n_eddy;            
                inlet.eddy_pos.setup(inlet.n_eddy,n_dims);                       
                inlet.sgn.setup(inlet.n_eddy,n_dims);
                inlet.eddy_pos.initialize_to_zero(); 
                inlet.sgn.initialize_to_zero();

            }
            
            
        }
        else{
            inlet.type=2;
            inlet.initialize=0;
        #ifdef _MPI
            MPI_Bcast(&inlet.mode,1, MPI_INT, 0, MPI_COMM_WORLD);
            MPI_Bcast(&inlet.vis_y,1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            MPI_Bcast(&inlet.turb_1,1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            MPI_Bcast(&inlet.turb_2,1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            MPI_Bcast(&inlet.n_eddy,1, MPI_INT, 0, MPI_COMM_WORLD);

            MPI_Bcast(inlet.eddy_pos.get_ptr_cpu(),inlet.n_eddy*3, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            MPI_Bcast(inlet.sgn.get_ptr_cpu(),inlet.n_eddy*3, MPI_INT, 0, MPI_COMM_WORLD);
        #endif

        }
    }

        

}
void bdy_inters::update_les_inlet(struct solution* FlowSol)
{
    if(n_inters!=0){
        int i,j,k,ibs;
                
        double max_fluc,fluc_bar;    

        

        if(FlowSol->rank==0)
            cout<<"Updateing LES inlet "<<endl;


        cal_inlet_rou_vel(FlowSol->time);

        cal_inlet_r_ij();

        
        // ofstream testw;

        // testw.open("inlet_sgn_eddy.dat");

        
        // for(i=0;i<inlet.n_eddy;i++){
        //     testw<<inlet.eddy_pos(i,0)<<" "<<inlet.eddy_pos(i,1)<<" "<<inlet.eddy_pos(i,2)<<" "<<endl;
        // }

        // for(i=0;i<inlet.n_eddy;i++){
        //     testw<<inlet.sgn(i,0)<<" "<<inlet.sgn(i,1)<<" "<<inlet.sgn(i,2)<<" "<<endl;
        // }
        // testw.close();


        if(inlet.type==1)
        {
            gen_fluc_random();
        }

        else if (inlet.type==2)
        {
            gen_fluc_sem(FlowSol);
        }  


   

        for(i=0;i<inlet.nbs;i++){
            for(j=0;j<n_fpts_per_inter;j++){
                for(k=0;k<3;k++){
                    inlet.fluctuations(j,i,k)/=run_input.uvw_ref;
                }
            }
        }  

        

        rescale_rij();
        correct_mass(FlowSol);



        max_fluc=-__DBL_MAX__;

        for (i=0; i<inlet.nbs; i++)
        {
            for (j=0; j<n_fpts_per_inter; j++)
            {
                fluc_bar=0.0;
                for(k=0; k<3;k++){
                    fluc_bar+=pow(inlet.fluctuations(j,i,k),2);
                }
                fluc_bar=sqrt(fluc_bar);
                max_fluc=max(max_fluc,fluc_bar);
            }
                
        }
        #ifdef _MPI
        MPI_Barrier(MPI_COMM_WORLD);
        double max_fluc_global;
        MPI_Reduce(&max_fluc, &max_fluc_global, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        max_fluc=max_fluc_global;
        #endif

        if(FlowSol->rank==0){
            cout<<"Maximum fluctuations magnitude="<<max_fluc<<endl;
        }

    }
    
}
    


void bdy_inters::gen_fluc_random(){

    int i,j,k;
    unsigned seed = chrono::system_clock::now().time_since_epoch().count();

    default_random_engine generator(seed);
    // default_random_engine e;
    // 第一个参数为高斯分布的平均值，第二个参数为标准差
    normal_distribution<double> distribution(0.0, 1.0);

    for(i=0;i<inlet.nbs;i++){
        for(j=0;j<n_fpts_per_inter;j++){
            for(k=0;k<n_dims;k++){
                inlet.fluctuations(j,i,k)=distribution(generator);
            }
        }
    }
}
void bdy_inters::gen_fluc_sem(struct solution* FlowSol){

    int i,j,k,m;
    int ibs;
    double ls_min;
    hf_array<double> ls;
    hf_array<double> temp_pos,temp_pos_2;
    double temp_random;
    hf_array<double> vel_c(n_dims);
    hf_array<double> bounding_box_min(3), bounding_box_max(3);
    hf_array<double> bounding_box_dimension(3);
    double bounding_box_vol;
    bool new_struct;
    hf_array<int> randomize(3);    
    hf_array<double> temp_dist(3),temp_dist2(3);
    double temp_dist_mag,temp_dist_mag2;
    double alpha,form_func;

    vel_c.initialize_to_zero();    
    ls.setup(inlet.nbs,3);
    ls.initialize_to_zero();
    inlet.fluctuations.initialize_to_zero();
    
    bounding_box_min(0)=__DBL_MAX__;
    bounding_box_max(0)=-__DBL_MAX__;

    ofstream sem_output;

    char file_name_s[256];

    char file_name_s_2[256];

    sprintf(file_name_s, "sem_output_rank%d.dat", FlowSol->rank);
    sem_output.open(file_name_s);

    //calculate bounding box axial width
    for (i=0;i<inlet.nbs;i++)
    {
        for(j=0;j<n_vtx;j++){
            temp_pos=cart2cyl(inlet.face_vtx_coord(j,i,0),inlet.face_vtx_coord(j,i,1),inlet.face_vtx_coord(j,i,2));
            bounding_box_min(0)=min(bounding_box_min(0),temp_pos(0));
            bounding_box_max(0)=max(bounding_box_max(0),temp_pos(0));
        }

    }

    

    // cal cutoff length-scale
    if (in_ele_type == TRI) //tri
    {            
    ls_min=FlowSol->mesh_eles_tris.calc_inlet_length_scale();
    }
    else if (in_ele_type == QUAD) // quad
    {
    ls_min=FlowSol->mesh_eles_quads.calc_inlet_length_scale();
    }
    else if (in_ele_type == TET) //tet
    {
    ls_min=FlowSol->mesh_eles_tets.calc_inlet_length_scale();
    }
    else if (in_ele_type == PRISM) //pri
    {
    ls_min=FlowSol->mesh_eles_pris.calc_inlet_length_scale();
    }
    else if (in_ele_type == HEX) //hex
    {
    ls_min=FlowSol->mesh_eles_hexas.calc_inlet_length_scale();
    }

    cout.precision(6);

    
    for(i=0;i<inlet.nbs;i++){
        for(j=0;j<3;j++){                
            if(inlet.mode==0){
            ls(i,j)=max(ls_min, pow(inlet.C_mu,0.75)*pow(inlet.turb_1,1.5)/inlet.turb_2);
            }
            //need to be added about wall_distance
            else if(inlet.mode==1){
                //ls(j, i) = max(ls_min, dh*(0.14 - 0.08*(1 - abs(wdis(icb))/dh)**2 - 0.06*(1 - abs(wdis(icb))/dh)**4))
            }
            

        }
    }
    if(FlowSol->rank==0){
        cout<<"ls_min="<<ls_min<<endl;
        cout<<"ls="<<pow(inlet.C_mu,0.75)*pow(inlet.turb_1,1.5)/inlet.turb_2<<endl;
    }
    

    bounding_box_min(1)=__DBL_MAX__;
    bounding_box_max(1)=-__DBL_MAX__; 
    bounding_box_min(2)=__DBL_MAX__;
    bounding_box_max(2)=-__DBL_MAX__;

    for(i=0;i<inlet.nbs;i++){
        for(j=0;j<n_vtx;j++){
            temp_pos=cart2cyl(inlet.face_vtx_coord(j,i,0),inlet.face_vtx_coord(j,i,1),inlet.face_vtx_coord(j,i,2));
            bounding_box_min(1)=min(bounding_box_min(1),temp_pos(1));
            bounding_box_max(1)=max(bounding_box_max(1),temp_pos(1));
            bounding_box_min(2)=min(bounding_box_min(2),temp_pos(2)-ls(i,0));
            bounding_box_max(2)=max(bounding_box_max(2),temp_pos(2)+ls(i,0));
        }
    }
    // no eddies generated in viscous sublayer
    bounding_box_min(0) = bounding_box_min(0) + inlet.vis_y;
    bounding_box_max(0) = bounding_box_max(0) - inlet.vis_y;

    sem_output<<"bounding_box 0"<<setw(18)<<bounding_box_min(0)<<" "<<setw(18)<<bounding_box_max(0)<<endl;

    sem_output<<"bounding_box 1"<<setw(18)<<bounding_box_min(1)<<" "<<setw(18)<<bounding_box_max(1)<<endl;

    sem_output<<"bounding_box 2"<<setw(18)<<bounding_box_min(2)<<" "<<setw(18)<<bounding_box_max(2)<<endl;

#ifdef _MPI
    hf_array<double> min_glob(3), max_glob(3);
    MPI_Allreduce(bounding_box_min.get_ptr_cpu(), min_glob.get_ptr_cpu(), 3, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(bounding_box_max.get_ptr_cpu(), max_glob.get_ptr_cpu(), 3, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    bounding_box_min=min_glob;
    bounding_box_max=max_glob;
#endif



    

    for(i=0;i<3;i++){
        bounding_box_dimension(i)=bounding_box_max(i)-bounding_box_min(i);
    }
    bounding_box_vol=(pow(bounding_box_max(0),2)-pow(bounding_box_min(0),2))*bounding_box_dimension(1)/2*bounding_box_dimension(2);

    
    //initialize sem
    srand(time(NULL));
    if(inlet.initialize == 1){

        if(FlowSol->rank==0){

            for(i=0;i<inlet.n_eddy;i++){

                for(j=0;j<3;j++){
                    temp_random=rand()/double(RAND_MAX);
                    if(temp_random<0.5)
                        inlet.sgn(i,j)=-1;
                    else
                        inlet.sgn(i,j)=1;
                        
                }

                for(j=0;j<3;j++){
                    temp_random=rand()/double(RAND_MAX);
                    temp_pos(j)=bounding_box_min(j)+temp_random*bounding_box_dimension(j);

                }
                temp_pos_2=cyl2cart(temp_pos(0),temp_pos(1),temp_pos(2));
                for(j=0;j<3;j++){
                    inlet.eddy_pos(i,j)=temp_pos_2(j);
                }

            }
        }
#ifdef _MPI
        
        MPI_Bcast(inlet.eddy_pos.get_ptr_cpu(),inlet.n_eddy*3, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(inlet.sgn.get_ptr_cpu(),inlet.n_eddy*3, MPI_DOUBLE, 0, MPI_COMM_WORLD);
#endif

        inlet.initialize=0;
    }

    //Estimation of the convection speed
    cal_convection_speed(vel_c);


    for (i=0; i<inlet.nbs; i++)
    {        
        for (j=0; j<n_fpts_per_inter; j++)
        {   sem_output<<"v0= "<<inlet.v(j,i,0)<<" v1= "<<inlet.v(j,i,1)<<" v2= "<<inlet.v(j,i,2)<<endl;
        }
    }
    
    
    sem_output<<"vel_c "<<setw(18)<<vel_c(0)<<" "<<setw(18)<<vel_c(1)<<" "<<setw(18)<<vel_c(2)<<endl;


    //advance eddies
    if (FlowSol->rank==0)
    {
        for ( i = 0; i < inlet.n_eddy; i++)
        {
            //cout<<"eddy pos before="<<inlet.eddy_pos(i,0)<<" "<<inlet.eddy_pos(i,1)<<" "<<inlet.eddy_pos(i,2)<<endl;
            for ( j = 0; j < 3; j++)
            {
                
                inlet.eddy_pos(i,j)=inlet.eddy_pos(i,j)+vel_c(j)*run_input.dt;
            }
            //cout<<"eddy pos after="<<inlet.eddy_pos(i,0)<<" "<<inlet.eddy_pos(i,1)<<" "<<inlet.eddy_pos(i,2)<<endl;
            
        }
        
        // cout.precision(9);
    
        for ( i = 0; i < inlet.n_eddy; i++)
        {
            new_struct=false;
            randomize.initialize_to_value(1);
            temp_pos=cart2cyl(inlet.eddy_pos(i,0),inlet.eddy_pos(i,1),inlet.eddy_pos(i,2));
            
            for(j=0;j<3;j++){

                if(temp_pos(j)<bounding_box_min(j)){
                    new_struct=true;
                    randomize(j)=0;
                    //cout<<"eddy pos cyl bf="<<temp_pos(0)<<" "<<temp_pos(1)<<" "<<temp_pos(2)<<endl;
                    temp_pos(j)+=bounding_box_dimension(j);
                    //cout<<"eddy pos cyl af="<<temp_pos(0)<<" "<<temp_pos(1)<<" "<<temp_pos(2)<<endl;
                }

                else if(temp_pos(j)>bounding_box_max(j)){
                    new_struct=true;
                    randomize(j)=0;
                    //cout<<"eddy pos cyl bf="<<temp_pos(0)<<" "<<temp_pos(1)<<" "<<temp_pos(2)<<endl;
                    temp_pos(j)-=bounding_box_dimension(j);
                    //cout<<"eddy pos cyl af="<<temp_pos(0)<<" "<<temp_pos(1)<<" "<<temp_pos(2)<<endl;
                }

            }
            
            if(new_struct==true){
               // cout<<randomize(0)<<" "<<randomize(1)<<" "<<randomize(2)<<endl;
                for(j=0;j<3;j++){

                    if(randomize(j)==1)
                    {
                        temp_random=rand()/double(RAND_MAX);
                        temp_pos(j)=bounding_box_min(j)+bounding_box_dimension(j)*temp_random;

                    }             
                }
                for(j=0;j<3;j++){

                    temp_random=rand()/double(RAND_MAX);
                    if(temp_random<0.5)
                        inlet.sgn(i,j)=-1;
                    else
                        inlet.sgn(i,j)=1;
                }
            }

            temp_pos_2=cyl2cart(temp_pos(0),temp_pos(1),temp_pos(2));

            for(j=0;j<3;j++){

                inlet.eddy_pos(i,j)=temp_pos_2(j);

            }

        
        }

    }
#ifdef _MPI
    MPI_Barrier(MPI_COMM_WORLD); //wait for master node
    MPI_Bcast(inlet.eddy_pos.get_ptr_cpu(),inlet.n_eddy*3, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(inlet.sgn.get_ptr_cpu(),inlet.n_eddy*3, MPI_INT, 0, MPI_COMM_WORLD);
#endif

    // sem_output.close();

    //calculate fluctuations

    alpha = sqrt(bounding_box_vol/inlet.n_eddy); 

    for(i=0;i<inlet.nbs;i++){

        ibs=inlet.ibslst[i];

        for(j=0;j<n_fpts_per_inter;j++){

            for(k=0;k<inlet.n_eddy;k++){

                for (m=0; m<n_dims; m++){

                    temp_dist(m) = abs((*pos_fpts(j,ibs,m))-inlet.eddy_pos(k,m));

                }
                temp_dist_mag=sqrt(pow(temp_dist(0),2)+pow(temp_dist(1),2)+pow(temp_dist(2),2));

                temp_pos=cart2cyl(inlet.eddy_pos(i,0),inlet.eddy_pos(i,1),inlet.eddy_pos(i,2));
                temp_pos(1)+=bounding_box_dimension(1);
                temp_pos_2=cyl2cart(temp_pos(0),temp_pos(1),temp_pos(2));

                for (m=0; m<n_dims; m++){

                    temp_dist2(m) = abs((*pos_fpts(j,ibs,m))-temp_pos_2(m));

                }
                temp_dist_mag2=sqrt(pow(temp_dist2(0),2)+pow(temp_dist2(1),2)+pow(temp_dist2(2),2));

                if(temp_dist_mag2<temp_dist_mag){
                    temp_dist=temp_dist2;
                    temp_dist_mag=temp_dist_mag2;
                }
        
                temp_pos(1)-=2*bounding_box_dimension(1);
                temp_pos_2=cyl2cart(temp_pos(0),temp_pos(1),temp_pos(2));

                for (m=0; m<n_dims; m++){

                    temp_dist2(m) = abs((*pos_fpts(j,ibs,m))-temp_pos_2(m));

                }
                temp_dist_mag2=sqrt(pow(temp_dist2(0),2)+pow(temp_dist2(1),2)+pow(temp_dist2(2),2));

                if(temp_dist_mag2<temp_dist_mag){
                    temp_dist=temp_dist2;                        
                }

                if(temp_dist(0)<ls(i,0) && temp_dist(1)<ls(i,1) && temp_dist(2)<ls(i,2)){

                    form_func=1.0;

                    for(m=0; m<n_dims; m++){

                        form_func=form_func*(1.0-temp_dist(m)/ls(i,m))/sqrt(2.0/3.0*ls(i,m));

                    }

                    for(m=0; m<n_dims; m++){

                        inlet.fluctuations(j,i,m)+=inlet.sgn(k,m)*form_func;

                    }
                    
                }                  

            }
            inlet.fluctuations(j,i,m)*=alpha;
        }
    }              
    
   
}

void bdy_inters::rescale_rij(){
    int i,j;
    hf_array<double> corr_mat(6);
    double u_corr,v_corr,w_corr;

    // ofstream testw;
    // testw.open("inlet_fluc_0.dat");

        
    // for(i=0;i<inlet.nbs;i++){
    //     for(j=0;j<n_fpts_per_inter;j++){
    //     testw<<inlet.fluctuations(j,i,0)<<" "<<inlet.fluctuations(j,i,1)<<" "<<inlet.fluctuations(j,i,2)<<" "<<endl;
    //     }
    // }

    
    // testw.close();

    for(i=0;i<inlet.nbs;i++){

        for(j=0;j<n_fpts_per_inter;j++){
            corr_mat.initialize_to_zero();
            corr_mat(0) = sqrt(inlet.r_ij(j,i,0));
            corr_mat(3) = inlet.r_ij(j,i,3)/corr_mat(0);
            corr_mat(1) = sqrt(inlet.r_ij(j,i,1) - corr_mat(3)*corr_mat(3));
            corr_mat(4) = inlet.r_ij(j,i,4)/corr_mat(0);
            corr_mat(5) = (inlet.r_ij(j,i,5) - corr_mat(3)*corr_mat(4))/corr_mat(1);
            corr_mat(2) = sqrt(max(inlet.r_ij(j,i,2) - corr_mat(4)*corr_mat(4) - corr_mat(5)*corr_mat(5), 0.0));
        }

        u_corr=corr_mat(0)*inlet.fluctuations(j,i,0);
        v_corr=corr_mat(3)*inlet.fluctuations(j,i,0)+corr_mat(1)*inlet.fluctuations(j,i,1);
        w_corr=corr_mat(4)*inlet.fluctuations(j,i,0)+corr_mat(5)*inlet.fluctuations(j,i,1)+corr_mat(2)*inlet.fluctuations(j,i,2);

        inlet.fluctuations(j,i,0)=u_corr;
        inlet.fluctuations(j,i,1)=v_corr;
        inlet.fluctuations(j,i,2)=w_corr;
        
    }
        

        // testw.open("inlet_fluc_2.dat");

        
        // for(i=0;i<inlet.nbs;i++){
        //     for(j=0;j<n_fpts_per_inter;j++){
        //     testw<<inlet.fluctuations(j,i,0)<<" "<<inlet.fluctuations(j,i,1)<<" "<<inlet.fluctuations(j,i,2)<<" "<<endl;
        //     }
        // }

       
        // testw.close();
}
void bdy_inters::correct_mass(struct solution* FlowSol){

    int i,j,k,ibs;

    double mass_flux;

    double detjac,wgt;

    ofstream mass_in,geo_in;

    char file_name_s[256];

    char file_name_s_2[256];

    sprintf(file_name_s, "fluc_mass_flux_rank%d.dat", FlowSol->rank);
    mass_in.open(file_name_s);

    
    mass_flux=0.0;

    for (i=0; i<inlet.nbs; i++)
    {
        ibs=inlet.ibslst[i];
        for (j=0; j<n_fpts_per_inter; j++)
        {
            wgt = *weight_fpts(j,ibs);
            detjac = *inter_detjac_inters_cubpts(j,ibs);
            mass_flux+=wgt*detjac*inlet.fluctuations(j,i,0)*inlet.rou(j,i);
            mass_in<<i<<" "<<j<<" "<<"flu="<<setw(18) <<inlet.fluctuations(j,i,0)<<" "<<setw(18) <<inlet.fluctuations(j,i,1)<<" "<<setw(18) <<inlet.fluctuations(j,i,2)<<" rou="<<inlet.rou(j,i)<<" mass = "<<mass_flux<<endl;

        }
    }

    mass_in<<mass_flux;
    mass_in.close();

    sprintf(file_name_s_2, "geo%d.dat", FlowSol->rank);
    geo_in.open(file_name_s_2);
    for(i=0;i<inlet.nbs;i++){
        for(j=0;j<n_vtx;j++){
            geo_in<<i<<" "<<j<<" "<<"pos="<<setw(18) <<inlet.face_vtx_coord(j,i,0)<<" "<<setw(18) <<inlet.face_vtx_coord(j,i,1)<<" "<<setw(18) <<inlet.face_vtx_coord(j,i,2)<<endl;

        }
    }   
    geo_in.close(); 

#ifdef _MPI
    double mass_flux_global;
    MPI_Allreduce(&mass_flux, &mass_flux_global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    mass_flux=mass_flux_global;
#endif

    if(FlowSol->rank==0){
        cout<<"mass flux err before correct"<<mass_flux<<endl;
    }

#ifdef _MPI
    MPI_Barrier(MPI_COMM_WORLD);
#endif



    sprintf(file_name_s, "fluc_mass_flux_2_rank%d.dat", FlowSol->rank);
    mass_in.open(file_name_s);

    


    for (i=0; i<inlet.nbs; i++)
    {
        ibs=inlet.ibslst[i];
        for (j=0; j<n_fpts_per_inter; j++){

            inlet.fluctuations(j,i,0)-=mass_flux/(inlet.total_area*inlet.rou(j,i));

        }
    }
    
    mass_flux=0;
    for (i=0; i<inlet.nbs; i++)
    {
        ibs=inlet.ibslst[i];
        for (j=0; j<n_fpts_per_inter; j++)
        {
            wgt = *weight_fpts(j,ibs);
            detjac = *inter_detjac_inters_cubpts(j,ibs);
            mass_flux+=wgt*detjac*inlet.fluctuations(j,i,0)*inlet.rou(j,i);
            mass_in<<i<<" "<<j<<" "<<"flu="<<setw(18) <<inlet.fluctuations(j,i,0)<<" "<<setw(18)<<inlet.fluctuations(j,i,1)<<" "<<setw(18) <<inlet.fluctuations(j,i,2)<<" rou="<<inlet.rou(j,i)<<" mass = "<<mass_flux<<endl;


        }
    }

   mass_in<<mass_flux;
   mass_in.close();

#ifdef _MPI
    double mass_flux_global_2;
    MPI_Allreduce(&mass_flux, &mass_flux_global_2, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    mass_flux=mass_flux_global_2;
#endif
    if(FlowSol->rank==0){
        cout<<"mass flux err after correct"<<mass_flux<<endl;
    }


}

void bdy_inters::cal_inlet_rou_vel(double time_bound){

    hf_array<double> norm(n_dims), fn(n_fields);

    //viscous
    hf_array<double> u_c(n_fields);

    int count;
    count=0;
    for(int i=0; i<n_inters; i++)//loop over boundary interfaces
    {
        int temp_bc_flag=run_input.bc_list(boundary_id(i)).get_bc_flag();
        if(temp_bc_flag==SUB_IN_SIMP||temp_bc_flag==SUB_IN_CHAR||temp_bc_flag==SUP_IN){
            
        
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

                //calculate inviscid boundary solution
                set_boundary_conditions(0, boundary_id(i), temp_u_l.get_ptr_cpu(), temp_u_r.get_ptr_cpu(),
                                        norm.get_ptr_cpu(), temp_loc.get_ptr_cpu(), run_input.gamma, run_input.R_ref, time_bound, run_input.equation); 
                
                for (int m=0; m<n_dims; m++){
                    inlet.v(j,count,m)= temp_u_r(m+1)/temp_u_r(0);
                }
                

                inlet.rou(j,count)=temp_u_r(0);

            }

            count++;
        }
    }

}
void bdy_inters::cal_inlet_r_ij(){
    int i,j,k;
    double u_bar;
    if(inlet.mode==0){                  
        for(i=0;i<inlet.nbs;i++){
            for(j=0;j<n_fpts_per_inter;j++){
                for(k=0;k<3;k++){
                    inlet.r_ij(j,i,k)=2.0/3.0*inlet.turb_1;
                }
            }
            
        }
    }
    else if(inlet.mode==1){
        for(i=0;i<inlet.nbs;i++){
            for(j=0;j<n_fpts_per_inter;j++){                   
                u_bar=0.;
                for (k=0; k<n_dims; k++){
                    u_bar+=inlet.v(j,i,k)*inlet.v(j,i,k);
                }
                u_bar=sqrt(u_bar);
        
                for(k=0;k<3;k++){
                    inlet.r_ij(j,i,k)=pow(inlet.turb_1*u_bar,2);
                }
            }
        }
    }

};
void bdy_inters::cal_convection_speed(hf_array<double> &vel_c){

    int i,j,k,ibs;
    double area;
    double wgt,detjac;

    area=0.0;

    for (i=0; i<inlet.nbs; i++)
    {
        ibs=inlet.ibslst[i];

        for (j=0; j<n_fpts_per_inter; j++)
        {
            wgt = *weight_fpts(j,ibs);
            detjac = *inter_detjac_inters_cubpts(j,ibs);
            for(k=0;k<3;k++){
                vel_c(k)+=inlet.v(j,i,k)*wgt*detjac;
            }

        }
        
    }

    // cout<<"vc ="<<vel_c(0)<<"vc ="<<vel_c(1)<<"vc ="<<vel_c(2)<<endl;

#ifdef _MPI
    hf_array<double> total_vel_c(3);
    total_vel_c.initialize_to_zero();
    MPI_Allreduce(vel_c.get_ptr_cpu(), total_vel_c.get_ptr_cpu(), 3, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    vel_c=total_vel_c;

#endif
    for(i=0;i<3;i++){
        vel_c(i)=vel_c(i)/inlet.total_area;
    }

}


double bdy_inters::cal_inlet_area(){

    int i,j,ibs;
    double area;
    double wgt,detjac;

    area=0.0;

    for (i=0; i<inlet.nbs; i++)
    {
        ibs=inlet.ibslst[i];

        for (j=0; j<n_fpts_per_inter; j++)
        {
            wgt = *weight_fpts(j,ibs);
            detjac = *inter_detjac_inters_cubpts(j,ibs);
            area+=wgt*detjac;

        }
    }
#ifdef _MPI
    double total_area;
    MPI_Allreduce(&area, &total_area, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    area=total_area;

#endif
    
    return area;

}


void bdy_inters::write_sem_restart(int in_file_num){
    if(n_inters!=0){
        ofstream sem_out;
        char file_name_s[256];
        char forcedir_s[256];
        struct stat st = {0};
        int i;
        // set name of directory to store output files
        sprintf(forcedir_s, "sem_files");

        // Create directory and set name of files

        if (inlet.type == 2)
        {
            if (stat(forcedir_s, &st) == -1)
                mkdir(forcedir_s, 0755);

            sprintf(file_name_s, "%s/sem_files_%09d.dat", forcedir_s, in_file_num);
            sem_out.open(file_name_s);
            //write mode
            sem_out<<inlet.mode<<endl;
            sem_out<<inlet.vis_y<<endl;
            //write turb 1 turb 2
            if(inlet.mode==0){
                sem_out<<inlet.turb_1<<" "<<inlet.turb_2<<endl;
            }
            else if(inlet.mode==1){
                sem_out<<inlet.turb_1<<endl;
            }
            //write n_eddy
            sem_out<<inlet.n_eddy<<endl;

            //write coordinates and sign
            for(i=0;i<inlet.n_eddy;i++){

                sem_out<<scientific << setw(18) << setprecision(12)<<inlet.eddy_pos(i,0)<<" "<< setw(18) << setprecision(12)<<inlet.eddy_pos(i,1)<<" "<< setw(18) << setprecision(12)<<inlet.eddy_pos(i,2)<<endl;

            }
            for(i=0;i<inlet.n_eddy;i++){

                sem_out<<inlet.sgn(i,0)<<" "<<inlet.sgn(i,1)<<" "<<inlet.sgn(i,2)<<endl;

            }
            
            sem_out.close();
        }
            
    }


}
void bdy_inters::read_sem_restart(int in_file_num,int &rest_info){
    char file_name_s[50];
    ifstream sem_in;
    int i;
    sprintf(file_name_s, "sem_files/sem_files_%09d.dat", in_file_num);
    sem_in.open(file_name_s);

    if(sem_in.is_open()){
        rest_info=1;
    }
    else{
        rest_info=0;
        return;
    }
    //read mode
    sem_in>>inlet.mode;
    sem_in>>inlet.vis_y;
    //write turb 1 turb 2
    if(inlet.mode==0){
        sem_in>>inlet.turb_1>>inlet.turb_2;
    }
        else if(inlet.mode==1){
        sem_in>>inlet.turb_1;
    }
    //read n_eddy
    sem_in>>inlet.n_eddy;
    //read coordinates and sign

    inlet.eddy_pos.setup(inlet.n_eddy,3);

    for(i=0;i<inlet.n_eddy;i++){
        sem_in>>inlet.eddy_pos(i,0)>>inlet.eddy_pos(i,1)>>inlet.eddy_pos(i,2);
    }

    inlet.sgn.setup(inlet.n_eddy,3);
    for(i=0;i<inlet.n_eddy;i++){
        sem_in>>inlet.sgn(i,0)>>inlet.sgn(i,1)>>inlet.sgn(i,2);
    }


    sem_in.close();



}
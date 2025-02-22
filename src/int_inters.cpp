/*!
 * \file int_inters.cpp
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

#include "../include/global.h"
#include "../include/int_inters.h"
#include "../include/solver.h"
#include "../include/flux.h"
#if defined _GPU
#include "../include/cuda_kernels.h"
#endif

using namespace std;

// #### constructors ####

// default constructor

int_inters::int_inters(){ }

int_inters::~int_inters() { }

// #### methods ####

// setup inters

void int_inters::setup(int in_n_inters,int in_inter_type)
{

  (*this).setup_inters(in_n_inters,in_inter_type);

      disu_fpts_r.setup(n_fpts_per_inter,n_inters,n_fields);
      norm_tconf_fpts_r.setup(n_fpts_per_inter,n_inters,n_fields);
      detjac_fpts_r.setup(n_fpts_per_inter,n_inters);
      tdA_fpts_r.setup(n_fpts_per_inter,n_inters);

      if(viscous)
        {
          grad_disu_fpts_r.setup(n_fpts_per_inter,n_inters,n_fields,n_dims);
          delta_disu_fpts_r.setup(n_fpts_per_inter,n_inters,n_fields);
        }
}

// set interior interface
void int_inters::set_interior(int in_inter, int in_ele_type_l, int in_ele_type_r, int in_ele_l, int in_ele_r, int in_local_inter_l, int in_local_inter_r, int rot_tag, struct solution* FlowSol)
{
  int i,j,k;
  int i_rhs,j_rhs;

      get_lut(rot_tag);

      for(i=0;i<n_fields;i++)
        {
          for(j=0;j<n_fpts_per_inter;j++)
            {
              j_rhs=lut(j);

              disu_fpts_l(j,in_inter,i)=get_disu_fpts_ptr(in_ele_type_l,in_ele_l,i,in_local_inter_l,j,FlowSol);
              disu_fpts_r(j,in_inter,i)=get_disu_fpts_ptr(in_ele_type_r,in_ele_r,i,in_local_inter_r,j_rhs,FlowSol);

              norm_tconf_fpts_l(j,in_inter,i)=get_norm_tconf_fpts_ptr(in_ele_type_l,in_ele_l,i,in_local_inter_l,j,FlowSol);
              norm_tconf_fpts_r(j,in_inter,i)=get_norm_tconf_fpts_ptr(in_ele_type_r,in_ele_r,i,in_local_inter_r,j_rhs,FlowSol);

              for (k=0;k<n_dims;k++)
                {
                  if(viscous)
                    {

                      delta_disu_fpts_l(j,in_inter,i)=get_delta_disu_fpts_ptr(in_ele_type_l,in_ele_l,i,in_local_inter_l,j,FlowSol);
                      delta_disu_fpts_r(j,in_inter,i)=get_delta_disu_fpts_ptr(in_ele_type_r,in_ele_r,i,in_local_inter_r,j_rhs,FlowSol);

                      grad_disu_fpts_l(j,in_inter,i,k) = get_grad_disu_fpts_ptr(in_ele_type_l,in_ele_l,in_local_inter_l,i,k,j,FlowSol);
                      grad_disu_fpts_r(j,in_inter,i,k) = get_grad_disu_fpts_ptr(in_ele_type_r,in_ele_r,in_local_inter_r,i,k,j_rhs,FlowSol);
                    }

                  // Subgrid-scale flux
                  if(LES)
                    {
                      sgsf_fpts_l(j,in_inter,i,k) = get_sgsf_fpts_ptr(in_ele_type_l,in_ele_l,in_local_inter_l,i,k,j,FlowSol);
                      sgsf_fpts_r(j,in_inter,i,k) = get_sgsf_fpts_ptr(in_ele_type_r,in_ele_r,in_local_inter_r,i,k,j_rhs,FlowSol);
                    }
                }
            }
        }


      for(i=0;i<n_fpts_per_inter;i++)
        {
          i_rhs=lut(i);

          tdA_fpts_l(i,in_inter)=get_tdA_fpts_ptr(in_ele_type_l,in_ele_l,in_local_inter_l,i,FlowSol);
          tdA_fpts_r(i,in_inter)=get_tdA_fpts_ptr(in_ele_type_r,in_ele_r,in_local_inter_r,i_rhs,FlowSol);

          for(j=0;j<n_dims;j++)
            {
              norm_fpts(i,in_inter,j)=get_norm_fpts_ptr(in_ele_type_l,in_ele_l,in_local_inter_l,i,j,FlowSol);
            }
        }
}

// move all from cpu to gpu

void int_inters::mv_all_cpu_gpu(void)
{
#ifdef _GPU

  disu_fpts_l.mv_cpu_gpu();
  norm_tconf_fpts_l.mv_cpu_gpu();
  tdA_fpts_l.mv_cpu_gpu();
  norm_fpts.mv_cpu_gpu();

  disu_fpts_r.mv_cpu_gpu();
  norm_tconf_fpts_r.mv_cpu_gpu();
  tdA_fpts_r.mv_cpu_gpu();

  //detjac_fpts_r.mv_cpu_gpu();
  //detjac_fpts_l.mv_cpu_gpu();

  delta_disu_fpts_l.mv_cpu_gpu();
  delta_disu_fpts_r.mv_cpu_gpu();

  if(viscous)
    {
      grad_disu_fpts_l.mv_cpu_gpu();
      //norm_tconvisf_fpts_l.mv_cpu_gpu();

      grad_disu_fpts_r.mv_cpu_gpu();
      //norm_tconvisf_fpts_r.mv_cpu_gpu();
    }

  sgsf_fpts_l.mv_cpu_gpu();
  sgsf_fpts_r.mv_cpu_gpu();

#endif
}

// calculate normal transformed continuous inviscid flux at the flux points
void int_inters::calculate_common_invFlux(void)
{

#ifdef _CPU
  hf_array<double> norm(n_dims), fn(n_fields);

  //viscous
  hf_array<double> u_c(n_fields);

  for(int i=0;i<n_inters;i++)
  {
    for(int j=0;j<n_fpts_per_inter;j++)
    {

      // calculate discontinuous solution at flux points
      for(int k=0;k<n_fields;k++) {
        temp_u_l(k)=(*disu_fpts_l(j,i,k));
        temp_u_r(k)=(*disu_fpts_r(j,i,k));
      }

      // Interface unit-normal vector
 
        for (int m=0;m<n_dims;m++)
          norm(m) = *norm_fpts(j,i,m);
      

      // Calling Riemann solver
        if (run_input.riemann_solve_type == 0 || run_input.riemann_solve_type == 2 || run_input.riemann_solve_type == 3) // Rusanov or RoeM or HLLC
        {
          // calculate flux from discontinuous solution at flux points
          if (n_dims == 2)
          {
            calc_invf_2d(temp_u_l, temp_f_l);
            calc_invf_2d(temp_u_r, temp_f_r);
          }
          else if (n_dims == 3)
          {
            calc_invf_3d(temp_u_l, temp_f_l);
            calc_invf_3d(temp_u_r, temp_f_r);
          }
          else
            FatalError("ERROR: Invalid number of dimensions ... ");
          if (run_input.riemann_solve_type == 0)
            rusanov_flux(temp_u_l, temp_u_r, temp_f_l, temp_f_r, norm, fn, n_dims, n_fields, run_input.gamma);
          else if (run_input.riemann_solve_type == 2)
            roeM_flux(temp_u_l, temp_u_r, temp_f_l, temp_f_r, norm, fn, n_dims, n_fields, run_input.gamma);
          else
            hllc_flux(temp_u_l, temp_u_r, temp_f_l, temp_f_r, norm, fn, n_dims, n_fields, run_input.gamma);
        }
        else if (run_input.riemann_solve_type == 1)
        { // Lax-Friedrich
          lax_friedrich(temp_u_l, temp_u_r, norm, fn, n_dims, n_fields, run_input.lambda, run_input.wave_speed);
        }
        else
          FatalError("Riemann solver not implemented");

        // Transform back to reference space from static physical space
        for(int k=0;k<n_fields;k++) {
          (*norm_tconf_fpts_l(j,i,k))= fn(k)*(*tdA_fpts_l(j,i));
          (*norm_tconf_fpts_r(j,i,k))=-fn(k)*(*tdA_fpts_r(j,i));
        }
      

      if(viscous)
      {
        // Calling viscous riemann solver
        if (run_input.vis_riemann_solve_type==0)
          ldg_solution(0,temp_u_l,temp_u_r,u_c,run_input.ldg_beta,norm);
        else
          FatalError("Viscous Riemann solver not implemented");

          for(int k=0;k<n_fields;k++) {
            *delta_disu_fpts_l(j,i,k) = (u_c(k) - temp_u_l(k));
            *delta_disu_fpts_r(j,i,k) = (u_c(k) - temp_u_r(k));
          }
        
      }

    }
  }
#endif

#ifdef _GPU
  if (n_inters!=0)
  {
    calculate_common_invFlux_gpu_kernel_wrapper(n_fpts_per_inter,n_dims,n_fields,n_inters,disu_fpts_l.get_ptr_gpu(),disu_fpts_r.get_ptr_gpu(),norm_tconf_fpts_l.get_ptr_gpu(),norm_tconf_fpts_r.get_ptr_gpu(),tdA_fpts_l.get_ptr_gpu(),tdA_fpts_r.get_ptr_gpu(),ndA_dyn_fpts_l.get_ptr_gpu(),ndA_dyn_fpts_r.get_ptr_gpu(),J_dyn_fpts_l.get_ptr_gpu(),J_dyn_fpts_r.get_ptr_gpu(),norm_fpts.get_ptr_gpu(),norm_dyn_fpts.get_ptr_gpu(),grid_vel_fpts.get_ptr_gpu(),run_input.riemann_solve_type,delta_disu_fpts_l.get_ptr_gpu(),delta_disu_fpts_r.get_ptr_gpu(),run_input.gamma,run_input.ldg_beta,viscous,motion,run_input.vis_riemann_solve_type,run_input.wave_speed(0),run_input.wave_speed(1),run_input.wave_speed(2),run_input.lambda,run_input.RANS);
    //cout << "Done with common invFlux" << endl;
  }
#endif
}


// calculate normal transformed continuous viscous flux at the flux points

void int_inters::calculate_common_viscFlux(void)
{

#ifdef _CPU
  hf_array<double> norm(n_dims), fn(n_fields);

  for(int i=0;i<n_inters;i++)
    {
      for(int j=0;j<n_fpts_per_inter;j++)
      {
        // obtain discontinuous solution at flux points

          for(int k=0;k<n_fields;k++)
          {
            temp_u_l(k)=(*disu_fpts_l(j,i,k));
            temp_u_r(k)=(*disu_fpts_r(j,i,k));
          }
        

          // obtain physical gradient of discontinuous solution at flux points

          for(int k=0;k<n_dims;k++)
            {
              for(int l=0;l<n_fields;l++)
                {
                  temp_grad_u_l(l,k) = *grad_disu_fpts_l(j,i,l,k);
                  temp_grad_u_r(l,k) = *grad_disu_fpts_r(j,i,l,k);
                }
            }

          // calculate flux from discontinuous solution at flux points

          if(n_dims==2)
            {
              calc_visf_2d(temp_u_l,temp_grad_u_l,temp_f_l);
              calc_visf_2d(temp_u_r,temp_grad_u_r,temp_f_r);
            }
          else if(n_dims==3)
            {
              calc_visf_3d(temp_u_l,temp_grad_u_l,temp_f_l);
              calc_visf_3d(temp_u_r,temp_grad_u_r,temp_f_r);
            }
          else
            FatalError("ERROR: Invalid number of dimensions ... ");

          // If LES, get physical SGS flux and add to viscous flux
          if (LES)
          {
            for (int k = 0; k < n_dims; k++)
            {
              for (int l = 0; l < n_fields; l++)
              {
                // pointers to subgrid-scale fluxes
                temp_sgsf_l(l, k) = *sgsf_fpts_l(j, i, l, k);
                temp_sgsf_r(l, k) = *sgsf_fpts_r(j, i, l, k);

                // Add SGS fluxes to viscous fluxes
                temp_f_l(l, k) += temp_sgsf_l(l, k);
                temp_f_r(l, k) += temp_sgsf_r(l, k);
              }
            }
          }

          // storing normal components
            for (int m=0;m<n_dims;m++)
              norm(m) = *norm_fpts(j,i,m);
          

          // Calling viscous riemann solver
          if (run_input.vis_riemann_solve_type==0)
            ldg_flux(0,temp_u_l,temp_u_r,temp_f_l,temp_f_r,norm,fn,n_dims,n_fields,run_input.ldg_tau,run_input.ldg_beta);
          else
            FatalError("Viscous Riemann solver not implemented");

          // Transform back to reference space
            for(int k=0;k<n_fields;k++) {
              (*norm_tconf_fpts_l(j,i,k))+=  fn(k)*(*tdA_fpts_l(j,i));
              (*norm_tconf_fpts_r(j,i,k))+= -fn(k)*(*tdA_fpts_r(j,i));
            }
          
        }
    }

#endif

#ifdef _GPU
  if (n_inters!=0)
    calculate_common_viscFlux_gpu_kernel_wrapper(n_fpts_per_inter,n_dims,n_fields,n_inters,disu_fpts_l.get_ptr_gpu(),disu_fpts_r.get_ptr_gpu(),grad_disu_fpts_l.get_ptr_gpu(),grad_disu_fpts_r.get_ptr_gpu(),norm_tconf_fpts_l.get_ptr_gpu(),norm_tconf_fpts_r.get_ptr_gpu(),tdA_fpts_l.get_ptr_gpu(),tdA_fpts_r.get_ptr_gpu(),ndA_dyn_fpts_l.get_ptr_gpu(),ndA_dyn_fpts_r.get_ptr_gpu(),J_dyn_fpts_l.get_ptr_gpu(),J_dyn_fpts_r.get_ptr_gpu(),norm_fpts.get_ptr_gpu(),norm_dyn_fpts.get_ptr_gpu(),sgsf_fpts_l.get_ptr_gpu(),sgsf_fpts_r.get_ptr_gpu(),run_input.riemann_solve_type,run_input.vis_riemann_solve_type,run_input.ldg_beta,run_input.ldg_tau,run_input.gamma,run_input.prandtl,run_input.rt_inf,run_input.mu_inf,run_input.c_sth,run_input.fix_vis,run_input.equation,run_input.diff_coeff,LES,motion,run_input.RANS,run_input.c_v1,run_input.omega,run_input.prandtl_t);
#endif
}


/*!
 * \file inters.cpp
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

#include "../include/inters.h"
#include "../include/solver.h"
#include "../include/flux.h"
#include "../include/solution.h"

#if defined _GPU
#include "../include/cuda_kernels.h"
#endif

using namespace std;

// #### constructors ####

// default constructor

inters::inters()
{
  order=run_input.order;
  viscous=run_input.viscous;
  LES = run_input.LES;
}

inters::~inters() { }

// #### methods ####

void inters::setup_inters(int in_n_inters, int in_inters_type)
{
  n_inters    = in_n_inters;
  inters_type = in_inters_type;

  if(inters_type==0) // segs
    {
      n_dims=2;

      if (run_input.equation==0)
        n_fields=4;
      else if (run_input.equation==1)
        n_fields=1;
      else
        FatalError("Equation not supported");

      n_fpts_per_inter=order+1;
    }
  else if(inters_type==1) // tris
    {
      n_dims=3;

      if (run_input.equation==0)
        n_fields=5;
      else if (run_input.equation==1)
        n_fields=1;
      else
        FatalError("Equation not supported");

      n_fpts_per_inter=(order+2)*(order+1)/2;
    }
  else if(inters_type==2) // quads
    {
      n_dims=3;
      if (run_input.equation==0)
        n_fields=5;
      else if (run_input.equation==1)
        n_fields=1;
      else
        FatalError("Equation not supported");

      n_fpts_per_inter=(order+1)*(order+1);
    }
  else
    {
      FatalError("ERROR: Invalid interface type ... ");
    }

  if (run_input.RANS==1)
    n_fields++;

      disu_fpts_l.setup(n_fpts_per_inter,n_inters,n_fields);
      norm_tconf_fpts_l.setup(n_fpts_per_inter,n_inters,n_fields);
      detjac_fpts_l.setup(n_fpts_per_inter,n_inters);
      tdA_fpts_l.setup(n_fpts_per_inter,n_inters);
      norm_fpts.setup(n_fpts_per_inter,n_inters,n_dims);
      pos_fpts.setup(n_fpts_per_inter,n_inters,n_dims);

      if(viscous)
        {
          delta_disu_fpts_l.setup(n_fpts_per_inter,n_inters,n_fields);
          grad_disu_fpts_l.setup(n_fpts_per_inter,n_inters,n_fields,n_dims);
          normal_disu_fpts_l.setup(n_fpts_per_inter,n_inters,n_fields);
          pos_disu_fpts_l.setup(n_fpts_per_inter,n_inters,n_dims);
        }

      if(LES) {
        sgsf_fpts_l.setup(n_fpts_per_inter,n_inters,n_fields,n_dims);
        sgsf_fpts_r.setup(n_fpts_per_inter,n_inters,n_fields,n_dims);
        temp_sgsf_l.setup(n_fields,n_dims);
        temp_sgsf_r.setup(n_fields,n_dims);
      }
      else {
        sgsf_fpts_l.setup(1);
        sgsf_fpts_r.setup(1);
      }

      temp_u_l.setup(n_fields);
      temp_u_r.setup(n_fields);

      temp_grad_u_l.setup(n_fields,n_dims);
      temp_grad_u_r.setup(n_fields,n_dims);

      temp_f_l.setup(n_fields,n_dims);
      temp_f_r.setup(n_fields,n_dims);

      temp_f.setup(n_fields,n_dims);


      temp_loc.setup(n_dims);

      lut.setup(n_fpts_per_inter);
}

// get look up table for flux point connectivity based on rotation tag
void inters::get_lut(int in_rot_tag)
{
  int i,j;

  if(inters_type==0) // segment
    {
      for(i=0;i<n_fpts_per_inter;i++)
        {
          lut(i)=n_fpts_per_inter-i-1;
        }
    }
  else if(inters_type==1) // triangle face
    {
      int index0,index1;
      if(in_rot_tag==0) // Example face 0 with 1
        {
          for(j=0;j<order+1;j++)
            {
              for (i=0;i<order-j+1;i++)
                {
                  index0 = j*(order+1) - (j-1)*j/2 + i;
                  index1 = i*(order+1) - (i-1)*i/2 + j;
                  lut(index0) = index1;

                }
            }
        }
      else if(in_rot_tag==1) // Example face 0 with 3
        {
          for(j=0;j<order+1;j++)
            {
              for (i=0;i<order+1-j;i++)
                {
                  index0 = j*(order+1) - (j-1)*j/2 + i;
                  index1 = (order+1)*(order+2)/2 -1 -(i+j)*(i+j+1)/2 -j;
                  lut(index0) = index1;

                }
            }
        }
      else if(in_rot_tag==2) // Example face 0 with 2
        {

          for(j=0;j<order+1;j++)
            {
              for (i=0;i<order+1-j;i++)
                {
                  index0 = j*(order+1) - (j-1)*j/2 + i;
                  index1 = j*(order+1) - (j-1)*j/2 + (order-j-i);
                  lut(index0) = index1;
                }
            }
        }
      else
        {
          cout << "ERROR: Unknown rotation of triangular face..." << endl;
        }
    }
  else if(inters_type==2) // quad face
    {
      if(in_rot_tag==0)
        {
          for(i=0;i<(order+1);i++)
            {
              for(j=0;j<(order+1);j++)
                {
                  lut((i*(order+1))+j)=((order+1)-1-j)+((order+1)*i);
                }
            }
        }
      else if(in_rot_tag==1)
        {
          for(i=0;i<(order+1);i++)
            {
              for(j=0;j<(order+1);j++)
                {
                  lut((i*(order+1))+j)=n_fpts_per_inter-((order+1)-1-j)-((order+1)*i)-1;
                }
            }
        }
      else if(in_rot_tag==2)
        {
          for(i=0;i<(order+1);i++)
            {
              for(j=0;j<(order+1);j++)
                {
                  lut((i*(order+1))+j)=((order+1)*j)+i;
                }
            }
        }
      else if(in_rot_tag==3)
        {
          for(i=0;i<(order+1);i++)
            {
              for(j=0;j<(order+1);j++)
                {
                  lut((i*(order+1))+j)=n_fpts_per_inter-((order+1)*j)-i-1;
                }
            }
        }
      else
        {
          cout << "ERROR: Unknown rotation tag ... " << endl;
        }
    }
  else
    {
      FatalError("ERROR: Invalid interface type ... ");
    }
}

// Rusanov inviscid numerical flux
void inters::right_flux(hf_array<double> &f_r, hf_array<double> &norm, hf_array<double> &fn, int n_dims, int n_fields, double gamma)
{
  // calculate normal flux from discontinuous solution at flux points
  for(int k=0;k<n_fields;k++) {
      fn(k)=0.;
      for(int l=0;l<n_dims;l++) {
          fn(k)+=f_r(k,l)*norm(l);
        }
    }
}

// Rusanov inviscid numerical flux(conservative form Riemann solver)
void inters::rusanov_flux(hf_array<double> &u_l, hf_array<double> &u_r, hf_array<double> &f_l, hf_array<double> &f_r, hf_array<double> &norm, hf_array<double> &fn, int n_dims, int n_fields, double gamma)
{
  double vn_l, vn_r, vsq_l, vsq_r, p_l, p_r, eig;
  hf_array<double> v_l(n_dims), v_r(n_dims);      //velocities
  hf_array<double> fn_l(n_fields), fn_r(n_fields);//noraml fluxes

  // calculate normal flux from discontinuous solution at flux points
#if defined _ACCELERATE_BLAS || defined _MKL_BLAS || defined _STANDARD_BLAS
  cblas_dgemv(CblasColMajor, CblasNoTrans, n_fields, n_dims, 1.0, f_l.get_ptr_cpu(), n_fields, norm.get_ptr_cpu(), 1, 0.0, fn_l.get_ptr_cpu(), 1);
  cblas_dgemv(CblasColMajor, CblasNoTrans, n_fields, n_dims, 1.0, f_r.get_ptr_cpu(), n_fields, norm.get_ptr_cpu(), 1, 0.0, fn_r.get_ptr_cpu(), 1);
#else
  fn_l.initialize_to_zero();
  fn_r.initialize_to_zero();
  for (int k = 0; k < n_fields; k++)
  {
    for (int l = 0; l < n_dims; l++)
    {
      fn_l(k) += f_l(k, l) * norm(l);
      fn_r(k) += f_r(k, l) * norm(l);
    }
  }
#endif

  vn_l = 0;
  vn_r = 0;
  vsq_l = 0;
  vsq_r = 0;

  // calculate wave speeds
  for (int i = 0; i < n_dims; i++)
  {
    v_l(i) = u_l(i + 1) / u_l(0);
    v_r(i) = u_r(i + 1) / u_r(0);
    vn_l += v_l(i) * norm(i);
    vn_r += v_r(i) * norm(i);
    vsq_l += pow(v_l(i), 2.);
    vsq_r += pow(v_r(i), 2.);
  }
  p_l = (gamma - 1.0) * (u_l(n_dims + 1) - 0.5 * u_l(0) * vsq_l);
  p_r = (gamma - 1.0) * (u_r(n_dims + 1) - 0.5 * u_r(0) * vsq_r);

  eig = sqrt(gamma * (p_l + p_r) / (u_l(0) + u_r(0))) + 0.5 * fabs(vn_l + vn_r);

  // calculate the normal continuous flux at the flux points

  for(int k=0;k<n_fields;k++)
    fn(k) = 0.5*( (fn_l(k)+fn_r(k)) - eig*(u_r(k)-u_l(k)) );
}

// RoeM inviscid numerical flux
void inters::roeM_flux(hf_array<double> &u_l, hf_array<double> &u_r, hf_array<double> &f_l, hf_array<double> &f_r, hf_array<double> &norm, hf_array<double> &fn, int n_dims, int n_fields, double gamma)
{
  // RoeM scheme (ref: JCP 185(2), 342-374)
  hf_array<double> v_l(n_dims), v_r(n_dims), va(n_dims), dv(n_dims); //velocities
  hf_array<double> du(n_fields), bdq(n_fields), fn_l(n_fields), fn_r(n_fields);//fluxes
  double p_l, p_r, h_l, h_r;
  double vn_l, vn_r, vsq_l, vsq_r;
  double drho, dp, dh, dvn;
  double sq_rho, rrho, ratr, ra, ha;
  double qq, va_n, aa, rcp_aa;
  double abs_ma, b1, b2, b1b2, rcp_b1_b2;
  double h, f, g;

  //calculate normal velocities and velocity squares
  vn_l = 0.;
  vsq_l = 0.;
  vn_r = 0.;
  vsq_r = 0.;
  for (int i = 0; i < n_dims; i++)
  {
    v_l(i) = u_l(i + 1) / u_l(0);
    v_r(i) = u_r(i + 1) / u_r(0);
    vn_l += v_l(i) * norm(i);
    vn_r += v_r(i) * norm(i);
    vsq_l += v_l(i) * v_l(i);
    vsq_r += v_r(i) * v_r(i);
    dv(i) = v_r(i) - v_l(i);
  }
  // Pressure, Specific enthalpy
  p_l = (gamma - 1.0) * (u_l(n_dims + 1) - 0.5 * u_l(0) * vsq_l);
  p_r = (gamma - 1.0) * (u_r(n_dims + 1) - 0.5 * u_r(0) * vsq_r);
  h_l = (u_l(n_dims + 1) + p_l) / u_l(0);
  h_r = (u_r(n_dims + 1) + p_r) / u_r(0);

  // Difference between two state
  drho = u_r(0) - u_l(0);
  dp = p_r - p_l;
  dh = h_r - h_l;
  dvn = vn_r - vn_l;

  // Compute Roe averaged density and enthalpy
  sq_rho = sqrt(u_r(0) / u_l(0));
  rrho = 1.0 / (1.0 + sq_rho);
  ratr = sq_rho * rrho;
  ra = sq_rho * u_l(0);
  ha = h_l * rrho + h_r * ratr;

  qq = 0.;
  va_n=0.;
  for (int i = 0; i < n_dims; i++)
  {
    va(i) = v_l(i) * rrho + v_r(i) * ratr;
    qq += va(i) * va(i);
    va_n += norm(i) * va(i);
  }

  aa = sqrt((gamma - 1) * (ha - 0.5 * qq));
  rcp_aa = 1.0 / aa;

  // Compute |M|, add a small number to avoid a possible singularity of f
  abs_ma = fabs(va_n * rcp_aa);

  // Eigen structure
  b1 = max(0.0, max(va_n + aa, vn_r + aa));
  b2 = min(0.0, min(va_n - aa, vn_l - aa));

  // Normalized wave speed
  b1b2 = b1 * b2;
  rcp_b1_b2 = 1.0 / (b1 - b2);
  b1 = b1 * rcp_b1_b2;
  b2 = b2 * rcp_b1_b2;
  b1b2 = b1b2 * rcp_b1_b2;

  // 1-D shock discontinuity sensing term and Mach number based function f,g

  h = 1.0 - ((p_l < p_r) ? (p_l / p_r) : (p_r / p_l));
  f = ((abs_ma != 0) ? pow(abs_ma, h) : 1.);
  g = f / (1.0 + abs_ma);

  // Difference of U, du
  for (int i = 0; i < n_fields-1; i++)
    du(i) = u_r(i) - u_l(i);

  du(n_dims + 1) = u_r(0) * h_r - u_l(0) * h_l;

  // BdQ
  bdq(0) = drho - f * dp * rcp_aa * rcp_aa;
  bdq(n_dims + 1) = bdq(0) * ha + ra * dh;
  for (int i = 0; i < n_dims; i++)
    bdq(i+1) = bdq(0)*va(i) + ra*(dv(i) - norm(i)*dvn);

  // calculate normal flux from discontinuous solution at flux points
#if defined _ACCELERATE_BLAS || defined _MKL_BLAS || defined _STANDARD_BLAS
  cblas_dgemv(CblasColMajor, CblasNoTrans, n_fields, n_dims, 1.0, f_l.get_ptr_cpu(), n_fields, norm.get_ptr_cpu(), 1, 0.0, fn_l.get_ptr_cpu(), 1);
  cblas_dgemv(CblasColMajor, CblasNoTrans, n_fields, n_dims, 1.0, f_r.get_ptr_cpu(), n_fields, norm.get_ptr_cpu(), 1, 0.0, fn_r.get_ptr_cpu(), 1);
#else
  fn_l.initialize_to_zero();
  fn_r.initialize_to_zero();
  for (int k = 0; k < n_fields; k++)
  {
    for (int l = 0; l < n_dims; l++)
    {
      fn_l(k) += f_l(k, l) * norm(l);
      fn_r(k) += f_r(k, l) * norm(l);
    }
  }
#endif
  // Flux
  for (int i = 0; i < n_fields; i++)
    fn(i) = (b1 * fn_l(i) - b2 * fn_r(i)) + b1b2 * (du(i) - g * bdq(i));
}

void inters::hllc_flux(hf_array<double> &u_l, hf_array<double> &u_r, hf_array<double> &f_l, hf_array<double> &f_r, hf_array<double> &norm, hf_array<double> &fn, int n_dims, int n_fields, double gamma)
{
  //declare arrays and variables
  hf_array<double> fn_l(n_fields), fn_r(n_fields); //normal fluxes
  double S_L, S_R, S_star;                         //wave speeds
  hf_array<double> v_l(n_dims), v_r(n_dims);  //velocities
  double p_l, p_r, h_l, h_r; //pressure, speed of sound, total enthalpy
  double vn_l, vn_r, vsq_l, vsq_r;     //normal velocities and velocity squared
  double sq_rho, rrho, h_m, a_m, vn_m; //roe average

  //calculate normal velocities and velocity squares
  vn_l = 0.;
  vsq_l = 0.;
  vn_r = 0.;
  vsq_r = 0.;
  for (int i = 0; i < n_dims; i++)
  {
    v_l(i) = u_l(i + 1) / u_l(0);
    v_r(i) = u_r(i + 1) / u_r(0);
    vn_l += v_l(i) * norm(i);
    vn_r += v_r(i) * norm(i);
    vsq_l += v_l(i)*v_l(i);
    vsq_r += v_r(i)*v_r(i);
  }

  //calculate pressure and speed of sound and total enthalpy of both sides
  p_l = (gamma - 1.0) * (u_l(n_dims + 1) - 0.5 * u_l(0) * vsq_l);
  p_r = (gamma - 1.0) * (u_r(n_dims + 1) - 0.5 * u_r(0) * vsq_r);
  //a_l = sqrt(gamma * p_l / u_l(0)); //speed of sound
  //a_r = sqrt(gamma * p_r / u_r(0));
  h_l = (u_l(n_dims + 1) + p_l) / u_l(0); //total enthalpy
  h_r = (u_r(n_dims + 1) + p_r) / u_r(0);

  // calculate normal flux from discontinuous solution at flux points

#if defined _ACCELERATE_BLAS || defined _MKL_BLAS || defined _STANDARD_BLAS
  cblas_dgemv(CblasColMajor, CblasNoTrans, n_fields, n_dims, 1.0, f_l.get_ptr_cpu(), n_fields, norm.get_ptr_cpu(), 1, 0.0, fn_l.get_ptr_cpu(), 1);
  cblas_dgemv(CblasColMajor, CblasNoTrans, n_fields, n_dims, 1.0, f_r.get_ptr_cpu(), n_fields, norm.get_ptr_cpu(), 1, 0.0, fn_r.get_ptr_cpu(), 1);
#else
  fn_l.initialize_to_zero();
  fn_r.initialize_to_zero();
  for (int k = 0; k < n_fields; k++)
  {
    for (int l = 0; l < n_dims; l++)
    {
      fn_l(k) += f_l(k, l) * norm(l);
      fn_r(k) += f_r(k, l) * norm(l);
    }
  }
#endif

  //calculate wave speed using roe average
  sq_rho = sqrt(u_r(0) / u_l(0));
  rrho = 1. / (sq_rho + 1.);

  //roe average velocity and total enthalpy
  vn_m = rrho * (vn_l + sq_rho * vn_r);
  h_m = rrho * (h_l + sq_rho * h_r);
  a_m = sqrt((gamma - 1.) * (h_m - 0.5 * vn_m * vn_m));

  S_R = vn_m + a_m;
  S_L = vn_m - a_m;
  S_star = (p_r - p_l + u_l(0) * vn_l * (S_L - vn_l) - u_r(0) * vn_r * (S_R - vn_r)) / (u_l(0) * (S_L - vn_l) - u_r(0) * (S_R - vn_r));

  //calculate flux
  if (S_L >= 0) //left flux
    fn = fn_l;
  else
  {
    if (S_star >= 0) //left star flux
    {
      double rcp_star = S_L - S_star;
      fn(0) = S_star * (S_L * u_l(0) - fn_l(0)) / rcp_star;
      for (int i = 0; i < n_dims; i++)
        fn(i + 1) = (S_star * (S_L * u_l(i + 1) - fn_l(i + 1)) + S_L * (p_l + u_l(0) * (S_L - vn_l) * (S_star - vn_l)) * norm(i)) / rcp_star;
      fn(n_dims + 1) = (S_star * (S_L * u_l(n_dims + 1) - fn_l(n_dims + 1)) + S_L * (p_l + u_l(0) * (S_L - vn_l) * (S_star - vn_l)) * S_star) / rcp_star;
    }
    else //right star flux or left flux
    {
      if (S_R >= 0) //right star flux
      {
        double rcp_star = S_R - S_star;
        fn(0) = S_star * (S_R * u_r(0) - fn_r(0)) / rcp_star;
        for (int i = 0; i < n_dims; i++)
          fn(i + 1) = (S_star * (S_R * u_r(i + 1) - fn_r(i + 1)) + S_R * (p_r + u_r(0) * (S_R - vn_r) * (S_star - vn_r)) * norm(i)) / rcp_star;
        fn(n_dims + 1) = (S_star * (S_R * u_r(n_dims + 1) - fn_r(n_dims + 1)) + S_R * (p_r + u_r(0) * (S_R - vn_r) * (S_star - vn_r)) * S_star) / rcp_star;
      }
      else //righ flux
      {
        fn = fn_r;
      }
    }
  }
}

// Lax-Friedrich inviscid numerical flux
void inters::lax_friedrich(hf_array<double> &u_l, hf_array<double> &u_r, hf_array<double> &norm, hf_array<double> &fn, int n_dims, int n_fields, double lambda, hf_array<double>& wave_speed)
{

  double u_av;
  double u_diff;

  u_av = 0.5*(u_l(0)+u_r(0));
  u_diff = (u_l(0)-u_r(0));

  double norm_speed = 0;
  for (int i=0;i<n_dims;i++)
    {
      norm_speed += wave_speed(i)*norm(i);
    }

  fn(0) = 0.;
  for (int i=0;i<n_dims;i++)
    {
      fn(0) += wave_speed(i)*norm(i)*u_av;
    }

  fn(0) += 0.5*lambda*abs(norm_speed)*u_diff;
}


// LDG viscous numerical flux
void inters::ldg_flux(int flux_spec, hf_array<double> &u_l, hf_array<double> &u_r, hf_array<double> &f_l, hf_array<double> &f_r, hf_array<double> &norm, hf_array<double> &fn, int n_dims, int n_fields, double ldg_tau, double ldg_beta)
{
  hf_array<double> f_c(n_fields,n_dims);//common flux

  if(flux_spec == 0) //Interior and mpi
  {
    //consistent switch
    if (ldg_beta != 0.)
    {
      if (norm(0) < 0.) //reverse beta
        ldg_beta = -ldg_beta;
      else if (norm(0) == 0.) //normal vector perpendicular to the test vector,use another test vector
      {
        if ((norm(0) + norm(1)) < 0.) //reverse beta
          ldg_beta = -ldg_beta;
        else if ((norm(0) + norm(1)) == 0) //normal vector perpendicular to the test vector,use another test vector
        {
          if ((norm(0) + norm(2)) < 0.) //reverse beta
            ldg_beta = -ldg_beta;
        }
      }
    }


    //f_c_i={f}_i+beta*(f_l-f_r)
    for (int k = 0; k < n_fields; k++)
      for (int i = 0; i < n_dims; i++)
        f_c(k, i) = (0.5 + ldg_beta) * f_l(k, i) + (0.5 - ldg_beta) * f_r(k, i);
  }
  else if (flux_spec == 1) //boundary flux
  {
    //f_c_i=f_r
    for (int k = 0; k < n_fields; k++)
      for (int i = 0; i < n_dims; i++)
        f_c(k, i) = f_r(k, i);
  }
  else
    FatalError("This variant of the LDG flux has not been implemented");

  // calculate normal common flux
#if defined _ACCELERATE_BLAS || defined _MKL_BLAS || defined _STANDARD_BLAS
  cblas_dgemv(CblasColMajor, CblasNoTrans, n_fields, n_dims, 1.0, f_c.get_ptr_cpu(), n_fields, norm.get_ptr_cpu(), 1, 0.0, fn.get_ptr_cpu(), 1);
#else
  fn.initialize_to_zero();
  for (int k = 0; k < n_fields; k++)
    for (int l = 0; l < n_dims; l++)
      fn(k) += f_c(k, l) * norm(l);
#endif
  for (int k = 0; k < n_fields; k++)
    fn(k) -= ldg_tau * (u_r(k) - u_l(k));
}


// LDG common solution
void inters::ldg_solution(int flux_spec, hf_array<double> &u_l, hf_array<double> &u_r, hf_array<double> &u_c, double ldg_beta, hf_array<double>& norm)
{
  if(flux_spec == 0) // Interior and mpi
    {
      //consistent switch
      if (ldg_beta != 0.)
      {
        if (norm(0) < 0.) //reverse beta
          ldg_beta = -ldg_beta;
        else if (norm(0) == 0.) //normal vector perpendicular to the test vector,use another test vector
        {
          if ((norm(0) + norm(1)) < 0.) //reverse beta
            ldg_beta = -ldg_beta;
          else if ((norm(0) + norm(1)) == 0) //normal vector perpendicular to the test vector,use another test vector
          {
            if ((norm(0) + norm(2)) < 0.) //reverse beta
              ldg_beta = -ldg_beta;
          }
        }
      }

      //u_c_k={u}-beta*(u_l-u_r)
        for (int k = 0; k < n_fields; k++)
          u_c(k) = 0.5 * (u_l(k) + u_r(k)) - ldg_beta * (u_l(k) - u_r(k));
    }
    else if (flux_spec == 1) //boundary
    {
      u_c = u_r;
    }
    else
      FatalError("This variant of the LDG flux has not been implemented");
}


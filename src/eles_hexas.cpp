/*!
 * \file eles_hexas.cpp
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

#include <iomanip>
#include <iostream>
#include <cmath>

#include "../include/global.h"
#include "../include/eles_hexas.h"
#include "../include/cubature_1d.h"
#include "../include/cubature_quad.h"
#include "../include/cubature_hexa.h"

using namespace std;

// #### constructors ####

// default constructor

eles_hexas::eles_hexas()
{

}

void eles_hexas::setup_ele_type_specific()
{
#ifndef _MPI
  cout << "Initializing hexas" << endl;
#endif

  ele_type=4;
  n_dims=3;

  if (run_input.equation==0)
    n_fields=5;
  else if (run_input.equation==1)
    n_fields=1;
  else
    FatalError("Equation not supported");

  if (run_input.RANS==1)
    n_fields++;

  n_inters_per_ele=6;
  length.setup(12);
  n_upts_per_ele=(order+1)*(order+1)*(order+1);
  upts_type=run_input.upts_type_hexa;
  set_loc_1d_upts();
  set_loc_upts();
  set_vandermonde1D();
  set_vandermonde3D();

  //set shock capturing arrays
  if (run_input.shock_cap)
  {
    if (run_input.shock_det == 0)//persson
      calc_norm_basis();
    else//concentration
      FatalError("Shock detector not implemented.");
    if (run_input.shock_cap == 1) //exp filter
      set_exp_filter();
    else
      FatalError("Shock capturing method not implemented.");
  }

  set_inters_cubpts();
  set_volume_cubpts(order, loc_volume_cubpts, weight_volume_cubpts);
  set_opp_volume_cubpts(loc_volume_cubpts, opp_volume_cubpts);

  //de-aliasing by over-integration
  if (run_input.over_int)
    set_over_int();

  n_ppts_per_ele=p_res*p_res*p_res;
  n_peles_per_ele=(p_res-1)*(p_res-1)*(p_res-1);
  n_verts_per_ele = 8;

  set_loc_ppts();
  set_opp_p();

  n_fpts_per_inter.setup(6);

  n_fpts_per_inter(0)=(order+1)*(order+1);
  n_fpts_per_inter(1)=(order+1)*(order+1);
  n_fpts_per_inter(2)=(order+1)*(order+1);
  n_fpts_per_inter(3)=(order+1)*(order+1);
  n_fpts_per_inter(4)=(order+1)*(order+1);
  n_fpts_per_inter(5)=(order+1)*(order+1);

  n_fpts_per_ele=n_inters_per_ele*(order+1)*(order+1);

  set_tloc_fpts();

  set_tnorm_fpts();

  set_opp_0(run_input.sparse_hexa);
  set_opp_1(run_input.sparse_hexa);
  set_opp_2(run_input.sparse_hexa);
  set_opp_3(run_input.sparse_hexa);

  if(viscous)
    {
      // Compute hex filter matrix
      if(LES_filter) compute_filter_upts();

      set_opp_4(run_input.sparse_hexa);
      set_opp_5(run_input.sparse_hexa);
      set_opp_6(run_input.sparse_hexa);

      temp_grad_u.setup(n_fields,n_dims);
    }

  temp_u.setup(n_fields);
  temp_f.setup(n_fields,n_dims);
}

// #### methods ####

void eles_hexas::set_connectivity_plot()
{
  int vertex_0,vertex_1,vertex_2,vertex_3,vertex_4,vertex_5,vertex_6,vertex_7;
  int count=0;
  int k,l,m;

  for(k=0;k<p_res-1;++k){
      for(l=0;l<p_res-1;++l){
          for(m=0;m<p_res-1;++m){
              vertex_0=m+(p_res*l)+(p_res*p_res*k);
              vertex_1=vertex_0+1;
              vertex_2=vertex_0+p_res+1;
              vertex_3=vertex_0+p_res;
              vertex_4=vertex_0+p_res*p_res;
              vertex_5=vertex_4+1;
              vertex_6=vertex_4+p_res+1;
              vertex_7=vertex_4+p_res;

              connectivity_plot(0,count) = vertex_0;
              connectivity_plot(1,count) = vertex_1;
              connectivity_plot(2,count) = vertex_2;
              connectivity_plot(3,count) = vertex_3;
              connectivity_plot(4,count) = vertex_4;
              connectivity_plot(5,count) = vertex_5;
              connectivity_plot(6,count) = vertex_6;
              connectivity_plot(7,count) = vertex_7;
              count++;
            }
        }
    }
}


// set location of 1d solution points in standard interval (required for tensor product elements)

void eles_hexas::set_loc_1d_upts(void)
{
  cubature_1d cub_1d(upts_type, order);
  loc_1d_upts.setup(order + 1);
  for (int i = 0; i < order + 1; i++)
    loc_1d_upts(i) = cub_1d.get_r(i);
}

// set location of 1d shape points in standard interval (required for tensor product element)

void eles_hexas::set_loc_1d_spts(hf_array<double> &loc_1d_spts, int in_n_1d_spts)
{
  int i;

  for(i=0;i<in_n_1d_spts;++i)
    {
      loc_1d_spts(i)=-1.0+((2.0*i)/(1.0*(in_n_1d_spts-1)));
    }
}



// set location of solution points in standard element

void eles_hexas::set_loc_upts(void)
{
  int i,j,k;

  int upt;

  loc_upts.setup(n_dims,n_upts_per_ele);

  for(i=0;i<(order+1);++i)
    {
      for(j=0;j<(order+1);++j)
        {
          for(k=0;k<(order+1);++k)
            {
              upt=k+(order+1)*j+(order+1)*(order+1)*i;

              loc_upts(0,upt)=loc_1d_upts(k);
              loc_upts(1,upt)=loc_1d_upts(j);
              loc_upts(2,upt)=loc_1d_upts(i);
            }
        }
    }
}

// set location of flux points in standard element

void eles_hexas::set_tloc_fpts(void)
{
  int i,j,k;

  int fpt;

  tloc_fpts.setup(n_dims,n_fpts_per_ele);

  for(i=0;i<n_inters_per_ele;++i)
    {
      for(j=0;j<(order+1);++j)
        {
          for(k=0;k<(order+1);++k)
            {
              fpt=k+((order+1)*j)+((order+1)*(order+1)*i);

              // for tensor prodiuct elements flux point location depends on solution point location

              if(i==0)
                {
                  tloc_fpts(0,fpt)=loc_1d_upts(order-k);
                  tloc_fpts(1,fpt)=loc_1d_upts(j);
                  tloc_fpts(2,fpt)=-1.0;

                }
              else if(i==1)
                {
                  tloc_fpts(0,fpt)=loc_1d_upts(k);
                  tloc_fpts(1,fpt)=-1.0;
                  tloc_fpts(2,fpt)=loc_1d_upts(j);
                }
              else if(i==2)
                {
                  tloc_fpts(0,fpt)=1.0;
                  tloc_fpts(1,fpt)=loc_1d_upts(k);
                  tloc_fpts(2,fpt)=loc_1d_upts(j);
                }
              else if(i==3)
                {
                  tloc_fpts(0,fpt)=loc_1d_upts(order-k);
                  tloc_fpts(1,fpt)=1.0;
                  tloc_fpts(2,fpt)=loc_1d_upts(j);
                }
              else if(i==4)
                {
                  tloc_fpts(0,fpt)=-1.0;
                  tloc_fpts(1,fpt)=loc_1d_upts(order-k);
                  tloc_fpts(2,fpt)=loc_1d_upts(j);
                }
              else if(i==5)
                {
                  tloc_fpts(0,fpt)=loc_1d_upts(k);
                  tloc_fpts(1,fpt)=loc_1d_upts(j);
                  tloc_fpts(2,fpt)=1.0;
                }
            }
        }
    }
}

void eles_hexas::set_inters_cubpts(void)
{

  n_cubpts_per_inter.setup(n_inters_per_ele);
  loc_inters_cubpts.setup(n_inters_per_ele);
  weight_inters_cubpts.setup(n_inters_per_ele);
  tnorm_inters_cubpts.setup(n_inters_per_ele);

  cubature_quad cub_quad(0,order);
  int n_cubpts_quad = cub_quad.get_n_pts();

  for (int i=0;i<n_inters_per_ele;++i)
    n_cubpts_per_inter(i) = n_cubpts_quad;

  for (int i=0;i<n_inters_per_ele;++i) {

      loc_inters_cubpts(i).setup(n_dims,n_cubpts_per_inter(i));
      weight_inters_cubpts(i).setup(n_cubpts_per_inter(i));
      tnorm_inters_cubpts(i).setup(n_dims,n_cubpts_per_inter(i));

      for (int j=0;j<n_cubpts_quad;++j) {

          if (i==0) {
              loc_inters_cubpts(i)(0,j)=cub_quad.get_r(j);
              loc_inters_cubpts(i)(1,j)=cub_quad.get_s(j);
              loc_inters_cubpts(i)(2,j)=-1.0;
            }
          else if (i==1) {
              loc_inters_cubpts(i)(0,j)=cub_quad.get_r(j);
              loc_inters_cubpts(i)(1,j)=-1.;
              loc_inters_cubpts(i)(2,j)=cub_quad.get_s(j);
            }
          else if (i==2) {
              loc_inters_cubpts(i)(0,j)=1.0;
              loc_inters_cubpts(i)(1,j)=cub_quad.get_r(j);
              loc_inters_cubpts(i)(2,j)=cub_quad.get_s(j);
            }
          else if (i==3) {
              loc_inters_cubpts(i)(0,j)=cub_quad.get_r(j);
              loc_inters_cubpts(i)(1,j)=1.0;
              loc_inters_cubpts(i)(2,j)=cub_quad.get_s(j);
            }
          else if (i==4) {
              loc_inters_cubpts(i)(0,j)=-1.0;
              loc_inters_cubpts(i)(1,j)=cub_quad.get_r(j);
              loc_inters_cubpts(i)(2,j)=cub_quad.get_s(j);
            }
          else if (i==5) {
              loc_inters_cubpts(i)(0,j)=cub_quad.get_r(j);
              loc_inters_cubpts(i)(1,j)=cub_quad.get_s(j);
              loc_inters_cubpts(i)(2,j)=1.0;
            }

          weight_inters_cubpts(i)(j) = cub_quad.get_weight(j);

          if (i==0) {
              tnorm_inters_cubpts(i)(0,j)= 0.;
              tnorm_inters_cubpts(i)(1,j)= 0.;
              tnorm_inters_cubpts(i)(2,j)= -1.;
            }
          else if (i==1) {
              tnorm_inters_cubpts(i)(0,j)= 0.;
              tnorm_inters_cubpts(i)(1,j)= -1.;
              tnorm_inters_cubpts(i)(2,j)= 0.;
            }
          else if (i==2) {
              tnorm_inters_cubpts(i)(0,j)= 1.;
              tnorm_inters_cubpts(i)(1,j)= 0.;
              tnorm_inters_cubpts(i)(2,j)= 0.;
            }
          else if (i==3) {
              tnorm_inters_cubpts(i)(0,j)= 0.;
              tnorm_inters_cubpts(i)(1,j)= 1.;
              tnorm_inters_cubpts(i)(2,j)= 0.;
            }
          else if (i==4) {
              tnorm_inters_cubpts(i)(0,j)= -1.;
              tnorm_inters_cubpts(i)(1,j)= 0.;
              tnorm_inters_cubpts(i)(2,j)= 0.;
            }
          else if (i==5) {
              tnorm_inters_cubpts(i)(0,j)= 0.;
              tnorm_inters_cubpts(i)(1,j)= 0.;
              tnorm_inters_cubpts(i)(2,j)= 1.;
            }

        }
    }

  set_opp_inters_cubpts();
}

void eles_hexas::set_volume_cubpts(int in_order, hf_array<double> &out_loc_volume_cubpts,hf_array<double> &out_weight_volume_cubpts)

{
  cubature_hexa cub_hexa(0,in_order);
  int n_cubpts_hexa = cub_hexa.get_n_pts();

  out_loc_volume_cubpts.setup(n_dims,n_cubpts_hexa);
  out_weight_volume_cubpts.setup(n_cubpts_hexa);

  for (int i=0;i<n_cubpts_hexa;++i)
    {
      out_loc_volume_cubpts(0,i) = cub_hexa.get_r(i);
      out_loc_volume_cubpts(1,i) = cub_hexa.get_s(i);
      out_loc_volume_cubpts(2,i) = cub_hexa.get_t(i);
      out_weight_volume_cubpts(i) = cub_hexa.get_weight(i);
    }
}

// Compute the surface jacobian determinant on a face
double eles_hexas::compute_inter_detjac_inters_cubpts(int in_inter,hf_array<double> d_pos)
{
  double output = 0.;
  double xr, xs, xt;
  double yr, ys, yt;
  double zr, zs, zt;
  double temp0,temp1,temp2;

  xr = d_pos(0,0);
  xs = d_pos(0,1);
  xt = d_pos(0,2);
  yr = d_pos(1,0);
  ys = d_pos(1,1);
  yt = d_pos(1,2);
  zr = d_pos(2,0);
  zs = d_pos(2,1);
  zt = d_pos(2,2);

  double xu=0.;
  double yu=0.;
  double zu=0.;
  double xv=0.;
  double yv=0.;
  double zv=0.;

  // From calculus, for a surface parameterized by two parameters
  // u and v, than jacobian determinant is
  //
  // || (xu i + yu j + zu k) cross ( xv i + yv j + zv k)  ||

  if (in_inter==0) // u=r, v=s
    {
      xu = xr;
      yu = yr;
      zu = zr;

      xv = xs;
      yv = ys;
      zv = zs;
    }
  else if (in_inter==1) // u=r, v=t
    {
      xu = xr;
      yu = yr;
      zu = zr;

      xv = xt;
      yv = yt;
      zv = zt;
    }
  else if (in_inter==2) //u=s, v=t
    {
      xu = xs;
      yu = ys;
      zu = zs;

      xv = xt;
      yv = yt;
      zv = zt;
    }
  else if (in_inter==3) //u=r,v=t
    {
      xu = xr;
      yu = yr;
      zu = zr;

      xv = xt;
      yv = yt;
      zv = zt;
    }
  else if (in_inter==4) //u=s,v=t
    {
      xu = xs;
      yu = ys;
      zu = zs;

      xv = xt;
      yv = yt;
      zv = zt;
    }
  else if (in_inter==5) //u=r,v=s
    {
      xu = xr;
      yu = yr;
      zu = zr;

      xv = xs;
      yv = ys;
      zv = zs;
    }

  temp0 = yu*zv-zu*yv;
  temp1 = zu*xv-xu*zv;
  temp2 = xu*yv-yu*xv;

  output = sqrt(temp0*temp0+temp1*temp1+temp2*temp2);

  return output;

}

// set location of plot points in standard element

void eles_hexas::set_loc_ppts(void)
{
  int i,j,k;

  int ppt;

  loc_ppts.setup(n_dims,n_ppts_per_ele);

  for(k=0;k<p_res;++k)
    {
      for(j=0;j<p_res;++j)
        {
          for(i=0;i<p_res;++i)
            {
              ppt=i+(p_res*j)+(p_res*p_res*k);

              loc_ppts(0,ppt)=-1.0+((2.0*i)/(1.0*(p_res-1)));
              loc_ppts(1,ppt)=-1.0+((2.0*j)/(1.0*(p_res-1)));
              loc_ppts(2,ppt)=-1.0+((2.0*k)/(1.0*(p_res-1)));
            }
        }
    }

}

// set transformed normal at flux points

void eles_hexas::set_tnorm_fpts(void)
{
  int i,j,k;

  int fpt;

  tnorm_fpts.setup(n_dims,n_fpts_per_ele);

  for(i=0;i<n_inters_per_ele;++i)
    {
      for(j=0;j<(order+1);++j)
        {
          for(k=0;k<(order+1);++k)
            {
              fpt=k+((order+1)*j)+((order+1)*(order+1)*i);

              if(i==0)
                {
                  tnorm_fpts(0,fpt)=0.0;
                  tnorm_fpts(1,fpt)=0.0;
                  tnorm_fpts(2,fpt)=-1.0;
                }
              else if(i==1)
                {
                  tnorm_fpts(0,fpt)=0.0;
                  tnorm_fpts(1,fpt)=-1.0;
                  tnorm_fpts(2,fpt)=0.0;
                }
              else if(i==2)
                {
                  tnorm_fpts(0,fpt)=1.0;
                  tnorm_fpts(1,fpt)=0.0;
                  tnorm_fpts(2,fpt)=0.0;
                }
              else if(i==3)
                {
                  tnorm_fpts(0,fpt)=0.0;
                  tnorm_fpts(1,fpt)=1.0;
                  tnorm_fpts(2,fpt)=0.0;
                }
              else if(i==4)
                {
                  tnorm_fpts(0,fpt)=-1.0;
                  tnorm_fpts(1,fpt)=0.0;
                  tnorm_fpts(2,fpt)=0.0;
                }
              else if(i==5)
                {
                  tnorm_fpts(0,fpt)=0.0;
                  tnorm_fpts(1,fpt)=0.0;
                  tnorm_fpts(2,fpt)=1.0;
                }
            }
        }
    }
}

// Filtering operators for use in subgrid-scale modelling
void eles_hexas::compute_filter_upts(void)
{
  int i,j,k,l,m,n,N,N2;
  double dlt, k_c, sum, norm;
  N = order+1;

  hf_array<double> X(N), B(N);
  hf_array<double> beta(N,N);

  filter_upts_1D.setup(N,N);

  X = loc_1d_upts;

  N2 = N/2;
  // If N is odd, round up N/2
  if(N % 2 != 0){N2 += 1;}
  // Cutoff wavenumber
  k_c = 1.0/run_input.filter_ratio;
  // Approx resolution in element (assumes uniform point spacing)
  // Interval is [-1:1]
  dlt = 2.0/order;

  // Normalised solution point separation
  for (i=0;i<N;++i)
    for (j=0;j<N;++j)
      beta(j,i) = (X(j)-X(i))/dlt;

  // Build high-order-commuting Vasilyev filter
  // Only use high-order filters for high enough order
  if(run_input.filter_type==0 and N>=3)
    {
      if (rank==0) cout<<"Building high-order-commuting Vasilyev filter"<<endl;
      hf_array<double> C(N);
      hf_array<double> A(N,N);

      for (i=0;i<N;++i)
        {
          B(i) = 0.0;
          C(i) = 0.0;
          for (j=0;j<N;++j)
            A(i,j) = 0.0;

        }
      // Populate coefficient matrix
      for (i=0;i<N;++i)
        {
          // Populate constraints matrix
          B(0) = 1.0;
          // Gauss filter weights
          B(1) =  exp(-pow(pi,2)/24.0);
          B(2) = -B(1)*pow(pi,2)/k_c/12.0;

          if(N % 2 == 1 && i+1 == N2)
            B(2) = 0.0;

          for (j=0;j<N;++j)
            {
              A(j,0) = 1.0;
              A(j,1) = cos(pi*k_c*beta(j,i));
              A(j,2) = -beta(j,i)*pi*sin(pi*k_c*beta(j,i));

              if(N % 2 == 1 && i+1 == N2)
                A(j,2) = pow(beta(j,i),3);

            }

          // Enforce filter moments
          for (k=3;k<N;++k)
            {
              for (j=0;j<N;++j)
                A(j,k) = pow(beta(j,i),k+1);

              B(k) = 0.0;
            }

          // Solve linear system by inverting A using
          // Gauss-Jordan method
          gaussj(N,A,B);
          for (j=0;j<N;++j)
            filter_upts_1D(j,i) = B(j);

        }
    }
  else if(run_input.filter_type==1) // Discrete Gaussian filter
    {
      if (rank==0) cout<<"Building discrete Gaussian filter"<<endl;
      int ctype, index;
      double k_R, k_L, coeff;
      double res_0, res_L, res_R;
      hf_array<double> alpha(N);
      cubature_1d cub_1d(0,order);
      int n_cubpts_1d = cub_1d.get_n_pts();
      hf_array<double> wf(n_cubpts_1d);
      for (j=0;j<n_cubpts_1d;j++)
        wf(j) = cub_1d.get_weight(j);

      // Determine corrected filter width for skewed quadrature points
      // using iterative constraining procedure
      // ctype options: (-1) no constraining, (0) constrain moment, (1) constrain cutoff frequency
      ctype = -1;
      if(ctype>=0)
        {
          for (i=0;i<N2;++i)
            {
              for (j=0;j<N;++j)
                B(j) = beta(j,i);

              k_L = 0.1; k_R = 1.0;
              res_L = flt_res(N,wf,B,k_L,k_c,ctype);
              res_R = flt_res(N,wf,B,k_R,k_c,ctype);
              alpha(i) = 0.5*(k_L+k_R);
              for (j=0;j<1000;++j)
                {
                  res_0 = flt_res(N,wf,B,k_c,alpha(i),ctype);
                  if(abs(res_0)<1.e-12) return;
                  if(res_0*res_L>0.0)
                    {
                      k_L = alpha(i);
                      res_L = res_0;
                    }
                  else
                    {
                      k_R = alpha(i);
                      res_R = res_0;
                    }
                  if(j==999)
                    {
                      alpha(i) = k_c;
                      ctype = -1;
                    }
                }
              alpha(N-i-1) = alpha(i);
            }
        }

      else if(ctype==-1) // no iterative constraining
        for (i=0;i<N;++i)
          alpha(i) = k_c;

      sum = 0.0;
      for (i=0;i<N;++i)
        {
          norm = 0.0;
          for (j=0;j<N;++j)
            {
              filter_upts_1D(i,j) = wf(j)*exp(-6.0*pow(alpha(i)*beta(i,j),2));
              norm += filter_upts_1D(i,j);
            }
          for (j=0;j<N;++j)
            {
              filter_upts_1D(i,j) /= norm;
              sum += filter_upts_1D(i,j);
            }
        }
    }
  else if(run_input.filter_type==2) // Modal coefficient filter
    {
      if (rank==0) cout<<"Building modal filter"<<endl;

      // Compute restriction-prolongation filter
      compute_modal_filter_1d(filter_upts_1D, vandermonde1D, inv_vandermonde1D, N, order);

      sum = 0;
      for(i=0;i<N;i++)
        for(j=0;j<N;j++)
          sum+=filter_upts_1D(i,j);
    }
  else // Simple average for low order
    {
      if (rank==0) cout<<"Building average filter"<<endl;
      sum=0;
      for (i=0;i<N;i++)
        {
          for (j=0;j<N;j++)
            {
              filter_upts_1D(i,j) = 1.0/N;
              sum+=1.0/N;
            }
        }
    }

  // Build 3D filter on ideal (reference) element.
  // This construction is unique to hexa elements but the resulting
  // matrix will be of the same(?) dimension for all 3D element types.
  int ii=0;
  filter_upts.setup(n_upts_per_ele,n_upts_per_ele);
  sum=0;
  for (i=0;i<N;++i)
    {
      for (j=0;j<N;++j)
        {
          for (k=0;k<N;++k)
            {
              int jj=0;
              for (l=0;l<N;++l)
                {
                  for (m=0;m<N;++m)
                    {
                      for (n=0;n<N;++n)
                        {
                          filter_upts(ii,jj) = filter_upts_1D(k,n)*filter_upts_1D(j,m)*filter_upts_1D(i,l);
                          sum+=filter_upts(ii,jj);
                          ++jj;
                        }
                    }
                }
              ++ii;
            }
        }
    }
}


//#### helper methods ####


int eles_hexas::read_restart_info_ascii(ifstream& restart_file)
{

  string str;
  // Move to triangle element
  while(1) {
      getline(restart_file,str);
      if (str=="HEXAS") break;

      if (restart_file.eof()) return 0;
    }

  getline(restart_file,str);
  restart_file >> order_rest;
  getline(restart_file,str);
  getline(restart_file,str);
  restart_file >> n_upts_per_ele_rest;
  getline(restart_file,str);
  getline(restart_file,str);

  loc_1d_upts_rest.setup(order_rest+1);

  for (int i=0;i<order_rest+1;++i)
    restart_file >> loc_1d_upts_rest(i);

  set_opp_r();

  return 1;
}

#ifdef _HDF5
void eles_hexas::read_restart_info_hdf5(hid_t &restart_file, int in_rest_order)
{
  hid_t dataset_id, plist_id;

  //open dataset
  dataset_id = H5Dopen2(restart_file, "HEXAS", H5P_DEFAULT);
  if (dataset_id < 0)
    FatalError("Cannot find hexa element property");

  plist_id = H5Pcreate(H5P_DATASET_XFER);
#ifdef _MPI
  //set collective read
  H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);
#endif

  if (n_eles)
  {
    order_rest = in_rest_order;
    n_upts_per_ele_rest = (order_rest + 1) * (order_rest + 1) * (order_rest + 1);
    loc_1d_upts_rest.setup(order_rest + 1);
    //read data
    H5Dread(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, plist_id, loc_1d_upts_rest.get_ptr_cpu());
    set_opp_r();
  }
#ifdef _MPI
  else //read empty
  {
    hid_t null_id=H5Dget_space(dataset_id);
    H5Sselect_none(null_id);
    H5Dread(dataset_id, H5T_NATIVE_DOUBLE, null_id, null_id, plist_id, NULL);
    H5Sclose(null_id);
  }
#endif

  //close objects
  H5Pclose(plist_id);
  H5Dclose(dataset_id);
}
#endif

// write restart info
#ifndef _HDF5
void eles_hexas::write_restart_info_ascii(ofstream& restart_file)
{
  restart_file << "HEXAS" << endl;

  restart_file << "Order" << endl;
  restart_file << order << endl;

  restart_file << "Number of solution points per hexahedral element" << endl;
  restart_file << n_upts_per_ele << endl;

  restart_file << "Location of solution points in 1D" << endl;
  for (int i=0;i<order+1;++i) {
      restart_file << loc_1d_upts(i) << " ";
    }
  restart_file << endl;



}
#endif

#ifdef _HDF5
void eles_hexas::write_restart_info_hdf5(hid_t &restart_file)
{
  hid_t dataset_id,plist_id,dataspace_id;
  hsize_t dim = run_input.order + 1;

  //create HEXAS dataset
  dataspace_id = H5Screate_simple(1, &dim, NULL);
  dataset_id = H5Dcreate2(restart_file, "HEXAS", H5T_NATIVE_DOUBLE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  plist_id = H5Pcreate(H5P_DATASET_XFER);
#ifdef _MPI
  //set collective read
  H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);
#endif
  //write loc_1d_upts
  if(n_eles)
  {
    H5Dwrite(dataset_id,H5T_NATIVE_DOUBLE,H5S_ALL,H5S_ALL,plist_id,loc_1d_upts.get_ptr_cpu());
  }
  #ifdef _MPI
  else
  {
    H5Sselect_none(dataspace_id);
    H5Dwrite(dataset_id,H5T_NATIVE_DOUBLE,dataspace_id,dataspace_id,plist_id,NULL);
  }
  #endif
  H5Pclose(plist_id);
  H5Dclose(dataset_id);
  H5Sclose(dataspace_id);
}
#endif

// initialize the vandermonde matrix
void eles_hexas::set_vandermonde1D(void)
{
  vandermonde1D.setup(order+1,order+1);

  for (int i=0;i<order+1;i++)
    for (int j=0;j<order+1;j++)
      vandermonde1D(i,j) = eval_legendre(loc_1d_upts(i),j);

  // Store its inverse
  inv_vandermonde1D = inv_array(vandermonde1D);
}

void eles_hexas::set_vandermonde3D(void)
{
    vandermonde.setup(n_upts_per_ele,n_upts_per_ele);
    hf_array<double> loc(n_dims);
    for (int i=0;i<n_upts_per_ele;i++)//location
    {
        loc(0)=loc_upts(0,i);
        loc(1)=loc_upts(1,i);
        loc(2)=loc_upts(2,i);
        for (int j=0;j<n_upts_per_ele;j++)//in mode
            vandermonde(i,j)=eval_legendre_basis_3D_hierarchical(j,loc,order);
    }
    inv_vandermonde=inv_array(vandermonde);
}

void eles_hexas::set_exp_filter(void)
{
  exp_filter.setup(n_upts_per_ele, n_upts_per_ele);
  exp_filter.initialize_to_zero();
  int i, j, k, l, mode;
  double eta_x, eta_y, eta_z;
  double eta_c = (double)run_input.expf_cutoff / (double)(order);

  mode = 0;
  for (l = 0; l < 3 * order + 1; l++) //sum of x,y,z mode
  {
    for (k = 0; k < l + 1; k++) //k no more than sum
    {
      for (j = 0; j < l - k + 1; j++) //j no more than sum-k
      {
        i = l - k - j; //i+j+k=l
        if (i <= order && j <= order && k <= order)
        {
          eta_x = (double)(i) / (double)(order);
          eta_y = (double)(j) / (double)(order);
          eta_z = (double)(k) / (double)(order);
          exp_filter(mode, mode) = 1.;
          if (eta_x > eta_c)
            exp_filter(mode, mode) *= exp(-run_input.expf_fac * pow((eta_x - eta_c) / (1. - eta_c), run_input.expf_order));
          if (eta_y > eta_c)
            exp_filter(mode, mode) *= exp(-run_input.expf_fac * pow((eta_y - eta_c) / (1. - eta_c), run_input.expf_order));
          if (eta_z > eta_c)
            exp_filter(mode, mode) *= exp(-run_input.expf_fac * pow((eta_z - eta_c) / (1. - eta_c), run_input.expf_order));
          mode++;
        }
      }
    }
    }

    exp_filter = mult_arrays(exp_filter, inv_vandermonde);
    exp_filter = mult_arrays(vandermonde, exp_filter);
}

void eles_hexas::calc_norm_basis(void)
{
  int n1, n2, n3;
  double norm1, norm2, norm3;
  norm_basis_persson.setup(n_upts_per_ele);
  for (int i = 0; i < n_upts_per_ele; i++)
  {
    get_legendre_basis_3D_index(i, order, n1, n2, n3);
    norm1 = 2.0 / (2.0 * n1 + 1.0);
    norm2 = 2.0 / (2.0 * n2 + 1.0);
    norm3 = 2.0 / (2.0 * n3 + 1.0);
    norm_basis_persson(i) = norm1 * norm2 * norm3;
  }
}

//detect shock use persson's method
void eles_hexas::shock_det_persson(void)
{
  hf_array<double> temp_modal(n_upts_per_ele, n_eles);     //store modal value

  if (run_input.shock_det_field == 0) //density
  {
//step 1. convert to modal value
#if defined _ACCELERATE_BLAS || defined _MKL_BLAS || defined _STANDARD_BLAS
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n_upts_per_ele, n_eles, n_upts_per_ele, 1.0, inv_vandermonde.get_ptr_cpu(), n_upts_per_ele, disu_upts(0).get_ptr_cpu(), n_upts_per_ele, 0.0, temp_modal.get_ptr_cpu(), n_upts_per_ele);
#else
    dgemm(n_upts_per_ele, n_eles, n_upts_per_ele, 1.0, 0.0, inv_vandermonde.get_ptr_cpu(), disu_upts(0).get_ptr_cpu(), temp_modal.get_ptr_cpu());
#endif
  }
  else if (run_input.shock_det_field == 1) //pressure
  {
    hf_array<double> temp_pressure(n_upts_per_ele);
    for (int i = 0; i < n_eles; i++) //for each element
    {
      //calculate pressure for each solution point
      for (int j = 0; j < n_upts_per_ele; j++) //for each solution points
      {
        double u_sqr = 0;//momentum squared
        for (int k = 0; k < n_dims; k++)
          u_sqr += disu_upts(0)(j, i, k + 1) * disu_upts(0)(j, i, k + 1);
        temp_pressure(j) = (run_input.gamma - 1.) * (disu_upts(0)(j, i, n_dims + 1) - 0.5 * u_sqr / disu_upts(0)(j, i, 0));
      }
      ////step 1. convert to modal value
#if defined _ACCELERATE_BLAS || defined _MKL_BLAS || defined _STANDARD_BLAS
      cblas_dgemv(CblasColMajor, CblasNoTrans, n_upts_per_ele, n_upts_per_ele, 1.0, inv_vandermonde.get_ptr_cpu(), n_upts_per_ele, temp_pressure.get_ptr_cpu(), 1, 0.0, temp_modal.get_ptr_cpu(0, i), 1);
#else
      dgemm(n_upts_per_ele, 1, n_upts_per_ele, 1.0, 0.0, inv_vandermonde.get_ptr_cpu(), temp_pressure.get_ptr_cpu(), temp_modal.get_ptr_cpu(0, i));
#endif
    }
  }
  else
    FatalError("Unrecognized shock detector field");

    //step 2. perform inplace \hat{u}^2 store in temp_modal
#if defined _ACCELERATE_BLAS || defined _MKL_BLAS || defined _STANDARD_BLAS
  vdSqr(n_upts_per_ele * n_eles, temp_modal.get_ptr_cpu(), temp_modal.get_ptr_cpu());
#else
  transform(temp_modal.get_ptr_cpu(), temp_modal.get_ptr_cpu(n_upts_per_ele * n_eles), temp_modal.get_ptr_cpu(), [](double x) { return x * x; });
#endif

  //step 3. use Parseval's theorem to calculate (u-u_n,u-u_n)
  int x, y, z;
  for (int ic = 0; ic < n_eles; ic++)
  {
    sensor(ic)=0;
    for (int j = 0; j < n_upts_per_ele; j++)
    {
      get_legendre_basis_3D_index(j, order, x, y, z);
      if (x == order || y == order || z == order)
        sensor(ic) += temp_modal(j, ic) * norm_basis_persson(j);
    }
  }

  //step 4. use Parseval's theorem to calculate (u,u),  and calculate ((u-u_n,u-u_n)/(u,u))
  for (int i = 0; i < n_eles; i++)
  {
#if defined _ACCELERATE_BLAS || defined _MKL_BLAS || defined _STANDARD_BLAS
    sensor(i) /= cblas_ddot(n_upts_per_ele, norm_basis_persson.get_ptr_cpu(), 1, temp_modal.get_ptr_cpu(0, i), 1);
#else
    sensor(i) /= inner_product(norm_basis_persson.get_ptr_cpu(), norm_basis_persson.get_ptr_cpu(n_upts_per_ele), temp_modal.get_ptr_cpu(0, i), 0.);
#endif
  }
}

void eles_hexas::set_concentration_array()
{
  int concen_type = 1;
  hf_array<double> concentration_factor(order+1);
  hf_array<double> grad_vandermonde;
  grad_vandermonde.setup(order+1,order+1);//1D gradient Vandermonde
  concentration_array.setup((order+1),(order+1));//concentration hf_array

    // create the vandermonde matrix
    for (int i=0;i<order+1;i++)
        for (int j=0;j<order+1;j++)
            grad_vandermonde(i,j) = eval_d_legendre(loc_1d_upts(i),j);

    // create concentration factor hf_array
    for(int j=0; j <order+1; j++){
        if(concen_type == 0){ // exponential
            if(j==0)
                concentration_factor(j) = 0;
            else
                concentration_factor(j) = exp(1/(6*j*(j+1)));
        }
        else if(concen_type == 1) // linear
            concentration_factor(j) = 1;

        else
            cout<<"Concentration factor not setup"<<endl;
        }

//set up concentration hf_array
    for (int i=0;i<order+1;i++)
                for (int j=0;j<order+1;j++)
                        concentration_array(j,i) = concentration_factor(j)*sqrt(1 - loc_1d_upts(i)*loc_1d_upts(i))*grad_vandermonde(i,j);//tanspose?

  }

  void eles_hexas::set_over_int(void)
  {
    //initialize over integration cubature points
    set_volume_cubpts(run_input.over_int_order, loc_over_int_cubpts, weight_over_int_cubpts);
    //set interpolation matrix from solution points to over integration cubature points
    set_opp_volume_cubpts(loc_over_int_cubpts, opp_over_int_cubpts);

    //set projection matrix from over integration cubature points to modal coefficients
    temp_u_over_int_cubpts.setup(loc_over_int_cubpts.get_dim(1), n_fields);
    temp_u_over_int_cubpts.initialize_to_zero();
    temp_tdisf_over_int_cubpts.setup(loc_over_int_cubpts.get_dim(1), n_fields, n_dims);
    hf_array<double> loc(n_dims);
    hf_array<double> temp_proj(n_upts_per_ele, loc_over_int_cubpts.get_dim(1));

    //step 1. nodal to L2 projected modal \hat{u_i}=\int{\phi_i*\l_j}=>\phi_i(j)*w(j)
    int n1, n2, n3;
    double norm1, norm2, norm3;
    for (int i = 0; i < n_upts_per_ele; i++)
    {
      get_legendre_basis_3D_index(i, order, n1, n2, n3);
      norm1 = 2.0 / (2.0 * n1 + 1.0);
      norm2 = 2.0 / (2.0 * n2 + 1.0);
      norm3 = 2.0 / (2.0 * n3 + 1.0);
      for (int j = 0; j < loc_over_int_cubpts.get_dim(1); j++)
      {
        loc(0) = loc_over_int_cubpts(0, j);
        loc(1) = loc_over_int_cubpts(1, j);
        loc(2) = loc_over_int_cubpts(2, j);
        temp_proj(i, j) = eval_legendre_basis_3D_hierarchical(i, loc, order) / (norm1 * norm2 * norm3) * weight_over_int_cubpts(j);
      }
    }
    //multiply modal coefficient by vandermonde matrix to get over_int_filter
    over_int_filter = mult_arrays(vandermonde, temp_proj);
  }
  // evaluate nodal basis

double eles_hexas::eval_nodal_basis(int in_index, hf_array<double> in_loc)
{
  int i,j,k;

  double nodal_basis;

  i=(in_index/((order+1)*(order+1)));
  j=(in_index-((order+1)*(order+1)*i))/(order+1);
  k=in_index-((order+1)*j)-((order+1)*(order+1)*i);

  nodal_basis=eval_lagrange(in_loc(0),k,loc_1d_upts)*eval_lagrange(in_loc(1),j,loc_1d_upts)*eval_lagrange(in_loc(2),i,loc_1d_upts);

  return nodal_basis;
}

// evaluate nodal basis using restart points
//
double eles_hexas::eval_nodal_basis_restart(int in_index, hf_array<double> in_loc)
{
  int i,j,k;

  double nodal_basis;

  i=(in_index/((order_rest+1)*(order_rest+1)));
  j=(in_index-((order_rest+1)*(order_rest+1)*i))/(order_rest+1);
  k=in_index-((order_rest+1)*j)-((order_rest+1)*(order_rest+1)*i);

  nodal_basis=eval_lagrange(in_loc(0),k,loc_1d_upts_rest)*eval_lagrange(in_loc(1),j,loc_1d_upts_rest)*eval_lagrange(in_loc(2),i,loc_1d_upts_rest);

  return nodal_basis;
}

// evaluate derivative of nodal basis

double eles_hexas::eval_d_nodal_basis(int in_index, int in_cpnt, hf_array<double> in_loc)
{
  int i,j,k;

  double d_nodal_basis;

  i=(in_index/((order+1)*(order+1)));
  j=(in_index-((order+1)*(order+1)*i))/(order+1);
  k=in_index-((order+1)*j)-((order+1)*(order+1)*i);

  if(in_cpnt==0)
    {
      d_nodal_basis=eval_d_lagrange(in_loc(0),k,loc_1d_upts)*eval_lagrange(in_loc(1),j,loc_1d_upts)*eval_lagrange(in_loc(2),i,loc_1d_upts);
    }
  else if(in_cpnt==1)
    {
      d_nodal_basis=eval_lagrange(in_loc(0),k,loc_1d_upts)*eval_d_lagrange(in_loc(1),j,loc_1d_upts)*eval_lagrange(in_loc(2),i,loc_1d_upts);
    }
  else if(in_cpnt==2)
    {
      d_nodal_basis=eval_lagrange(in_loc(0),k,loc_1d_upts)*eval_lagrange(in_loc(1),j,loc_1d_upts)*eval_d_lagrange(in_loc(2),i,loc_1d_upts);
    }
  else
    {
      cout << "ERROR: Invalid component requested ... " << endl;
    }

  return d_nodal_basis;
}

// evaluate nodal shape basis

double eles_hexas::eval_nodal_s_basis(int in_index, hf_array<double> in_loc, int in_n_spts)
{
  int i,j,k;
  double nodal_s_basis;

  if (is_perfect_cube(in_n_spts))
    {
      int n_1d_spts = round(pow(in_n_spts,1./3.));
      hf_array<double> loc_1d_spts(n_1d_spts);
      set_loc_1d_spts(loc_1d_spts,n_1d_spts);

      i=(in_index/(n_1d_spts*n_1d_spts));
      j=(in_index-(n_1d_spts*n_1d_spts*i))/n_1d_spts;
      k=in_index-(n_1d_spts*j)-(n_1d_spts*n_1d_spts*i);

      nodal_s_basis=eval_lagrange(in_loc(0),k,loc_1d_spts)*eval_lagrange(in_loc(1),j,loc_1d_spts)*eval_lagrange(in_loc(2),i,loc_1d_spts);
    }
  else if (in_n_spts==20) // Quadratic hex with 20 nodes
    {
      if (in_index==0)
        nodal_s_basis = (1./8.*(-1+in_loc(0)))*(-1+in_loc(1))*(-1+in_loc(2))*(in_loc(0)+2+in_loc(1)+in_loc(2));
      else if (in_index==1)
        nodal_s_basis = -(1./8.*(in_loc(0)+1))*(-1+in_loc(1))*(-1+in_loc(2))*(-in_loc(0)+2+in_loc(1)+in_loc(2));
      else if (in_index==2)
        nodal_s_basis = -(1./8.*(in_loc(0)+1))*(in_loc(1)+1)*(-1+in_loc(2))*(in_loc(0)-2+in_loc(1)-in_loc(2));
      else if (in_index==3)
        nodal_s_basis = (1./8.*(-1+in_loc(0)))*(in_loc(1)+1)*(-1+in_loc(2))*(-in_loc(0)-2+in_loc(1)-in_loc(2));
      else if (in_index==4)
        nodal_s_basis = -(1./8.*(-1+in_loc(0)))*(-1+in_loc(1))*(in_loc(2)+1)*(in_loc(0)+2+in_loc(1)-in_loc(2));
      else if (in_index==5)
        nodal_s_basis = (1./8.*(in_loc(0)+1))*(-1+in_loc(1))*(in_loc(2)+1)*(-in_loc(0)+2+in_loc(1)-in_loc(2));
      else if (in_index==6)
        nodal_s_basis = (1./8.*(in_loc(0)+1))*(in_loc(1)+1)*(in_loc(2)+1)*(in_loc(0)-2+in_loc(1)+in_loc(2));
      else if (in_index==7)
        nodal_s_basis = -(1./8.*(-1+in_loc(0)))*(in_loc(1)+1)*(in_loc(2)+1)*(-in_loc(0)-2+in_loc(1)+in_loc(2));
      else if (in_index==8)
        nodal_s_basis = -(1./4.*(-1+in_loc(1)))*(-1+in_loc(2))*(in_loc(0)*in_loc(0)-1);
      else if (in_index==9)
        nodal_s_basis = (1./4.*(in_loc(0)+1))*(-1+in_loc(2))*(in_loc(1)*in_loc(1)-1);
      else if (in_index==10)
        nodal_s_basis  = (1./4.*(in_loc(1)+1))*(-1+in_loc(2))*(in_loc(0)*in_loc(0)-1);
      else if (in_index==11)
        nodal_s_basis  = -(1./4.*(-1+in_loc(0)))*(-1+in_loc(2))*(in_loc(1)*in_loc(1)-1);
      else if (in_index==12)
        nodal_s_basis  = -(1./4.*(-1+in_loc(0)))*(-1+in_loc(1))*(in_loc(2)*in_loc(2)-1);
      else if (in_index==13)
        nodal_s_basis  = (1./4.*(in_loc(0)+1))*(-1+in_loc(1))*(in_loc(2)*in_loc(2)-1);
      else if (in_index==14)
        nodal_s_basis  = -(1./4.*(in_loc(0)+1))*(in_loc(1)+1)*(in_loc(2)*in_loc(2)-1);
      else if (in_index==15)
        nodal_s_basis  = (1./4.*(-1+in_loc(0)))*(in_loc(1)+1)*(in_loc(2)*in_loc(2)-1);
      else if (in_index==16)
        nodal_s_basis  = (1./4.*(-1+in_loc(1)))*(in_loc(2)+1)*(in_loc(0)*in_loc(0)-1);
      else if (in_index==17)
        nodal_s_basis  = -(1./4.*(in_loc(0)+1))*(in_loc(2)+1)*(in_loc(1)*in_loc(1)-1);
      else if (in_index==18)
        nodal_s_basis  = -(1./4.*(in_loc(1)+1))*(in_loc(2)+1)*(in_loc(0)*in_loc(0)-1);
      else if (in_index==19)
        nodal_s_basis  = (1./4.*(-1+in_loc(0)))*(in_loc(2)+1)*(in_loc(1)*in_loc(1)-1);
    }
  else
    {
      cout << "in_n_spts = " << in_n_spts << endl;
      FatalError("Shape basis not implemented yet, exiting");
    }

  return nodal_s_basis;
}

// evaluate derivative of nodal shape basis

void eles_hexas::eval_d_nodal_s_basis(hf_array<double> &d_nodal_s_basis, hf_array<double> in_loc, int in_n_spts)
{
  int i,j,k;

  if (is_perfect_cube(in_n_spts))
    {
      int n_1d_spts = round(pow(in_n_spts,1./3.));
      hf_array<double> loc_1d_spts(n_1d_spts);
      set_loc_1d_spts(loc_1d_spts,n_1d_spts);

      for (int m=0;m<in_n_spts;++m)
        {
          i=(m/(n_1d_spts*n_1d_spts));
          j=(m-(n_1d_spts*n_1d_spts*i))/n_1d_spts;
          k=m-(n_1d_spts*j)-(n_1d_spts*n_1d_spts*i);

          d_nodal_s_basis(m,0)=eval_d_lagrange(in_loc(0),k,loc_1d_spts)*eval_lagrange(in_loc(1),j,loc_1d_spts)*eval_lagrange(in_loc(2),i,loc_1d_spts);
          d_nodal_s_basis(m,1)=eval_lagrange(in_loc(0),k,loc_1d_spts)*eval_d_lagrange(in_loc(1),j,loc_1d_spts)*eval_lagrange(in_loc(2),i,loc_1d_spts);
          d_nodal_s_basis(m,2)=eval_lagrange(in_loc(0),k,loc_1d_spts)*eval_lagrange(in_loc(1),j,loc_1d_spts)*eval_d_lagrange(in_loc(2),i,loc_1d_spts);

        }

    }
  else if (in_n_spts==20)
    {
      d_nodal_s_basis(0 ,0) = (1./8.*(in_loc(2)-1))*(-1+in_loc(1))*(in_loc(1)+in_loc(2)+2*in_loc(0)+1);
      d_nodal_s_basis(1 ,0) =  -(1./8.*(in_loc(2)-1))*(-1+in_loc(1))*(in_loc(1)+in_loc(2)-2*in_loc(0)+1) ;
      d_nodal_s_basis(2 ,0) =  -(1./8.*(in_loc(2)-1))*(in_loc(1)+1)*(in_loc(1)-in_loc(2)+2*in_loc(0)-1);
      d_nodal_s_basis(3 ,0) =  (1./8.*(in_loc(2)-1))*(in_loc(1)+1)*(in_loc(1)-in_loc(2)-2*in_loc(0)-1);
      d_nodal_s_basis(4 ,0) =  -(1./8.*(in_loc(2)+1))*(-1+in_loc(1))*(in_loc(1)-in_loc(2)+2*in_loc(0)+1);
      d_nodal_s_basis(5 ,0) =  (1./8.*(in_loc(2)+1))*(-1+in_loc(1))*(in_loc(1)-in_loc(2)-2*in_loc(0)+1);
      d_nodal_s_basis(6 ,0) =  (1./8.*(in_loc(2)+1))*(in_loc(1)+1)*(in_loc(1)+in_loc(2)+2*in_loc(0)-1);
      d_nodal_s_basis(7 ,0) =  -(1./8.*(in_loc(2)+1))*(in_loc(1)+1)*(in_loc(1)+in_loc(2)-1-2*in_loc(0));
      d_nodal_s_basis(8 ,0) =  -(1./2.)*in_loc(0)*(in_loc(2)-1)*(-1+in_loc(1));
      d_nodal_s_basis(9 ,0) =  (1./4.*(in_loc(2)-1))*(in_loc(1)*in_loc(1)-1);
      d_nodal_s_basis(10,0) =   (1./2.)*in_loc(0)*(in_loc(2)-1)*(in_loc(1)+1);
      d_nodal_s_basis(11,0) =   -(1./4.*(in_loc(2)-1))*(in_loc(1)*in_loc(1)-1);
      d_nodal_s_basis(12,0) =   -(1./4.*(-1+in_loc(1)))*(in_loc(2)*in_loc(2)-1);
      d_nodal_s_basis(13,0) =   (1./4.*(-1+in_loc(1)))*(in_loc(2)*in_loc(2)-1);
      d_nodal_s_basis(14,0) =   -(1./4.*(in_loc(1)+1))*(in_loc(2)*in_loc(2)-1);
      d_nodal_s_basis(15,0) =   (1./4.*(in_loc(1)+1))*(in_loc(2)*in_loc(2)-1);
      d_nodal_s_basis(16,0) =   (1./2.)*in_loc(0)*(in_loc(2)+1)*(-1+in_loc(1));
      d_nodal_s_basis(17,0) =   -(1./4.*(in_loc(2)+1))*(in_loc(1)*in_loc(1)-1);
      d_nodal_s_basis(18,0) =   -(1./2.)*in_loc(0)*(in_loc(2)+1)*(in_loc(1)+1);
      d_nodal_s_basis(19,0) =   (1./4.*(in_loc(2)+1))*(in_loc(1)*in_loc(1)-1);

      d_nodal_s_basis(0 ,1) = (1./8.*(in_loc(2)-1))*(-1+in_loc(0))*(in_loc(0)+in_loc(2)+2*in_loc(1)+1);
      d_nodal_s_basis(1 ,1) =  -(1./8.*(in_loc(2)-1))*(in_loc(0)+1)*(-in_loc(0)+in_loc(2)+2*in_loc(1)+1);
      d_nodal_s_basis(2 ,1) =  -(1./8.*(in_loc(2)-1))*(in_loc(0)+1)*(in_loc(0)-in_loc(2)+2*in_loc(1)-1);
      d_nodal_s_basis(3 ,1) =  (1./8.*(in_loc(2)-1))*(-1+in_loc(0))*(-in_loc(0)-in_loc(2)+2*in_loc(1)-1);
      d_nodal_s_basis(4 ,1) =  -(1./8.*(in_loc(2)+1))*(-1+in_loc(0))*(in_loc(0)-in_loc(2)+2*in_loc(1)+1);
      d_nodal_s_basis(5 ,1) =  (1./8.*(in_loc(2)+1))*(in_loc(0)+1)*(-in_loc(0)-in_loc(2)+2*in_loc(1)+1);
      d_nodal_s_basis(6 ,1) =  (1./8.*(in_loc(2)+1))*(in_loc(0)+1)*(in_loc(0)+in_loc(2)-1+2*in_loc(1));
      d_nodal_s_basis(7 ,1) =  -(1./8.*(in_loc(2)+1))*(-1+in_loc(0))*(-in_loc(0)+in_loc(2)-1+2*in_loc(1));
      d_nodal_s_basis(8 ,1) =  -(1./4.*(in_loc(2)-1))*(in_loc(0)*in_loc(0)-1);
      d_nodal_s_basis(9 ,1) =  (1./2.)*in_loc(1)*(in_loc(2)-1)*(in_loc(0)+1);
      d_nodal_s_basis(10,1) =   (1./4.*(in_loc(2)-1))*(in_loc(0)*in_loc(0)-1);
      d_nodal_s_basis(11,1) =   -(1./2.)*in_loc(1)*(in_loc(2)-1)*(-1+in_loc(0));
      d_nodal_s_basis(12,1) =   -(1./4.*(-1+in_loc(0)))*(in_loc(2)*in_loc(2)-1);
      d_nodal_s_basis(13,1) =   (1./4.*(in_loc(0)+1))*(in_loc(2)*in_loc(2)-1);
      d_nodal_s_basis(14,1) =   -(1./4.*(in_loc(0)+1))*(in_loc(2)*in_loc(2)-1);
      d_nodal_s_basis(15,1) =   (1./4.*(-1+in_loc(0)))*(in_loc(2)*in_loc(2)-1);
      d_nodal_s_basis(16,1) =   (1./4.*(in_loc(2)+1))*(in_loc(0)*in_loc(0)-1);
      d_nodal_s_basis(17,1) =   -(1./2.)*in_loc(1)*(in_loc(2)+1)*(in_loc(0)+1);
      d_nodal_s_basis(18,1) =   -(1./4.*(in_loc(2)+1))*(in_loc(0)*in_loc(0)-1);
      d_nodal_s_basis(19,1) =   (1./2.)*in_loc(1)*(in_loc(2)+1)*(-1+in_loc(0));

      d_nodal_s_basis(0 ,2) = (1./8.*(-1+in_loc(0)))*(-1+in_loc(1))*(in_loc(1)+in_loc(0)+2*in_loc(2)+1);
      d_nodal_s_basis(1 ,2) =  -(1./8.*(in_loc(1)-1))*(in_loc(0)+1)*(-in_loc(0)+in_loc(1)+2*in_loc(2)+1);
      d_nodal_s_basis(2 ,2) =  -(1./8.*(in_loc(0)+1))*(in_loc(1)+1)*(in_loc(1)+in_loc(0)-2*in_loc(2)-1);
      d_nodal_s_basis(3 ,2) =  (1./8.*(-1+in_loc(0)))*(in_loc(1)+1)*(in_loc(1)-in_loc(0)-2*in_loc(2)-1);
      d_nodal_s_basis(4 ,2) =  -(1./8.*(-1+in_loc(0)))*(-1+in_loc(1))*(in_loc(1)+in_loc(0)-2*in_loc(2)+1);
      d_nodal_s_basis(5 ,2) =  (1./8.*(in_loc(0)+1))*(-1+in_loc(1))*(in_loc(1)-in_loc(0)-2*in_loc(2)+1);
      d_nodal_s_basis(6 ,2) =  (1./8.*(in_loc(0)+1))*(in_loc(1)+1)*(in_loc(1)+in_loc(0)+2*in_loc(2)-1);
      d_nodal_s_basis(7 ,2) =  -(1./8.*(-1+in_loc(0)))*(in_loc(1)+1)*(in_loc(1)-in_loc(0)+2*in_loc(2)-1);
      d_nodal_s_basis(8 ,2) =  -(1./4.*(-1+in_loc(1)))*(in_loc(0)*in_loc(0)-1);
      d_nodal_s_basis(9 ,2) =  (1./4.*(in_loc(0)+1))*(in_loc(1)*in_loc(1)-1);
      d_nodal_s_basis(10,2) =   (1./4.*(in_loc(1)+1))*(in_loc(0)*in_loc(0)-1);
      d_nodal_s_basis(11,2) =   -(1./4.*(-1+in_loc(0)))*(in_loc(1)*in_loc(1)-1);
      d_nodal_s_basis(12,2) =   -(1./2.)*in_loc(2)*(-1+in_loc(0))*(-1+in_loc(1));
      d_nodal_s_basis(13,2) =   (1./2.)*in_loc(2)*(in_loc(0)+1)*(-1+in_loc(1));
      d_nodal_s_basis(14,2) =   -(1./2.)*in_loc(2)*(in_loc(0)+1)*(in_loc(1)+1);
      d_nodal_s_basis(15,2) =   (1./2.)*in_loc(2)*(-1+in_loc(0))*(in_loc(1)+1);
      d_nodal_s_basis(16,2) =   (1./4.*(-1+in_loc(1)))*(in_loc(0)*in_loc(0)-1);
      d_nodal_s_basis(17,2) =   -(1./4.*(in_loc(0)+1))*(in_loc(1)*in_loc(1)-1);
      d_nodal_s_basis(18,2) =   -(1./4.*(in_loc(1)+1))*(in_loc(0)*in_loc(0)-1);
      d_nodal_s_basis(19,2) =   (1./4.*(-1+in_loc(0)))*(in_loc(1)*in_loc(1)-1);
    }
  else
    {
      FatalError("Shape basis not implemented yet, exiting");
    }
}

//evaluate 3D Legendre basis
double eles_hexas::eval_legendre_basis_3D_hierarchical(int in_mode, hf_array<double> in_loc, int in_basis_order)
{
  double leg_basis;

  int n_dof=(in_basis_order+1)*(in_basis_order+1)*(in_basis_order+1);

  if(in_mode<n_dof)
  {
    int i,j,k,l;
    int mode;

    mode = 0; //mode=x+y*(N_x+1)+z*(N_x+1)*(N_z+1)
    for (l=0; l<3*in_basis_order+1; l++) // sum range from 0 to 3*order
    {
      for (k=0; k<l+1; k++) // k no more than the sum
      {
        for (j=0; j<l-k+1; j++)// j no more than sum-k
        {
          i = l-k-j;
          if(i<=in_basis_order && j<=in_basis_order && k<=in_basis_order)
          {

            if(mode==in_mode) // found the correct mode
            {
              leg_basis=eval_legendre(in_loc(0),i)*eval_legendre(in_loc(1),j)*eval_legendre(in_loc(2),k);
              return leg_basis;
            }
            mode++;
          }
        }
      }
    }
  }
  else
  {
    cout << "ERROR: Invalid mode when evaluating Legendre basis ...." << endl;
  }

FatalError("No mode is founded");
}

//get the indics of 3D Legendre basis
int eles_hexas::get_legendre_basis_3D_index(int in_mode, int in_basis_order, int &out_i, int &out_j, int &out_k)
{
  int n_dof = (in_basis_order + 1) * (in_basis_order + 1) * (in_basis_order + 1);

  if (in_mode < n_dof)
  {
    int i, j, k, l;
    int mode;
    mode = 0;                                    //mode=x+y*(N_x+1)+z*(N_x+1)*(N_z+1)
    for (l = 0; l < 3 * in_basis_order + 1; l++) // sum range from 0 to 3*order
    {
      for (k = 0; k < l + 1; k++) // k no more than the sum
      {
        for (j = 0; j < l - k + 1; j++) // j no more than sum-k
        {
          i = l - k - j;
          if (i <= in_basis_order && j <= in_basis_order && k <= in_basis_order)
          {
            if (mode == in_mode) // found the correct mode
            {
              out_i = i;
              out_j = j;
              out_k = k;
              return 0;
            }
            mode++;
          }
        }
      }
    }
  }
  else
  {
    cout << "ERROR: Invalid mode ...." << endl;
  }
  return -1;
}

void eles_hexas::fill_opp_3(hf_array<double>& opp_3)
{
  int i,j,k;
  hf_array<double> loc(n_dims);

  for(i=0;i<n_fpts_per_ele;++i)
    {
      for(j=0;j<n_upts_per_ele;++j)
        {
          for(k=0;k<n_dims;++k)
            {
              loc(k)=loc_upts(k,j);
            }

          opp_3(j,i)=eval_div_vcjh_basis(i,loc);
        }
    }
}

// evaluate divergence of vcjh basis

double eles_hexas::eval_div_vcjh_basis(int in_index, hf_array<double>& loc)
{
  int i,j,k;
  double eta;
  double div_vcjh_basis;
  int scheme = run_input.vcjh_scheme_hexa;

  if (scheme==0)
    eta = run_input.eta_hexa;
  else if (scheme < 5)
    eta = compute_eta(run_input.vcjh_scheme_hexa,order);

  i=(in_index/n_fpts_per_inter(0));
  j=(in_index-(n_fpts_per_inter(0)*i))/(order+1);
  k=in_index-(n_fpts_per_inter(0)*i)-((order+1)*j);

  if (scheme < 5) {

    if(i==0)
      div_vcjh_basis = -eval_lagrange(loc(0),order-k,loc_1d_upts) * eval_lagrange(loc(1),j,loc_1d_upts) * eval_d_vcjh_1d(loc(2),0,order,eta);
    else if(i==1)
      div_vcjh_basis = -eval_lagrange(loc(0),k,loc_1d_upts) * eval_lagrange(loc(2),j,loc_1d_upts) * eval_d_vcjh_1d(loc(1),0,order,eta);
    else if(i==2)
      div_vcjh_basis = eval_lagrange(loc(1),k,loc_1d_upts) * eval_lagrange(loc(2),j,loc_1d_upts) * eval_d_vcjh_1d(loc(0),1,order,eta);
    else if(i==3)
      div_vcjh_basis = eval_lagrange(loc(0),order-k,loc_1d_upts) * eval_lagrange(loc(2),j,loc_1d_upts) * eval_d_vcjh_1d(loc(1),1,order,eta);
    else if(i==4)
      div_vcjh_basis = -eval_lagrange(loc(1),order-k,loc_1d_upts) * eval_lagrange(loc(2),j,loc_1d_upts) * eval_d_vcjh_1d(loc(0),0,order,eta);
    else if(i==5)
      div_vcjh_basis = eval_lagrange(loc(0),k,loc_1d_upts) * eval_lagrange(loc(1),j,loc_1d_upts) * eval_d_vcjh_1d(loc(2),1,order,eta);

  }
  // OFR scheme
  else if (scheme == 5) {

    if(i==0)
      div_vcjh_basis = -eval_lagrange(loc(0),order-k,loc_1d_upts) * eval_lagrange(loc(1),j,loc_1d_upts) * eval_d_ofr_1d(loc(2),0,order);
    else if(i==1)
      div_vcjh_basis = -eval_lagrange(loc(0),k,loc_1d_upts) * eval_lagrange(loc(2),j,loc_1d_upts) * eval_d_ofr_1d(loc(1),0,order);
    else if(i==2)
      div_vcjh_basis = eval_lagrange(loc(1),k,loc_1d_upts) * eval_lagrange(loc(2),j,loc_1d_upts) * eval_d_ofr_1d(loc(0),1,order);
    else if(i==3)
      div_vcjh_basis = eval_lagrange(loc(0),order-k,loc_1d_upts) * eval_lagrange(loc(2),j,loc_1d_upts) * eval_d_ofr_1d(loc(1),1,order);
    else if(i==4)
      div_vcjh_basis = -eval_lagrange(loc(1),order-k,loc_1d_upts) * eval_lagrange(loc(2),j,loc_1d_upts) * eval_d_ofr_1d(loc(0),0,order);
    else if(i==5)
      div_vcjh_basis = eval_lagrange(loc(0),k,loc_1d_upts) * eval_lagrange(loc(1),j,loc_1d_upts) * eval_d_ofr_1d(loc(2),1,order);

  }
  // OESFR scheme
  else if (scheme == 6) {

    if(i==0)
      div_vcjh_basis = -eval_lagrange(loc(0),order-k,loc_1d_upts) * eval_lagrange(loc(1),j,loc_1d_upts) * eval_d_oesfr_1d(loc(2),0,order);
    else if(i==1)
      div_vcjh_basis = -eval_lagrange(loc(0),k,loc_1d_upts) * eval_lagrange(loc(2),j,loc_1d_upts) * eval_d_oesfr_1d(loc(1),0,order);
    else if(i==2)
      div_vcjh_basis = eval_lagrange(loc(1),k,loc_1d_upts) * eval_lagrange(loc(2),j,loc_1d_upts) * eval_d_oesfr_1d(loc(0),1,order);
    else if(i==3)
      div_vcjh_basis = eval_lagrange(loc(0),order-k,loc_1d_upts) * eval_lagrange(loc(2),j,loc_1d_upts) * eval_d_oesfr_1d(loc(1),1,order);
    else if(i==4)
      div_vcjh_basis = -eval_lagrange(loc(1),order-k,loc_1d_upts) * eval_lagrange(loc(2),j,loc_1d_upts) * eval_d_oesfr_1d(loc(0),0,order);
    else if(i==5)
      div_vcjh_basis = eval_lagrange(loc(0),k,loc_1d_upts) * eval_lagrange(loc(1),j,loc_1d_upts) * eval_d_oesfr_1d(loc(2),1,order);

  }

  return div_vcjh_basis;
}

// Get position of 1d solution point
double eles_hexas::get_loc_1d_upt(int in_index)
{
  return loc_1d_upts(in_index);
}

/*! Calculate element volume */
double eles_hexas::calc_ele_vol(double& detjac)
{
  double vol;
  // Element volume = |Jacobian|*width*height*span of reference element
  vol = detjac*8.;
  return vol;
}

/*! Calculate element reference length for timestep calculation */
double eles_hexas::calc_h_ref_specific(int in_ele)
  {
    double out_h_ref;

    // Compute edge lengths (bottom to top, Counter-clockwise)
    length(0) = sqrt(pow(shape(0,0,in_ele) - shape(0,1,in_ele),2.0) + pow(shape(1,0,in_ele) - shape(1,1,in_ele),2.0) + pow(shape(2,0,in_ele) - shape(2,1,in_ele),2.0));
    length(1) = sqrt(pow(shape(0,1,in_ele) - shape(0,3,in_ele),2.0) + pow(shape(1,1,in_ele) - shape(1,3,in_ele),2.0) + pow(shape(2,1,in_ele) - shape(2,3,in_ele),2.0));
    length(2) = sqrt(pow(shape(0,3,in_ele) - shape(0,2,in_ele),2.0) + pow(shape(1,3,in_ele) - shape(1,2,in_ele),2.0) + pow(shape(2,3,in_ele) - shape(2,2,in_ele),2.0));
    length(3) = sqrt(pow(shape(0,2,in_ele) - shape(0,0,in_ele),2.0) + pow(shape(1,2,in_ele) - shape(1,0,in_ele),2.0) + pow(shape(2,2,in_ele) - shape(2,0,in_ele),2.0));
    length(4) = sqrt(pow(shape(0,4,in_ele) - shape(0,5,in_ele),2.0) + pow(shape(1,4,in_ele) - shape(1,5,in_ele),2.0) + pow(shape(2,4,in_ele) - shape(2,5,in_ele),2.0));
    length(5) = sqrt(pow(shape(0,5,in_ele) - shape(0,7,in_ele),2.0) + pow(shape(1,5,in_ele) - shape(1,7,in_ele),2.0) + pow(shape(2,5,in_ele) - shape(2,7,in_ele),2.0));
    length(6) = sqrt(pow(shape(0,7,in_ele) - shape(0,6,in_ele),2.0) + pow(shape(1,7,in_ele) - shape(1,6,in_ele),2.0) + pow(shape(2,7,in_ele) - shape(2,6,in_ele),2.0));
    length(7) = sqrt(pow(shape(0,6,in_ele) - shape(0,4,in_ele),2.0) + pow(shape(1,6,in_ele) - shape(1,4,in_ele),2.0) + pow(shape(2,6,in_ele) - shape(2,4,in_ele),2.0));
    length(8) = sqrt(pow(shape(0,1,in_ele) - shape(0,5,in_ele),2.0) + pow(shape(1,1,in_ele) - shape(1,5,in_ele),2.0) + pow(shape(2,1,in_ele) - shape(2,5,in_ele),2.0));
    length(9) = sqrt(pow(shape(0,3,in_ele) - shape(0,7,in_ele),2.0) + pow(shape(1,3,in_ele) - shape(1,7,in_ele),2.0) + pow(shape(2,3,in_ele) - shape(2,7,in_ele),2.0));
    length(10) = sqrt(pow(shape(0,0,in_ele) - shape(0,4,in_ele),2.0) + pow(shape(1,0,in_ele) - shape(1,4,in_ele),2.0) + pow(shape(2,0,in_ele) - shape(2,4,in_ele),2.0));
    length(11) = sqrt(pow(shape(0,2,in_ele) - shape(0,6,in_ele),2.0) + pow(shape(1,2,in_ele) - shape(1,6,in_ele),2.0) + pow(shape(2,2,in_ele) - shape(2,6,in_ele),2.0));
    // Get minimum edge length
    out_h_ref = length.get_min();
    return out_h_ref;
  }

  int eles_hexas::calc_p2c(hf_array<double> &in_pos)
  {
    hf_array<double> plane_coeff;
    hf_array<double> pos_centroid;
    hf_array<int> vertex_index_loc(3);
    hf_array<double> pos_plane_pts(n_dims, 3);
    for (int i = 0; i < n_eles; i++) //for each element
    {
      int alpha = 1; //indicator

      //calculate centroid
      hf_array<double> temp_pos_s_pts(n_dims, n_spts_per_ele(i));
      for (int j = 0; j < n_spts_per_ele(i); j++)
        for (int k = 0; k < n_dims; k++)
          temp_pos_s_pts(k, j) = shape(k, j, i);
      pos_centroid = calc_centroid(temp_pos_s_pts);

      int num_f_per_c = 6;

      for (int j = 0; j < num_f_per_c; j++) //for each face
      {
        if (is_perfect_cube(n_spts_per_ele(i)))
        {
          int n_spts_1d = 2;
          int shift = n_spts_1d * n_spts_1d * (n_spts_1d - 1);
          //store local vertex index
          if (j == 0)
          {
            vertex_index_loc(0) = n_spts_1d - 1;
            vertex_index_loc(1) = 0;
            vertex_index_loc(2) = n_spts_1d * (n_spts_1d - 1);
          }
          else if (j == 1)
          {
            vertex_index_loc(0) = 0;
            vertex_index_loc(1) = n_spts_1d - 1;
            vertex_index_loc(2) = n_spts_1d - 1 + shift;
          }
          else if (j == 2)
          {
            vertex_index_loc(0) = n_spts_1d - 1;
            vertex_index_loc(1) = n_spts_1d * n_spts_1d - 1;
            vertex_index_loc(2) = n_spts_per_ele(i) - 1;
          }
          else if (j == 3)
          {
            vertex_index_loc(0) = n_spts_1d * n_spts_1d - 1;
            vertex_index_loc(1) = n_spts_1d * (n_spts_1d - 1);
            vertex_index_loc(2) = n_spts_per_ele(i) - n_spts_1d;
          }
          else if (j == 4)
          {
            vertex_index_loc(0) = n_spts_1d * (n_spts_1d - 1);
            vertex_index_loc(1) = 0;
            vertex_index_loc(2) = shift;
          }
          else if (j == 5)
          {
            vertex_index_loc(0) = shift;
            vertex_index_loc(1) = n_spts_1d - 1 + shift;
            vertex_index_loc(2) = n_spts_per_ele(i) - 1;
          }
        }
        else if (n_spts_per_ele(i) == 20)
        {
          if (j == 0)
          {
            vertex_index_loc(0) = 1;
            vertex_index_loc(1) = 0;
            vertex_index_loc(2) = 3;
          }
          else if (j == 1)
          {
            vertex_index_loc(0) = 0;
            vertex_index_loc(1) = 1;
            vertex_index_loc(2) = 5;
          }
          else if (j == 2)
          {
            vertex_index_loc(0) = 1;
            vertex_index_loc(1) = 2;
            vertex_index_loc(2) = 6;
          }
          else if (j == 3)
          {
            vertex_index_loc(0) = 2;
            vertex_index_loc(1) = 3;
            vertex_index_loc(2) = 7;
          }
          else if (j == 4)
          {
            vertex_index_loc(0) = 3;
            vertex_index_loc(1) = 0;
            vertex_index_loc(2) = 4;
          }
          else if (j == 5)
          {
            vertex_index_loc(0) = 4;
            vertex_index_loc(1) = 5;
            vertex_index_loc(2) = 6;
          }
        }
        else
        {
          FatalError("elemment type not implemented");
        }
        //store position of vertex on each face
        for (int k = 0; k < 3; k++)        //number of points needed to define a plane
          for (int l = 0; l < n_dims; l++) //dims
            pos_plane_pts(l, k) = shape(l, vertex_index_loc(k), i);

        //calculate plane coeff
        plane_coeff = calc_plane(pos_plane_pts);

        alpha = alpha * ((plane_coeff(0) * in_pos(0) + plane_coeff(1) * in_pos(1) + plane_coeff(2) * in_pos(2) + plane_coeff(3)) *
                             (plane_coeff(0) * pos_centroid(0) + plane_coeff(1) * pos_centroid(1) + plane_coeff(2) * pos_centroid(2) + plane_coeff(3)) >= 0);
        if (alpha == 0)
          break;
      }
      if (alpha > 0)
        return i;
    }

    return -1;
}

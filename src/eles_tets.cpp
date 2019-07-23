/*!
 * \file eles_tets.cpp
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
#include "../include/eles_tets.h"
#include "../include/cubature_tri.h"
#include "../include/cubature_tet.h"

using namespace std;

// #### constructors ####

// default constructor

eles_tets::eles_tets()
{
}

// #### methods ####


void eles_tets::setup_ele_type_specific()
{
#ifndef _MPI
  cout << "Initializing tets" << endl;
#endif

  ele_type=2;
  n_dims=3;

  if (run_input.equation==0)
    n_fields=5;
  else if (run_input.equation==1)
    n_fields=1;
  else
    FatalError("Equation not supported");

  if (run_input.RANS==1)
    n_fields++;

  n_inters_per_ele=4;

  n_upts_per_ele=(order+3)*(order+2)*(order+1)/6;
  upts_type=run_input.upts_type_tet;
  set_loc_upts();
  set_vandermonde();

  //set shock capturing arrays
  if(run_input.shock_cap)
  {
    if (run_input.shock_det != 0)
      FatalError("Shock detector not implemented.");
    if (run_input.shock_cap == 1)//exp filter
      set_exp_filter();
    else
      FatalError("Shock capturing method not implemented.");
  }

  n_ppts_per_ele=(p_res+2)*(p_res+1)*p_res/6;
  n_peles_per_ele = (p_res-1)*(p_res)*(p_res+1)/6 + 4*(p_res-2)*(p_res-1)*(p_res)/6 +(p_res-3)*(p_res-2)*(p_res-1)/6;
  n_verts_per_ele = 4;

  set_loc_ppts();
  set_opp_p();

  set_inters_cubpts();
  set_volume_cubpts(order, loc_volume_cubpts, weight_volume_cubpts);
  set_opp_volume_cubpts(loc_volume_cubpts, opp_volume_cubpts);

  if(run_input.over_int)
    set_over_int();

  n_fpts_per_inter.setup(4);

  n_fpts_per_inter(0)=(order+2)*(order+1)/2;
  n_fpts_per_inter(1)=(order+2)*(order+1)/2;
  n_fpts_per_inter(2)=(order+2)*(order+1)/2;
  n_fpts_per_inter(3)=(order+2)*(order+1)/2;

  n_fpts_per_ele=n_inters_per_ele*(order+2)*(order+1)/2;

  fpts_type=run_input.fpts_type_tet;

  set_tloc_fpts();

  //set_loc_spts();

  set_tnorm_fpts();

  set_opp_0(run_input.sparse_tet);
  set_opp_1(run_input.sparse_tet);
  set_opp_2(run_input.sparse_tet);
  set_opp_3(run_input.sparse_tet);

  if(viscous)
    {
      set_opp_4(run_input.sparse_tet);
      set_opp_5(run_input.sparse_tet);
      set_opp_6(run_input.sparse_tet);

      temp_grad_u.setup(n_fields,n_dims);

      // Compute tet filter matrix
      if(LES_filter) compute_filter_upts();
    }

  temp_u.setup(n_fields);
  temp_f.setup(n_fields,n_dims);
}

void eles_tets::set_connectivity_plot()
{
  int vertex_0,vertex_1,vertex_2,vertex_3,vertex_4,vertex_5;
  int count=0;
  int temp = (p_res)*(p_res+1)*(p_res+2)/6;

  /*! Loop over the plot sub-elements. */
  /*! For tets there are 3 sets with different orientations */
  /*! First set */
  for(int k=0;k<p_res-1;++k){
      for(int j=0;j<p_res-1-k;++j){
          for(int i=0;i<p_res-1-k-j;++i){

              vertex_0 = temp - (p_res-k)*(p_res+1-k)*(p_res+2-k)/6 + j*(p_res-k) - (j-1)*j/2 + i;
              vertex_1 = temp - (p_res-k)*(p_res+1-k)*(p_res+2-k)/6 + j*(p_res-k) - (j-1)*j/2 + i + 1;
              vertex_2 = temp - (p_res-k)*(p_res+1-k)*(p_res+2-k)/6 + (j+1)*(p_res-k) - (j)*(j+1)/2 + i;
              vertex_3 = temp - (p_res-(k+1))*(p_res+1-(k+1))*(p_res+2-(k+1))/6 + j*(p_res-(k+1)) - (j-1)*j/2 + i;

              connectivity_plot(0,count) = vertex_0;
              connectivity_plot(1,count) = vertex_1;
              connectivity_plot(2,count) = vertex_2;
              connectivity_plot(3,count) = vertex_3;
              count++;
            }
        }
    }

  /*! Second set */
  for(int k=0;k<p_res-2;++k){
      for(int j=0;j<p_res-2-k;++j){
          for(int i=0;i<p_res-2-k-j;++i){
              vertex_0 = temp - (p_res-k)*(p_res+1-k)*(p_res+2-k)/6 + j*(p_res-k) - (j-1)*j/2 + i + 1;
              vertex_1 = temp - (p_res-k)*(p_res+1-k)*(p_res+2-k)/6 + (j+1)*(p_res-k) - (j)*(j+1)/2 + i + 1;
              vertex_2 = temp - (p_res-(k+1))*(p_res+1-(k+1))*(p_res+2-(k+1))/6 + j*(p_res-(k+1)) - (j-1)*j/2 + i + 1;
              vertex_3 = temp - (p_res-(k+1))*(p_res+1-(k+1))*(p_res+2-(k+1))/6 + (j+1)*(p_res-(k+1)) - (j)*(j+1)/2 + (i-1) + 1;
              vertex_4 = temp - (p_res-(k+1))*(p_res+1-(k+1))*(p_res+2-(k+1))/6 + (j)*(p_res-(k+1)) - (j-1)*(j)/2 + (i-1) + 1;
              vertex_5 = temp - (p_res-(k))*(p_res+1-(k))*(p_res+2-(k))/6 + (j+1)*(p_res-(k)) - (j)*(j+1)/2 + (i-1) + 1;


              connectivity_plot(0,count) = vertex_0;
              connectivity_plot(1,count) = vertex_2;
              connectivity_plot(2,count) = vertex_1;
              connectivity_plot(3,count) = vertex_4;
              count++;

              connectivity_plot(0,count) = vertex_2;
              connectivity_plot(1,count) = vertex_3;
              connectivity_plot(2,count) = vertex_1;
              connectivity_plot(3,count) = vertex_4;
              count++;

              connectivity_plot(0,count) = vertex_5;
              connectivity_plot(1,count) = vertex_1;
              connectivity_plot(2,count) = vertex_3;
              connectivity_plot(3,count) = vertex_4;
              count++;

              connectivity_plot(0,count) = vertex_0;
              connectivity_plot(1,count) = vertex_4;
              connectivity_plot(2,count) = vertex_1;
              connectivity_plot(3,count) = vertex_5;
              count++;
            }
        }
    }

  /*! Third set */
  for(int k=0;k<p_res-3;++k){
      for(int j=0;j<p_res-3-k;++j){
          for(int i=0;i<p_res-3-k-j;++i){

              vertex_0 = temp - (p_res-k)*(p_res+1-k)*(p_res+2-k)/6 + (j+1)*(p_res-k) - (j)*(j+1)/2 + i + 1;
              vertex_1 = temp - (p_res-(k+1))*(p_res+1-(k+1))*(p_res+2-(k+1))/6 + (j)*(p_res-(k+1)) - (j-1)*(j)/2 + i + 1;
              vertex_2 = temp - (p_res-(k+1))*(p_res+1-(k+1))*(p_res+2-(k+1))/6 + (j+1)*(p_res-(k+1)) - (j)*(j+1)/2 + i ;
              vertex_3 = temp - (p_res-(k+1))*(p_res+1-(k+1))*(p_res+2-(k+1))/6 + (j+1)*(p_res-(k+1)) - (j)*(j+1)/2 + i + 1;

              connectivity_plot(0,count) = vertex_0;
              connectivity_plot(1,count) = vertex_1;
              connectivity_plot(2,count) = vertex_2;
              connectivity_plot(3,count) = vertex_3;
              count++;
            }
        }
    }
}



// set location of solution points in standard element

void eles_tets::set_loc_upts(void)
{
  loc_upts.setup(n_dims,n_upts_per_ele);
  cubature_tet cub_tet(upts_type, order);
  for (int i = 0; i < n_upts_per_ele; i++)
  {
    loc_upts(0, i) = cub_tet.get_r(i);
    loc_upts(1, i) = cub_tet.get_s(i);
    loc_upts(2, i) = cub_tet.get_t(i);
  }
}

// set location of flux points in standard element

void eles_tets::set_tloc_fpts(void)
{

  int i,j;
  tloc_fpts.setup(n_dims,n_fpts_per_ele);
  
  hf_array<double> loc_tri_fpts(n_fpts_per_inter(0),2);
  cubature_tri cub_tri(fpts_type, order);
  for (int i = 0; i < n_fpts_per_inter(0); i++)
  {
    loc_tri_fpts(i, 0) = cub_tri.get_r(i);
    loc_tri_fpts(i, 1) = cub_tri.get_s(i);
  }

  // Now map these points to 3D

  int i_tri_fpts, i_alpha;
  for(j=0;j<order+1;j++)
    {
      for (i=0;i<order+1-j;i++)
        {

          i_tri_fpts = j*(order+1) -(j-1)*j/2+i;
          i_alpha = j*(order+1) - (j-1)*j/2 + (order-j-i);
          tloc_fpts(0,i_tri_fpts) = loc_tri_fpts(i_alpha,0);

          i_alpha = i_tri_fpts;
          tloc_fpts(1,i_tri_fpts) = loc_tri_fpts(i_alpha,0);
          tloc_fpts(2,i_tri_fpts) = loc_tri_fpts(i_alpha,1);

          tloc_fpts(0,n_fpts_per_inter(0)+i_tri_fpts) = -1;
          tloc_fpts(1,n_fpts_per_inter(0)+i_tri_fpts) = loc_tri_fpts(i_alpha,1);
          tloc_fpts(2,n_fpts_per_inter(0)+i_tri_fpts) = loc_tri_fpts(i_alpha,0);

          tloc_fpts(0,2*n_fpts_per_inter(0)+i_tri_fpts) = loc_tri_fpts(i_alpha,0);
          tloc_fpts(1,2*n_fpts_per_inter(0)+i_tri_fpts) = -1;
          tloc_fpts(2,2*n_fpts_per_inter(0)+i_tri_fpts) = loc_tri_fpts(i_alpha,1);

          tloc_fpts(0,3*n_fpts_per_inter(0)+i_tri_fpts) = loc_tri_fpts(i_alpha,1);
          tloc_fpts(1,3*n_fpts_per_inter(0)+i_tri_fpts) = loc_tri_fpts(i_alpha,0);
          tloc_fpts(2,3*n_fpts_per_inter(0)+i_tri_fpts) = -1;
        }
    }

}

// set location of plot points in standard element

void eles_tets::set_loc_ppts(void)
{
  loc_ppts.setup(n_dims,n_ppts_per_ele);
  int index;

  for(int k=0;k<p_res;k++)
    {
      for(int j=0;j<p_res-k;j++)
        {
          for(int i=0;i<p_res-k-j;i++)
            {

              index = (p_res)*(p_res+1)*(p_res+2)/6 -
                  (p_res-k)*(p_res+1-k)*(p_res+2-k)/6 +
                  j*(p_res-k)-(j-1)*j/2
                  + i;

              loc_ppts(0,index) = -1.0+(2.0*i/(p_res-1));
              loc_ppts(1,index) = -1.0+(2.0*j/(p_res-1));
              loc_ppts(2,index) = -1.0+(2.0*k/(p_res-1));
            }
        }
    }
}

void eles_tets::set_inters_cubpts(void)
{


  n_cubpts_per_inter.setup(n_inters_per_ele);
  loc_inters_cubpts.setup(n_inters_per_ele);
  weight_inters_cubpts.setup(n_inters_per_ele);
  tnorm_inters_cubpts.setup(n_inters_per_ele);

  cubature_tri cub_tri(0,order);
  int n_cubpts_tri = cub_tri.get_n_pts();

  for (int i=0;i<n_inters_per_ele;i++)
    n_cubpts_per_inter(i) = n_cubpts_tri;

  for (int i=0;i<n_inters_per_ele;i++) {

      loc_inters_cubpts(i).setup(n_dims,n_cubpts_per_inter(i));
      weight_inters_cubpts(i).setup(n_cubpts_per_inter(i));
      tnorm_inters_cubpts(i).setup(n_dims,n_cubpts_per_inter(i));

      for (int j=0;j<n_cubpts_tri;j++) {

          if (i==0) {
              loc_inters_cubpts(i)(0,j)=cub_tri.get_r(j);
              loc_inters_cubpts(i)(1,j)=cub_tri.get_s(j);
              loc_inters_cubpts(i)(2,j)=-1.-cub_tri.get_r(j)-cub_tri.get_s(j);
            }
          else if (i==1) {
              loc_inters_cubpts(i)(0,j)=-1;
              loc_inters_cubpts(i)(1,j)=cub_tri.get_r(j);
              loc_inters_cubpts(i)(2,j)=cub_tri.get_s(j);
            }
          else if (i==2) {
              loc_inters_cubpts(i)(0,j)=cub_tri.get_r(j);
              loc_inters_cubpts(i)(1,j)=-1.;
              loc_inters_cubpts(i)(2,j)=cub_tri.get_s(j);
            }
          else if (i==3) {
              loc_inters_cubpts(i)(0,j)=cub_tri.get_r(j);
              loc_inters_cubpts(i)(1,j)=cub_tri.get_s(j);
              loc_inters_cubpts(i)(2,j)=-1.;
            }

          weight_inters_cubpts(i)(j) = cub_tri.get_weight(j);

          if (i==0) {
              tnorm_inters_cubpts(i)(0,j)= 1./sqrt(3.);
              tnorm_inters_cubpts(i)(1,j)= 1./sqrt(3.);
              tnorm_inters_cubpts(i)(2,j)= 1./sqrt(3.);
            }
          else if (i==1) {
              tnorm_inters_cubpts(i)(0,j)= -1.0;
              tnorm_inters_cubpts(i)(1,j)= 0.;
              tnorm_inters_cubpts(i)(2,j)= 0.;
            }
          else if (i==2) {
              tnorm_inters_cubpts(i)(0,j)= 0.;
              tnorm_inters_cubpts(i)(1,j)= -1.;
              tnorm_inters_cubpts(i)(2,j)= 0.;
            }
          else if (i==3) {
              tnorm_inters_cubpts(i)(0,j)= 0.;
              tnorm_inters_cubpts(i)(1,j)= 0.;
              tnorm_inters_cubpts(i)(2,j)= -1.;
            }

        }
    }
  set_opp_inters_cubpts();

}


void eles_tets::set_volume_cubpts(int in_order, hf_array<double> &out_loc_volume_cubpts, hf_array<double> &out_weight_volume_cubpts)
{
  cubature_tet cub_tet(0,in_order);
  int n_cubpts_tet = cub_tet.get_n_pts();

  out_loc_volume_cubpts.setup(n_dims,n_cubpts_tet);
  out_weight_volume_cubpts.setup(n_cubpts_tet);

  for (int i=0;i<n_cubpts_tet;i++)
    {
      out_loc_volume_cubpts(0,i) = cub_tet.get_r(i);
      out_loc_volume_cubpts(1,i) = cub_tet.get_s(i);
      out_loc_volume_cubpts(2,i) = cub_tet.get_t(i);
      out_weight_volume_cubpts(i) = cub_tet.get_weight(i);
    }
}


// Compute the surface jacobian determinant on a face
double eles_tets::compute_inter_detjac_inters_cubpts(int in_inter,hf_array<double> d_pos)
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

  if (in_inter==0) // r=u,s=v,t=-u-v;
    {
      xu = xr-xt;
      yu = yr-yt;
      zu = zr-zt;

      xv = xs-xt;
      yv = ys-yt;
      zv = zs-zt;
      //cout << "xu=" << xu << "yu=" << yu << "zu=" << zu << endl;
      //cout << "xv=" << xv << "yv=" << yv << "zv=" << zv << endl;
    }
  else if (in_inter==1) // u=s, v=t
    {
      xu = xr;
      yu = yr;
      zu = zr;

      xv = xt;
      yv = yt;
      zv = zt;
    }
  else if (in_inter==2) //u=r, v=t
    {
      xu = xr;
      yu = yr;
      zu = zr;

      xv = xt;
      yv = yt;
      zv = zt;
    }
  else if (in_inter==3) //u=r,v=s
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
  //cout << "temp0=" << temp0 << "temp1" << temp1 << "temp2" << temp2 << endl;
  //cout << "in_inter=" << in_inter << "output=" << output << endl;


  return output;
}

// set location of shape points in standard element
/*
void eles_tets::set_loc_spts(void)
{

// First order

    // Node 0 at (-1,-1,-1)
    // Node 1 at (1,-1,-1)
    // Node 2 at (-1,1,-1)
    // Node 3 at (-1,-1,1)

// 	z
//	|      y
// 	      /
//  3
//  |   2
//  |  /
//	| /
//	|/
//  0--------1    ----> x

// Second order

// Node 0 at (-1,-1,-1)
// Node 1 at (1,-1,-1)
// Node 2 at (-1,1,-1)
// Node 3 at (-1,-1,1)
// Node 4 at (0,-1,-1)
// Node 5 at (-1,0,-1)
// Node 6 at (-1,-1,0)
// Node 7 at (0,0,-1)
// Node 8 at (-1,0,0)
// Node 9 at (0,-1,0)

// 	z
//	|      y
// 	      /
//  3
//  |   2
//  6  / \
//	| 5   7
//	|/     \
//  0---4----1    ----> x

}
*/

// set transformed normal at flux points

void eles_tets::set_tnorm_fpts(void)
{
  int i,j,fpt;
  tnorm_fpts.setup(n_dims,n_fpts_per_ele);

  for (i=0;i<n_inters_per_ele;i++)
    {
      for(j=0;j<(order+1)*(order+2)/2;j++)
        {
          fpt = (order+1)*(order+2)/2*i+j;

          if (i==0) {
              tnorm_fpts(0,fpt)= 1./sqrt(3.);
              tnorm_fpts(1,fpt)= 1./sqrt(3.);
              tnorm_fpts(2,fpt)= 1./sqrt(3.);
            }
          else if (i==1) {
              tnorm_fpts(0,fpt)= -1.0;
              tnorm_fpts(1,fpt)= 0.;
              tnorm_fpts(2,fpt)= 0.;
            }
          else if (i==2) {
              tnorm_fpts(0,fpt)= 0.;
              tnorm_fpts(1,fpt)= -1.0;
              tnorm_fpts(2,fpt)= 0.;
            }
          else if (i==3) {
              tnorm_fpts(0,fpt)= 0.;
              tnorm_fpts(1,fpt)= 0.;
              tnorm_fpts(2,fpt)= -1.0;
            }
        }
    }
}

// Filtering operators for use in subgrid-scale modelling
void eles_tets::compute_filter_upts(void)
{
  int i,j,k,l,N,N2;
  double dlt, k_c, sum, vol, norm;
  N = n_upts_per_ele;

  hf_array<double> X(n_dims,N);
  hf_array<double> beta(N,N);
  hf_array<double> B(N);

  filter_upts.setup(N,N);

  X = loc_upts;

  N2 = N/2;
  // If N is odd, round up N/2
  if(N % 2 != 0){N2 += 1;}
  // Cutoff wavenumber
  k_c = 1.0/run_input.filter_ratio;

  // Approx resolution in element (assumes uniform point spacing)
  dlt = 2.0/order;

  // Normalised solution point separation: r = sqrt((x_a-x_b)^2 + (y_a-y_b)^2)
  for (i=0;i<N;i++)
    for (j=i;j<N;j++)
      beta(i,j) = sqrt(pow(X(0,i)-X(0,j),2) + pow(X(1,i)-X(1,j),2) + pow(X(2,i)-X(2,j),2))/dlt;
  for (i=0;i<N;i++)
    for (j=0;j<i;j++)
      beta(i,j) = beta(j,i);

  if(run_input.filter_type==0) // Vasilyev filter
    {
      FatalError("Vasilyev filters not implemented for tris. Exiting.");
    }
  else if(run_input.filter_type==1) // Discrete Gaussian filter
    {
      if (rank==0) cout<<"Building discrete Gaussian filter"<<endl;
      FatalError("Gaussian filter not implemented for tris. Exiting.");
    }
  else if(run_input.filter_type==2) // Modal coefficient filter
    {
      if (rank==0) cout<<"Building modal filter"<<endl;

      // Compute modal filter
      compute_modal_filter_tet();

    }
  else // Simple average for low order
    {
      if (rank==0) cout<<"Building average filter"<<endl;
      sum=0;
      for(i=0;i<N;i++)
        {
          for(j=0;j<N;j++)
            {
              filter_upts(i,j) = 1.0/N;
              sum+=1.0/N;
            }
        }
    }

  // Ensure symmetry and normalisation
  for(i=0;i<N2;i++)
    {
      for(j=0;j<N;j++)
        {
          filter_upts(i,j) = 0.5*filter_upts(i,j) + filter_upts(N-i-1,N-j-1);
          filter_upts(N-i-1,N-j-1) = filter_upts(i,j);
        }
    }

  for(i=0;i<N2;i++)
    {
      norm = 0.0;
      for(j=0;j<N;j++)
        norm += filter_upts(i,j); // or abs(filter_upts(i,j))?
      for(j=0;j<N;j++)
        filter_upts(i,j) /= norm;
      for(j=0;j<N;j++)
        filter_upts(N-i-1,N-j-1) = filter_upts(i,j);
    }

  sum = 0;
  for(i=0;i<N;i++)
    for(j=0;j<N;j++)
      sum+=filter_upts(i,j);
}

// Compute a modal filter matrix for a tetrahedral element, given Vandermonde matrix and inverse
void eles_tets::compute_modal_filter_tet()
{
	int i;
  //int j,k,ind=0;
  int N=n_upts_per_ele;
	//double Cp=0.1;     // Dubiner SVV filter strength coeff.
	//double p=order;    // filter exponent
	//double alpha;
  double eta;

	filter_upts.initialize_to_zero();

  // Exponential filter (SVV method) (similar to Meister et al 2009)

  // Full form: alpha = Cp*(p+1)*dt/delta
  /*alpha = Cp*p;

  for(i=0;i<p+1;i++) {
    for(j=0;j<p-i+1;j++) {
      for(k=0;k<p-i-j+1;k++) {
        eta = (i+j+k)/(p+1.0);
        modal(ind,ind) = exp(-alpha*pow(eta,2*p));
        ind++;
      }
    }
  }*/

  // Gaussian filter in modal space (from SD3D)
  for(i=0;i<N;i++) {
    eta = i/double(N);
    filter_upts(i,i) = exp(-pow(2.0*eta,2.0)/48.0);
  }
  filter_upts = mult_arrays(vandermonde,filter_upts);
  filter_upts = mult_arrays(filter_upts,inv_vandermonde);
}

//#### helper methods ####

// initialize the vandermonde matrix
void eles_tets::set_vandermonde(void)
{
  vandermonde.setup(n_upts_per_ele,n_upts_per_ele);

  // create the vandermonde matrix
  for (int i=0;i<n_upts_per_ele;i++)
    for (int j=0;j<n_upts_per_ele;j++)
      vandermonde(i,j) = eval_dubiner_basis_3d(loc_upts(0,i),loc_upts(1,i),loc_upts(2,i),j,order);

  // Store its inverse
  inv_vandermonde = inv_array(vandermonde);
}

void eles_tets::set_exp_filter(void)
{
  exp_filter.setup(n_upts_per_ele, n_upts_per_ele);
  exp_filter.initialize_to_zero();
  int i, j, k, mode;
  double eta;
  double eta_c = (double)run_input.expf_cutoff / (double)order;

  mode = 0;
  for (k = 0; k < order + 1; k++) //sum of x,y,z mode
  {
    for (j = 0; j < k + 1; j++) //j<=sum
    {
      for (i = 0; i < k - j + 1; i++) //i<=sum-j
      {
        eta = (double)k / (double)(order);
      if (eta <= eta_c)
        exp_filter(mode, mode) = 1;
      else
        exp_filter(mode, mode) = exp(-run_input.expf_fac * pow((eta - eta_c) / (1. - eta_c), run_input.expf_order));
        mode++;
      }
    }
  }

  exp_filter = mult_arrays(exp_filter, inv_vandermonde);
  exp_filter = mult_arrays(vandermonde, exp_filter);
}

//detect shock use persson's method
void eles_tets::shock_det_persson(void)
{
  //calculate number of order-1 element
  int n_mode_under = order * (order + 1) * (order + 2) / 6;

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
  for (int i = 0; i < n_eles; i++)
  {
#if defined _ACCELERATE_BLAS || defined _MKL_BLAS || defined _STANDARD_BLAS
    sensor(i) = cblas_dasum(n_upts_per_ele - n_mode_under, temp_modal.get_ptr_cpu(n_mode_under, i), 1);
#else
    sensor(i) = accumulate(temp_modal.get_ptr_cpu(n_mode_under, i), temp_modal.get_ptr_cpu(0, i + 1), 0.);
#endif
  }
  
  //step 4. use Parseval's theorem to calculate (u,u) and calculate ((u-u_n,u-u_n)/(u,u))
  for (int i = 0; i < n_eles; i++)
  {
#if defined _ACCELERATE_BLAS || defined _MKL_BLAS || defined _STANDARD_BLAS
    sensor(i) /= cblas_dasum(n_upts_per_ele, temp_modal.get_ptr_cpu(0, i), 1);
#else
    sensor(i) /= accumulate(temp_modal.get_ptr_cpu(0, i), temp_modal.get_ptr_cpu(0, i + 1), 0.);
#endif
  }
}

// initialize the vandermonde matrix
void eles_tets::set_vandermonde_restart()
{
  vandermonde_rest.setup(n_upts_per_ele_rest,n_upts_per_ele_rest);

  // create the vandermonde matrix
  for (int i=0;i<n_upts_per_ele_rest;i++)
    for (int j=0;j<n_upts_per_ele_rest;j++)
      vandermonde_rest(i,j) = eval_dubiner_basis_3d(loc_upts_rest(0,i),loc_upts_rest(1,i),loc_upts_rest(2,i),j,order_rest);

  // Store its inverse
  inv_vandermonde_rest = inv_array(vandermonde_rest);
}

int eles_tets::read_restart_info_ascii(ifstream& restart_file)
{
  string str;
  // Move to triangle element
  while(1) {
      getline(restart_file,str);
      if (str=="TETS") break;
      if (restart_file.eof()) return 0;
    }

  getline(restart_file,str);
  restart_file >> order_rest;
  getline(restart_file,str);
  getline(restart_file,str);
  restart_file >> n_upts_per_ele_rest;
  getline(restart_file,str);
  getline(restart_file,str);

  loc_upts_rest.setup(n_dims,n_upts_per_ele_rest);

  for (int i=0;i<n_upts_per_ele_rest;i++) {
      for (int j=0;j<n_dims;j++) {
          restart_file >> loc_upts_rest(j,i);
        }
    }

  set_vandermonde_restart();
  set_opp_r();

  return 1;
}

#ifdef _HDF5
void eles_tets::read_restart_info_hdf5(hid_t &restart_file, int in_rest_order)
{
  hid_t dataset_id, plist_id;

  //open dataset
  dataset_id = H5Dopen2(restart_file, "TETS", H5P_DEFAULT);
  if (dataset_id < 0)
    FatalError("Cannot find tets property");

  plist_id = H5Pcreate(H5P_DATASET_XFER);
#ifdef _MPI
  //set collective read
  H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);
#endif

  if (n_eles)
  {
    order_rest = in_rest_order;
    n_upts_per_ele_rest = (order_rest + 3) * (order_rest + 2) * (order_rest + 1) / 6;
    loc_upts_rest.setup(n_dims, n_upts_per_ele_rest);
    //read data
    H5Dread(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, plist_id, loc_upts_rest.get_ptr_cpu());

    set_vandermonde_restart();
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
void eles_tets::write_restart_info_ascii(ofstream& restart_file)
{
  restart_file << "TETS" << endl;

  restart_file << "Order" << endl;
  restart_file << order << endl;

  restart_file << "Number of solution points per element" << endl;
  restart_file << n_upts_per_ele << endl;

  restart_file << "Location of solution points in tetrahedral elements" << endl;
  for (int i=0;i<n_upts_per_ele;i++) {
      for (int j=0;j<n_dims;j++) {
          restart_file << loc_upts(j,i) << " ";
        }
      restart_file << endl;
    }

}
#endif

#ifdef _HDF5
void eles_tets::write_restart_info_hdf5(hid_t &restart_file)
{
  hid_t dataset_id,plist_id,dataspace_id;
  hsize_t dim = 3 * (run_input.order + 3) * (run_input.order + 2) * (run_input.order + 1) / 6;

  //create TETS dataset
  dataspace_id=H5Screate_simple(1,&dim,NULL);
  dataset_id=H5Dcreate2(restart_file,"TETS",H5T_NATIVE_DOUBLE,dataspace_id,H5P_DEFAULT,H5P_DEFAULT,H5P_DEFAULT);
  plist_id = H5Pcreate(H5P_DATASET_XFER);
#ifdef _MPI
  //set collective read
  H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);
#endif
  //write loc_upts
  if(n_eles)
  {
    H5Dwrite(dataset_id,H5T_NATIVE_DOUBLE,H5S_ALL,H5S_ALL,plist_id,loc_upts.get_ptr_cpu());
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

void eles_tets::set_over_int(void)
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
  for (int i = 0; i < n_upts_per_ele; i++)
  {
    for (int j = 0; j < loc_over_int_cubpts.get_dim(1); j++)
    {
      loc(0) = loc_over_int_cubpts(0, j);
      loc(1) = loc_over_int_cubpts(1, j);
      loc(2) = loc_over_int_cubpts(2, j);
      temp_proj(i, j) = eval_dubiner_basis_3d(loc(0), loc(1), loc(2), i, order) * weight_over_int_cubpts(j);
    }
  }
  //multiply modal coefficient by vandermonde matrix to get over_int_filter
  over_int_filter = mult_arrays(vandermonde, temp_proj);
}

// evaluate nodal basis

double eles_tets::eval_nodal_basis(int in_index, hf_array<double> in_loc)
{
  hf_array<double> dubiner_basis_at_loc(n_upts_per_ele);
  double out_nodal_basis_at_loc;

  // First evaluate the normalized Dubiner basis at position in_loc
  for (int i=0;i<n_upts_per_ele;i++)
    dubiner_basis_at_loc(i) = eval_dubiner_basis_3d(in_loc(0),in_loc(1),in_loc(2),i,order);

  // From Hesthaven, equation 3.3, V^T * l = P, or l = (V^-1)^T P
  out_nodal_basis_at_loc = 0.;
  for (int i=0;i<n_upts_per_ele;i++)
    out_nodal_basis_at_loc += inv_vandermonde(i,in_index)*dubiner_basis_at_loc(i);

  return out_nodal_basis_at_loc;
}

// evaluate nodal basis

double eles_tets::eval_nodal_basis_restart(int in_index, hf_array<double> in_loc)
{
  hf_array<double> dubiner_basis_at_loc(n_upts_per_ele_rest);
  double out_nodal_basis_at_loc;

  // First evaluate the normalized Dubiner basis at position in_loc
  for (int i=0;i<n_upts_per_ele_rest;i++)
    dubiner_basis_at_loc(i) = eval_dubiner_basis_3d(in_loc(0),in_loc(1),in_loc(2),i,order_rest);

  // From Hesthaven, equation 3.3, V^T * l = P, or l = (V^-1)^T P
  out_nodal_basis_at_loc = 0.;
  for (int i=0;i<n_upts_per_ele_rest;i++)
    out_nodal_basis_at_loc += inv_vandermonde_rest(i,in_index)*dubiner_basis_at_loc(i);

  return out_nodal_basis_at_loc;
}

// evaluate derivative of nodal basis

double eles_tets::eval_d_nodal_basis(int in_index, int in_cpnt, hf_array<double> in_loc)
{
  hf_array<double> d_dubiner_basis_at_loc(n_upts_per_ele);
  double out_d_nodal_basis_at_loc;

  // First evaluate the derivative normalized Dubiner basis at position in_loc
  for (int i=0;i<n_upts_per_ele;i++)
    d_dubiner_basis_at_loc(i) = eval_grad_dubiner_basis_3d(in_loc(0),in_loc(1),in_loc(2),i,order,in_cpnt);

  // From Hesthaven, equation 3.3, V^T * l = P, or l = (V^-1)^T P
  out_d_nodal_basis_at_loc = 0.;
  for (int i=0;i<n_upts_per_ele;i++)
    out_d_nodal_basis_at_loc += inv_vandermonde(i,in_index)*d_dubiner_basis_at_loc(i);

  return out_d_nodal_basis_at_loc;
}

// evaluate nodal shape basis

double eles_tets::eval_nodal_s_basis(int in_index, hf_array<double> in_loc, int in_n_spts)
{
  double nodal_s_basis;

  if (in_n_spts==4) {
      if (in_index==0)
        nodal_s_basis = -0.5*(in_loc(0)+in_loc(1)+in_loc(2)+1.);
      else if (in_index==1)
        nodal_s_basis = 0.5*(in_loc(0)+1.);
      else if (in_index==2)
        nodal_s_basis = 0.5*(in_loc(1)+1.);
      else if (in_index==3)
        nodal_s_basis = 0.5*(in_loc(2)+1.);
    }
  else if (in_n_spts==10) {
      if (in_index==0)
        nodal_s_basis = (1./2.*(2.+in_loc(0)+in_loc(1)+in_loc(2)))*(in_loc(0)+1.+in_loc(1)+in_loc(2));
      else if (in_index==1)
        nodal_s_basis = (1./2.)*in_loc(0)*(in_loc(0)+1.);
      else if (in_index==2)
        nodal_s_basis = (1./2.)*in_loc(1)*(in_loc(1)+1.);
      else if (in_index==3)
        nodal_s_basis = (1./2.)*in_loc(2)*(in_loc(2)+1.);
      else if (in_index==4)
        nodal_s_basis = -(in_loc(0)+1.+in_loc(1)+in_loc(2))*(in_loc(0)+1.);
      else if (in_index==5)
        nodal_s_basis = -(in_loc(0)+1.+in_loc(1)+in_loc(2))*(in_loc(1)+1.);
      else if (in_index==6)
        nodal_s_basis = -(in_loc(0)+1.+in_loc(1)+in_loc(2))*(in_loc(2)+1.);
      else if (in_index==7)
        nodal_s_basis = (in_loc(0)+1.)*(in_loc(1)+1.);
      else if (in_index==8)
        nodal_s_basis = (in_loc(1)+1.)*(in_loc(2)+1.);
      else if (in_index==9)
        nodal_s_basis = (in_loc(2)+1.)*(in_loc(0)+1.);
    }
  else
    {
      cout << "Shape order not implemented yet, exiting" << endl;
      cout << "n_spt = " << in_n_spts << endl;
      exit(1);
    }
  return nodal_s_basis;
}

// evaluate derivative of nodal shape basis

void eles_tets::eval_d_nodal_s_basis(hf_array<double> &d_nodal_s_basis, hf_array<double> in_loc, int in_n_spts)
{

  if (in_n_spts==4) {
      d_nodal_s_basis(0,0) = -0.5;
      d_nodal_s_basis(1,0) = 0.5;
      d_nodal_s_basis(2,0) = 0.;
      d_nodal_s_basis(3,0) = 0.;

      d_nodal_s_basis(0,1) = -0.5;
      d_nodal_s_basis(1,1) = 0.;
      d_nodal_s_basis(2,1) = 0.5;
      d_nodal_s_basis(3,1) = 0.;

      d_nodal_s_basis(0,2) = -0.5;
      d_nodal_s_basis(1,2) = 0.;
      d_nodal_s_basis(2,2) = 0.;
      d_nodal_s_basis(3,2) = 0.5;
    }
  else if (in_n_spts==10) {

      d_nodal_s_basis(0,0) = 1.5+in_loc(0)+in_loc(1)+in_loc(2);
      d_nodal_s_basis(1,0) = in_loc(0)+0.5;
      d_nodal_s_basis(2,0) = 0.;
      d_nodal_s_basis(3,0) = 0.;
      d_nodal_s_basis(4,0) = -2.*in_loc(0)-2.-in_loc(1)-in_loc(2);
      d_nodal_s_basis(5,0) = -in_loc(1)-1.;
      d_nodal_s_basis(6,0) = -in_loc(2)-1.;
      d_nodal_s_basis(7,0) = in_loc(1)+1.;
      d_nodal_s_basis(8,0) = 0.;
      d_nodal_s_basis(9,0) = in_loc(2)+1.;

      d_nodal_s_basis(0,1) = 1.5+in_loc(0)+in_loc(1)+in_loc(2);
      d_nodal_s_basis(1,1) = 0.;
      d_nodal_s_basis(2,1) = in_loc(1)+0.5;
      d_nodal_s_basis(3,1) = 0.;
      d_nodal_s_basis(4,1) = -in_loc(0)-1.;
      d_nodal_s_basis(5,1) = -2.*in_loc(1)-2.-in_loc(0)-in_loc(2);
      d_nodal_s_basis(6,1) = -in_loc(2)-1.;
      d_nodal_s_basis(7,1) = in_loc(0)+1.;
      d_nodal_s_basis(8,1) = in_loc(2)+1.;
      d_nodal_s_basis(9,1) = 0.;

      d_nodal_s_basis(0,2)= 1.5+in_loc(0)+in_loc(1)+in_loc(2);
      d_nodal_s_basis(1,2)= 0.;
      d_nodal_s_basis(2,2)= 0.;
      d_nodal_s_basis(3,2)= in_loc(2)+0.5;
      d_nodal_s_basis(4,2)= -in_loc(0)-1.;
      d_nodal_s_basis(5,2)= -in_loc(1)-1.;
      d_nodal_s_basis(6,2)= -2.*in_loc(2)-2.-in_loc(0)-in_loc(1);
      d_nodal_s_basis(7,2)= 0.;
      d_nodal_s_basis(8,2)= in_loc(1)+1.;
      d_nodal_s_basis(9,2)= in_loc(0)+1.;
    }
  else
    {
      cout << "Shape order not implemented yet, exiting" << endl;
      cout << "n_spt = " << in_n_spts << endl;
      exit(1);
    }

}

void eles_tets::fill_opp_3(hf_array<double>& opp_3)
{

  hf_array <double> Filt(n_upts_per_ele,n_upts_per_ele);
  hf_array <double> opp_3_dg(n_upts_per_ele, n_fpts_per_ele);
  hf_array <double> m_temp(n_upts_per_ele, n_fpts_per_ele);

  compute_filt_matrix_tet(Filt,run_input.vcjh_scheme_tet, run_input.c_tet);

  get_opp_3_dg_tet(opp_3_dg);

  //cout << "opp_3_dg" << endl;
  //opp_3_dg.print();
  //cout << endl;

  m_temp = mult_arrays(Filt,opp_3_dg);

  //cout << "opp_3_vcjh" << endl;
  //m_temp.print();
  //cout << endl;
  opp_3 = hf_array<double>(m_temp);
}


void eles_tets::get_opp_3_dg_tet(hf_array<double>& opp_3_dg)
{
  int i,j,k;
  hf_array<double> loc(n_dims);

  for(i=0;i<n_fpts_per_ele;i++)
    {
      for(j=0;j<n_upts_per_ele;j++)
        {
          for(k=0;k<n_dims;k++)
            {
              loc(k)=loc_upts(k,j);
            }

          opp_3_dg(j,i)=eval_div_dg_tet(i,loc);
        }
    }
}


// evaluate divergence of dg basis

double eles_tets::eval_div_dg_tet(int in_index, hf_array<double>& loc)
{
  int face, face_fpt;
  double r,s,t;
  double r_face,s_face,face_jac;
  double integral, edge_length, gdotn_at_cubpt;
  double div_vcjh_basis;

  hf_array<double> mtemp_0(n_fpts_per_inter(0),n_fpts_per_inter(0));
  hf_array<double> gdotn(n_fpts_per_inter(0),1);
  hf_array<double> coeff_gdotn(n_fpts_per_inter(0),1);
  hf_array<double> coeff_divg(n_upts_per_ele,1);

  face = in_index/n_fpts_per_inter(0);
  face_fpt = in_index-(n_fpts_per_inter(0)*face);

  // Compute the coefficients of vjch basis in Dubiner basis
  // i.e. sigma_i = integral of (h cdot n)*L_i over the edge


  // 1. Construct a polynomial for g*n over the edge
  //    Store the coefficient of g*n in terms of jacobi basis in coeff_gdotn
  for (int i=0;i<n_fpts_per_inter(0);i++) {

      if (i==face_fpt)
        gdotn(i,0) = 1.;
      else
        gdotn(i,0) = 0.;

      r = tloc_fpts(0,face*n_fpts_per_inter(0)+i);
      s = tloc_fpts(1,face*n_fpts_per_inter(0)+i);
      t = tloc_fpts(2,face*n_fpts_per_inter(0)+i);

      if (face==0) {
          r_face = r;
          s_face = t;
        }
      else if (face==1) {
          r_face = t;
          s_face = s;
        }
      else if (face==2) {
          r_face = r;
          s_face = t;
        }
      else if (face==3) {
          r_face = s;
          s_face = r;
        }

      for (int j=0;j<n_fpts_per_inter(0);j++)
        mtemp_0(i,j) = eval_dubiner_basis_2d(r_face,s_face,j,order);
    }

  mtemp_0 = inv_array(mtemp_0);
  coeff_gdotn = mult_arrays(mtemp_0,gdotn);

  if (std::isnan(coeff_gdotn(0,0)))
    exit(1);

  // 2. Perform the edge integrals to obtain coefficients sigma_i
  for (int i=0;i<n_upts_per_ele;i++)
    {
      cubature_tri cub2d(0,7); //TODO: Check if strong enough
      integral = 0.;

      for (int j=0;j<cub2d.get_n_pts();j++)
        {
          r_face = cub2d.get_r(j);
          s_face = cub2d.get_s(j);

          // Get the position along the edge
          if (face==0) {
              face_jac = sqrt(3.);
              r = r_face;
              t = s_face;
              s = -1. -t -r;
            }
          else if (face==1) {
              face_jac = 1.;
              r = -1.0;
              s = s_face;
              t = r_face;
            }
          else if (face==2) {
              face_jac = 1.;
              r = r_face;
              s = -1.0;
              t = s_face;
            }
          else if (face==3) {
              face_jac = 1.;
              r = s_face;
              s = r_face;
              t = -1.0;
            }

          gdotn_at_cubpt = 0.;
          for (int k=0;k<n_fpts_per_inter(0);k++)
            gdotn_at_cubpt += coeff_gdotn(k,0)*eval_dubiner_basis_2d(r_face,s_face,k,order);

          integral += cub2d.get_weight(j)*eval_dubiner_basis_3d(r,s,t,i,order)*gdotn_at_cubpt;
        }
      coeff_divg(i,0) = integral*face_jac;
    }

  div_vcjh_basis = 0.;
  for (int i=0;i<n_upts_per_ele;i++)
    div_vcjh_basis += coeff_divg(i,0)*eval_dubiner_basis_3d(loc(0),loc(1),loc(2),i,order);

  return div_vcjh_basis;

}


void eles_tets::compute_filt_matrix_tet(hf_array<double>& Filt, int vcjh_scheme_tet, double c_tet)
{

  // -----------------
  // VCJH Filter
  // -----------------
  int Ncoeff, indx;
  double ap;
  double c_plus;
  double c_plus_1d, c_sd_1d, c_hu_1d;

  Ncoeff = (order+1)*(order+2)/2;

  hf_array <double> c_coeff(Ncoeff);
  hf_array <double> mtemp_0, mtemp_1;
  hf_array <double> K(n_upts_per_ele,n_upts_per_ele);
  hf_array <double> Identity(n_upts_per_ele,n_upts_per_ele);
  hf_array <double> Filt_dubiner(n_upts_per_ele,n_upts_per_ele);
  hf_array <double> Dr(n_upts_per_ele,n_upts_per_ele);
  hf_array <double> Ds(n_upts_per_ele,n_upts_per_ele);
  hf_array <double> Dt(n_upts_per_ele,n_upts_per_ele);
  hf_array <double> tempr(n_upts_per_ele,n_upts_per_ele);
  hf_array <double> temps(n_upts_per_ele,n_upts_per_ele);
  hf_array <double> tempt(n_upts_per_ele,n_upts_per_ele);
  hf_array <double> D_high_order_trans(n_upts_per_ele,n_upts_per_ele);
  hf_array <double> vandermonde_trans(n_upts_per_ele,n_upts_per_ele);

  hf_array<hf_array <double> > D_high_order;
  hf_array<hf_array <double> > D_T_D;

  // 1D prep
  ap = 1./pow(2.0,order)*factorial(2*order)/ (factorial(order)*factorial(order));

  c_sd_1d = (2*order)/((2*order+1)*(order+1)*(factorial(order)*ap)*(factorial(order)*ap));
  c_hu_1d = (2*(order+1))/((2*order+1)*order*(factorial(order)*ap)*(factorial(order)*ap));

  if(vcjh_scheme_tet>1)
    {
      //1D c+
      if (order==2)
        c_plus_1d = 0.206;
      else if (order==3)
        c_plus_1d = 3.80e-3;
      else if (order==4)
        c_plus_1d = 4.67e-5;
      else if (order==5)
        c_plus_1d = 4.28e-7;
      else
        FatalError("C_plus scheme not implemented for this order");

      //3D c+
      if (order==2)
        c_plus = 3.07e-2;
      else if (order==3)
        c_plus = 5.44e-4;
      else if (order==4)
        c_plus = 9.92e-6;
      else if (order==5)
        c_plus = 1.10e-7;
      else
        FatalError("C_plus scheme not implemented for this order");
    }


  if (vcjh_scheme_tet==0)
    {
      //c_tet set by user
    }
  else if (vcjh_scheme_tet==1) // DG
    {
      c_tet = 0.;
    }
  else if (vcjh_scheme_tet==2) // SD-like
    {
      c_tet = (c_sd_1d/c_plus_1d)*c_plus;
    }
  else if (vcjh_scheme_tet==3) // HU-like
    {
      c_tet = (c_hu_1d/c_plus_1d)*c_plus;
    }
  else if (vcjh_scheme_tet==4) // Cplus scheme
    {
      c_tet = c_plus;
    }
  else
    FatalError("VCJH tetrahedral scheme not recognized");

//  cout << "c_tet " << c_tet << endl;

  run_input.c_tet = c_tet;

  // Evaluate the derivative normalized of Dubiner basis at position in_loc
  for (int i=0;i<n_upts_per_ele;i++) {
      for (int j=0;j<n_upts_per_ele;j++) {
          tempr(i,j) = eval_grad_dubiner_basis_3d(loc_upts(0,i),loc_upts(1,i),loc_upts(2,i),j,order,0);
          temps(i,j) = eval_grad_dubiner_basis_3d(loc_upts(0,i),loc_upts(1,i),loc_upts(2,i),j,order,1);
          tempt(i,j) = eval_grad_dubiner_basis_3d(loc_upts(0,i),loc_upts(1,i),loc_upts(2,i),j,order,2);
        }
    }

  //Convert to nodal derivatives
  Dr = mult_arrays(tempr,inv_vandermonde);
  Ds = mult_arrays(temps,inv_vandermonde);
  Dt = mult_arrays(tempt,inv_vandermonde);

  //cout << "Dr nodal" << endl;
  //Dr.print();
  //cout << endl;
  //cout << "Dr dubiner" << endl;
  //(Dr*vandermonde).print();
  //cout << endl;

  //cout << "Ds nodal" << endl;
  //Ds.print();
  //cout << endl;
  //cout << "Ds dubiner" << endl;
  //(Ds*vandermonde).print();
  //cout << endl;

  //cout << "Dt nodal" << endl;
  //Dt.print();
  //cout << endl;
  //cout << "Dt dubiner" << endl;
  //(Dt*vandermonde).print();
  //cout << endl;

  //Create identity matrix
  Identity.initialize_to_zero();

  for (int i=0;i<n_upts_per_ele;i++)
    Identity(i,i) = 1.;

  // Set hf_array with trinomial coefficients multiplied by value of c
  indx = 0;
  for(int v=1; v<=(order+1); v++) {
      for(int w=1; w<=v; w++) {
          c_coeff(indx) = (1./n_upts_per_ele)*(factorial(order)/( factorial(v-1)*factorial(order-(v-1)) ))*(factorial(v-1)/(factorial(w-1)*factorial((v-1)-(w-1))));
          //cout << "v=" << v << " w=" << w << " indx=" << indx << " coeff= " << c_coeff(indx) << endl;
          indx++;
        }
    }

  // Initialize K to zero
  K.initialize_to_zero();

  // Compute D_transpose*D
  D_high_order.setup(Ncoeff);
  D_T_D.setup(Ncoeff);

  indx = 0;
  for(int v=1; v<=(order+1); v++)
    {
      for(int w=1; w<=v; w++)
        {
          D_high_order(indx) = hf_array<double>(Identity);

          for (int i=1; i<=(order-v+1); i++)
            D_high_order(indx) = mult_arrays(D_high_order(indx),Dr);
          for (int i=1; i<=(v-w); i++)
            D_high_order(indx) = mult_arrays(D_high_order(indx),Ds);
          for (int i=1; i<=(w-1); i++)
            D_high_order(indx) = mult_arrays(D_high_order(indx),Dt);

          D_high_order_trans = transpose_array(D_high_order(indx));
          D_T_D(indx) = mult_arrays(D_high_order_trans,D_high_order(indx));

          //cout << "indx=" << indx << endl;
          //(D_high_order(indx)*vandermonde).print();

          //cout << endl;
          //mtemp_2 = vandermonde.get_trans()*D_T_D(indx)*vandermonde;
          //mtemp_2.print();
          //cout << endl;

          // Scale by c_coeff
          for (int i=0;i<n_upts_per_ele;i++) {
              for (int j=0;j<n_upts_per_ele;j++) {
                  D_T_D(indx)(i,j) = c_tet*c_coeff(indx)*D_T_D(indx)(i,j);
                  K(i,j) += D_T_D(indx)(i,j); //without jacobian scaling
                }
            }
          indx++;
        }
    }

  //mass matrix
  vandermonde_trans = transpose_array(vandermonde);
  mtemp_0 = mult_arrays(vandermonde,vandermonde_trans);

  //filter
  mtemp_1 = hf_array<double>(mtemp_0);
  mtemp_1 = mult_arrays(mtemp_1,K);

  for (int i=0;i<n_upts_per_ele;i++)
    for (int j=0;j<n_upts_per_ele;j++)
      mtemp_1(i,j) += Identity(i,j);

  Filt = inv_array(mtemp_1);
  Filt_dubiner = mult_arrays(inv_vandermonde,Filt);
  Filt_dubiner = mult_arrays(Filt_dubiner,vandermonde);

  //cout << "Filt" << endl;
  //Filt.print();
  //cout << endl;

  //cout << "Filt_dubiner" << endl;
  //Filt_dubiner.print();


  /*
  // ------------------------
  // Diagonal filter
  // ------------------------
  matrix Filt_dubiner(n_upts_per_ele,n_upts_per_ele);
  int n_upts_lower = (order+1)*order/2;

  double frac;

  if (vcjh_scheme_tri==0)
  {
    double c_1d = c_tet*2*order;
    double cp = 1./pow(2.0,order)*factorial(2*order)/ (factorial(order)*factorial(order));
    double kappa = (2*order+1)/2*(factorial(order)*cp)*(factorial(order)*cp);
    frac = 1./ (1+c_1d*kappa);
  }
  else if (vcjh_scheme_tri==1) // DG
  {
    frac = 1.0;
  }
  else if (vcjh_scheme_tri==2) // SD-like
  {
    frac = (order+1.)/(2.*order+1);
  }
  else if (vcjh_scheme_tri==3) // HU-like
  {
    frac = (order)/(2.*order+1);
  }
  else if (vcjh_scheme_tri==4) // Cplus scheme
  {
    if (order==2)
      c_tet = 4.3e-2;
    else if (order==3)
      c_tet = 6.4e-4;
    else if (order==4)
      c_tet = 5.3e-6;
    else
      FatalError("C_plus scheme not implemented for this order");

    double c_1d = c_tet*2*order;
    double cp = 1./pow(2.0,order)*factorial(2*order)/ (factorial(order)*factorial(order));
    double kappa = (2*order+1)/2*(factorial(order)*cp)*(factorial(order)*cp);
    frac = 1./ (1+c_1d*kappa);
  }
  else
    FatalError("VCJH triangular scheme not recognized");

  cout << "Filtering fraction=" << frac << endl;

  for (int j=0;j<n_upts_per_ele;j++) {
    for (int k=0;k<n_upts_per_ele;k++) {
      if (j==k) {
        if (j < n_upts_lower)
          Filt_dubiner(j,k) = 1.;
        else
          Filt_dubiner(j,k) = frac;
      }
      else {
        Filt_dubiner(j,k) = 0.;
      }
    }
  }

  Filt = vandermonde_tri*Filt_dubiner*inv_vandermonde_tri;

  //cout << "Filt_dubiner_diag" << endl;
  //Filt_dubiner.print();

  cout << "Filt_diag" << endl;
  Filt.print();
  */

}

/*! Calculate element volume */
double eles_tets::calc_ele_vol(double& detjac)
{
  double vol;
  // Element volume = |Jacobian|*1/6*width*height*depth of reference element
  vol = detjac*8./6.;
  return vol;
}

/*! Calculate element reference length for timestep calculation */
double eles_tets::calc_h_ref_specific(int in_ele)
{
    double vol,s_a,s_b,s_c,s_d;
    hf_array<double> a,b,c,d,e;
    a.setup(n_dims);
    b.setup(n_dims);
    c.setup(n_dims);
    d.setup(n_dims);
    e.setup(n_dims);
    double out_h_ref;
    s_a=0.;
    s_b=0.;
    s_c=0.;
    s_d=0.;

    //compute edge vector
    for(int i=0; i<n_dims; i++)
    {
        a(i)=shape(i,1,in_ele)-shape(i,0,in_ele);//0-1
        b(i)=shape(i,2,in_ele)-shape(i,0,in_ele);//0-2
        c(i)=shape(i,3,in_ele)-shape(i,0,in_ele);//0-3
        d(i)=shape(i,2,in_ele)-shape(i,1,in_ele);//1-2
        e(i)=shape(i,3,in_ele)-shape(i,1,in_ele);//1-3
    }
    //calculate volume
    vol=1./6.*trip_prod(a,b,c);

    //calculate area of each face;
    s_a=0.5*sqrt(pow(a(1)*b(2)-a(2)*b(1),2)+pow(a(0)*b(2)-a(2)*b(0),2)+pow(a(0)*b(1)-a(1)*b(0),2));
    s_b=0.5*sqrt(pow(a(1)*c(2)-a(2)*c(1),2)+pow(a(0)*c(2)-a(2)*c(0),2)+pow(a(0)*c(1)-a(1)*c(0),2));
    s_c=0.5*sqrt(pow(b(1)*c(2)-b(2)*c(1),2)+pow(b(0)*c(2)-b(2)*c(0),2)+pow(b(0)*c(1)-b(1)*c(0),2));
    s_d=0.5*sqrt(pow(d(1)*e(2)-d(2)*e(1),2)+pow(d(0)*e(2)-d(2)*e(0),2)+pow(d(0)*e(1)-d(1)*e(0),2));

    //insphere diameter
    out_h_ref=6.*vol/(s_a+s_b+s_c+s_d);
    return out_h_ref;
}

int eles_tets::calc_p2c(hf_array<double>& in_pos)
{
    hf_array<double> plane_coeff;
    hf_array<double> pos_centroid;
    hf_array<int> vertex_index_loc(3);
    hf_array<double> pos_plane_pts(n_dims,3);

    for (int i=0; i<n_eles; i++)//for each element
    {
        int alpha=1;//indicator

        //calculate centroid
        hf_array<double> temp_pos_s_pts(n_dims,n_spts_per_ele(i));
        for (int j=0; j<n_spts_per_ele(i); j++)
            for (int k=0; k<n_dims; k++)
                temp_pos_s_pts(k,j)=shape(k,j,i);
        pos_centroid=calc_centroid(temp_pos_s_pts);

            int num_f_per_c = 4;

            for(int j=0; j<num_f_per_c; j++)//for each face
            {
                //store local vertex index
                if(j==0)
                {
                    vertex_index_loc(0) = 1;
                    vertex_index_loc(1) = 2;
                    vertex_index_loc(2) = 3;
                }
                else if(j==1)
                {
                    vertex_index_loc(0) = 0;
                    vertex_index_loc(1) = 3;
                    vertex_index_loc(2) = 2;

                }
                else if(j==2)
                {
                    vertex_index_loc(0) = 0;
                    vertex_index_loc(1) = 1;
                    vertex_index_loc(2) = 3;
                }
                else if(j==3)
                {
                    vertex_index_loc(0) = 0;
                    vertex_index_loc(1) = 2;
                    vertex_index_loc(2) = 1;
                }

                //store position of vertex on each face
                for (int k=0; k<3; k++) //number of points needed to define a plane
                    for (int l=0; l<n_dims; l++) //dims
                        pos_plane_pts(l,k)=shape(l,vertex_index_loc(k),i);

                //calculate plane coeff
                plane_coeff=calc_plane(pos_plane_pts);

                alpha=alpha*((plane_coeff(0)*in_pos(0)+plane_coeff(1)*in_pos(1)+plane_coeff(2)*in_pos(2)+plane_coeff(3))*
                        (plane_coeff(0)*pos_centroid(0)+plane_coeff(1)*pos_centroid(1)+plane_coeff(2)*pos_centroid(2)+plane_coeff(3))>=0);
                if (alpha==0)
                    break;
            }

        if (alpha>0)
            return i;
    }

    return -1;
}

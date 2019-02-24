/*!
 * \file eles_pris.cpp
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
#include "../include/eles_pris.h"
#include "../include/cubature_1d.h"
#include "../include/cubature_tri.h"
#include "../include/cubature_quad.h"
#include "../include/cubature_pris.h"

using namespace std;

// #### constructors ####

// default constructor

eles_pris::eles_pris()
{
}

// #### methods ####

void eles_pris::setup_ele_type_specific()
{

#ifndef _MPI
  cout << "Initializing pris" << endl;
#endif

  ele_type=3;
  n_dims=3;

  if (run_input.equation==0)
    n_fields=5;
  else if (run_input.equation==1)
    n_fields=1;
  else
    FatalError("Equation not supported");

  if (run_input.RANS==1)
    n_fields++;

  n_inters_per_ele=5;
  length.setup(5);
  n_upts_per_ele=(order+2)*(order+1)*(order+1)/2;
  upts_type_pri_tri = run_input.upts_type_pri_tri;
  upts_type_pri_1d = run_input.upts_type_pri_1d;
  set_loc_upts();
  set_vandermonde_tri();
  set_vandermonde3D();

  //set shock capturing arrays
  if(run_input.shock_cap)
  {
    if (run_input.shock_det == 0)//persson
      calc_norm_basis();
      else
      FatalError("Concentration method not implmented.");
    if (run_input.shock_cap == 1)//exp filter
      set_exp_filter();
  }

  set_inters_cubpts();
  set_volume_cubpts();
  set_opp_volume_cubpts();
  set_vandermonde_vol_cub();

  n_ppts_per_ele=(p_res+1)*(p_res)*(p_res)/2;
  n_peles_per_ele=( (p_res-1)*(p_res-1)*(p_res-1) );
  n_verts_per_ele = 6;

  set_loc_ppts();
  set_opp_p();

  //de-aliasing by over-integration
  if (run_input.over_int)
    set_over_int_filter();

  n_fpts_per_inter.setup(5);

  n_fpts_per_inter(0)=(order+2)*(order+1)/2;
  n_fpts_per_inter(1)=(order+2)*(order+1)/2;
  n_fpts_per_inter(2)=(order+1)*(order+1);
  n_fpts_per_inter(3)=(order+1)*(order+1);
  n_fpts_per_inter(4)=(order+1)*(order+1);

  n_fpts_per_ele=3*(order+1)*(order+1)+(order+2)*(order+1);

  // Check consistency between tet-pri interface
  if (upts_type_pri_tri != run_input.fpts_type_tet)
    FatalError("upts_type_pri_tri != fpts_type_tet");

  // Check consistency between hex-pri interface
  if (upts_type_pri_1d != run_input.upts_type_hexa)
    FatalError("upts_type_pri_1d != upts_type_hexa");

  set_tloc_fpts();

  set_tnorm_fpts();

  set_opp_0(run_input.sparse_pri);
  set_opp_1(run_input.sparse_pri);
  set_opp_2(run_input.sparse_pri);
  set_opp_3(run_input.sparse_pri);

  if(viscous)
    {
      // Compute hex filter matrix
      //if(LES_filter) compute_filter_upts();

      set_opp_4(run_input.sparse_pri);
      set_opp_5(run_input.sparse_pri);
      set_opp_6(run_input.sparse_pri);

      temp_grad_u.setup(n_fields,n_dims);
    }

  temp_u.setup(n_fields);
  temp_f.setup(n_fields,n_dims);
}

// set shape

/*
void eles_pris::set_shape(int in_s_order)
{
    // fill in
}
*/

void eles_pris::set_connectivity_plot()
{
  int vertex_0, vertex_1, vertex_2, vertex_3, vertex_4, vertex_5;
  int count = 0;
  int temp = (p_res) * (p_res + 1) / 2; //number of pts each 1d layer

  for (int l = 0; l < p_res - 1; ++l)
  { // level 1d
    for (int j = 0; j < p_res - 1; ++j)
    { // level_tri
      for (int k = 0; k < p_res - j - 1; ++k)
      { // starting pt

        vertex_0 = k + (j * (p_res + 1)) - ((j * (j + 1)) / 2) + l * temp;
        vertex_1 = vertex_0 + 1;
        vertex_2 = k + ((j + 1) * (p_res + 1)) - (((j + 1) * (j + 2)) / 2) + l * temp;

        vertex_3 = vertex_0 + temp;
        vertex_4 = vertex_1 + temp;
        vertex_5 = vertex_2 + temp;

        connectivity_plot(0, count) = vertex_0;
        connectivity_plot(1, count) = vertex_1;
        connectivity_plot(2, count) = vertex_2;
        connectivity_plot(3, count) = vertex_3;
        connectivity_plot(4, count) = vertex_4;
        connectivity_plot(5, count) = vertex_5;
        count++;
      }
    }
  }

  for (int l = 0; l < p_res - 1; ++l)
  {
    for (int j = 0; j < p_res - 2; ++j)
    {
      for (int k = 0; k < p_res - j - 2; ++k)
      {
        vertex_0 = k + 1 + (j * (p_res)) - ((j * (j - 1)) / 2) + l * temp;
        vertex_1 = vertex_0 + p_res - j;
        vertex_2 = vertex_1 - 1;

        vertex_3 = vertex_0 + temp;
        vertex_4 = vertex_1 + temp;
        vertex_5 = vertex_2 + temp;

        connectivity_plot(0, count) = vertex_0;
        connectivity_plot(1, count) = vertex_1;
        connectivity_plot(2, count) = vertex_2;
        connectivity_plot(3, count) = vertex_3;
        connectivity_plot(4, count) = vertex_4;
        connectivity_plot(5, count) = vertex_5;
        count++;
      }
    }
  }
}




// set location of solution points in standard element

void eles_pris::set_loc_upts(void)
{

  loc_upts.setup(n_dims,n_upts_per_ele);

  n_upts_tri = (order+1)*(order+2)/2;
  n_upts_1d = order+1;

  loc_upts_pri_1d.setup(n_upts_1d);
  loc_upts_pri_tri.setup(2,n_upts_tri);

  cubature_1d cub_1d(upts_type_pri_1d,order);
  cubature_tri cub_tri(upts_type_pri_tri,order);
  
  for (int i = 0; i < n_upts_1d;i++)
    loc_upts_pri_1d(i) = cub_1d.get_r(i);

  for (int i=0;i<n_upts_tri;i++) {
    loc_upts_pri_tri(0,i) = cub_tri.get_r(i);
    loc_upts_pri_tri(1,i) = cub_tri.get_s(i);
  } 

  // Now set loc_upts
  for (int i=0;i<n_upts_1d;i++) {
      for (int j=0;j<n_upts_tri;j++) {
          loc_upts(0,n_upts_tri*i+j) = loc_upts_pri_tri(0,j);
          loc_upts(1,n_upts_tri*i+j) = loc_upts_pri_tri(1,j);
          loc_upts(2,n_upts_tri*i+j) = loc_upts_pri_1d(i);
        }
    }

}

// set location of flux points in standard element

void eles_pris::set_tloc_fpts(void)
{

  tloc_fpts.setup(n_dims,n_fpts_per_ele);

  hf_array<double> loc_tri_fpts( (order+1)*(order+2)/2,2);
  loc_1d_fpts.setup(order+1);

  cubature_1d cub_1d(upts_type_pri_1d,order);
  cubature_tri cub_tri(upts_type_pri_tri,order);

  for (int i = 0; i < n_fpts_per_inter(0); i++)
  {
    loc_tri_fpts(i, 0) = cub_tri.get_r(i);
    loc_tri_fpts(i, 1) = cub_tri.get_s(i);
  }
      for (int i = 0; i < order + 1; i++)
    loc_1d_fpts(i) = cub_1d.get_r(i);

  // Now need to map these points on faces of prisms
  // Inter 0
  for (int i=0;i<n_fpts_per_inter(0);i++)
    {
      tloc_fpts(0,i) = loc_tri_fpts(i,1);//note need to notice
      tloc_fpts(1,i) = loc_tri_fpts(i,0);
      tloc_fpts(2,i) = -1.;
    }

  // Inter 1
  for (int i=0;i<n_fpts_per_inter(1);i++)
    {
      tloc_fpts(0,n_fpts_per_inter(0)+i) = loc_tri_fpts(i,0);
      tloc_fpts(1,n_fpts_per_inter(0)+i) = loc_tri_fpts(i,1);
      tloc_fpts(2,n_fpts_per_inter(0)+i) = 1.;
    }

  // Inters 2,3,4
  int offset = n_fpts_per_inter(0)*2;
  for (int face=0;face<3;face++) {
      for (int i=0;i<order+1;i++) {
          for (int j=0;j<order+1;j++) {

              if (face==0) {
                  tloc_fpts(0,offset+face*(order+1)*(order+1)+i*(order+1)+j) = loc_1d_fpts(j);
                  tloc_fpts(1,offset+face*(order+1)*(order+1)+i*(order+1)+j) = -1;;
                }
              else if (face==1) {
                  tloc_fpts(0,offset+face*(order+1)*(order+1)+i*(order+1)+j) = loc_1d_fpts(order-j);//x from r to l
                  tloc_fpts(1,offset+face*(order+1)*(order+1)+i*(order+1)+j) = loc_1d_fpts(j);//y from bot to up
                }
              else if (face==2) {
                  tloc_fpts(0,offset+face*(order+1)*(order+1)+i*(order+1)+j) = -1.;
                  tloc_fpts(1,offset+face*(order+1)*(order+1)+i*(order+1)+j) = loc_1d_fpts(order-j);;
                }

              tloc_fpts(2,offset+face*(order+1)*(order+1)+i*(order+1)+j) = loc_1d_fpts(i);
            }
        }
    }

}


void eles_pris::set_inters_cubpts(void)
{

  n_cubpts_per_inter.setup(n_inters_per_ele);
  loc_inters_cubpts.setup(n_inters_per_ele);
  weight_inters_cubpts.setup(n_inters_per_ele);
  tnorm_inters_cubpts.setup(n_inters_per_ele);

  cubature_tri cub_tri(0,order);
  cubature_quad cub_quad(0,order);

  int n_cubpts_tri = cub_tri.get_n_pts();
  int n_cubpts_quad = cub_quad.get_n_pts();

  for (int i=0;i<n_inters_per_ele;i++)
    {
      if (i==0 || i==1) {
          n_cubpts_per_inter(i) = n_cubpts_tri;
        }
      else if (i==2 || i==3 || i==4) {
          n_cubpts_per_inter(i) = n_cubpts_quad;
        }

    }

  for (int i=0;i<n_inters_per_ele;i++) {

      loc_inters_cubpts(i).setup(n_dims,n_cubpts_per_inter(i));
      weight_inters_cubpts(i).setup(n_cubpts_per_inter(i));
      tnorm_inters_cubpts(i).setup(n_dims,n_cubpts_per_inter(i));

      for (int j=0;j<n_cubpts_per_inter(i);j++) {

          if (i==0) {
              loc_inters_cubpts(i)(0,j)=cub_tri.get_r(j);
              loc_inters_cubpts(i)(1,j)=cub_tri.get_s(j);
              loc_inters_cubpts(i)(2,j)=-1.;
            }
          else if (i==1) {
              loc_inters_cubpts(i)(0,j)=cub_tri.get_r(j);
              loc_inters_cubpts(i)(1,j)=cub_tri.get_s(j);
              loc_inters_cubpts(i)(2,j)=1.;
            }
          else if (i==2) {
              loc_inters_cubpts(i)(0,j)=cub_quad.get_r(j);
              loc_inters_cubpts(i)(1,j)=-1.;
              loc_inters_cubpts(i)(2,j)=cub_quad.get_s(j);
            }
          else if (i==3) {
              loc_inters_cubpts(i)(0,j)=cub_quad.get_r(j);
              loc_inters_cubpts(i)(1,j)=-cub_quad.get_r(j);
              loc_inters_cubpts(i)(2,j)=cub_quad.get_s(j);
            }
          else if (i==4) {
              loc_inters_cubpts(i)(0,j)=-1.;
              loc_inters_cubpts(i)(1,j)=cub_quad.get_r(j);
              loc_inters_cubpts(i)(2,j)=cub_quad.get_s(j);
            }

          if (i==0 || i==1)
            weight_inters_cubpts(i)(j) = cub_tri.get_weight(j);
          else if (i==2 || i==3 || i==4)
            weight_inters_cubpts(i)(j) = cub_quad.get_weight(j);

          if (i==0) {
              tnorm_inters_cubpts(i)(0,j)= 0.;
              tnorm_inters_cubpts(i)(1,j)= 0.;
              tnorm_inters_cubpts(i)(2,j)= -1.;
            }
          else if (i==1) {
              tnorm_inters_cubpts(i)(0,j)= 0.;
              tnorm_inters_cubpts(i)(1,j)= 0.;
              tnorm_inters_cubpts(i)(2,j)= 1.;
            }
          else if (i==2) {
              tnorm_inters_cubpts(i)(0,j)= 0.;
              tnorm_inters_cubpts(i)(1,j)= -1.;
              tnorm_inters_cubpts(i)(2,j)= 0.;
            }
          else if (i==3) {
              tnorm_inters_cubpts(i)(0,j)= 1./sqrt(2.);
              tnorm_inters_cubpts(i)(1,j)= 1./sqrt(2.);
              tnorm_inters_cubpts(i)(2,j)= 0.;
            }
          else if (i==4) {
              tnorm_inters_cubpts(i)(0,j)= -1.;
              tnorm_inters_cubpts(i)(1,j)= 0.;
              tnorm_inters_cubpts(i)(2,j)= 0.;
            }

        }
    }
  set_opp_inters_cubpts();

}

void eles_pris::set_volume_cubpts(void)
{
  cubature_pris cub_pri(0, 0, order);
  n_cubpts_per_ele = cub_pri.get_n_pts();
  loc_volume_cubpts.setup(n_dims, n_cubpts_per_ele);
  weight_volume_cubpts.setup(n_cubpts_per_ele);

  for (int i = 0; i < n_cubpts_per_ele; i++)
  {
    loc_volume_cubpts(0, i) = cub_pri.get_r(i);
    loc_volume_cubpts(1, i) = cub_pri.get_s(i);
    loc_volume_cubpts(2, i) = cub_pri.get_t(i);
    weight_volume_cubpts(i) = cub_pri.get_weight(i);
  }
}

// Compute the surface jacobian determinant on a face
double eles_pris::compute_inter_detjac_inters_cubpts(int in_inter,hf_array<double> d_pos)
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
  else if (in_inter==1) // u=s, v=s
    {
      xu = xr;
      yu = yr;
      zu = zr;

      xv = xs;
      yv = ys;
      zv = zs;
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
  else if (in_inter==3) //r=u,t=v,s=1-u
    {
      xu = xr-xs;
      yu = yr-ys;
      zu = zr-zs;

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


  temp0 = yu*zv-zu*yv;
  temp1 = zu*xv-xu*zv;
  temp2 = xu*yv-yu*xv;

  output = sqrt(temp0*temp0+temp1*temp1+temp2*temp2);

  return output;
}




// set location of plot points in standard element

void eles_pris::set_loc_ppts(void)
{
  int i,j,k,index;

  loc_ppts.setup(3,p_res*(p_res+1)/2*p_res);

  for(k=0;k<p_res;k++)//z index
    {
      for(j=0;j<p_res;j++)//y index
        {
          for(i=0;i<p_res-j;i++)//x index
            {
              index = (p_res*(p_res+1)/2)*k + (i+(j*(p_res+1))-((j*(j+1))/2));/*|2\    bottom to up
                                                                              //|0_1\*/
              loc_ppts(0,index)=-1.0+((2.0*i)/(1.0*(p_res-1)));
              loc_ppts(1,index)=-1.0+((2.0*j)/(1.0*(p_res-1)));
              loc_ppts(2,index)=-1.0+((2.0*k)/(1.0*(p_res-1)));
            }
        }
    }
}



// set location of shape points in standard element
/*
void eles_pris::set_loc_spts(void)
{
    // fill in
}
*/

// set transformed normal at flux points

void eles_pris::set_tnorm_fpts(void)
{

  tnorm_fpts.setup(n_dims,n_fpts_per_ele);

  int fpt = -1;
  for (int i=0;i<n_inters_per_ele;i++)
    {
      for (int j=0;j<n_fpts_per_inter(i);j++)
        {
          fpt++;
          if (i==0) {
              tnorm_fpts(0,fpt) = 0.;
              tnorm_fpts(1,fpt) = 0.;
              tnorm_fpts(2,fpt) = -1.;
            }
          else if (i==1) {
              tnorm_fpts(0,fpt) = 0.;
              tnorm_fpts(1,fpt) = 0.;
              tnorm_fpts(2,fpt) = 1.;
            }
          else if (i==2) {
              tnorm_fpts(0,fpt) = 0.;
              tnorm_fpts(1,fpt) = -1.;
              tnorm_fpts(2,fpt) = 0.;
            }
          else if (i==3) {
              tnorm_fpts(0,fpt) = 1./sqrt(2.);
              tnorm_fpts(1,fpt) = 1./sqrt(2.);
              tnorm_fpts(2,fpt) = 0.;
            }
          else if (i==4) {
              tnorm_fpts(0,fpt) = -1.;
              tnorm_fpts(1,fpt) = 0.;
              tnorm_fpts(2,fpt) = 0.;
            }
        }
    }
  //cout << "tnorm_fpts" << endl;
  //tnorm_fpts.print();
}

//#### helper methods ####

// initialize the vandermonde matrix
void eles_pris::set_vandermonde_tri()
{
  vandermonde_tri.setup(n_upts_tri,n_upts_tri);

  // create the vandermonde matrix
  for (int i=0;i<n_upts_tri;i++)
    for (int j=0;j<n_upts_tri;j++)
      vandermonde_tri(i,j) = eval_dubiner_basis_2d(loc_upts_pri_tri(0,i),loc_upts_pri_tri(1,i),j,order);

  // Store its inverse
  inv_vandermonde_tri = inv_array(vandermonde_tri);
}

void eles_pris::set_vandermonde3D(void)
{
  vandermonde.setup(n_upts_per_ele, n_upts_per_ele);
  hf_array<double> loc(n_dims);
  // create the vandermonde matrix
  for (int i = 0; i < n_upts_per_ele; i++)
  {
    loc(0) = loc_upts(0, i);
    loc(1) = loc_upts(1, i);
    loc(2) = loc_upts(2, i);
    for (int j = 0; j < n_upts_per_ele; j++)
    {
      vandermonde(i, j) = eval_pris_basis_hierarchical(j, loc, order);
    }
  }

  // Store its inverse
  inv_vandermonde = inv_array(vandermonde);
}

void eles_pris::set_vandermonde_vol_cub(void)
{
  vandermonde_vol_cub.setup(n_cubpts_per_ele, n_cubpts_per_ele);
  hf_array<double> loc(n_dims);
  // create the vandermonde matrix
  for (int i = 0; i < n_cubpts_per_ele; i++)
  {
    loc(0) = loc_volume_cubpts(0, i);
    loc(1) = loc_volume_cubpts(1, i);
    loc(2) = loc_volume_cubpts(2, i);
    for (int j = 0; j < n_cubpts_per_ele; j++)
    {
      vandermonde_vol_cub(i, j) = eval_pris_basis_hierarchical(j, loc, order);
    }
  }

  // Store its inverse
  inv_vandermonde_vol_cub = inv_array(vandermonde_vol_cub);
}

void eles_pris::set_exp_filter(void)
{
  exp_filter.setup(n_upts_per_ele, n_upts_per_ele);
  exp_filter.initialize_to_zero();
  int i, j, k, l, mode;
  double eta;

    mode = 0;
    for (l = 0; l < 2 * order + 1; l++) //sum of x,y,z mode
    {
      for (k = 0; k < l + 1; k++) //k<=sum
      {
        for (j = 0; j < l - k + 1; j++) //j<=sum-k
        {
          i = l - k - j;
          if (k <= order && i + j <= order)
          {
              eta = (double)l / (double)(2*order);
              exp_filter(mode, mode) = exp(-run_input.expf_fac * pow(eta, run_input.expf_order));
            mode++;
          }
        }
      }
    }

  exp_filter = mult_arrays(exp_filter, inv_vandermonde);
  exp_filter = mult_arrays(vandermonde, exp_filter);
}

void eles_pris::calc_norm_basis(void)
{
  int n1, n2, n3;
  double norm3;
  norm_basis_persson.setup(n_upts_per_ele);
  for (int i = 0; i < n_upts_per_ele; i++)
  {
    get_pris_basis_index(i, order, n1, n2, n3);
    norm3 = 2.0 / (2.0 * n3 + 1.0);
    norm_basis_persson(i) = norm3;
  }
}

//detect shock use persson's method
void eles_pris::shock_det_persson(void)
{

  hf_array<double> temp_modal(n_upts_per_ele, n_eles); //store modal value

//step 1. convert to modal value
#if defined _ACCELERATE_BLAS || defined _MKL_BLAS || defined _STANDARD_BLAS
  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n_upts_per_ele, n_eles, n_upts_per_ele, 1.0, inv_vandermonde.get_ptr_cpu(), n_upts_per_ele, disu_upts(0).get_ptr_cpu(), n_upts_per_ele, 0.0, temp_modal.get_ptr_cpu(), n_upts_per_ele);
#else
  dgemm(n_upts_per_ele, n_eles, n_upts_per_ele, 1.0, 0.0, inv_vandermonde.get_ptr_cpu(), disu_upts(0).get_ptr_cpu(), temp_modal.get_ptr_cpu());
#endif

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
    sensor(ic) = 0;
    for (int j = 0; j < n_upts_per_ele; j++)
    {
      get_pris_basis_index(j, order, x, y, z);
      if (run_input.over_int)
      {
        if (x + y >= run_input.N_under || z >= run_input.N_under)
          sensor(ic) += temp_modal(j, ic) * norm_basis_persson(j);
      }
      else
      {
        if (x + y == order || z == order)
          sensor(ic) += temp_modal(j, ic) * norm_basis_persson(j);
      }
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

// initialize the vandermonde matrix
void eles_pris::set_vandermonde_tri_restart()
{
  vandermonde_tri_rest.setup(n_upts_tri_rest,n_upts_tri_rest);

  // create the vandermonde matrix
  for (int i=0;i<n_upts_tri_rest;i++)
    for (int j=0;j<n_upts_tri_rest;j++)
      vandermonde_tri_rest(i,j) = eval_dubiner_basis_2d(loc_upts_pri_tri_rest(0,i),loc_upts_pri_tri_rest(1,i),j,order_rest);

  // Store its inverse
  inv_vandermonde_tri_rest = inv_array(vandermonde_tri_rest);
}

int eles_pris::read_restart_info_ascii(ifstream& restart_file)
{

  string str;
  // Move to triangle element
  while(1) {
      getline(restart_file,str);
      if (str=="PRIS") break;

      if (restart_file.eof()) return 0;
    }

  getline(restart_file,str);
  restart_file >> order_rest;
  getline(restart_file,str);
  getline(restart_file,str);
  restart_file >> n_upts_per_ele_rest;
  getline(restart_file,str);
  getline(restart_file,str);
  restart_file >> n_upts_tri_rest;
  getline(restart_file,str);
  getline(restart_file,str);

  loc_upts_pri_1d_rest.setup(order_rest+1);
  loc_upts_pri_tri_rest.setup(2,n_upts_tri_rest);

  for (int i=0;i<order_rest+1;i++) {
      restart_file >> loc_upts_pri_1d_rest(i);
    }
  getline(restart_file,str);
  getline(restart_file,str);

  for (int i=0;i<n_upts_tri_rest;i++) {
      for (int j=0;j<2;j++) {
          restart_file >> loc_upts_pri_tri_rest(j,i);
        }
    }

  set_vandermonde_tri_restart();
  set_opp_r();

  return 1;

}

#ifdef _HDF5
void eles_pris::read_restart_info_hdf5(hid_t &restart_file, int in_rest_order)
{
  hid_t dataset_id, plist_id, memspace_id, dataspace_id;
  hsize_t count;  // number of blocks
  hsize_t offset; // start
  //open dataset
  dataset_id = H5Dopen2(restart_file, "PRIS", H5P_DEFAULT);
  if (dataset_id < 0)
    FatalError("Cannot find pris property");

  plist_id = H5Pcreate(H5P_DATASET_XFER);
#ifdef _MPI
  //set collective read
  H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);
#endif

  if (n_eles)
  {
    order_rest = in_rest_order;
    n_upts_per_ele_rest = (order_rest + 2) * (order_rest + 1) * (order_rest + 1) / 2;
    n_upts_tri_rest = (order_rest + 1) * (order_rest + 2) / 2;
    loc_upts_pri_1d_rest.setup(order_rest + 1);
    loc_upts_pri_tri_rest.setup(2, n_upts_tri_rest);

    //read data
    offset = 0;
    count = order_rest + 1;
    memspace_id = H5Screate_simple(1, &count, NULL); //row major: n_eles by n_upts_per_ele_rest* n_fields
    dataspace_id = H5Dget_space(dataset_id);

    if (H5Sselect_hyperslab(dataspace_id, H5S_SELECT_SET, &offset, NULL, &count, NULL) < 0)
      FatalError("Failed to get hyperslab");
    H5Dread(dataset_id, H5T_NATIVE_DOUBLE, memspace_id, dataspace_id, plist_id, loc_upts_pri_1d_rest.get_ptr_cpu()); //read 1d
    H5Sclose(memspace_id);

    offset = order_rest + 1;
    count = 2 * n_upts_tri_rest;
    memspace_id = H5Screate_simple(1, &count, NULL); //row major: n_eles by n_upts_per_ele_rest* n_fields

    if (H5Sselect_hyperslab(dataspace_id, H5S_SELECT_SET, &offset, NULL, &count, NULL) < 0)
      FatalError("Failed to get hyperslab");
    H5Dread(dataset_id, H5T_NATIVE_DOUBLE, memspace_id, dataspace_id, plist_id, loc_upts_pri_tri_rest.get_ptr_cpu()); //read tri
    H5Sclose(memspace_id);
    H5Sclose(dataspace_id);
    
    set_vandermonde_tri_restart();
    set_opp_r();
  }
#ifdef _MPI
  else //read empty
  {
    dataspace_id = H5Dget_space(dataset_id);
    H5Sselect_none(dataspace_id);
    H5Dread(dataset_id, H5T_NATIVE_DOUBLE, dataspace_id, dataspace_id, plist_id, NULL);
    H5Dread(dataset_id, H5T_NATIVE_DOUBLE, dataspace_id, dataspace_id, plist_id, NULL);
    H5Sclose(dataspace_id);
  }
#endif

  //close objects
  H5Pclose(plist_id);
  H5Dclose(dataset_id);
}
#endif

void eles_pris::write_restart_info_ascii(ofstream& restart_file)
{
  restart_file << "PRIS" << endl;

  restart_file << "Order" << endl;
  restart_file << order << endl;

  restart_file << "Number of solution points per prismatic element" << endl;
  restart_file << n_upts_per_ele << endl;

  restart_file << "Number of solution points in triangle" << endl;
  restart_file << n_upts_tri << endl;

  restart_file << "Location of solution points in 1D" << endl;
  for (int i=0;i<order+1;i++) {
      restart_file << loc_upts_pri_1d(i) << " ";
    }
  restart_file << endl;

  restart_file << "Location of solution points in triangle" << endl;
  for (int i=0;i<n_upts_tri;i++) {
      for (int j=0;j<2;j++) {
          restart_file << loc_upts_pri_tri(j,i) << " ";
        }
      restart_file << endl;
    }
}

#ifdef _HDF5
void eles_pris::write_restart_info_hdf5(hid_t &restart_file)
{
  hid_t dataset_id, plist_id, dataspace_id, memspace_id;
  hsize_t count, offset;
  hsize_t dim = run_input.order + 1 + 2 * (run_input.order + 1) * (run_input.order + 2) / 2;

  //create PRIS dataset
  dataspace_id = H5Screate_simple(1, &dim, NULL);
  dataset_id = H5Dcreate2(restart_file, "PRIS", H5T_NATIVE_DOUBLE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  plist_id = H5Pcreate(H5P_DATASET_XFER);
#ifdef _MPI
  //set collective read
  H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);
#endif

  //write loc_upts_pri_1d and loc_upts_pri_tri
  if (n_eles)
  {
    offset = 0;
    count = order + 1;
    memspace_id = H5Screate_simple(1, &count, NULL);
    if (H5Sselect_hyperslab(dataspace_id, H5S_SELECT_SET, &offset, NULL, &count, NULL) < 0)
      FatalError("Failed to get hyperslab");
    H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, memspace_id, dataspace_id, plist_id, loc_upts_pri_1d.get_ptr_cpu());
    H5Sclose(memspace_id);

    offset = order + 1;
    count = 2 * (order + 1) * (order + 2) / 2;
    memspace_id = H5Screate_simple(1, &count, NULL);
    if (H5Sselect_hyperslab(dataspace_id, H5S_SELECT_SET, &offset, NULL, &count, NULL) < 0)
      FatalError("Failed to get hyperslab");
    H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, memspace_id, dataspace_id, plist_id, loc_upts_pri_tri.get_ptr_cpu());
    H5Sclose(memspace_id);
  }
#ifdef _MPI
  else
  {
    H5Sselect_none(dataspace_id);
    H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, dataspace_id, dataspace_id, plist_id, NULL);
    H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, dataspace_id, dataspace_id, plist_id, NULL);
  }
#endif
  H5Pclose(plist_id);
  H5Dclose(dataset_id);
  H5Sclose(dataspace_id);
}
#endif

void eles_pris::set_over_int_filter()
{
  int N_under = run_input.N_under;
  int n_mode_under = (N_under + 1) * (N_under + 1) * (N_under + 2) / 2; //projected n_upts_per_ele
  //int n_mode_under_tri = (N_under + 1) * (N_under + 2) / 2;
  //int n_mode_under_1d = N_under + 1;
  hf_array<double> temp_proj(n_mode_under, n_cubpts_per_ele);
  hf_array<double> temp_vand(n_upts_per_ele, n_mode_under);
  hf_array<double> loc(n_dims);

  //step 1. nodal to L2 projected modal \hat{u_i}=\int{\phi_i*l_j}=>\phi_i(j)*w(j)
  int dummy, dummy1, order_t;
  double norm_t;
  for (int i = 0; i < n_mode_under; i++)
  {
    get_pris_basis_index(i, N_under, dummy, dummy1, order_t);
    norm_t = 2.0 / (2.0 * order_t + 1.0);
    for (int j = 0; j < n_cubpts_per_ele; j++)
    {
      loc(0) = loc_volume_cubpts(0, j);
      loc(1) = loc_volume_cubpts(1, j);
      loc(2) = loc_volume_cubpts(2, j);
      temp_proj(i, j) = eval_pris_basis_hierarchical(i, loc, N_under) / norm_t * weight_volume_cubpts(j);
    }
  }
  //step 2. projected modal back to nodal to get filtered solution \tilde{u_j}=V_{ji}*\hat{u_i}
  for (int j = 0; j < n_upts_per_ele; j++)
  {
    loc(0) = loc_upts(0, j);
    loc(1) = loc_upts(1, j);
    loc(2) = loc_upts(2, j);
    for (int i = 0; i < n_mode_under; i++)
    {
      temp_vand(j, i) = eval_pris_basis_hierarchical(i, loc, N_under);
    }
  }
  over_int_filter = mult_arrays(temp_proj, opp_volume_cubpts);
  over_int_filter = mult_arrays(temp_vand, over_int_filter);
}

// evaluate nodal basis

double eles_pris::eval_nodal_basis(int in_index, hf_array<double> in_loc)
{
  double oned_nodal_basis_at_loc;
  double tri_nodal_basis_at_loc;

  int index_tri = in_index%n_upts_tri;
  int index_1d = in_index/n_upts_tri;

  // 1. First evaluate the triangular nodal basis at loc(0) and loc(1)

  // First evaluate the normalized Dubiner basis at position in_loc
  hf_array<double> dubiner_basis_at_loc(n_upts_tri);
  for (int i=0;i<n_upts_tri;i++)
    dubiner_basis_at_loc(i) = eval_dubiner_basis_2d(in_loc(0),in_loc(1),i,order);

  // From Hesthaven, equation 3.3, V^T * l = P, or l = (V^-1)^T P
  tri_nodal_basis_at_loc = 0.;
  for (int i=0;i<n_upts_tri;i++)
    tri_nodal_basis_at_loc += inv_vandermonde_tri(i,index_tri)*dubiner_basis_at_loc(i);

  // 2. Now evaluate the 1D lagrange basis at loc(2)
  oned_nodal_basis_at_loc = eval_lagrange(in_loc(2),index_1d,loc_upts_pri_1d);

  return (tri_nodal_basis_at_loc*oned_nodal_basis_at_loc);

}

// evaluate nodal basis for restart

double eles_pris::eval_nodal_basis_restart(int in_index, hf_array<double> in_loc)
{
  double oned_nodal_basis_at_loc;
  double tri_nodal_basis_at_loc;

  int index_tri = in_index%n_upts_tri_rest;
  int index_1d = in_index/n_upts_tri_rest;

  // 1. First evaluate the triangular nodal basis at loc(0) and loc(1)

  // First evaluate the normalized Dubiner basis at position in_loc
  hf_array<double> dubiner_basis_at_loc(n_upts_tri_rest);
  for (int i=0;i<n_upts_tri_rest;i++)
    dubiner_basis_at_loc(i) = eval_dubiner_basis_2d(in_loc(0),in_loc(1),i,order_rest);

  // From Hesthaven, equation 3.3, V^T * l = P, or l = (V^-1)^T P
  tri_nodal_basis_at_loc = 0.;
  for (int i=0;i<n_upts_tri_rest;i++)
    tri_nodal_basis_at_loc += inv_vandermonde_tri_rest(i,index_tri)*dubiner_basis_at_loc(i);

  // 2. Now evaluate the 1D lagrange basis at loc(2)
  oned_nodal_basis_at_loc = eval_lagrange(in_loc(2),index_1d,loc_upts_pri_1d_rest);

  return (tri_nodal_basis_at_loc*oned_nodal_basis_at_loc);
}

// evaluate derivative of nodal basis

double eles_pris::eval_d_nodal_basis(int in_index, int in_cpnt, hf_array<double> in_loc)
{
  double out_d_nodal_basis_at_loc;

  int index_tri = in_index%n_upts_tri;
  int index_1d = in_index/n_upts_tri;

  if (in_cpnt == 0 || in_cpnt == 1)
    {
      double d_tri_nodal_basis_at_loc;
      double oned_nodal_basis_at_loc;

      // 1. Evaluate the derivative of triangular nodal basis at loc(0) and loc(1)

      // Evalute the derivative normalized Dubiner basis at position in_loc
      hf_array<double> d_dubiner_basis_at_loc(n_upts_per_ele);
      for (int i=0;i<n_upts_tri;i++) {
          if (in_cpnt==0)
            d_dubiner_basis_at_loc(i) = eval_dr_dubiner_basis_2d(in_loc(0),in_loc(1),i,order);
          else if (in_cpnt==1)
            d_dubiner_basis_at_loc(i) = eval_ds_dubiner_basis_2d(in_loc(0),in_loc(1),i,order);
        }

      // From Hesthaven, equation 3.3, V^T * l = P, or l = (V^-1)^T P
      d_tri_nodal_basis_at_loc = 0.;
      for (int i=0;i<n_upts_tri;i++)
        d_tri_nodal_basis_at_loc += inv_vandermonde_tri(i,index_tri)*d_dubiner_basis_at_loc(i);

      // 2. Evaluate the 1d nodal basis at loc(2)
      oned_nodal_basis_at_loc = eval_lagrange(in_loc(2),index_1d,loc_upts_pri_1d);

      out_d_nodal_basis_at_loc = d_tri_nodal_basis_at_loc*oned_nodal_basis_at_loc;
    }
  else if (in_cpnt==2)
    {

      double tri_nodal_basis_at_loc;
      double d_oned_nodal_basis_at_loc;

      // 1. First evaluate the triangular nodal basis at loc(0) and loc(1)

      // Evaluate the normalized Dubiner basis at position in_loc
      hf_array<double> dubiner_basis_at_loc(n_upts_tri);
      for (int i=0;i<n_upts_tri;i++)
        dubiner_basis_at_loc(i) = eval_dubiner_basis_2d(in_loc(0),in_loc(1),i,order);

      // From Hesthaven, equation 3.3, V^T * l = P, or l = (V^-1)^T P
      tri_nodal_basis_at_loc = 0.;
      for (int i=0;i<n_upts_tri;i++)
        tri_nodal_basis_at_loc += inv_vandermonde_tri(i,index_tri)*dubiner_basis_at_loc(i);

      // 2. Then evaluate teh derivative of 1d nodal basis at loc(2)
      d_oned_nodal_basis_at_loc = eval_d_lagrange(in_loc(2),index_1d,loc_upts_pri_1d);

      out_d_nodal_basis_at_loc = tri_nodal_basis_at_loc*d_oned_nodal_basis_at_loc;

    }


  return out_d_nodal_basis_at_loc;
  // fill in
}

// evaluate nodal shape basis

double eles_pris::eval_nodal_s_basis(int in_index, hf_array<double> in_loc, int in_n_spts)
{

  double nodal_s_basis;

  if (in_n_spts==6) {
      if (in_index==0)
        nodal_s_basis =  1./4.*(in_loc(0)+in_loc(1)) *(in_loc(2)-1.);
      else if (in_index==1)
        nodal_s_basis = -1./4.*(in_loc(0)+1.)*(in_loc(2)-1.);
      else if (in_index==2)
        nodal_s_basis = -1./4.*(in_loc(1)+1.)*(in_loc(2)-1.);
      else if (in_index==3)
        nodal_s_basis = -1./4.*(in_loc(0)+in_loc(1))*(in_loc(2)+1.);
      else if (in_index==4)
        nodal_s_basis =  1./4.*(in_loc(0)+1.)*(in_loc(2)+1.);
      else if (in_index==5)
        nodal_s_basis =  1./4.*(in_loc(1)+1.)*(in_loc(2)+1.);
    }
  else if (in_n_spts==15) {
      if (in_index==0)
        nodal_s_basis = (1./4*(in_loc(0)+in_loc(1)))*(in_loc(0)+in_loc(1)+1.)*in_loc(2)*(in_loc(2)-1.);
      else if (in_index==1)
        nodal_s_basis = (1./4)*in_loc(0)*(in_loc(0)+1.)*in_loc(2)*(in_loc(2)-1.);
      else if (in_index==2)
        nodal_s_basis = (1./4)*in_loc(1)*(in_loc(1)+1.)*in_loc(2)*(in_loc(2)-1.);
      else if (in_index==3)
        nodal_s_basis = (1./4*(in_loc(0)+in_loc(1)))*(in_loc(0)+in_loc(1)+1.)*in_loc(2)*(in_loc(2)+1.);
      else if (in_index==4)
        nodal_s_basis = (1./4)*in_loc(0)*(in_loc(0)+1.)*in_loc(2)*(in_loc(2)+1.);
      else if (in_index==5)
        nodal_s_basis = (1./4)*in_loc(1)*(in_loc(1)+1.)*in_loc(2)*(in_loc(2)+1.);
      else if (in_index==6)
        nodal_s_basis = -(1./2*(in_loc(0)+in_loc(1)))*(in_loc(0)+1.)*in_loc(2)*(in_loc(2)-1.);
      else if (in_index==7)
        nodal_s_basis = (1./2*(in_loc(0)+1.))*(in_loc(1)+1.)*in_loc(2)*(in_loc(2)-1.);
      else if (in_index==8)
        nodal_s_basis = -(1./2*(in_loc(0)+in_loc(1)))*(in_loc(1)+1.)*in_loc(2)*(in_loc(2)-1.);
      else if (in_index==9)
        nodal_s_basis = (1./2*(in_loc(0)+in_loc(1)))*(in_loc(2)*in_loc(2)-1.);
      else if (in_index==10)
        nodal_s_basis = -(1./2*(in_loc(0)+1.))*(in_loc(2)*in_loc(2)-1.);
      else if (in_index==11)
        nodal_s_basis = -(1./2*(in_loc(1)+1.))*(in_loc(2)*in_loc(2)-1.);
      else if (in_index==12)
        nodal_s_basis = -(1./2*(in_loc(0)+in_loc(1)))*(in_loc(0)+1.)*in_loc(2)*(in_loc(2)+1.);
      else if (in_index==13)
        nodal_s_basis = (1./2*(in_loc(0)+1.))*(in_loc(1)+1.)*in_loc(2)*(in_loc(2)+1.);
      else if (in_index==14)
        nodal_s_basis = -(1./2*(in_loc(0)+in_loc(1)))*(in_loc(1)+1.)*in_loc(2)*(in_loc(2)+1.);
    }
  else
    {
      FatalError("Shape order not implemented yet, exiting");
    }
  return nodal_s_basis;

}

// evaluate derivative of nodal shape basis

void eles_pris::eval_d_nodal_s_basis(hf_array<double> &d_nodal_s_basis, hf_array<double> in_loc, int in_n_spts)
{

  if (in_n_spts==6) {
      d_nodal_s_basis(0,0) =  1./4.*(in_loc(2)-1.);
      d_nodal_s_basis(1,0) = -1./4.*(in_loc(2)-1.);
      d_nodal_s_basis(2,0) = 0;
      d_nodal_s_basis(3,0) = -1./4.*(in_loc(2)+1.);
      d_nodal_s_basis(4,0) =  1./4.*(in_loc(2)+1.);
      d_nodal_s_basis(5,0) =  0.;

      d_nodal_s_basis(0,1) =  1./4.*(in_loc(2)-1.);
      d_nodal_s_basis(1,1) = 0.;
      d_nodal_s_basis(2,1) = -1./4.*(in_loc(2)-1.);
      d_nodal_s_basis(3,1) = -1./4.*(in_loc(2)+1.);
      d_nodal_s_basis(4,1) =  0.;
      d_nodal_s_basis(5,1) =  1./4.*(in_loc(2)+1.);

      d_nodal_s_basis(0,2) =  1./4.*(in_loc(0)+in_loc(1));
      d_nodal_s_basis(1,2) = -1./4.*(in_loc(0)+1.);
      d_nodal_s_basis(2,2) = -1./4.*(in_loc(1)+1.);
      d_nodal_s_basis(3,2) = -1./4.*(in_loc(0)+in_loc(1));
      d_nodal_s_basis(4,2) =  1./4.*(in_loc(0)+1.);
      d_nodal_s_basis(5,2) =  1./4.*(in_loc(1)+1.);
    }
  else if (in_n_spts==15) {

      d_nodal_s_basis(0 ,0) = (1./4)*in_loc(2)*(in_loc(2)-1.)*(2*in_loc(0)+2*in_loc(1)+1.);
      d_nodal_s_basis(1 ,0) = (1./4)*in_loc(2)*(in_loc(2)-1.)*(2*in_loc(0)+1.);
      d_nodal_s_basis(2 ,0) = 0.;
      d_nodal_s_basis(3 ,0) = (1./4)*in_loc(2)*(in_loc(2)+1.)*(2*in_loc(0)+2*in_loc(1)+1.);
      d_nodal_s_basis(4 ,0) =(1./4)*in_loc(2)*(in_loc(2)+1.)*(2*in_loc(0)+1.);
      d_nodal_s_basis(5 ,0) = 0.;
      d_nodal_s_basis(6 ,0) = -(1./2)*in_loc(2)*(in_loc(2)-1.)*(2*in_loc(0)+1.+in_loc(1));
      d_nodal_s_basis(7 ,0) = (1./2*(in_loc(1)+1.))*in_loc(2)*(in_loc(2)-1.);
      d_nodal_s_basis(8 ,0) = -(1./2*(in_loc(1)+1.))*in_loc(2)*(in_loc(2)-1.);
      d_nodal_s_basis(9 ,0) = (1./2)*in_loc(2)*in_loc(2)-1./2;
      d_nodal_s_basis(10,0) = -(1./2)*in_loc(2)*in_loc(2)+1./2;
      d_nodal_s_basis(11,0) = 0.;
      d_nodal_s_basis(12,0) = -(1./2)*in_loc(2)*(in_loc(2)+1.)*(2*in_loc(0)+1.+in_loc(1));
      d_nodal_s_basis(13,0) = (1./2*(in_loc(1)+1.))*in_loc(2)*(in_loc(2)+1.);
      d_nodal_s_basis(14,0) = -(1./2*(in_loc(1)+1.))*in_loc(2)*(in_loc(2)+1.);


      d_nodal_s_basis(0 ,1) = (1./4)*in_loc(2)*(in_loc(2)-1.)*(2*in_loc(0)+2*in_loc(1)+1.);
      d_nodal_s_basis(1 ,1) = 0.;
      d_nodal_s_basis(2 ,1) = (1./4)*in_loc(2)*(in_loc(2)-1.)*(2*in_loc(1)+1.);
      d_nodal_s_basis(3 ,1) = (1./4)*in_loc(2)*(in_loc(2)+1.)*(2*in_loc(0)+2*in_loc(1)+1.);
      d_nodal_s_basis(4 ,1) = 0.;
      d_nodal_s_basis(5 ,1) = (1./4)*in_loc(2)*(in_loc(2)+1.)*(2*in_loc(1)+1.);
      d_nodal_s_basis(6 ,1) = -(1./2*(in_loc(0)+1.))*in_loc(2)*(in_loc(2)-1.);
      d_nodal_s_basis(7 ,1) = (1./2*(in_loc(0)+1.))*in_loc(2)*(in_loc(2)-1.);
      d_nodal_s_basis(8 ,1) = -(1./2)*in_loc(2)*(in_loc(2)-1.)*(2*in_loc(1)+1.+in_loc(0));
      d_nodal_s_basis(9 ,1) = (1./2)*in_loc(2)*in_loc(2)-1./2;
      d_nodal_s_basis(10,1) = 0.;
      d_nodal_s_basis(11,1) = -(1./2)*in_loc(2)*in_loc(2)+1./2;
      d_nodal_s_basis(12,1) = -(1./2*(in_loc(0)+1.))*in_loc(2)*(in_loc(2)+1.);
      d_nodal_s_basis(13,1) = (1./2*(in_loc(0)+1.))*in_loc(2)*(in_loc(2)+1.);
      d_nodal_s_basis(14,1) = -(1./2)*in_loc(2)*(in_loc(2)+1.)*(2*in_loc(1)+1.+in_loc(0));

      d_nodal_s_basis(0 ,2) = (1./4*(in_loc(0)+in_loc(1)+1.))*(in_loc(0)+in_loc(1))*(2*in_loc(2)-1.);
      d_nodal_s_basis(1 ,2) = (1./4)*in_loc(0)*(2*in_loc(2)-1.)*(in_loc(0)+1.);
      d_nodal_s_basis(2 ,2) = (1./4)*in_loc(1)*(2*in_loc(2)-1.)*(in_loc(1)+1.);
      d_nodal_s_basis(3 ,2) = (1./4*(in_loc(0)+in_loc(1)+1.))*(in_loc(0)+in_loc(1))*(2*in_loc(2)+1.);
      d_nodal_s_basis(4 ,2) = (1./4)*in_loc(0)*(2*in_loc(2)+1.)*(in_loc(0)+1.);
      d_nodal_s_basis(5 ,2) = (1./4)*in_loc(1)*(2*in_loc(2)+1.)*(in_loc(1)+1.);
      d_nodal_s_basis(6 ,2) = -(1./2*(2*in_loc(2)-1.))*(in_loc(0)+1.)*(in_loc(0)+in_loc(1));
      d_nodal_s_basis(7 ,2) = (1./2*(2*in_loc(2)-1.))*(in_loc(1)+1.)*(in_loc(0)+1.);
      d_nodal_s_basis(8 ,2) = -(1./2*(2*in_loc(2)-1.))*(in_loc(1)+1.)*(in_loc(0)+in_loc(1));
      d_nodal_s_basis(9 ,2) = in_loc(2)*(in_loc(0)+in_loc(1));
      d_nodal_s_basis(10,2) = -in_loc(2)*(in_loc(0)+1.);
      d_nodal_s_basis(11,2) = -in_loc(2)*(in_loc(1)+1.);
      d_nodal_s_basis(12,2) = -(1./2*(2*in_loc(2)+1.))*(in_loc(0)+1.)*(in_loc(0)+in_loc(1));
      d_nodal_s_basis(13,2) = (1./2*(2*in_loc(2)+1.))*(in_loc(1)+1.)*(in_loc(0)+1.);
      d_nodal_s_basis(14,2) = -(1./2*(2*in_loc(2)+1.))*(in_loc(1)+1.)*(in_loc(0)+in_loc(1));
    }
  else
    {
      FatalError("Shape order not implemented yet, exiting");
    }
}

double eles_pris::eval_pris_basis_hierarchical(int in_mode, hf_array<double> in_loc, int in_order)
{

  double pris_basis;

  int n_dof = ((in_order + 1) * (in_order + 1) * (in_order + 2)) / 2; //total number of solution pts/basis

  if (in_mode < n_dof)
  {
    int i, j, k, l;
    int mode;
    double jacobi_0, jacobi_1;
    hf_array<double> ab;

    ab = rs_to_ab(in_loc(0), in_loc(1)); //transform r,s coord to a,b in dubiner basis

    mode = 0;
    for (l = 0; l < 2 * in_order + 1; l++) //sum of r,s,t modes less than 2*order+1
    {
      for (k = 0; k < l + 1; k++) //t basis from 0 to sum
      {
        for (j = 0; j < l - k + 1; j++) //s basis from 0 to sum-k
        {
          i = l - k - j; //r basis
          if (k <= in_order && i + j <= in_order)
          {
            if (mode == in_mode) // found the correct mode
            {
              jacobi_0 = eval_jacobi(ab(0), 0, 0, i);
              jacobi_1 = eval_jacobi(ab(1), (2 * i) + 1, 0, j);
              pris_basis = sqrt(2.0) * jacobi_0 * jacobi_1 * pow(1.0 - ab(1), i) * eval_legendre(in_loc(2), k);
              return pris_basis;
            }
            mode++;
          }
        }
      }
    }
  }
  else
  {
    cout << "ERROR: Invalid mode when evaluating Dubiner basis ...." << endl;
  }
  FatalError("No mode is founded");
}

int eles_pris::get_pris_basis_index(int in_mode, int in_order, int &out_r, int &out_s, int &out_t)
{
  int n_dof = ((in_order + 1) * (in_order + 1) * (in_order + 2)) / 2; //total number of solution pts/basis

  if (in_mode < n_dof)
  {
    int i, j, k, l;
    int mode;

    mode = 0;
    for (l = 0; l < 2 * in_order + 1; l++) //sum of r,s,t modes from 0 to 2*order
    {
      for (k = 0; k < l + 1; k++) //t basis from 0 to sum
      {
        for (j = 0; j < l - k + 1; j++) //s basis from 0 to sum-k
        {
          i = l - k - j; //r basis
          if (k <= in_order && i + j <= in_order)
          {
            if (mode == in_mode) // found the correct mode
            {
              out_r = i;
              out_s = j;
              out_t = k;
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
    cout << "ERROR: Invalid mode when evaluating Dubiner basis ...." << endl;
  }
  return -1;
}

void eles_pris::fill_opp_3(hf_array<double>& opp_3)
{

  hf_array<double> loc(3);
  hf_array<double> opp_3_tri(n_upts_tri,3*(order+1));
  get_opp_3_tri(opp_3_tri,loc_upts_pri_tri,loc_1d_fpts,vandermonde_tri, inv_vandermonde_tri,n_upts_tri,order,run_input.c_tri,run_input.vcjh_scheme_tri);

  // Compute value of eta
  double eta;
  if (run_input.vcjh_scheme_pri_1d == 0)
    eta = run_input.eta_pri;
  else
    eta = compute_eta(run_input.vcjh_scheme_pri_1d,order);

  for (int upt=0;upt<n_upts_per_ele;upt++)
    {
      loc(0)=loc_upts(0,upt);
      loc(1)=loc_upts(1,upt);
      loc(2)=loc_upts(2,upt);

      int upt_1d = upt/n_upts_tri;
      int upt_tri = upt%n_upts_tri;

      for (int in_index=0;in_index<n_fpts_per_ele;in_index++)
        {
          // Face 0
          if (in_index < n_fpts_per_inter(0))
            {
              int face_fpt = in_index;
              if (face0_map(face_fpt)==upt_tri)
                opp_3(upt,in_index)= -eval_d_vcjh_1d(loc(2),0,order,eta);
              else
                opp_3(upt,in_index)= 0.;
            }
          // Face 1
          else if (in_index < n_fpts_per_inter(0)+n_fpts_per_inter(1))
            {
              int face_fpt = in_index-n_fpts_per_inter(0);
              if (face_fpt == upt_tri)
                opp_3(upt,in_index)= eval_d_vcjh_1d(loc(2),1,order,eta);
              else
                opp_3(upt,in_index)= 0.;
            }
          // face 2
          else if (in_index < n_fpts_per_inter(0)+n_fpts_per_inter(1)+n_fpts_per_inter(2))
            {
              int face_fpt = in_index-2*n_fpts_per_inter(0);
              int edge_fpt = face_fpt%(order+1);
              int edge = 0;

              if ( face_fpt/(order+1)==upt_1d)
                //opp_3(upt,in_index)= eval_div_dg_tri(loc,edge,edge_fpt,order,loc_upts_pri_1d);
                opp_3(upt,in_index)= opp_3_tri(upt_tri,edge*(order+1)+edge_fpt);
              else
                opp_3(upt,in_index)= 0.;
            }
          // face 3
          else if (in_index < n_fpts_per_inter(0)+n_fpts_per_inter(1)+n_fpts_per_inter(2)+n_fpts_per_inter(3))
            {
              int face_fpt = in_index-2*n_fpts_per_inter(0)-n_fpts_per_inter(2);
              int edge_fpt = face_fpt%(order+1);
              int edge = 1;

              if (face_fpt/(order+1) == upt_1d)
                //opp_3(upt,in_index)= eval_div_dg_tri(loc,edge,edge_fpt,order,loc_upts_pri_1d);
                opp_3(upt,in_index)= opp_3_tri(upt_tri,edge*(order+1)+edge_fpt);
              else
                opp_3(upt,in_index)= 0.;
            }
          // face 4
          else if (in_index < n_fpts_per_inter(0)+n_fpts_per_inter(1)+n_fpts_per_inter(2)+n_fpts_per_inter(3)+n_fpts_per_inter(4))
            {
              int face_fpt = (in_index-2*n_fpts_per_inter(0)-n_fpts_per_inter(2)-n_fpts_per_inter(3));
              int edge_fpt = face_fpt%(order+1);
              int edge = 2;

              if (face_fpt/(order+1) == upt_1d)
                //opp_3(upt,in_index)= eval_div_dg_tri(loc,edge,edge_fpt,order,loc_upts_pri_1d);
                opp_3(upt,in_index)= opp_3_tri(upt_tri,edge*(order+1)+edge_fpt);
              else
                opp_3(upt,in_index)= 0.;
            }
        }
    }


}

// evaluate divergence of vcjh basis

double eles_pris::eval_div_vcjh_basis(int in_index, hf_array<double>& loc)
{
  double div_vcjh_basis;
  double tol = 1e-12;
  double eta;

  // Check that loc is at one of the solution points, otherwise procedure doesn't work
  int flag = 1;
  int upt;
  for (int i=0;i<n_upts_per_ele;i++) {
      if (   abs(loc(0)-loc_upts(0,i)) < tol
             && abs(loc(1)-loc_upts(1,i)) < tol
             && abs(loc(2)-loc_upts(2,i)) < tol) {
          flag = 0;
          upt = i;
          break;
        }
    }
  if (flag==1) FatalError("eval_div_vcjh_basis is not at solution point, exiting");

  int upt_1d = upt/n_upts_tri;
  int upt_tri = upt%n_upts_tri;

  // Compute value of eta
  if (run_input.vcjh_scheme_pri_1d == 0)
    eta = run_input.eta_pri;
  else
    eta = compute_eta(run_input.vcjh_scheme_pri_1d,order);

  // Compute value of c_tri
  double c_tri =  0.; // HACK

  // Face 0
  if (in_index < n_fpts_per_inter(0))
    {
      int face_fpt = in_index;
      if (face0_map(face_fpt)==upt_tri)
        div_vcjh_basis = -eval_d_vcjh_1d(loc(2),0,order,eta);
      else
        div_vcjh_basis = 0.;
    }
  // Face 1
  else if (in_index < n_fpts_per_inter(0)+n_fpts_per_inter(1))
    {
      int face_fpt = in_index-n_fpts_per_inter(0);
      if (face_fpt == upt_tri)
        div_vcjh_basis = eval_d_vcjh_1d(loc(2),1,order,eta);
      else
        div_vcjh_basis = 0.;
    }
  // face 2
  else if (in_index < n_fpts_per_inter(0)+n_fpts_per_inter(1)+n_fpts_per_inter(2))
    {
      int face_fpt = in_index-2*n_fpts_per_inter(0);
      int edge_fpt = face_fpt%(order+1);
      int edge = 0;

      if ( face_fpt/(order+1)==upt_1d)
        div_vcjh_basis = eval_div_dg_tri(loc,edge,edge_fpt,order,loc_upts_pri_1d);
      else
        div_vcjh_basis = 0.;
    }
  // face 3
  else if (in_index < n_fpts_per_inter(0)+n_fpts_per_inter(1)+n_fpts_per_inter(2)+n_fpts_per_inter(3))
    {
      int face_fpt = in_index-2*n_fpts_per_inter(0)-n_fpts_per_inter(2);
      int edge_fpt = face_fpt%(order+1);
      int edge = 1;

      if (face_fpt/(order+1) == upt_1d)
        div_vcjh_basis = eval_div_dg_tri(loc,edge,edge_fpt,order,loc_upts_pri_1d);
      else
        div_vcjh_basis = 0.;
    }
  // face 4
  else if (in_index < n_fpts_per_inter(0)+n_fpts_per_inter(1)+n_fpts_per_inter(2)+n_fpts_per_inter(3)+n_fpts_per_inter(4))
    {
      int face_fpt = (in_index-2*n_fpts_per_inter(0)-n_fpts_per_inter(2)-n_fpts_per_inter(3));
      int edge_fpt = face_fpt%(order+1);
      int edge = 2;

      if (face_fpt/(order+1) == upt_1d)
        div_vcjh_basis = eval_div_dg_tri(loc,edge,edge_fpt,order,loc_upts_pri_1d);
      else
        div_vcjh_basis = 0.;
    }

  return div_vcjh_basis;
}

int eles_pris::face0_map(int index)
{

  int k;
  for(int j=0;j<(order+1);j++)
    {
      for (int i=0;i<(order+1)-j;i++)
        {
          k= j*(order+1) -(j-1)*j/2+i;
          if (k==index)
            {
              return (i*(order+1) - (i-1)*i/2+j);
            }
        }
    }
  cout << "Should not be here in face0_map, exiting" << endl;
  exit(1);
}


/*! Calculate element volume */
double eles_pris::calc_ele_vol(double& detjac)
{
  double vol;
  // Element volume = |Jacobian|*width*height of reference element
  vol = detjac*4.;
  return vol;

}

/*! Calculate element reference length for timestep calculation */
double eles_pris::calc_h_ref_specific(int in_ele)
{
    double a,b,c,d,s;
    double out_h_ref;

    // Compute edge lengths (Counter-clockwise)
    for (int i=0; i<3; i++)
        length(i) = sqrt(pow(shape(0,i,in_ele) - shape(0,i+3,in_ele),2.0) + pow(shape(1,i,in_ele) - shape(1,i+3,in_ele),2.0)+pow(shape(2,i,in_ele) - shape(2,i+3,in_ele),2.0));
    for (int i=3; i<5; i++)
    {
        d=(i-3)*3;
        a = sqrt(pow(shape(0,d,in_ele) - shape(0,d+1,in_ele),2.0) + pow(shape(1,d,in_ele) - shape(1,d+1,in_ele),2.0)+pow(shape(2,d,in_ele) - shape(2,d+1,in_ele),2.0));
        b = sqrt(pow(shape(0,d+1,in_ele) - shape(0,d+2,in_ele),2.0) + pow(shape(1,d+1,in_ele) - shape(1,d+2,in_ele),2.0)+pow(shape(2,d+1,in_ele) - shape(2,d+2,in_ele),2.0));
        c = sqrt(pow(shape(0,d+2,in_ele) - shape(0,d,in_ele),2.0) + pow(shape(1,d+2,in_ele) - shape(1,d,in_ele),2.0)+pow(shape(2,d+2,in_ele) - shape(2,d,in_ele),2.0));
        s = 0.5*(a+b+c);
        length(i) = 2*sqrt(((s-a)*(s-b)*(s-c))/s);

    }
    // Get minimum edge length
    out_h_ref = length.get_min();

    return out_h_ref;
}

int eles_pris::calc_p2c(hf_array<double>& in_pos)
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

        int num_f_per_c = 5;

        for(int j=0; j<num_f_per_c; j++)//for each face
        {
            //store local vertex index
            if(j==0)
            {
                vertex_index_loc(0) = 0;
                vertex_index_loc(1) = 2;
                vertex_index_loc(2) = 1;
            }
            else if(j==1)
            {
                vertex_index_loc(0) = 3;
                vertex_index_loc(1) = 4;
                vertex_index_loc(2) = 5;

            }
            else if(j==2)
            {
                vertex_index_loc(0) = 0;
                vertex_index_loc(1) = 1;
                vertex_index_loc(2) = 4;
            }
            else if(j==3)
            {
                vertex_index_loc(0) = 1;
                vertex_index_loc(1) = 2;
                vertex_index_loc(2) = 5;
            }
            else if(j==4)
            {
                vertex_index_loc(0) = 2;
                vertex_index_loc(1) = 0;
                vertex_index_loc(2) = 3;
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

/*!
 * \file eles_quads.cpp
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

#include <iomanip>
#include <iostream>
#include <cmath>

#include "../include/global.h"
#include "../include/eles.h"
#include "../include/eles_quads.h"
#include "../include/hf_array.h"
#include "../include/funcs.h"
#include "../include/cubature_1d.h"
#include "../include/cubature_quad.h"

using namespace std;

// #### constructors ####

// default constructor

eles_quads::eles_quads()
{

}

// #### methods ####

void eles_quads::setup_ele_type_specific()
{

#ifndef _MPI
  cout << "Initializing quads" << endl;
#endif

  ele_type=1;
  n_dims=2;

  if (run_input.equation==0)
    n_fields=4;
  else if (run_input.equation==1)
    n_fields=1;
  else
    FatalError("Equation not supported");

  if (run_input.RANS==1)
    n_fields++;

  n_inters_per_ele=4;
  length.setup(4);

  n_upts_per_ele=(order+1)*(order+1);
  upts_type=run_input.upts_type_quad;
  set_loc_1d_upts();
  set_loc_upts();
  set_vandermonde1D();
  set_vandermonde2D();

  //set shock capturing arrays
  if(run_input.shock_cap)
  {
    if (run_input.shock_det == 0)//persson
      calc_norm_basis();
    else//concentration
      set_concentration_array();
    if (run_input.shock_cap == 1)//exp filter
      set_exp_filter();
  }

  n_ppts_per_ele=p_res*p_res;
  n_peles_per_ele=(p_res-1)*(p_res-1);
  n_verts_per_ele = 4;

  set_loc_ppts();
  set_opp_p();

  set_inters_cubpts();
  set_volume_cubpts();
  set_opp_volume_cubpts();
  set_vandermonde_vol_cub();

  //de-aliasing by over-integration
  if (run_input.over_int)
    set_over_int_filter();

  n_fpts_per_inter.setup(4);

  n_fpts_per_inter(0)=(order+1);
  n_fpts_per_inter(1)=(order+1);
  n_fpts_per_inter(2)=(order+1);
  n_fpts_per_inter(3)=(order+1);

  n_fpts_per_ele=n_inters_per_ele*(order+1);

  set_tloc_fpts();

  set_tnorm_fpts();

  set_opp_0(run_input.sparse_quad);
  set_opp_1(run_input.sparse_quad);
  set_opp_2(run_input.sparse_quad);
  set_opp_3(run_input.sparse_quad);

  if(viscous)
    {
      set_opp_4(run_input.sparse_quad);
      set_opp_5(run_input.sparse_quad);
      set_opp_6(run_input.sparse_quad);

      temp_grad_u.setup(n_fields,n_dims);

      // Compute quad filter matrix
      if(LES_filter) compute_filter_upts();
    }

  temp_u.setup(n_fields);
  temp_f.setup(n_fields,n_dims);
}

void eles_quads::set_connectivity_plot()
{
  int vertex_0,vertex_1,vertex_2,vertex_3;
  int count=0;

  for(int k=0;k<p_res-1;++k){
      for(int l=0;l<p_res-1;++l){

          vertex_0=l+(p_res*k);
          vertex_1=vertex_0+1;
          vertex_2=vertex_0+p_res+1;
          vertex_3=vertex_0+p_res;

          connectivity_plot(0,count) = vertex_0;
          connectivity_plot(1,count) = vertex_1;
          connectivity_plot(2,count) = vertex_2;
          connectivity_plot(3,count) = vertex_3;
          count++;
        }
    }
}

// set location of 1d solution points in standard interval (required for tensor product elements)

void eles_quads::set_loc_1d_upts(void)
{
  cubature_1d cub_1d(upts_type, order);
  loc_1d_upts.setup(order + 1);
  for (int i = 0; i < order + 1; i++)
    loc_1d_upts(i) = cub_1d.get_r(i);
}

// set location of 1d shape points in standard interval (required for tensor product element)

void eles_quads::set_loc_1d_spts(hf_array<double> &loc_1d_spts, int in_n_1d_spts)
{
  int i;

  for(i=0;i<in_n_1d_spts;i++)
    {
      loc_1d_spts(i)=-1.0+((2.0*i)/(1.0*(in_n_1d_spts-1)));
    }
}




// set location of solution points in standard element

void eles_quads::set_loc_upts(void)
{
  int i,j;

  int upt;

  loc_upts.setup(n_dims,n_upts_per_ele);

  for(i=0;i<(order+1);i++)
    {
      for(j=0;j<(order+1);j++)
        {
          upt=j+((order+1)*i);

          loc_upts(0,upt)=loc_1d_upts(j);
          loc_upts(1,upt)=loc_1d_upts(i);
        }
    }
}

// set location of flux points in standard element

void eles_quads::set_tloc_fpts(void)
{
  int i,j;

  int fpt;

  tloc_fpts.setup(n_dims,n_fpts_per_ele);

  for(i=0;i<n_inters_per_ele;i++)
    {
      for(j=0;j<(order+1);j++)
        {
          fpt=j+((order+1)*i);

          // for tensor product elements flux point location depends on solution point location

          if(i==0)
            {
              tloc_fpts(0,fpt)=loc_1d_upts(j);
              tloc_fpts(1,fpt)=-1.0;
            }
          else if(i==1)
            {
              tloc_fpts(0,fpt)=1.0;
              tloc_fpts(1,fpt)=loc_1d_upts(j);
            }
          else if(i==2)
            {
              tloc_fpts(0,fpt)=loc_1d_upts(order-j);
              tloc_fpts(1,fpt)=1.0;
            }
          else if(i==3)
            {
              tloc_fpts(0,fpt)=-1.0;
              tloc_fpts(1,fpt)=loc_1d_upts(order-j);
            }
        }
    }
}

// set location and weights of interface cubature points in standard element

void eles_quads::set_inters_cubpts(void)
{

  n_cubpts_per_inter.setup(n_inters_per_ele);
  loc_inters_cubpts.setup(n_inters_per_ele);
  weight_inters_cubpts.setup(n_inters_per_ele);
  tnorm_inters_cubpts.setup(n_inters_per_ele);

  cubature_1d cub_1d(0,order);
  int n_cubpts_1d = cub_1d.get_n_pts();

  for (int i=0;i<n_inters_per_ele;i++)
    n_cubpts_per_inter(i) = n_cubpts_1d;

  for (int i=0;i<n_inters_per_ele;i++) {

      loc_inters_cubpts(i).setup(n_dims,n_cubpts_per_inter(i));
      weight_inters_cubpts(i).setup(n_cubpts_per_inter(i));
      tnorm_inters_cubpts(i).setup(n_dims,n_cubpts_per_inter(i));

      for (int j=0;j<n_cubpts_1d;j++) {

          if (i==0) {
              loc_inters_cubpts(i)(0,j)=cub_1d.get_r(j);
              loc_inters_cubpts(i)(1,j)=-1.;
            }
          else if (i==1) {
              loc_inters_cubpts(i)(0,j)=1.;
              loc_inters_cubpts(i)(1,j)=cub_1d.get_r(j);
            }
          else if (i==2) {
              loc_inters_cubpts(i)(0,j)=cub_1d.get_r(n_cubpts_1d-j-1);
              loc_inters_cubpts(i)(1,j)=1.0;
            }
          else if (i==3) {
              loc_inters_cubpts(i)(0,j)=-1.0;
              loc_inters_cubpts(i)(1,j)=cub_1d.get_r(n_cubpts_1d-j-1);
            }

          weight_inters_cubpts(i)(j) = cub_1d.get_weight(j);

          if (i==0) {
              tnorm_inters_cubpts(i)(0,j)= 0.;
              tnorm_inters_cubpts(i)(1,j)= -1.;
            }
          else if (i==1) {
              tnorm_inters_cubpts(i)(0,j)= 1.;
              tnorm_inters_cubpts(i)(1,j)= 0.;
            }
          else if (i==2) {
              tnorm_inters_cubpts(i)(0,j)= 0.;
              tnorm_inters_cubpts(i)(1,j)= 1.;
            }
          else if (i==3) {
              tnorm_inters_cubpts(i)(0,j)= -1.;
              tnorm_inters_cubpts(i)(1,j)= 0.;
            }

        }
    }

  set_opp_inters_cubpts();


}

void eles_quads::set_volume_cubpts(void)
{
  cubature_quad cub_quad(0,order);
  int n_cubpts_quad = cub_quad.get_n_pts();
  n_cubpts_per_ele = n_cubpts_quad;
  loc_volume_cubpts.setup(n_dims,n_cubpts_quad);
  weight_volume_cubpts.setup(n_cubpts_quad);

  for (int i=0;i<n_cubpts_quad;i++)
    {
      loc_volume_cubpts(0,i) = cub_quad.get_r(i);
      loc_volume_cubpts(1,i) = cub_quad.get_s(i);
      //cout << "x=" << loc_volume_cubpts(0,i) << endl;
      //cout << "y=" << loc_volume_cubpts(1,i) << endl;
      weight_volume_cubpts(i) = cub_quad.get_weight(i);
      //cout<<"wgt=" << weight_volume_cubpts(i) << endl;
    }
}

// Compute the surface jacobian determinant on a face
double eles_quads::compute_inter_detjac_inters_cubpts(int in_inter,hf_array<double> d_pos)
{
  double output = 0.;
  double xr, xs;
  double yr, ys;

  xr = d_pos(0,0);
  xs = d_pos(0,1);

  yr = d_pos(1,0);
  ys = d_pos(1,1);

  if (in_inter==0)
    {
      output = sqrt(xr*xr+yr*yr);
    }
  else if (in_inter==1)
    {
      output = sqrt(xs*xs+ys*ys);
    }
  else if (in_inter==2)
    {
      output = sqrt(xr*xr+yr*yr);
    }
  else if (in_inter==3)
    {
      output = sqrt(xs*xs+ys*ys);
    }

  return output;
}

// set location of plot points in standard element

void eles_quads::set_loc_ppts(void)
{
  int i,j;

  int ppt;

  loc_ppts.setup(n_dims,n_ppts_per_ele);

  for(j=0;j<p_res;j++)
    {
      for(i=0;i<p_res;i++)
        {
          ppt=i+(p_res*j);

          loc_ppts(0,ppt)=-1.0+((2.0*i)/(1.0*(p_res-1)));
          loc_ppts(1,ppt)=-1.0+((2.0*j)/(1.0*(p_res-1)));
        }
    }
}

// set transformed normal at flux points

void eles_quads::set_tnorm_fpts(void)
{
  int i,j;

  int fpt;

  tnorm_fpts.setup(n_dims,n_fpts_per_ele);

  for(i=0;i<n_inters_per_ele;i++)
    {
      for(j=0;j<(order+1);j++)
        {
          fpt=j+((order+1)*i);

          if(i==0)
            {
              tnorm_fpts(0,fpt)=0.0;
              tnorm_fpts(1,fpt)=-1.0;
            }
          else if(i==1)
            {
              tnorm_fpts(0,fpt)=1.0;
              tnorm_fpts(1,fpt)=0.0;
            }
          else if(i==2)
            {
              tnorm_fpts(0,fpt)=0.0;
              tnorm_fpts(1,fpt)=1.0;
            }
          else if(i==3)
            {
              tnorm_fpts(0,fpt)=-1.0;
              tnorm_fpts(1,fpt)=0.0;
            }
        }
    }
}

// Filtering operators for use in subgrid-scale modelling
void eles_quads::compute_filter_upts(void)
{
  int i,j,k,l,N,N2;
  double dlt, k_c, sum, norm;
  N = order+1; // order is of basis polynomials NOT truncation error!

  hf_array<double> X(N),B(N);
  hf_array<double> beta(N,N);

  filter_upts_1D.setup(N,N);

  X = loc_1d_upts;

  N2 = N/2;
  if(N % 2 != 0){N2 += 1;}
  // Cutoff wavenumber
  k_c = 1.0/run_input.filter_ratio;
  // Approx resolution in element (assumes uniform point spacing)
  // Interval is [-1:1]
  dlt = 2.0/order;

  // Normalised solution point separation
  for (i=0;i<N;i++)
    for (j=0;j<N;j++)
      beta(j,i) = (X(j)-X(i))/dlt;

  // Build high-order-commuting Vasilyev filter
  // Only use high-order filters for high enough order
  if(run_input.filter_type==0 and N>=3)
    {
      if (rank==0) cout<<"Building high-order-commuting Vasilyev filter"<<endl;
      hf_array<double> C(N);
      hf_array<double> A(N,N);

      for (i=0;i<N;i++)
        {
          B(i) = 0.0;
          C(i) = 0.0;
          for (j=0;j<N;j++)
            A(i,j) = 0.0;

        }
      // Populate coefficient matrix
      for (i=0;i<N;i++)
        {
          // Populate constraints matrix
          B(0) = 1.0;
          // Gauss filter weights
          B(1) =  exp(-pow(pi,2)/24.0);
          B(2) = -B(1)*pow(pi,2)/k_c/12.0;

          if(N % 2 == 1 && i+1 == N2)
            B(2) = 0.0;

          for (j=0;j<N;j++)
            {
              A(j,0) = 1.0;
              A(j,1) = cos(pi*k_c*beta(j,i));
              A(j,2) = -beta(j,i)*pi*sin(pi*k_c*beta(j,i));

              if(N % 2 == 1 && i+1 == N2)
                A(j,2) = pow(beta(j,i),3);

            }

          // Enforce filter moments
          for (k=3;k<N;k++)
            {
              for (j=0;j<N;j++)
                A(j,k) = pow(beta(j,i),k+1);

              B(k) = 0.0;
            }

          // Solve linear system by inverting A using
          // Gauss-Jordan method
          gaussj(N,A,B);
          sum=0;
          for (j=0;j<N;j++)
            filter_upts_1D(j,i) = B(j);

        }
    }
  else if(run_input.filter_type==1) // Discrete Gaussian filter
    {
      if (rank==0) cout<<"Building discrete Gaussian filter"<<endl;
      int ctype,index;
      double k_R, k_L, coeff;
      double res_0, res_L, res_R;
      hf_array<double> alpha(N);
      cubature_1d cub_1d(0,order);
      int n_cubpts_1d = cub_1d.get_n_pts();
      hf_array<double> wf(n_cubpts_1d);
      for (j=0;j<n_cubpts_1d;++j)
        wf(j) = cub_1d.get_weight(j);

      // Determine corrected filter width for skewed quadrature points
      // using iterative constraining procedure
      // ctype options: (-1) no constraining, (0) constrain moment, (1) constrain cutoff frequency
      ctype = -1;
      if(ctype>=0)
        {
          for (i=0;i<N2;i++)
            {
              for (j=0;j<N;j++)
                B(j) = beta(j,i);

              k_L = 0.1; k_R = 1.0;
              res_L = flt_res(N,wf,B,k_L,k_c,ctype);
              res_R = flt_res(N,wf,B,k_R,k_c,ctype);
              alpha(i) = 0.5*(k_L+k_R);
              for (j=0;j<1000;j++)
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
        for (i=0;i<N;i++)
          alpha(i) = k_c;

      sum = 0.0;
      for (i=0;i<N;i++)
        {
          norm = 0.0;
          for (j=0;j<N;j++)
            {
              filter_upts_1D(i,j) = wf(j)*exp(-6.0*pow(alpha(i)*beta(i,j),2));
              norm += filter_upts_1D(i,j);
            }
          for (j=0;j<N;j++)
            {
              filter_upts_1D(i,j) /= norm;
              sum += filter_upts_1D(i,j);
            }
        }
    }
  else if(run_input.filter_type==2) // Modal coefficient filter
    {
      if (rank==0) cout<<"Building modal filter"<<endl;

      // Compute modal filter
      compute_modal_filter_1d(filter_upts_1D, vandermonde1D, inv_vandermonde1D, N, order);

      sum = 0;
      for(i=0;i<N;i++)
        for(j=0;j<N;j++)
          sum+=filter_upts_1D(i,j);
    }
  else // Simple average
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

  // Build 2D filter on ideal (reference) element.
  int ii=0;
  filter_upts.setup(n_upts_per_ele,n_upts_per_ele);
  sum=0;
  for (i=0;i<N;i++)
    {
      for (j=0;j<N;j++)
        {
          int jj=0;
          for (k=0;k<N;k++)
            {
              for (l=0;l<N;l++)
                {
                  filter_upts(ii,jj) = filter_upts_1D(j,l)*filter_upts_1D(i,k);
                  sum+=filter_upts(ii,jj);
                  ++jj;
                }
            }
          ++ii;
        }
    }
}


//#### helper methods ####

int eles_quads::read_restart_info_ascii(ifstream& restart_file)
{

  string str;
  // Move to triangle element
  while(1) {
      getline(restart_file,str);
      if (str=="QUADS") break;

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

  for (int i=0;i<order_rest+1;i++)
    restart_file >> loc_1d_upts_rest(i);

  set_opp_r();

  return 1;
}

#ifdef _HDF5
void eles_quads::read_restart_info_hdf5(hid_t &restart_file, int in_rest_order)
{
  hid_t dataset_id, plist_id;

  //open dataset
  dataset_id = H5Dopen2(restart_file, "QUADS", H5P_DEFAULT);
  if (dataset_id < 0)
    FatalError("Cannot find quads property");

  plist_id = H5Pcreate(H5P_DATASET_XFER);
#ifdef _MPI
  //set collective read
  H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);
#endif

  if (n_eles)
  {
    order_rest = in_rest_order;
    n_upts_per_ele_rest = (order_rest + 1) * (order_rest + 1);
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

void eles_quads::write_restart_info_ascii(ofstream& restart_file)
{
  restart_file << "QUADS" << endl;

  restart_file << "Order" << endl;
  restart_file << order << endl;

  restart_file << "Number of solution points per quadrilateral element" << endl;
  restart_file << n_upts_per_ele << endl;

  restart_file << "Location of solution points in 1D" << endl;
  for (int i=0;i<order+1;i++) {
      restart_file << loc_1d_upts(i) << " ";
    }
  restart_file << endl;

}

#ifdef _HDF5
void eles_quads::write_restart_info_hdf5(hid_t &restart_file)
{
  hid_t dataset_id,plist_id,dataspace_id;
  hsize_t dim = run_input.order + 1;

  //create QUADS dataset
  dataspace_id=H5Screate_simple(1,&dim,NULL);
  dataset_id=H5Dcreate2(restart_file,"QUADS",H5T_NATIVE_DOUBLE,dataspace_id,H5P_DEFAULT,H5P_DEFAULT,H5P_DEFAULT);
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
void eles_quads::set_vandermonde1D(void)
{
  vandermonde1D.setup(order+1,order+1);

  for (int i=0;i<order+1;i++)
    for (int j=0;j<order+1;j++)
      vandermonde1D(i,j) = eval_legendre(loc_1d_upts(i),j);

  // Store its inverse
  inv_vandermonde1D = inv_array(vandermonde1D);
}

// Set the 2D inverse Vandermonde hf_array needed for shock capturing
void eles_quads::set_vandermonde2D(void)
{
  vandermonde.setup(n_upts_per_ele,n_upts_per_ele);
  hf_array <double> loc(n_dims);

  // create the vandermonde matrix
  for (int i=0;i<n_upts_per_ele;i++){
      loc(0) = loc_upts(0,i);
      loc(1) = loc_upts(1,i);

      for (int j=0;j<n_upts_per_ele;j++)
          vandermonde(i,j) = eval_legendre_basis_2D_hierarchical(j,loc,order);
  }

  // Store its inverse
  inv_vandermonde = inv_array(vandermonde);
}

void eles_quads::set_vandermonde_vol_cub(void)
{
  vandermonde_vol_cub.setup(n_cubpts_per_ele, n_cubpts_per_ele);
  hf_array<double> loc(n_dims);
  // create the temp vandermonde matrix at cub pts
  for (int i = 0; i < n_cubpts_per_ele; i++)
  {
    loc(0) = loc_volume_cubpts(0, i);
    loc(1) = loc_volume_cubpts(1, i);

    for (int j = 0; j < n_cubpts_per_ele; j++)
      vandermonde_vol_cub(i, j) = eval_legendre_basis_2D_hierarchical(j, loc, order);
  }

  // Store its inverse
  inv_vandermonde_vol_cub = inv_array(vandermonde_vol_cub);
}

void eles_quads::set_exp_filter(void)
{
  exp_filter.setup(n_upts_per_ele, n_upts_per_ele);
  exp_filter.initialize_to_zero();
  int i, j, k, mode;
  double eta;
  
  mode = 0;
  for (k = 0; k < 2 * order + 1; k++) //sum of x,y mode
  {
    for (j = 0; j < k + 1; j++) //j<=sum
    {
      i = k - j; //i+j=sum
      if (i <= order && j <= order)
      {
        eta = (double)(k) / (double)(2 * order);
        exp_filter(mode, mode) = exp(-run_input.expf_fac * pow(eta, run_input.expf_order));
        mode++;
      }
    }
  }

  exp_filter = mult_arrays(exp_filter, inv_vandermonde);
  exp_filter = mult_arrays(vandermonde, exp_filter);
}

void eles_quads::calc_norm_basis(void)
{
  int n1, n2;
  double norm1, norm2;
  norm_basis_persson.setup(n_upts_per_ele);
  for (int i = 0; i < n_upts_per_ele; i++)
  {
    get_legendre_basis_2D_index(i, order, n1, n2);
    norm1 = 2.0 / (2.0 * n1 + 1.0);
    norm2 = 2.0 / (2.0 * n2 + 1.0);
    norm_basis_persson(i) = norm1 * norm2;
  }
}

//detect shock use persson's method
void eles_quads::shock_det_persson(void)
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
  int x, y;
  for (int ic = 0; ic < n_eles; ic++)
  {
    sensor(ic) = 0;
    for (int j = 0; j < n_upts_per_ele; j++)
    {
      get_legendre_basis_2D_index(j, order, x, y);
      if (run_input.over_int)
      {
        if (x >= run_input.N_under || y >= run_input.N_under)
          sensor(ic) += temp_modal(j, ic) * norm_basis_persson(j);
      }
      else
      {
        if (x == order || y == order)
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

// Set the 1D concentration matrix based on 1D-loc_upts
void eles_quads::set_concentration_array()
{
  int concen_type = 1;
  hf_array<double> concentration_factor(order+1);
  hf_array<double> grad_vandermonde;
  grad_vandermonde.setup(order+1,order+1);
  concentration_array.setup((order+1)*(order+1));

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


    for (int i=0;i<order+1;i++)
                for (int j=0;j<order+1;j++)
                        concentration_array(j + i*(order+1)) = concentration_factor(j)*sqrt(1 - loc_1d_upts(i)*loc_1d_upts(i))*grad_vandermonde(i,j);
}

void eles_quads::set_over_int_filter(void)
{
  int N_under = run_input.N_under;
  int n_mode_under = (N_under + 1) * (N_under + 1); //projected n_upts_per_ele
  hf_array<double> temp_proj(n_mode_under, n_cubpts_per_ele);
  hf_array<double> temp_vand(n_upts_per_ele, n_mode_under);
  hf_array<double> loc(n_dims);

  //step 1. nodal to L2 projected modal \hat{u_i}=\int{\phi_i*l_j}=>\phi_i(j)*w(j)
  int n1, n2;
  double norm1, norm2;
  for (int i = 0; i < n_mode_under; i++)
  {
    get_legendre_basis_2D_index(i, N_under, n1, n2);
    norm1 = 2.0 / (2.0 * n1 + 1.0);
    norm2 = 2.0 / (2.0 * n2 + 1.0);
    for (int j = 0; j < n_cubpts_per_ele; j++)
    {
      loc(0) = loc_volume_cubpts(0,j);
      loc(1) = loc_volume_cubpts(1,j);
      temp_proj(i, j) = eval_legendre_basis_2D_hierarchical(i, loc, N_under) / (norm1 * norm2) * weight_volume_cubpts(j);
    }
  }
  //step 2. projected modal back to nodal to get filtered solution \tilde{u_j}=V_{ji}*\hat{u_i}
  for (int j = 0; j < n_upts_per_ele; j++)
  {
    loc(0) = loc_upts(0, j);
    loc(1) = loc_upts(1, j);
    for (int i = 0; i < n_mode_under; i++)
      temp_vand(j, i) = eval_legendre_basis_2D_hierarchical(i, loc, N_under);
  }
  over_int_filter = mult_arrays(temp_proj, opp_volume_cubpts);
  over_int_filter = mult_arrays(temp_vand, over_int_filter);
}

// evaluate nodal basis

double eles_quads::eval_nodal_basis(int in_index, hf_array<double> in_loc)
{
  int i,j;

  double nodal_basis;

  i=in_index/(order+1);
  j=in_index-((order+1)*i);

  nodal_basis=eval_lagrange(in_loc(0),j,loc_1d_upts)*eval_lagrange(in_loc(1),i,loc_1d_upts);

  return nodal_basis;
}


// evaluate nodal basis using restart points

double eles_quads::eval_nodal_basis_restart(int in_index, hf_array<double> in_loc)
{
  int i,j;

  double nodal_basis;

  i=in_index/(order_rest+1);
  j=in_index-((order_rest+1)*i);

  nodal_basis=eval_lagrange(in_loc(0),j,loc_1d_upts_rest)*eval_lagrange(in_loc(1),i,loc_1d_upts_rest);

  return nodal_basis;
}

// evaluate derivative of nodal basis

double eles_quads::eval_d_nodal_basis(int in_index, int in_cpnt, hf_array<double> in_loc)
{
  int i,j;

  double d_nodal_basis;

  i=in_index/(order+1);
  j=in_index-((order+1)*i);

  if(in_cpnt==0)
    {
      d_nodal_basis=eval_d_lagrange(in_loc(0),j,loc_1d_upts)*eval_lagrange(in_loc(1),i,loc_1d_upts);
    }
  else if(in_cpnt==1)
    {
      d_nodal_basis=eval_lagrange(in_loc(0),j,loc_1d_upts)*eval_d_lagrange(in_loc(1),i,loc_1d_upts);
    }
  else
    {
      cout << "ERROR: Invalid component requested ... " << endl;
    }

  return d_nodal_basis;
}

// evaluate nodal shape basis

double eles_quads::eval_nodal_s_basis(int in_index, hf_array<double> in_loc, int in_n_spts)
{
  int i,j;
  double nodal_s_basis;

  if (is_perfect_square(in_n_spts))
    {
      int n_1d_spts = round(sqrt(1.0*in_n_spts));
      hf_array<double> loc_1d_spts(n_1d_spts);
      set_loc_1d_spts(loc_1d_spts,n_1d_spts);

      j=in_index/n_1d_spts;
      i=in_index-(n_1d_spts*j);
      nodal_s_basis=eval_lagrange(in_loc(0),i,loc_1d_spts)*eval_lagrange(in_loc(1),j,loc_1d_spts);
    }
  else if (in_n_spts==8)
    {
      if (in_index==0)
        nodal_s_basis = -0.25*(1.-in_loc(0))*(1.-in_loc(1))*(1.+in_loc(0)+in_loc(1));
      else if (in_index==1)
        nodal_s_basis = -0.25*(1.+in_loc(0))*(1.-in_loc(1))*(1.-in_loc(0)+in_loc(1));
      else if (in_index==2)
        nodal_s_basis = -0.25*(1.+in_loc(0))*(1.+in_loc(1))*(1.-in_loc(0)-in_loc(1));
      else if (in_index==3)
        nodal_s_basis = -0.25*(1.-in_loc(0))*(1.+in_loc(1))*(1.+in_loc(0)-in_loc(1));
      else if (in_index==4)
        nodal_s_basis = 0.5*(1.-in_loc(0))*(1.+in_loc(0))*(1.-in_loc(1));
      else if (in_index==5)
        nodal_s_basis = 0.5*(1.+in_loc(0))*(1.+in_loc(1))*(1.-in_loc(1));
      else if (in_index==6)
        nodal_s_basis = 0.5*(1.-in_loc(0))*(1.+in_loc(0))*(1.+in_loc(1));
      else if (in_index==7)
        nodal_s_basis = 0.5*(1.-in_loc(0))*(1.+in_loc(1))*(1.-in_loc(1));
    }
  else
    {
      cout << "Shape basis not implemented yet, exiting" << endl;
      exit(1);
    }

  return nodal_s_basis;
}

// evaluate derivative of nodal shape basis

void eles_quads::eval_d_nodal_s_basis(hf_array<double> &d_nodal_s_basis, hf_array<double> in_loc, int in_n_spts)
{
  int i,j;

  if (is_perfect_square(in_n_spts))
    {
      int n_1d_spts = round(sqrt(1.0*in_n_spts));
      hf_array<double> loc_1d_spts(n_1d_spts);
      set_loc_1d_spts(loc_1d_spts,n_1d_spts);

      for (int k=0;k<in_n_spts;k++)
        {
          i=k/n_1d_spts;
          j=k-(n_1d_spts*i);

          d_nodal_s_basis(k,0)=eval_d_lagrange(in_loc(0),j,loc_1d_spts)*eval_lagrange(in_loc(1),i,loc_1d_spts);
          d_nodal_s_basis(k,1)=eval_lagrange(in_loc(0),j,loc_1d_spts)*eval_d_lagrange(in_loc(1),i,loc_1d_spts);
        }
    }
  else if (in_n_spts==8)
    {
      d_nodal_s_basis(0,0) = -0.25*(-1.+in_loc(1))*(2.*in_loc(0)+in_loc(1));
      d_nodal_s_basis(1,0) = 0.25*(-1.+in_loc(1))*(in_loc(1)-2.*in_loc(0));
      d_nodal_s_basis(2,0)	 = 0.25*(1.+in_loc(1))*(2.*in_loc(0)+in_loc(1));
      d_nodal_s_basis(3,0)	 = -0.25*(1.+in_loc(1))*(in_loc(1)-2.*in_loc(0));
      d_nodal_s_basis(4,0) = in_loc(0)*(-1.+in_loc(1));
      d_nodal_s_basis(5,0) = -0.5*(1.+in_loc(1))*(-1.+in_loc(1));
      d_nodal_s_basis(6,0) = -in_loc(0)*(1.+in_loc(1));
      d_nodal_s_basis(7,0) = 0.5*(1.+in_loc(1))*(-1.+in_loc(1));


      d_nodal_s_basis(0,1) = -0.25*(-1.+in_loc(0))*(in_loc(0)+ 2.*in_loc(1));
      d_nodal_s_basis(1,1) = 0.25*(1.+in_loc(0))*(2.*in_loc(1)-in_loc(0));
      d_nodal_s_basis(2,1) = 0.25*(1.+in_loc(0))*(in_loc(0)+2.*in_loc(1));
      d_nodal_s_basis(3,1) = -0.25*(-1.+in_loc(0))*(2.*in_loc(1)-in_loc(0));
      d_nodal_s_basis(4,1) = 0.5*(1.+in_loc(0))*(-1.+in_loc(0));
      d_nodal_s_basis(5,1) = -in_loc(1)*(1.+in_loc(0));
      d_nodal_s_basis(6,1) = -0.5*(1.+in_loc(0))*(-1.+in_loc(0));
      d_nodal_s_basis(7,1) = in_loc(1)*(-1.+in_loc(0));
    }
  else
    {
      cout << "Shape basis not implemented yet, exiting" << endl;
      exit(1);
    }

}

// Evaluate 2D legendre basis
double eles_quads::eval_legendre_basis_2D_hierarchical(int in_mode, hf_array<double> in_loc, int in_basis_order)
{
        double leg_basis;

        int n_dof=(in_basis_order+1)*(in_basis_order+1);

        if(in_mode<n_dof)
          {
            int i,j,k;
            int mode;

            mode=0;
             //mode=j+i*(N_x+1)
        for (k=0; k<2*in_basis_order+1; k++) //sum of x,y mode range from 0 to 2*order
        {
            for (j=0; j<k+1; j++) //j no more than the sum
            {
                i=k-j;//i+j=k,i>=0

                if (i<=in_basis_order && j<=in_basis_order)//i and j both no more than order to be valid
                {
                    if(mode==in_mode) // found the correct mode
                    {
                        leg_basis=eval_legendre(in_loc(0),i)*eval_legendre(in_loc(1),j);
                        return leg_basis;
                    }

                    mode++;
                }
            }
        }
          }
        else
          {
            cout << "ERROR: Invalid mode when evaluating Legendre basis ...." << endl;
          }

FatalError("No mode is found");
}

int eles_quads::get_legendre_basis_2D_index(int in_mode, int in_basis_order, int &out_i, int &out_j)
{
  int n_dof = (in_basis_order + 1) * (in_basis_order + 1);
  if (in_mode < n_dof)
  {
    int i, j, k;
    int mode;

    mode = 0;
    //mode=j+i*(N_x+1)
    for (k = 0; k < 2 * in_basis_order + 1; k++) //sum of x,y mode range from 0 to 2*order
    {
      for (j = 0; j < k + 1; j++) //j no more than the sum
      {
        i = k - j; //i+j=k,i>=0

        if (i <= in_basis_order && j <= in_basis_order) //i and j both no more than order to be valid
        {
          if (mode == in_mode) // found the correct mode
          {
            out_i = i;
            out_j = j;
            return 0;
          }
          mode++;
        }
      }
    }
  }
  else
  {
    cout << "ERROR: Invalid mode when evaluating Legendre basis ...." << endl;
  }
return -1;
}

void eles_quads::fill_opp_3(hf_array<double>& opp_3)
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

          opp_3(j,i)=eval_div_vcjh_basis(i,loc);
        }
    }
}

// evaluate divergence of vcjh basis

double eles_quads::eval_div_vcjh_basis(int in_index, hf_array<double>& loc)
{
  int i,j;
  double eta;
  double div_vcjh_basis;
  int scheme = run_input.vcjh_scheme_quad;

  if (scheme==0)
    eta = run_input.eta_quad;
  else if (scheme < 5)
    eta = compute_eta(run_input.vcjh_scheme_quad,order);

  i=in_index/n_fpts_per_inter(0);
  j=in_index-(n_fpts_per_inter(0)*i);

  if (scheme < 5) {

    if(i==0)
      div_vcjh_basis = -eval_lagrange(loc(0),j,loc_1d_upts) * eval_d_vcjh_1d(loc(1),0,order,eta);
    else if(i==1)
      div_vcjh_basis = eval_lagrange(loc(1),j,loc_1d_upts) * eval_d_vcjh_1d(loc(0),1,order,eta);
    else if(i==2)
      div_vcjh_basis = eval_lagrange(loc(0),order-j,loc_1d_upts) * eval_d_vcjh_1d(loc(1),1,order,eta);
    else if(i==3)
      div_vcjh_basis = -eval_lagrange(loc(1),order-j,loc_1d_upts) * eval_d_vcjh_1d(loc(0),0,order,eta);

  }
  // OFR scheme
  else if (scheme == 5) {

    if(i==0)
      div_vcjh_basis = -eval_lagrange(loc(0),j,loc_1d_upts) * eval_d_ofr_1d(loc(1),0,order);
    else if(i==1)
      div_vcjh_basis = eval_lagrange(loc(1),j,loc_1d_upts) * eval_d_ofr_1d(loc(0),1,order);
    else if(i==2)
      div_vcjh_basis = eval_lagrange(loc(0),order-j,loc_1d_upts) * eval_d_ofr_1d(loc(1),1,order);
    else if(i==3)
      div_vcjh_basis = -eval_lagrange(loc(1),order-j,loc_1d_upts) * eval_d_ofr_1d(loc(0),0,order);

  }
  // OESFR scheme
  else if (scheme == 6) {

    if(i==0)
      div_vcjh_basis = -eval_lagrange(loc(0),j,loc_1d_upts) * eval_d_oesfr_1d(loc(1),0,order);
    else if(i==1)
      div_vcjh_basis = eval_lagrange(loc(1),j,loc_1d_upts) * eval_d_oesfr_1d(loc(0),1,order);
    else if(i==2)
      div_vcjh_basis = eval_lagrange(loc(0),order-j,loc_1d_upts) * eval_d_oesfr_1d(loc(1),1,order);
    else if(i==3)
      div_vcjh_basis = -eval_lagrange(loc(1),order-j,loc_1d_upts) * eval_d_oesfr_1d(loc(0),0,order);

  }

  return div_vcjh_basis;
}

// Get position of 1d solution point
double eles_quads::get_loc_1d_upt(int in_index)
{
  return loc_1d_upts(in_index);
}

/*! Calculate element volume */
double eles_quads::calc_ele_vol(double& detjac)
{
  double vol;
  // Element volume = |Jacobian|*width*height of reference element
  vol = detjac*4.;
  return vol;
}

/*! Calculate element reference length for timestep calculation */
double eles_quads::calc_h_ref_specific(int in_ele)
  {
    double out_h_ref;

    // Compute edge lengths (Counter-clockwise)
    length(0) = sqrt(pow(shape(0,0,in_ele) - shape(0,1,in_ele),2.0) + pow(shape(1,0,in_ele) - shape(1,1,in_ele),2.0));
    length(1) = sqrt(pow(shape(0,1,in_ele) - shape(0,3,in_ele),2.0) + pow(shape(1,1,in_ele) - shape(1,3,in_ele),2.0));
    length(2) = sqrt(pow(shape(0,3,in_ele) - shape(0,2,in_ele),2.0) + pow(shape(1,3,in_ele) - shape(1,2,in_ele),2.0));
    length(3) = sqrt(pow(shape(0,2,in_ele) - shape(0,0,in_ele),2.0) + pow(shape(1,2,in_ele) - shape(1,0,in_ele),2.0));

    // Get minimum edge length
    out_h_ref = length.get_min();

    return out_h_ref;
  }

int eles_quads::calc_p2c(hf_array<double>& in_pos)
{
    hf_array<double> line_coeff;
    hf_array<double> pos_centroid;
    hf_array<int> vertex_index_loc(2);
    hf_array<double> pos_line_pts(n_dims,2);

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

        for (int j = 0; j < num_f_per_c; j++) //for each face
        {
          if (is_perfect_square(n_spts_per_ele(i)))
          {
            int n_spts_1d = 2;
            if (j == 0)
            {
              vertex_index_loc(0) = 0;
              vertex_index_loc(1) = n_spts_1d - 1;
            }
            else if (j == 1)
            {
              vertex_index_loc(0) = n_spts_1d - 1;
              vertex_index_loc(1) = n_spts_per_ele(i) - 1;
            }
            else if (j == 2)
            {
              vertex_index_loc(0) = n_spts_per_ele(i) - 1;
              vertex_index_loc(1) = n_spts_per_ele(i) - n_spts_1d;
            }
            else if (j == 3)
            {
              vertex_index_loc(0) = n_spts_per_ele(i) - n_spts_1d;
              vertex_index_loc(1) = 0;
            }
          }
          else if (n_spts_per_ele(i) == 8)
          {
            if (j == 0)
            {
              vertex_index_loc(0) = 0;
              vertex_index_loc(1) = 1;
            }
            else if (j == 1)
            {
              vertex_index_loc(0) = 1;
              vertex_index_loc(1) = 2;
            }
            else if (j == 2)
            {
              vertex_index_loc(0) = 2;
              vertex_index_loc(1) = 3;
            }
            else if (j == 3)
            {
              vertex_index_loc(0) = 3;
              vertex_index_loc(1) = 0;
            }
          }
          else
          {
            FatalError("cell type not implemented");
          }

          //store position of vertex on each face
          for (int k = 0; k < 2; k++)//number of points needed to define a line
            for (int l = 0; l < n_dims; l++) //dims
              pos_line_pts(l, k) = shape(l, vertex_index_loc(k), i);

          //calculate plane coeff
          line_coeff = calc_line(pos_line_pts);

          alpha = alpha * ((line_coeff(0) * in_pos(0) + line_coeff(1) * in_pos(1) + line_coeff(2)) *
                               (line_coeff(0) * pos_centroid(0) + line_coeff(1) * pos_centroid(1) + line_coeff(2)) >= 0);
          if (alpha == 0)
            break;
        }

        if (alpha>0)
            return i;
    }

    return -1;
}

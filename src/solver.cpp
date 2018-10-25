/*!
 * \file solver.cpp
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
#include <sstream>
#include <cmath>

#include "../include/global.h"
#include "../include/hf_array.h"
#include "../include/input.h"
#include "../include/geometry.h"
#include "../include/solver.h"
#include "../include/output.h"
#include "../include/funcs.h"
#include "../include/error.h"
#include "../include/solution.h"

#ifdef _TECIO
#include "TECIO.h"
#endif

#ifdef _MPI
#include "mpi.h"
#include "metis.h"
#include "parmetis.h"
#endif

#ifdef _GPU
#include "../include/util.h"
#endif

using namespace std;

void CalcResidual(int in_file_num, int in_rk_stage, struct solution* FlowSol) {

  int i;                            /*!< Loop iterator */

  /*! If at first RK step and using certain LES models, compute some model-related quantities. */
  if (run_input.LES == 1 && in_rk_stage == 0)
  {
    if (run_input.SGS_model == 2 || run_input.SGS_model == 3 || run_input.SGS_model == 4)
    { //similarity and svv
      for (i = 0; i < FlowSol->n_ele_types; i++)
        FlowSol->mesh_eles(i)->calc_sgs_terms();
    }
  }

    /*! Compute the solution at the flux points. */
    for(i=0; i<FlowSol->n_ele_types; i++)
      FlowSol->mesh_eles(i)->extrapolate_solution();

#ifdef _MPI
  /*! Send the solution at the flux points across the MPI interfaces. */
  if (FlowSol->nproc>1)
    for(i=0; i<FlowSol->n_mpi_inter_types; i++)
      FlowSol->mesh_mpi_inters(i).send_solution();
#endif

  if (FlowSol->viscous) {
      /*! Compute the uncorrected transformed gradient of the solution at the solution points. */
      for(i=0; i<FlowSol->n_ele_types; i++)
        FlowSol->mesh_eles(i)->calculate_gradient();
    }

  /*! Compute the transformed inviscid flux at the solution points and store in total transformed flux storage. */
  for(i=0; i<FlowSol->n_ele_types; i++)
    FlowSol->mesh_eles(i)->evaluate_invFlux();


  // If running periodic channel or periodic hill cases,
  // calculate body forcing and add to source term
  if(run_input.forcing==1 and in_rk_stage==0 and run_input.equation==0 and FlowSol->n_dims==3)
  {
#ifdef _GPU
    // copy disu_upts for body force calculation
    for(i=0; i<FlowSol->n_ele_types; i++)
      FlowSol->mesh_eles(i)->cp_disu_upts_gpu_cpu();
#endif

    for(i=0;i<FlowSol->n_ele_types;i++)
    {
      FlowSol->mesh_eles(i)->evaluate_body_force(in_file_num);
    }
  }

  /*! Compute the transformed normal inviscid numerical fluxes.
   Compute the common solution and solution corrections (viscous only). */
  for(i=0; i<FlowSol->n_int_inter_types; i++)
    FlowSol->mesh_int_inters(i).calculate_common_invFlux();

  for(i=0; i<FlowSol->n_bdy_inter_types; i++)
    FlowSol->mesh_bdy_inters(i).evaluate_boundaryConditions_invFlux(FlowSol->time);//TODO:use RK_time instead

#ifdef _MPI
  /*! Send the previously computed values across the MPI interfaces. */
  if (FlowSol->nproc>1) {
      for(i=0; i<FlowSol->n_mpi_inter_types; i++)
        FlowSol->mesh_mpi_inters(i).receive_solution();

      for(i=0; i<FlowSol->n_mpi_inter_types; i++)
        FlowSol->mesh_mpi_inters(i).calculate_common_invFlux();
    }
#endif

    if (FlowSol->viscous)
    {
      /*! Compute physical corrected gradient of the solution at the solution and flux points. */
      for(i=0; i<FlowSol->n_ele_types; i++)
        FlowSol->mesh_eles(i)->correct_gradient();

#ifdef _MPI
      /*! Send the corrected physical gradients across the MPI interface. */
      if (FlowSol->nproc>1)
      {
        for(i=0; i<FlowSol->n_mpi_inter_types; i++)
          FlowSol->mesh_mpi_inters(i).send_corrected_gradient();
      }
#endif

      /*! Compute discontinuous transformed viscous flux at upts and add to total transformed flux at upts. */
      /*! If using LES, compute the transformed SGS flux and add to total transformed flux at solution points. */
      for(i=0; i<FlowSol->n_ele_types; i++)
        FlowSol->mesh_eles(i)->evaluate_viscFlux();

      //If using LES, extrapolate transformed SGS flux to flux points
      if (run_input.LES)
      {
        for(i=0; i<FlowSol->n_ele_types; i++)
          FlowSol->mesh_eles(i)->extrapolate_sgsFlux();
      }

//If using MPI and LES, send SGS flux across processors
#ifdef _MPI
      if (FlowSol->nproc > 1)
      {
        if (run_input.LES)
        {
          for (i = 0; i < FlowSol->n_mpi_inter_types; i++)
            FlowSol->mesh_mpi_inters(i).send_sgsf_fpts();
        }
      }
#endif
    }
    /*! For viscous or inviscid, compute the transformed normal discontinuous flux at flux points. */
    for(i=0; i<FlowSol->n_ele_types; i++)
      FlowSol->mesh_eles(i)->extrapolate_totalFlux();

    /*! For viscous or inviscid, compute the transformed divergence of total flux at solution points. */
    for(i=0; i<FlowSol->n_ele_types; i++)
      FlowSol->mesh_eles(i)->calculate_divergence();

    if (FlowSol->viscous) {
      /*! Compute transformed normal interface viscous flux and add to transformed normal inviscid flux. */
      for(i=0; i<FlowSol->n_int_inter_types; i++)
        FlowSol->mesh_int_inters(i).calculate_common_viscFlux();

      for(i=0; i<FlowSol->n_bdy_inter_types; i++)
        FlowSol->mesh_bdy_inters(i).evaluate_boundaryConditions_viscFlux(FlowSol->time);//TODO: use RK_time instead

#if _MPI
      /*! Evaluate the MPI interfaces. */
      if (FlowSol->nproc>1) {
          for(i=0; i<FlowSol->n_mpi_inter_types; i++)
            FlowSol->mesh_mpi_inters(i).receive_corrected_gradient();

          if (run_input.LES) {
            for(i=0; i<FlowSol->n_mpi_inter_types; i++)
            FlowSol->mesh_mpi_inters(i).receive_sgsf_fpts();
          }

          for(i=0; i<FlowSol->n_mpi_inter_types; i++)
            FlowSol->mesh_mpi_inters(i).calculate_common_viscFlux();
        }
#endif
    }

  /*! Compute the transformed divergence of the continuous flux. */
  for(i=0; i<FlowSol->n_ele_types; i++)
    FlowSol->mesh_eles(i)->calculate_corrected_divergence();

  /*! Compute source term */
  if (run_input.turb_model==1) {
    for (i=0; i<FlowSol->n_ele_types; i++)
      FlowSol->mesh_eles(i)->calc_src_upts_SA();
  }

  /*! De-aliasing using over-integration*/
  if (run_input.over_int)
    for (i = 0; i < FlowSol->n_ele_types; i++)
      FlowSol->mesh_eles(i)->dealias_over_integration();
}

#ifdef _MPI
void set_rank_nproc(int in_rank, int in_nproc, struct solution* FlowSol)
{
  FlowSol->rank = in_rank;
  FlowSol->nproc = in_nproc;
  FlowSol->error_states.setup(FlowSol->nproc);
}
#endif

// get pointer to transformed discontinuous solution at a flux point

double* get_disu_fpts_ptr(int in_ele_type, int in_ele, int in_field, int in_local_inter, int in_fpt, struct solution* FlowSol)
{
  return FlowSol->mesh_eles(in_ele_type)->get_disu_fpts_ptr(in_fpt,in_local_inter,in_field,in_ele);
}

// get pointer to normal continuous transformed inviscid flux at a flux point

double* get_norm_tconf_fpts_ptr(int in_ele_type, int in_ele, int in_field, int in_local_inter, int in_fpt, struct solution* FlowSol)
{
  return FlowSol->mesh_eles(in_ele_type)->get_norm_tconf_fpts_ptr(in_fpt,in_local_inter,in_field,in_ele);
}

// get pointer to subgrid-scale flux at a flux point

double* get_sgsf_fpts_ptr(int in_ele_type, int in_ele, int in_local_inter, int in_field, int in_dim, int in_fpt, struct solution* FlowSol)
{
	return FlowSol->mesh_eles(in_ele_type)->get_sgsf_fpts_ptr(in_fpt,in_local_inter,in_field,in_dim,in_ele);
}

// get pointer to determinant of jacobian at a flux point

double* get_detjac_fpts_ptr(int in_ele_type, int in_ele, int in_ele_local_inter, int in_inter_local_fpt, struct solution* FlowSol)
{
  return FlowSol->mesh_eles(in_ele_type)->get_detjac_fpts_ptr(in_inter_local_fpt,in_ele_local_inter,in_ele);
}


// get pointer to magntiude of normal dot inverse of (determinant of jacobian multiplied by jacobian) at a flux point

double* get_tdA_fpts_ptr(int in_ele_type, int in_ele, int in_ele_local_inter, int in_inter_local_fpt, struct solution* FlowSol)
{
  return FlowSol->mesh_eles(in_ele_type)->get_tdA_fpts_ptr(in_inter_local_fpt,in_ele_local_inter,in_ele);
}

// get pointer to the normal at a flux point

double* get_norm_fpts_ptr(int in_ele_type, int in_ele, int in_local_inter, int in_fpt, int in_dim, struct solution* FlowSol)
{
  return FlowSol->mesh_eles(in_ele_type)->get_norm_fpts_ptr(in_fpt,in_local_inter,in_dim,in_ele);
}

// get CPU pointer to the coordinates at a flux point.
// See bdy_inters for reasons for this CPU/GPU split.

double* get_loc_fpts_ptr_cpu(int in_ele_type, int in_ele, int in_local_inter, int in_fpt, int in_dim, struct solution* FlowSol)
{
  return FlowSol->mesh_eles(in_ele_type)->get_loc_fpts_ptr_cpu(in_fpt,in_local_inter,in_dim,in_ele);
}

// get GPU pointer to the coordinates at a flux point

double* get_loc_fpts_ptr_gpu(int in_ele_type, int in_ele, int in_local_inter, int in_fpt, int in_dim, struct solution* FlowSol)
{
  return FlowSol->mesh_eles(in_ele_type)->get_loc_fpts_ptr_gpu(in_fpt,in_local_inter,in_dim,in_ele);
}

// get pointer to delta of the transformed discontinuous solution at a flux point

double* get_delta_disu_fpts_ptr(int in_ele_type, int in_ele, int in_field, int in_local_inter, int in_fpt, struct solution* FlowSol)
{
  return FlowSol->mesh_eles(in_ele_type)->get_delta_disu_fpts_ptr(in_fpt,in_local_inter,in_field,in_ele);
}

// get pointer to gradient of the discontinuous solution at a flux point
double* get_grad_disu_fpts_ptr(int in_ele_type, int in_ele, int in_local_inter, int in_field, int in_dim, int in_fpt, struct solution* FlowSol)
{
  return FlowSol->mesh_eles(in_ele_type)->get_grad_disu_fpts_ptr(in_fpt,in_local_inter,in_dim,in_field,in_ele);
}

// get pointer to the discontinuous solution (close normal) at a flux point
double* get_normal_disu_fpts_ptr(int in_ele_type, int in_ele, int in_local_inter, int in_field, int in_fpt, struct solution* FlowSol, hf_array<double> temp_loc, double temp_pos[3])
{
  return FlowSol->mesh_eles(in_ele_type)->get_normal_disu_fpts_ptr(in_fpt,in_local_inter,in_field,in_ele, temp_loc, temp_pos);
}

void patch_solution(struct solution* FlowSol)
{
    for(int i=0; i<FlowSol->n_ele_types; i++)
    {
        if (FlowSol->mesh_eles(i)->get_n_eles()!=0)
            FlowSol->mesh_eles(i)->set_patch();
    }
}

void InitSolution(struct solution* FlowSol)
{
  // set initial conditions
  if (FlowSol->rank==0) cout << "Setting initial conditions... " << flush;

  if (run_input.restart_flag==0) {
      for(int i=0;i<FlowSol->n_ele_types;i++) {
          if (FlowSol->mesh_eles(i)->get_n_eles()!=0)

            FlowSol->mesh_eles(i)->set_ics(FlowSol->time);
        }

    }
  else
    {
      FlowSol->ini_iter = run_input.restart_iter;
      read_restart(run_input.restart_iter,run_input.n_restart_files,FlowSol);
    }
    
#ifdef _MPI
    MPI_Barrier(MPI_COMM_WORLD);
#endif // _MPI
    if (FlowSol->rank == 0)
      cout << "done" << endl;
      // copy solution to gpu
#ifdef _GPU
  for(int i=0;i<FlowSol->n_ele_types;i++) {
      if (FlowSol->mesh_eles(i)->get_n_eles()!=0) {
          FlowSol->mesh_eles(i)->cp_disu_upts_cpu_gpu();

        }
    }
#endif

}

void read_restart(int in_file_num, int in_n_files, struct solution* FlowSol)
{

  char file_name_s[50];
  ifstream restart_file;
  restart_file.precision(15);

  // Open the restart files and read info

  for (int i=0;i<FlowSol->n_ele_types;i++) {
      if (FlowSol->mesh_eles(i)->get_n_eles()!=0) {

          for (int j=0;j<in_n_files;j++)
            {
              if(in_n_files!=1)
                  sprintf(file_name_s,"Rest_%.09d/Rest_%.09d_p%.04d.dat",in_file_num,in_file_num,j);//in folder
              else
                  sprintf(file_name_s,"Rest_%.09d_p%.04d.dat",in_file_num,j);
              restart_file.open(file_name_s);
              if (!restart_file)
                FatalError("Could not open restart file ");

              restart_file >> FlowSol->time;

              int info_found = FlowSol->mesh_eles(i)->read_restart_info(restart_file);
              restart_file.close();

              if (info_found)
                break;
            }
        }
    }

  // Now open all the restart files one by one and store data belonging to this processor

  for (int j=0;j<in_n_files;j++)
    {
      //cout <<  "Reading restart file " << j << endl;
      if (in_n_files!=1)
          sprintf(file_name_s,"Rest_%.09d/Rest_%.09d_p%.04d.dat",in_file_num,in_file_num,j);//in folder
      else
          sprintf(file_name_s,"Rest_%.09d_p%.04d.dat",in_file_num,j);
      restart_file.open(file_name_s);

      if (restart_file.fail())
        FatalError(strcat((char *)"Could not open restart file ",file_name_s));

      for (int i=0;i<FlowSol->n_ele_types;i++)  {
          if (FlowSol->mesh_eles(i)->get_n_eles()!=0) {

              FlowSol->mesh_eles(i)->read_restart_data(restart_file);

            }
        }
      restart_file.close();
    }

    //check if all elements are read into the program
    for (int i = 0; i < FlowSol->n_ele_types; i++)
    {
      if (FlowSol->mesh_eles(i)->get_n_eles() != 0 && FlowSol->mesh_eles(i)->restart_counter)
        FatalError("Some elements are not found in the restart files, check n_restart_files");
    }
}

void calc_time_step(struct solution *FlowSol)
{
  if (run_input.dt_type == 1) //global time step
  {
    // If using global minimum timestep based on CFL, determine
    // global minimum
    double dt_globe=1e12;//initialize to large value
    double dt_globe_new;
    for (int j = 0; j < FlowSol->n_ele_types; j++) //for each type of element
    {
      if (FlowSol->mesh_eles(j)->get_n_eles() != 0) //if have element
      {
        for (int ic = 0; ic < FlowSol->mesh_eles(j)->get_n_eles(); ic++) //loop over each element
        {
          dt_globe_new = FlowSol->mesh_eles(j)->calc_dt_local(ic);
          if (dt_globe_new < dt_globe)
            dt_globe = dt_globe_new;
        }
      }
    }

#ifdef _MPI
    // If in parallel and using global minumum timestep, allocate storage
    // for minimum timesteps in each partition
    if (FlowSol->nproc > 1)
    {
      double dt_globe_mpi;
      MPI_Allreduce(&dt_globe, &dt_globe_mpi, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
      dt_globe = dt_globe_mpi;
    }
#endif
    run_input.dt = dt_globe; //copy to run_input.dt
  }
  // If using local timestepping, just compute and store all local
  // timesteps
  else if (run_input.dt_type == 2)
  {
    double dt_local_min = 1e12; //initialize to large number
    double dt_local_min_new;

    for (int j = 0; j < FlowSol->n_ele_types; j++) //for each type of element
    {
      if (FlowSol->mesh_eles(j)->get_n_eles() != 0) //if have element
      {
        for (int ic = 0; ic < FlowSol->mesh_eles(j)->get_n_eles(); ic++) //loop over each element
        {
          FlowSol->mesh_eles(j)->dt_local(ic) = FlowSol->mesh_eles(j)->calc_dt_local(ic);
        }
        //get local minimum time step
        dt_local_min_new = FlowSol->mesh_eles(j)->dt_local.get_min();
        if (dt_local_min_new < dt_local_min)
          dt_local_min = dt_local_min_new;
      }
    }
    //if run in parallel, find out global minimum time step
#ifdef _MPI
    if (FlowSol->nproc > 1)
    {
      double dt_local_mpi;
      MPI_Allreduce(&dt_local_min, &dt_local_mpi, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
      dt_local_min = dt_local_mpi;
    }
#endif
    run_input.dt = dt_local_min; //copy to run_input.dt
  }
}

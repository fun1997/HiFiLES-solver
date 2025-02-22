/*!
 * \file HiFiLES.cpp
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
#include <fstream>
#include <unistd.h>

#include "../include/global.h"
#include "../include/geometry.h"
#include "../include/solver.h"
#include "../include/output.h"
#include "../include/solution.h"
#include "../include/mesh.h"

#ifdef _GPU
#include "util.h"
#endif

using namespace std;

int main(int argc, char *argv[]) {

  int rank = 0;
  int i, j;                           /*!< Loop iterators */
  int i_steps = 0;                    /*!< Iteration index */
  int RKSteps;                        /*!< Number of RK steps */
  clock_t init_time, final_time;      /*!< To control the time */
  struct solution FlowSol;            /*!< Main structure with the flow solution and geometry */        
  ofstream write_hist;                /*!< Output files (forces, statistics, and history) */
  mesh* mesh_data=new mesh();         /*!< Store mesh information*/

  // {
  //   int k=0;
  //   while (0 == k){
  //     sleep(5);
  //   }

  // }
  sleep(5);
  /*! Initialize MPI. */

#ifdef _MPI
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif

  /*! Check the command line input. */

  if (argc < 2)
  {
    if (rank == 0)
      cout << "No input file specified. For help use -h or --help " << endl;
#ifdef _MPI
    MPI_Finalize();
#endif
    return (0);
  }
  else if (!strcmp(argv[1], "-h") || !strcmp(argv[1], "-help"))
  {
    if (rank == 0)
    {
      cout << "To run, use HiFiLES <input_file>" << endl;
      cout << " For more details, go to https://github.com/weiqishen/HiFiLES-solver/wiki" << endl;
    }
#ifdef _MPI
    MPI_Finalize();
#endif
    return (0);
  }

  if (rank == 0) {
    cout<<endl;
    cout << "██╗  ██╗██╗███████╗██╗      ██╗     ███████╗███████╗" << endl;
    cout << "██║  ██║██║██╔════╝██║      ██║     ██╔════╝██╔════╝" << endl;
    cout << "███████║██║█████╗  ██║█████╗██║     █████╗  ███████╗" << endl;
    cout << "██╔══██║██║██╔══╝  ██║╚════╝██║     ██╔══╝  ╚════██║" << endl;
    cout << "██║  ██║██║██║     ██║      ███████╗███████╗███████║" << endl;
    cout << "╚═╝  ╚═╝╚═╝╚═╝     ╚═╝      ╚══════╝╚══════╝╚══════╝" << endl;
    cout << "UF Edition by Theoretical Fluid Dynamics and Turbulence Group" << endl;
    cout << argc<<endl;
    for(int i=0;i<argc;i++){
      cout << "argumen "<<i<<" "<<argv[i]<<endl;
    }
    
  }

  /////////////////////////////////////////////////
  /// Read config file and mesh
  /////////////////////////////////////////////////

  /*! Read the config file and store the information in run_input. */

  run_input.setup(argv[1], rank);

  /*! Set the input values in the FlowSol structure. */

  SetInput(&FlowSol);

  /*! Read the mesh file from a file. */

  GeoPreprocess(&FlowSol, *mesh_data);
  delete mesh_data;

  /*! initialize object to output result/restart files */   

  output run_output(&FlowSol); 

  /*! Initialize solution and patch solution if needed */

  InitSolution(&FlowSol);

  /*! Read the probe file if needed and store the information in run_probe. */

  if (run_input.probe)
    run_probe.setup(argv[1], &FlowSol, rank); 

  /////////////////////////////////////////////////
  /// Pre-processing
  /////////////////////////////////////////////////

  /*! Variable initialization. */

  if (run_input.adv_type == 0)
    RKSteps = 1; //Euler
  else if (run_input.adv_type == 1 || run_input.adv_type == 2)
    RKSteps = 4; //RK24/34
  else if (run_input.adv_type == 3)
    RKSteps = 5; //RK45
  else if (run_input.adv_type == 4)
    RKSteps = 14; //RK414

  /*! Initialize forces, integral quantities, and residuals. */

  if (FlowSol.rank == 0)
    FlowSol.norm_residual.setup(6);
  if (run_input.calc_force)
  {
    FlowSol.inv_force.setup(FlowSol.n_dims);
    FlowSol.vis_force.setup(FlowSol.n_dims);
  }
  if (run_input.n_integral_quantities)
    FlowSol.integral_quantities.setup(run_input.n_integral_quantities);

  /*! Copy solution and gradients from GPU to CPU, ready for the following routines */
#ifdef _GPU

  CopyGPUCPU(&FlowSol);

#endif

  /*! Dump initial Paraview, tecplot or CGNS files. */

  if (run_input.write_type == 0)
    run_output.write_vtu(FlowSol.ini_iter + i_steps);
  else if (run_input.write_type == 1)
    run_output.write_tec(FlowSol.ini_iter + i_steps);
#ifdef _CGNS
  else if (run_input.write_type == 2)
    run_output.write_CGNS(FlowSol.ini_iter + i_steps);
#endif
  else
    FatalError("ERROR: Trying to write unrecognized file format ... ");

  if (FlowSol.rank == 0) cout << endl;

  /////////////////////////////////////////////////
  /// Flow solver
  /////////////////////////////////////////////////

  init_time = clock();

  /*! Main solver loop (outer loop). */

  while (i_steps < run_input.n_steps)
  {

    //compute time step if using automatic time step

    calc_time_step(&FlowSol);

    for (i = 0; i < RKSteps; i++)
    {
      /*! Spatial integration. */

      CalcResidual(FlowSol.ini_iter + i_steps, i, &FlowSol);

      /*! Time integration using a RK scheme */

      for (j = 0; j < FlowSol.n_ele_types; j++)
        FlowSol.mesh_eles(j)->AdvanceSolution(i, run_input.adv_type);
    
        /*! Shock capturing */

      if (run_input.shock_cap)
        for (j = 0; j < FlowSol.n_ele_types; j++)
          FlowSol.mesh_eles(j)->shock_capture();
    }

    /*! Update total time, and increase the iteration index. */

    FlowSol.time += run_input.dt;
    run_input.time = FlowSol.time;
    i_steps++;
    if(run_input.pressure_ramp)
    run_input.ramp_counter++;

    /*! Copy solution and gradients from GPU to CPU, ready for the following routines */
#ifdef _GPU

    if(i_steps == 1 || i_steps%run_input.plot_freq == 0 ||
       i_steps%run_input.monitor_res_freq == 0 || i_steps%run_input.restart_dump_freq==0) {

      run_output.CopyGPUCPU(&FlowSol);

    }

#endif

    /*! ============================Outputs=============================== */

    /*! Compute time-averaged quantities. */
    if ( i_steps==1)//set start time for averaging
        run_input.spinup_time=FlowSol.time;
    if(run_input.n_average_fields)
      run_output.CalcTimeAverageQuantities();

    if (i_steps == 1 || i_steps % run_input.monitor_res_freq == 0)
    {

      /*! Compute the value of the forces. */

      if (run_input.calc_force != 0)
        run_output.CalcForces(FlowSol.ini_iter + i_steps, (i_steps == 1 || i_steps % run_input.monitor_cp_freq == 0));

      /*! Compute integral quantities. */

      if (run_input.n_integral_quantities) //if calculate integral quantities
        run_output.CalcIntegralQuantities();

      /*! Compute the norm of the residual. */

      run_output.CalcNormResidual();

      /*! Output the history file. */

      run_output.HistoryOutput(FlowSol.ini_iter + i_steps, init_time, &write_hist);

      if (FlowSol.rank == 0)
        cout << endl;
    }
    /*! Dump Paraview, Tecplot or CGNS files. */

    if (i_steps % run_input.plot_freq == 0)
    {
      if (run_input.write_type == 0)
        run_output.write_vtu(FlowSol.ini_iter + i_steps);
      else if (run_input.write_type == 1)
        run_output.write_tec(FlowSol.ini_iter + i_steps);
#ifdef _CGNS
      else if (run_input.write_type == 2)
        run_output.write_CGNS(FlowSol.ini_iter + i_steps);
#endif
      else
        FatalError("ERROR: Trying to write unrecognized file format ... ");
    }

    /*! Write probe file. */

        if(run_input.probe!=0)
        {
            if((i_steps%run_probe.probe_freq==0))
#ifdef _HDF5
              run_output.write_probe_hdf5();
#else
              run_output.write_probe_ascii();
#endif
        }

    /*! Dump restart file. */

        if (i_steps % run_input.restart_dump_freq == 0)
        {
          if(FlowSol.rank==0){
            for(int i=0; i<FlowSol.n_bdy_inter_types; i++){ 
              if(FlowSol.mesh_bdy_inters(i).inlet.type != 0){           
                FlowSol.mesh_bdy_inters(i).write_sem_restart(FlowSol.ini_iter + i_steps);
              }
            }
          }
        
#ifdef _HDF5
          run_output.write_restart_hdf5(FlowSol.ini_iter + i_steps);
#else
          run_output.write_restart_ascii(FlowSol.ini_iter + i_steps);
#endif
        }
  }

  /////////////////////////////////////////////////
  /// End simulation
  /////////////////////////////////////////////////

  /*! Calculate Error */
  if (run_input.test_case)
    run_output.compute_error(FlowSol.ini_iter + i_steps);
    
  /*! Close convergence history file. */

  if (rank == 0) {
    write_hist.close();

  /*! Compute execution time. */

  final_time = clock()-init_time;
  printf("Execution time= %f s\n", (double) final_time/((double) CLOCKS_PER_SEC));
    }
  /*! Finalize MPI. */

#ifdef _MPI
  MPI_Finalize();
#endif
return 0;
}

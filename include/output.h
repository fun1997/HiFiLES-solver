/*!
 * \file output.h
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

#pragma once

#include <string>
#include "global.h"
#include "solution.h"

#ifdef _CGNS
#ifdef _MPI
#include "pcgnslib.h"
#else
#include "cgnslib.h"
#endif
#endif

#ifdef _MPI
#include "mpi_inters.h"
#endif

#ifdef _GPU
#include "util.h"
#endif
class output
{

  public:
    // #### constructors ####

    // default constructor

    output(struct solution *in_sol);

    // default destructor

    ~output();

#ifdef _CGNS
    void setup_CGNS();
#endif

    //driver methods

    /*! write an output file in Tecplot ASCII format */
    void write_tec(int in_file_num);

    /*! write an output file in VTK ASCII format */
    void write_vtu(int in_file_num);

#ifdef _CGNS
    /*! write an output file in CGNS format */
    void write_CGNS(int in_file_num);
#endif

    /*! write an probe data file */

#ifdef _HDF5
    void write_probe_hdf5(void);
#else
    void write_probe_ascii(void);
#endif

/*! writing a restart file */
#ifdef _HDF5
    void write_restart_hdf5(int in_file_num);
#else
    void write_restart_ascii(int in_file_num);
#endif

    /*! monitor convergence of residual */
    void HistoryOutput(int in_file_num, clock_t init, ofstream *write_hist);

    /*! compute forces on wall faces*/
    void CalcForces(int in_file_num, bool write_forces);
    
    //helper methods

    /*! compute integral diagnostic quantities */
    void CalcIntegralQuantities(void);

    /*! Calculate time averaged diagnostic quantities */
    void CalcTimeAverageQuantities(void);

    /*! compute error */
    void compute_error(int in_file_num);

    /*! calculate residual */
    void CalcNormResidual(void);

    /*! check if the solution is bounded !*/
    void check_stability(void);

#ifdef _CGNS
    /*! calculate connectivity for each element (used by cgns)*/
    void calc_connectivity(hf_array<cgsize_t> &out_conn,int ele_type,cgsize_t &start_index);
#endif

#ifdef _GPU
    /*! copy solution and gradients from GPU to CPU for above routines !*/
    void CopyGPUCPU(void);
#endif

  protected:
    //data members

    struct solution *FlowSol; //the solution structure
#ifdef _CGNS
    int cell_dim, phy_dim;
    cgsize_t glob_npnodes; //total number of plot nodes globally
    cgsize_t glob_npeles;  //total number of plot elements globally
    hf_array<cgsize_t> pele_start, pele_end;//local element list start end index
    hf_array<int> sum_npele;//global number of each type of plot element
    cgsize_t pnode_start;//local node list start index
    
#endif
};
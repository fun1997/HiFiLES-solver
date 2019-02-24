/*!
 * \file bdy_inters.h
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

#include "inters.h"
#include "solution.h"

class bdy_inters: public inters
{
public:

  // #### constructors ####

  // default constructor

  bdy_inters();

  // default destructor

  ~bdy_inters();

  // #### methods ####

  /*! setup inters */
  void setup(int in_n_inters, int in_inter_type);

  /*! Set bdy interface */
  void set_boundary(int in_inter, int bc_id, int in_ele_type_l, int in_ele_l, int in_local_inter_l, struct solution* FlowSol);

  /*! Compute right hand side state at boundaries */
  void set_boundary_conditions(int sol_spec,int bc_id, double* u_l, double* u_r, double *norm, double *loc,double gamma, double R_ref, double time_bound, int equation);

  /*! Compute right hand side gradient at boundaries */
void set_boundary_gradients(int bc_id, hf_array<double> &u_l, hf_array<double> &u_r, hf_array<double> &grad_ul, hf_array<double> &grad_ur, hf_array<double> &norm, hf_array<double> &loc, double gamma, double R_ref, double time_bound, int equation);

  /*! move all from cpu to gpu */
  void mv_all_cpu_gpu(void);

  /*! calculate normal transformed continuous inviscid flux at the flux points on boundaries*/
  void evaluate_boundaryConditions_invFlux(double time_bound);

  /*! calculate delta in transformed discontinuous solution at flux points */
  void calc_delta_disu_fpts_boundary(void);

  /*! calculate normal transformed continuous viscous flux at the flux points on boundaries*/
  void evaluate_boundaryConditions_viscFlux(double time_bound);

protected:

  // #### members ####

  hf_array<int> boundary_id;


};

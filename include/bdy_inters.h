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
#include <vector>
#include "turbinlet.h"

class bdy_inters: public inters
{
public:
 //the vertex of inter
  hf_array<double> pos_bdr_face_vtx;
  int flag=0;
  int in_ele_type;
  turbinlet inlet; 

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
  void evaluate_boundaryConditions_invFlux(solution* FlowSol, double time_bound);

  /*! calculate delta in transformed discontinuous solution at flux points */
  void calc_delta_disu_fpts_boundary(void);

  /*! calculate normal transformed continuous viscous flux at the flux points on boundaries*/
  void evaluate_boundaryConditions_viscFlux(double time_bound);

  // about turbulent inlet

  /*! add LES inlet condition */
  void add_les_inlet(int in_file_numstruct, solution* FlowSol);

  /*! update the velocity profile */
  void update_les_inlet(struct solution* FlowSol);

  /*! Generate turbulent fluctuations using synthetic eddy method */
  void gen_fluc_sem(struct solution* FlowSol);

  /*! Generate turbulent fluctuations with Gaussian random noise */
  void gen_fluc_random();

  /*! rescale according to r_ij */
  void rescale_rij();

  /*! Correct mass flux on the inlet interface */
  void correct_mass(struct solution* FlowSol);

  void cal_inlet_rou_vel(double time_bound);

  void cal_inlet_r_ij();

  void cal_convection_speed(hf_array<double> &vel_c);

  double cal_inlet_area();

  void write_sem_restart(int in_file_num);

  void read_sem_restart(int in_file_num,int & rest_info);



protected:

  // #### members ####

  hf_array<int> boundary_id;
  vector<hf_array<double *>> wm_disu_ptr; //pointer to the solution at input point of wall model
  vector<double> wm_dist;             //distance from input point to wall
};

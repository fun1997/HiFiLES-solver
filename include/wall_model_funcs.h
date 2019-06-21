#pragma once

#include <cmath>
#include "global.h"
#include "solution.h"

//get the pointer to the wall model input solution variables
double *get_wm_disu_ptr(int in_ele_type, int in_ele, int in_upt, int in_field, struct solution *Flowsol);
//use wall normal and arbitrary flux point on interface to calculate the farthest point in the wall adjacant cell to the wall and return the upts index
double calc_wm_upts_dist(int in_ele_type, int in_ele, int in_local_inter, struct solution *FlowSol, int &out_upt);
//calculate wall stress from input solutions
void calc_wall_stress(hf_array<double> &in_u_wm, hf_array<double> &in_uw, double in_dist, hf_array<double> &in_norm, hf_array<double> &out_fn);

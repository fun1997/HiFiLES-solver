/*!
 * \file bc.h
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

#include "hf_array.h"
#include "error.h"
#include <string>
#include <map>
#include <iterator>
#include <algorithm>
class bc
{
public:

  friend class mesh_reader;

// default constructor/destructor
bc();
~bc();

void setup(const string &in_bc_name);//setup the maps

//get functions
int get_bc_flag(void);
string get_bc_type(void);
string get_bc_name(void);

//set functions
int set_bc_flag(string &in_type);

//bc params
double mach;
double rho;
double nx,ny,nz;
double p_total,T_total,p_ramp_coeff,T_ramp_coeff,p_total_old,T_total_old;
double p_static,T_static;
hf_array<double> velocity;
int pressure_ramp;
int use_wm;
// turbulent inlet
int type;
int mode;
double vis_y;
double turb_1;
double turb_2;
int n_eddy;

private:

map<string,int> bc_type2flag;
map<int,string> bc_flag2type;
string bc_name;
int bc_flag;
};
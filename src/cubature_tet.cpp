/*!
 * \file cubature_tet.cpp
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
#include <cmath>
#include <string>
#include <sstream>

#include "../include/global.h"
#include "../include/cubature_tet.h"

using namespace std;

// #### constructors ####

// default constructor

cubature_tet::cubature_tet()
{	
  order=0;
  n_pts=0;
  locs.setup(0,0);
  weights.setup(0);
}

// constructor 1

cubature_tet::cubature_tet(int in_rule, int in_order) // set by order
{
  if (HIFILES_DIR == NULL)
    FatalError("environment variable HIFILES_HOME is undefined");

  order = in_order;
  n_pts = (order + 1) * (order + 2) * (order + 3) / 6;
  locs.setup(n_pts, 3);
  weights.setup(n_pts);
  string filename;
  filename = HIFILES_DIR;
  ifstream datfile;
  int upper_order_lim, lower_order_lim;

  //set rule-dependent variables
  if (in_rule == 0) //internal
  {
    lower_order_lim = 0;
    upper_order_lim = 6;
    if_weight = 1;
    filename += "/data/tet_inter.bin";
  }
  else if (in_rule == 1) //alpha
  {
    lower_order_lim = 1;
    upper_order_lim = 15;
    if_weight = 0;
    filename += "/data/tet_alpha.bin";
  }
  else
  FatalError("Cubature rule not implemented");

  if (order <= upper_order_lim && order >= lower_order_lim)
  {
    hf_array<double> temp_loc_0(n_pts);
    hf_array<double> temp_loc_1(n_pts);
    hf_array<double> temp_loc_2(n_pts);

    //open file
    datfile.open(filename.c_str(), ifstream::binary);
    if (!datfile)
      FatalError("Unable to open cubature file");
    //skip lines
    int skip = 0;
    for (int i = lower_order_lim; i < order; i++)
      skip += (3+if_weight) * (i + 1) * (i + 2) * (i + 3) / 6;
    datfile.seekg(sizeof(double) * skip, ios_base::beg);
    // read file
    datfile.read((char *)temp_loc_0.get_ptr_cpu(), sizeof(double) * n_pts);
    datfile.read((char *)temp_loc_1.get_ptr_cpu(), sizeof(double) * n_pts);
    datfile.read((char *)temp_loc_2.get_ptr_cpu(), sizeof(double) * n_pts);
    if (if_weight)
      datfile.read((char *)weights.get_ptr_cpu(), sizeof(double) * n_pts);
    //close file
    datfile.close();
    //assign values to locs
    for (int i = 0; i < n_pts; i++)
    {
      locs(i, 0) = temp_loc_0(i);
      locs(i, 1) = temp_loc_1(i);
      locs(i, 2) = temp_loc_2(i);
    }
  }
  else
    FatalError("cubature order not implemented.");
}

// copy constructor

cubature_tet::cubature_tet(const cubature_tet& in_cubature)
{
  order=in_cubature.order;
  n_pts=in_cubature.n_pts;
  locs=in_cubature.locs;
  weights=in_cubature.weights;
}

// assignment

cubature_tet& cubature_tet::operator=(const cubature_tet& in_cubature)
{
  // check for self asignment
  if(this == &in_cubature)
    {
      return (*this);
    }
  else
    {
      order=in_cubature.order;
      n_pts=in_cubature.n_pts;
      locs=in_cubature.locs;
      weights=in_cubature.weights;
      return *this;
    }
}


// destructor

cubature_tet::~cubature_tet()
{

}

// #### methods ####

// method to get number of cubature points

int cubature_tet::get_n_pts(void)
{
  return n_pts;
}

// method to get r location of cubature point

double cubature_tet::get_r(int in_pos)
{
  return locs(in_pos,0);
}

// method to get s location of cubature point
double cubature_tet::get_s(int in_pos)
{
  return locs(in_pos,1);
}

// method to get s location of cubature point
double cubature_tet::get_t(int in_pos)
{
  return locs(in_pos,2);
}

// method to get weight location of cubature point

double cubature_tet::get_weight(int in_pos)
{
  if (if_weight)
    return weights(in_pos);
  else
    FatalError("No weights for this rule");
  return -1;
}

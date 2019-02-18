/*!
 * \file cubature_hexa.cpp
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
#include <cmath>
#include <string>
#include <sstream>

#include "../include/global.h"
#include "../include/cubature_1d.h"
#include "../include/cubature_hexa.h"

using namespace std;

// #### constructors ####

// default constructor

cubature_hexa::cubature_hexa()
{	
  order=0;
  n_pts=0;
  locs.setup(0,0);
  weights.setup(0);
}

// constructor 1

cubature_hexa::cubature_hexa(int in_rule, int in_order) // set by order
{

  order = in_order;
  n_pts = (order + 1) * (order + 1) * (order + 1);
  locs.setup(n_pts, 3);
  weights.setup(n_pts);

  //get 1D points and weights
  cubature_1d cub_1d(in_rule, order);

    for (int i = 0; i < (order + 1); i++) //z
    {
      for (int j = 0; j < (order + 1); j++) //y
      {
        for (int k = 0; k < (order + 1); k++) //x
        {
          int index = k + (order + 1) * j + (order + 1) * (order + 1) * i;
          locs(index, 0) = cub_1d.get_r(k);
          locs(index, 1) = cub_1d.get_r(j);
          locs(index, 2) = cub_1d.get_r(i);
          weights(index) = cub_1d.get_weight(i) * cub_1d.get_weight(j) * cub_1d.get_weight(k);
        }
      }
    }
}

// copy constructor

cubature_hexa::cubature_hexa(const cubature_hexa& in_cubature)
{
  order=in_cubature.order;
  n_pts=in_cubature.n_pts;
  locs=in_cubature.locs;
  weights=in_cubature.weights;
}

// assignment

cubature_hexa& cubature_hexa::operator=(const cubature_hexa& in_cubature)
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

cubature_hexa::~cubature_hexa()
{

}

// #### methods ####

// method to get number of cubature points

int cubature_hexa::get_n_pts(void)
{
  return n_pts;
}

// method to get r location of cubature point

double cubature_hexa::get_r(int in_pos)
{
  return locs(in_pos,0);
}

// method to get s location of cubature point
double cubature_hexa::get_s(int in_pos)
{
  return locs(in_pos,1);
}

// method to get s location of cubature point
double cubature_hexa::get_t(int in_pos)
{
  return locs(in_pos,2);
}

// method to get weight location of cubature point

double cubature_hexa::get_weight(int in_pos)
{
  return weights(in_pos);
}



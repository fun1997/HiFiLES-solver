/*!
 * \file cubature_1d.cpp
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

using namespace std;

// #### constructors ####

// default constructor

cubature_1d::cubature_1d()
{	
  order=0;
  n_pts=0;
  locs.setup(0);
  weights.setup(0);
}

// constructor 1

cubature_1d::cubature_1d(int in_rule, int in_order) // set by order of quadrature rule, n_pts=order+1,accuracy=2*n_pts-1
{
  if (HIFILES_DIR == NULL)
    FatalError("environment variable HIFILES_HOME is undefined");

  order = in_order;
  n_pts = order + 1;
  locs.setup(n_pts);
  weights.setup(n_pts);
  string filename;
  filename = HIFILES_DIR;
  ifstream datfile;

//set file name
  if (in_rule == 0) //Gauss
    filename += "/data/JacobiGQ.bin";
  else if (in_rule == 1) //Gauss Lobatto
    filename += "/data/JacobiGL.bin";
  else
    FatalError("cubature rule not implemented.");

  if (order <= 15 && order >= 0)
  {
    //open file
    datfile.open(filename.c_str(), ios_base::binary);
  if (!datfile)
    FatalError("Unable to open cubature file");

    //skip lines
    int skip = (1+order)*order;
    datfile.seekg(sizeof(double) * skip, ios_base::beg);
    //read file
    datfile.read((char *)locs.get_ptr_cpu(), sizeof(double) * n_pts);
    datfile.read((char *)weights.get_ptr_cpu(), sizeof(double) * n_pts);
    //close file
    datfile.close();
  }
  else
  {
    datfile.close();
    FatalError("cubature order not implemented.");
  }
}

// copy constructor

cubature_1d::cubature_1d(const cubature_1d& in_cubature_1d)
{
  order=in_cubature_1d.order;
  n_pts=in_cubature_1d.n_pts;
  locs=in_cubature_1d.locs;
  weights=in_cubature_1d.weights;
}

// assignment

cubature_1d& cubature_1d::operator=(const cubature_1d& in_cubature_1d)
{
  // check for self asignment
  if(this == &in_cubature_1d)
    {
      return (*this);
    }
  else
    {
      order=in_cubature_1d.order;
      n_pts=in_cubature_1d.n_pts;
      locs=in_cubature_1d.locs;
      weights=in_cubature_1d.weights;
    }
}

// destructor

cubature_1d::~cubature_1d()
{

}

// #### methods ####

// method to get number of cubature_1d points

int cubature_1d::get_n_pts(void)
{
  return n_pts;
}

// method to get r location of cubature_1d point

double cubature_1d::get_r(int in_pos)
{
  return locs(in_pos);
}

// method to get weight location of cubature_1d point

double cubature_1d::get_weight(int in_pos)
{
  return weights(in_pos);
}


/*!
 * \file cubature_hexa.h
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
#include "hf_array.h"

class cubature_hexa
{
public:

  // #### constructors ####

  // default constructor
  cubature_hexa();

  // constructor 1
  cubature_hexa(int in_rule,int in_order); // set by order

  // copy constructor
  cubature_hexa(const cubature_hexa& in_cubature);

  // assignment
  cubature_hexa& operator=(const cubature_hexa& in_cubature);

  // destructor
  ~cubature_hexa();

  // #### methods ####

  // method to get number of cubature points
  int get_n_pts(void);

  // method to get r location of cubature point
  double get_r(int in_pos);

  // method to get s location of cubature point
  double get_s(int in_pos);

  // method to get s location of cubature point
  double get_t(int in_pos);

  // method to get weight location of cubature point
  double get_weight(int in_pos);

  // #### members ####

  // cubature rule
  int order;

  // number of cubature points
  int n_pts;

  // location of cubature points
  hf_array<double> locs;

  // weight of cubature points
  hf_array<double> weights;

};

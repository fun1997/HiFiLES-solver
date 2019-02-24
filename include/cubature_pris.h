/*!
 * \file cubature_pris.h
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

class cubature_pris
{
public:

  // #### constructors ####

  // default constructor
  cubature_pris();

  // constructor 1
  cubature_pris(int in_rule_tri,int in_rule_1d,int in_order); // set by order

  // copy constructor
  cubature_pris(const cubature_pris& in_cubature_pris);

  // assignment
  cubature_pris& operator=(const cubature_pris& in_cubature_pris);

  // destructor
  ~cubature_pris();

  // #### methods ####

  // method to get number of cubature_pris points
  int get_n_pts(void);

  // method to get r location of cubature_pris point
  double get_r(int in_pos);

  // method to get s location of cubature_pris point
  double get_s(int in_pos);

  // method to get t location of cubature_pris point
  double get_t(int in_pos);

  // method to get weight location of cubature_pris point
  double get_weight(int in_pos);

  // #### members ####

  // cubature_pris order
  int order;

  // number of cubature_pris points
  int n_pts;

  // location of cubature_pris points
  hf_array<double> locs;

  // weight of cubature_pris points
  hf_array<double> weights;

  //switch if calculate weigths
  int if_weight;

};

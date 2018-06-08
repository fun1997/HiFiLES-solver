/*!
 * \file cubature_pris.h
 * \author - Weiqi Shen 
 * \Unifersity of Florida
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

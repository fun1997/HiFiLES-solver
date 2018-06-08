/*!
 * \file cubature_pris.cpp
 * \author - Weiqi Shen
 */

#include <iostream>
#include <cmath>
#include <string>
#include <sstream>

#include "../include/global.h"
#include "../include/cubature_1d.h"
#include "../include/cubature_tri.h"
#include "../include/cubature_pris.h"

using namespace std;

// #### constructors ####

// default constructor

cubature_pris::cubature_pris()
{	
  order=0;
  n_pts=0;
  locs.setup(0,0);
  weights.setup(0);
}

// constructor 1

cubature_pris::cubature_pris(int in_rule_tri,int in_rule_1d, int in_order) // set by order
{
  order = in_order;
  n_pts = (order+2)*(order+1)*(order+1)/2;
  locs.setup(n_pts, 3);
  weights.setup(n_pts);

  //get tri/1D points and weights
  cubature_tri cub_tri(in_rule_tri, order);
  cubature_1d cub_1d(in_rule_1d,order);

  if (in_rule_tri == 0)
    if_weight = 1;
  else
    if_weight = 0;

      for (int i = 0; i < cub_1d.get_n_pts(); i++) //z
      {
        for (int j = 0; j < cub_tri.get_n_pts(); j++) //xy
        {
          int index = j + cub_tri.get_n_pts() * i;
          locs(index, 0) = cub_tri.get_r(j);
          locs(index, 1) = cub_tri.get_s(j);
          locs(index, 2)=cub_1d.get_r(i);
          if (if_weight)
            weights(index) = cub_tri.get_weight(j) * cub_1d.get_weight(i);
        }
      }

}

// copy constructor

cubature_pris::cubature_pris(const cubature_pris& in_cubature_pris)
{
  order=in_cubature_pris.order;
  n_pts=in_cubature_pris.n_pts;
  locs=in_cubature_pris.locs;
  weights=in_cubature_pris.weights;
}

// assignment

cubature_pris& cubature_pris::operator=(const cubature_pris& in_cubature_pris)
{
  // check for self asignment
  if(this == &in_cubature_pris)
    {
      return (*this);
    }
  else
    {
      order=in_cubature_pris.order;
      n_pts=in_cubature_pris.n_pts;
      locs=in_cubature_pris.locs;
      weights=in_cubature_pris.weights;
    }
}

// destructor

cubature_pris::~cubature_pris()
{

}

// #### methods ####

// method to get number of cubature_pris points

int cubature_pris::get_n_pts(void)
{
  return n_pts;
}

// method to get r location of cubature_pris point

double cubature_pris::get_r(int in_pos)
{
  return locs(in_pos,0);
}

// method to get s location of cubature_pris point

double cubature_pris::get_s(int in_pos)
{
  return locs(in_pos,1);
}

// method to get s location of cubature_pris point

double cubature_pris::get_t(int in_pos)
{
  return locs(in_pos,2);
}

// method to get weight location of cubature_pris point

double cubature_pris::get_weight(int in_pos)
{
  if (if_weight)
    return weights(in_pos);
  else
    FatalError("No weights for this rule");
}



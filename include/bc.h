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
private:

map<string,int> bc_type2flag;
map<int,string> bc_flag2type;
string bc_name;
int bc_flag;
};
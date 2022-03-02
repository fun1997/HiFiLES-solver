#pragma once
#include "hf_array.h"
class turbinlet
{
public:

    //constants
    const double C_mu = 0.09 ,kappa = 0.42; //k-eps model constant, Von-Karman constant
    const double gamma_gas = 1.4;
    // !< type of turbulent inlet
    // !! - type = 0  not used
    // !! - type = 1  gaussian noise
    // !! - type = 2  synthetic eddy method
    int type=0;
    // !< operation mode
    // !! - mode = 0  given tke (and dissipation rate)
    // !! - mode = 1  given turbulent intensity (and use pipe/channel law for length scale)
    // !! - mode = 2  given tke (and dissipation rate profile)
    int mode;
    // inlet parameters
    // !! @param [optional] in_turb_1  if in_mode=0, turbulent kinetic energy; if in_mode=1, turbulent intensity
    // !! @param [optional] in_turb_2  if in_type=1 and in_mode=0, dissipation rate, otherwise not used
    // !! @param [optional] in_vis_y   thickness of viscous sublayer if the wall boundary, typically y+<11
    // !! @param [optional] n_eddy     number of eddies, used if in_type=1
    double vis_y;
    double turb_1, turb_2;

    //!< sem related
    int n_eddy; //!< number of eddies
    hf_array<double> eddy_pos;
    hf_array<int> sgn;
    int initialize = 1;// !< initialize flag

    //!< geometry information
    int id;
    int nbs;// !< local number of boundary surface
    double total_area;// !< boundary surface area
    
    hf_array<int> ibslst;// !< local boundary face id

    hf_array<int> ibslst_inv;

    hf_array<double> face_vtx_coord;// !< vertex coordinates of bdy face, 3*4*nbs

    hf_array<double> rou;//(n_fpts_per_inter,inlet.nbs)
   
    hf_array<double> v; //(n_fpts_per_inter,inlet.nbs,n_dims)

    hf_array<double> r_ij;//(n_fpts_per_inter,inlet.nbs,n_dims)

    hf_array<double> fluctuations;//(n_fpts_per_inter,inlet.nbs,n_dims)


};

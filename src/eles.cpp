/*!
 * \file eles.cpp
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
#include <iomanip>
#include <cmath>
#include <numeric>

#include "../include/global.h"
#include "../include/eles.h"
#if defined _GPU
#include "cuda.h"
#include "cuda_runtime_api.h"
#include "cublas.h"
#include "../include/cuda_kernels.h"
#endif
#include "../include/flux.h"
#include "../include/source.h"

using namespace std;

// #### constructors ####

// default constructor

eles::eles()
{
}

// default destructor

eles::~eles() {}

// #### methods ####

// set number of elements

void eles::setup(int in_n_eles, int in_max_n_spts_per_ele)
{

    n_eles=in_n_eles;
    max_n_spts_per_ele = in_max_n_spts_per_ele;

    if (n_eles!=0)
    {

        order=run_input.order;
        p_res=run_input.p_res;
        viscous =run_input.viscous;
        LES = run_input.LES;
        sgs_model = run_input.SGS_model;

        // Set filter flag before calling setup_ele_type_specific
        LES_filter = 0;
        if (LES && (sgs_model == 3 || sgs_model == 2 || sgs_model == 4))
            LES_filter = 1;

        // Initialize the element specific static members
        (*this).setup_ele_type_specific();

        //initialize probes
        if(run_input.probe)
            opp_probe.setup(n_upts_per_ele);

        if(run_input.adv_type==0)//Euler
        {
            n_adv_levels=1;
        }

        else if(run_input.adv_type==1||run_input.adv_type==2||run_input.adv_type==3||run_input.adv_type==4)//SSP-RK24/SSP-RK34/RK45/SSP-RK414
        {
            n_adv_levels=2;
        }
        else
        {
            cout << "ERROR: Type of time integration scheme not recongized ... " << endl;
        }

        // Allocate storage for solution
        disu_upts.setup(n_adv_levels);
        for(int i=0; i<n_adv_levels; i++)
        {
            disu_upts(i).setup(n_upts_per_ele,n_eles,n_fields);
        }
        //initialize the second register for solution value to 0
        if (n_adv_levels != 1)
        {
            for (int m = 1; m < disu_upts.get_dim(0); m++)
                disu_upts(m).initialize_to_zero();
        }

        // Allocate storage for timestep
        // If using local, one timestep per element
        if(run_input.dt_type == 2)
            dt_local.setup(n_eles);

        // Set no. of diagnostic fields
        n_diagnostic_fields = run_input.n_diagnostic_fields;

        // Set no. of diagnostic fields
        n_average_fields = run_input.n_average_fields;

        // Allocate storage for time-averaged velocity components
        if(n_average_fields > 0)
        {
            disu_average_upts.setup(n_upts_per_ele,n_eles,n_average_fields);
            disu_average_upts.initialize_to_zero();
        }

        // Allocate extra arrays for LES models
        if(LES)
        {

            sgsf_upts.setup(n_upts_per_ele,n_eles,n_fields,n_dims);
            sgsf_fpts.setup(n_fpts_per_ele,n_eles,n_fields,n_dims);
            sgsf_fpts.initialize_to_zero();
             // SVV model requires filtered solution
            if(LES_filter)
            {
                disuf_upts.setup(n_upts_per_ele,n_eles,n_fields);
                disuf_upts.initialize_to_zero();
            }

            // Similarity model requires product terms and Leonard tensors
            if(sgs_model==2 || sgs_model==4)
            {

                // Leonard tensor and velocity-velocity product for momentum SGS term
                if(n_dims==2)
                {
                    Lu.setup(n_upts_per_ele,n_eles,3);
                    uu.setup(n_upts_per_ele,n_eles,3);
                }
                else if(n_dims==3)
                {
                    Lu.setup(n_upts_per_ele,n_eles,6);
                    uu.setup(n_upts_per_ele,n_eles,6);
                }
                Lu.initialize_to_zero();
                // Leonard tensor and velocity-energy product for energy SGS term
                Le.setup(n_upts_per_ele,n_eles,n_dims);
                ue.setup(n_upts_per_ele,n_eles,n_dims);
                Le.initialize_to_zero();
            }
            // Allocate SGS flux hf_array if using LES
            temp_sgsf.setup(n_fields,n_dims);
        }

        // Allocate hf_array for wall distance or each solution point if using a RANS-based turbulence model or LES wall model
        if (run_input.RANS)//S-A
        {
            wall_distance.setup(n_upts_per_ele,n_eles,n_dims);
            wall_distance_mag.setup(n_upts_per_ele,n_eles);
        }
        else if (LES && sgs_model == 0) //Smagorinsky model use wall distance for near-wall damping
            wall_distance.setup(n_upts_per_ele, n_eles, n_dims);

        // Initialize source term
        src_upts.setup(n_upts_per_ele, n_eles, n_fields);
        src_upts.initialize_to_zero();


        set_shape(in_max_n_spts_per_ele);//set shape array and n_spts_per_ele

        // Allocate storage for the s_nodal_basis
        d_nodal_s_basis.setup(max_n_spts_per_ele,n_dims);

        ele2global_ele.setup(n_eles);
        bcid.setup(n_eles,n_inters_per_ele);

        n_fields_mul_n_eles=n_fields*n_eles;
        n_dims_mul_n_upts_per_ele=n_dims*n_upts_per_ele;

        //over-integration need second register to store divergence of transformed continuous flux
        div_tconf_upts.setup(1);
        div_tconf_upts(0).setup(n_upts_per_ele,n_eles,n_fields);
        div_tconf_upts(0).initialize_to_zero();

        disu_fpts.setup(n_fpts_per_ele,n_eles,n_fields);
        disu_fpts.initialize_to_zero();
        tdisf_upts.setup(n_upts_per_ele,n_eles,n_fields,n_dims);
        norm_tdisf_fpts.setup(n_fpts_per_ele,n_eles,n_fields);
        norm_tdisf_fpts.initialize_to_zero();
        norm_tconf_fpts.setup(n_fpts_per_ele,n_eles,n_fields);

        if(viscous)
        {
            delta_disu_fpts.setup(n_fpts_per_ele,n_eles,n_fields);
            grad_disu_upts.setup(n_upts_per_ele,n_eles,n_fields,n_dims);
            grad_disu_upts.initialize_to_zero();
            grad_disu_fpts.setup(n_fpts_per_ele,n_eles,n_fields,n_dims);
            grad_disu_fpts.initialize_to_zero();
        }

        if(run_input.shock_cap)
        {
            sensor.setup(n_eles);
            sensor.initialize_to_zero();
        }

        // Set connectivity hf_array. Needed for output.

        connectivity_plot.setup(n_verts_per_ele,n_peles_per_ele);

        set_connectivity_plot();
    }

}

hf_array<int> eles::get_connectivity_plot()
{
    return connectivity_plot;
}

// set initial conditions

void eles::set_ics(double& time)
{
    int i,j,k;

    double rho,vx,vy,vz,p;
    double gamma=run_input.gamma;
    time = 0.;

    hf_array<double> pos(n_dims);
    hf_array<double> ics(n_fields);

    hf_array<double> grad_rho(n_dims);

    for(i=0; i<n_eles; i++)
    {
        for(j=0; j<n_upts_per_ele; j++)
        {
            // calculate position of solution point
            for(k=0; k<n_dims; k++)
            {
                pos(k)=pos_upts(j,i,k);
            }

            // evaluate solution at solution point
            if(run_input.ic_form==0)//isentropic vortex at (0,0)
            {
                eval_isentropic_vortex(pos,time,rho,vx,vy,vz,p,n_dims);

                ics(0)=rho;
                ics(1)=rho*vx;
                ics(2)=rho*vy;
                if(n_dims==2)
                {
                    ics(3)=(p/(gamma-1.0))+(0.5*rho*((vx*vx)+(vy*vy)));
                }
                else if(n_dims==3)
                {
                    ics(3)=rho*vz;
                    ics(4)=(p/(gamma-1.0))+(0.5*rho*((vx*vx)+(vy*vy)+(vz*vz)));
                }
                else
                {
                    cout << "ERROR: Invalid number of dimensions ... " << endl;
                }
            }
            else if(run_input.ic_form==1)
            {
                rho=run_input.rho_c_ic;
                vx=run_input.u_c_ic;
                vy=run_input.v_c_ic;
                vz=run_input.w_c_ic;
                p=run_input.p_c_ic;

                ics(0)=rho;
                ics(1)=rho*vx;
                ics(2)=rho*vy;
                if(n_dims==2)
                {
                    ics(3)=(p/(gamma-1.0))+(0.5*rho*((vx*vx)+(vy*vy)));

                    if (run_input.RANS==1)
                    {
                        ics(4) = run_input.mu_tilde_c_ic;
                    }
                }
                else if(n_dims==3)
                {
                    ics(3)=rho*vz;
                    ics(4)=(p/(gamma-1.0))+(0.5*rho*((vx*vx)+(vy*vy)+(vz*vz)));

                    if(run_input.RANS==1)
                    {
                        ics(5) = run_input.mu_tilde_c_ic;
                    }
                }
                else
                {
                    cout << "ERROR: Invalid number of dimensions ... " << endl;
                }

            }
            else if (run_input.ic_form==2) // Sine wave (single)
            {
                eval_sine_wave_single(pos,run_input.wave_speed,run_input.diff_coeff,time,rho,grad_rho,n_dims);
                ics(0) = rho;
            }
            else if (run_input.ic_form==3) // Sine wave (group)
            {
                eval_sine_wave_group(pos,run_input.wave_speed,run_input.diff_coeff,time,rho,grad_rho,n_dims);
                ics(0) = rho;
            }
            else if (run_input.ic_form==4) // Spherical distribution
            {
                eval_sphere_wave(pos,run_input.wave_speed,time,rho,n_dims);
                ics(0) = rho;
            }
            else if (run_input.ic_form==5) // Constant for adv-diff
            {
                ics(0) = run_input.rho_c_ic;
            }
            else if (run_input.ic_form==6) // const rho p,Up to 4th order polynomials for u, v, w
            {
                rho=run_input.rho_c_ic;
                p=run_input.p_c_ic;
                eval_poly_ic(pos,rho,ics,n_dims);
                ics(0) = rho;
                if(n_dims==2)
                    ics(3)=p/(gamma-1.0)+0.5*(ics(1)*ics(1)+ics(2)*ics(2))/rho;
                else if(n_dims==3)
                    ics(4)=p/(gamma-1.0)+0.5*(ics(1)*ics(1)+ics(2)*ics(2)+ics(3)*ics(3))/rho;
            }
            else if (run_input.ic_form==7) // Taylor-Green Vortex initial conditions
            {
                double V_0=run_input.uvw_c_ic / run_input.uvw_ref;
                if (n_dims == 2)
                {
                    // Simple 2D div-free vortex
                    p = run_input.p_c_ic + run_input.rho_c_ic * pow(V_0, 2) / 4.0 *
                                               (cos(2.0 * pos(0)) + cos(2.0 * pos(1)));     //p_0+rho_0*V_0^2/4*(cos(2x/L)+cos(2*y/L))
                    ics(0) = p / (run_input.R_ref * run_input.T_c_ic);                      //p/(R*T_0)
                    ics(1) = ics(0) * V_0 * sin(pos(0)) * cos(pos(1));                      //rho*V_0*sin(x/L)*cos(y/L)
                    ics(2) = -ics(0) * V_0 * cos(pos(0)) * sin(pos(1));                     //-rho*V_0*cos(x/L)*sin(y/L)
                    ics(3) = p / (gamma - 1.0) + 0.5 * (ics(1) * ics(1) + ics(2) * ics(2))/ics(0); //p/(gamma-1)+0.5(rhoV)^2/rho
                }
                else if(n_dims==3)
                {
                    p = run_input.p_c_ic + run_input.rho_c_ic * pow(V_0, 2) / 16.0 *
                                               (cos(2.0 * pos(0)) + cos(2.0 * pos(1))) * (cos(2.0 * pos(2)) + 2.0); //p_0+rho_0*V_0^2/16*(cos(2x/L)+cos(2*y/L))*(cos(2*z/L)+2)
                    ics(0) = p / (run_input.R_ref * run_input.T_c_ic);                                              //p/(R*T_0)
                    ics(1) = ics(0) * V_0 * sin(pos(0)) * cos(pos(1)) * cos(pos(2));                                //rho*V_0*sin(x/L)*cos(y/L)*cos(z/L)
                    ics(2) = -ics(0) * V_0 * cos(pos(0)) * sin(pos(1)) * cos(pos(2));                               //rho*V_0*cos(x/L)*sin(y/L)*cos(z/L)
                    ics(3) = 0.0;
                    ics(4) = p / (gamma - 1.0) + 0.5 * (ics(1) * ics(1) + ics(2) * ics(2) + ics(3) * ics(3)) / ics(0); //p/(gamma-1)+0.5(rhoV)^2/rho
                }
            }
            else if(run_input.ic_form==9)//stationary shock
            {
                int found=0;
                for (k = 0; k < run_input.bc_list.get_dim(0); k++)
                {
                    if (run_input.bc_list(k).get_bc_flag() == SUP_IN || run_input.bc_list(k).get_bc_flag() == CHAR)
                    {
                        if (pos(0) <= run_input.x_shock_ic) //supersonic zone
                        {
                            rho = run_input.bc_list(k).rho;
                            vx = run_input.bc_list(k).velocity(0);
                            vy = run_input.bc_list(k).velocity(1);
                            vz = run_input.bc_list(k).velocity(2);
                            p = run_input.bc_list(k).p_static;
                        }
                        else //subsonic zone initialize
                        {
                            rho = run_input.rho_c_ic;
                            vx = run_input.u_c_ic;
                            vy = run_input.v_c_ic;
                            vz = run_input.w_c_ic;
                            p = run_input.p_c_ic;
                        }
                        found =1;
                        break;
                    }
                }
                if(found==0)
                    FatalError("Must have a Sup_In or Char boundary condition");

                /*! initialize uniform flow with normal shock stationary condition*/

                ics(0)=rho;
                ics(1)=rho*vx;
                ics(2)=rho*vy;
                if(n_dims==2)
                {
                    ics(3)=(p/(gamma-1.0))+(0.5*rho*((vx*vx)+(vy*vy)));

                    if (run_input.RANS==1)
                    {
                        ics(4) = run_input.mu_tilde_c_ic;
                    }
                }
                else if(n_dims==3)
                {
                    ics(3)=rho*vz;
                    ics(4)=(p/(gamma-1.0))+(0.5*rho*((vx*vx)+(vy*vy)+(vz*vz)));

                    if(run_input.RANS==1)
                    {
                        ics(5) = run_input.mu_tilde_c_ic;
                    }
                }

                else
                {
                    cout << "ERROR: Invalid number of dimensions ... " << endl;
                }
            }
            else if (run_input.ic_form==10) //shock tube initial condition
            {
                vx=0.;
                vy=0.;
                vz=0.;
                if (pos(0)<=run_input.x_shock_ic)
                {
                    if(run_input.viscous)
                    {
                        p=100000./run_input.p_ref;
                        rho=1.0/run_input.rho_ref;
                    }
                    else
                    {
                        p=100000.;
                        rho=1.0;
                    }
                }
                else
                {
                    if(run_input.viscous)
                    {
                        p=10000./run_input.p_ref;
                        rho=0.125/run_input.rho_ref;
                    }
                    else
                    {
                        p=10000.;
                        rho=0.125;
                    }
                }
                ics(0)=rho;
                ics(1)=rho*vx;
                ics(2)=rho*vy;
                if(n_dims==2)
                {
                    ics(3)=(p/(gamma-1.0))+(0.5*rho*((vx*vx)+(vy*vy)));

                    if (run_input.RANS==1)
                    {
                        ics(4) = run_input.mu_tilde_c_ic;
                    }
                }
                else if(n_dims==3)
                {
                    ics(3)=rho*vz;
                    ics(4)=(p/(gamma-1.0))+(0.5*rho*((vx*vx)+(vy*vy)+(vz*vz)));

                    if(run_input.RANS==1)
                    {
                        ics(5) = run_input.mu_tilde_c_ic;
                    }
                }
            }
            else
            {
               FatalError("ERROR: Invalid form of initial condition ...");
            }

            // Add perturbation to channel
            if(run_input.perturb_ic==1 and n_dims==3)
            {
                // Constructs the function
                // u = alpha exp(-((x-L_x/2)/L_x)^2) exp(-(y/L_y)^2) cos(4 pi z/L_z)
                // Where L_x, L_y, L_z are the domain dimensions, u_0=bulk velocity,
                // alpha=scale factor, u=wall normal velocity
                double alpha, L_x, L_y, L_z;
                alpha = 0.1;
                L_x = 2.*pi;
                L_y = pi;
                L_z = 2.;
                ics(3) += alpha*exp(-pow((pos(0)-L_x/2.)/L_x,2))*exp(-pow(pos(1)/L_y,2))*cos(4.*pi*pos(2)/L_z);
            }

            // set solution at solution point
            for(k=0; k<n_fields; k++)
            {
                disu_upts(0)(j,i,k)=ics(k);
            }
        }
    }

    // If required, calculate element reference lengths
    if (run_input.dt_type > 0)
    {
        // Allocate hf_array
        h_ref.setup(n_eles);
        // Call element specific function to obtain length
        for (int i=0; i<n_eles; i++)
        {
            h_ref(i) = (*this).calc_h_ref_specific(i);
        }
    }
    else
    {
        h_ref.setup(1);
    }
#ifdef _GPU
    h_ref.cp_cpu_gpu();
#endif
}


// set patch

void eles::set_patch(void)
{
    double rho,vx,vy,vz,p,temper;
    double gamma=run_input.gamma;
    double temp_R;
    hf_array<double> pos(n_dims);
    double rho_temp;
    if (viscous) //Navier
        temp_R = run_input.R_ref;
    else //Euler
        temp_R = run_input.R_gas;
    for(int i=0; i<n_eles; i++)
    {
        for(int j=0; j<n_upts_per_ele; j++)
        {
            for(int k=0; k<n_dims; k++)
            {
                pos(k)=pos_upts(j,i,k);
            }
            if(run_input.patch_type==0)//vortex
            {
                //set parameters
                double Mv=run_input.Mv;//vortex strength
                double ra=run_input.ra;//inner radii
                double rb=run_input.rb;//outer radii
                double xc=run_input.xc;//core location x
                double yc=run_input.yc;//core location y
                double r=sqrt(pow(pos(0)-xc,2)+pow(pos(1)-yc,2));//distance to core

                if (r<=rb)//in range of vortex set rho u v p
                {
                    /*! copy solution to rho, vx, vy, vz, p*/
                    rho=disu_upts(0)(j,i,0);
                    vx=disu_upts(0)(j,i,1)/disu_upts(0)(j,i,0);
                    vy=disu_upts(0)(j,i,2)/disu_upts(0)(j,i,0);

                    if(n_dims==2)
                    {
                        p=(disu_upts(0)(j,i,3)-0.5*rho*(vx*vx+vy*vy))*(gamma-1.0);
                    }
                    else
                    {
                        vz=disu_upts(0)(j,i,3)/disu_upts(0)(j,i,0) ;
                        p=(disu_upts(0)(j,i,4)-0.5*rho*(vx*vx+vy*vy+vz*vz))*(gamma-1.0);
                    }

                    double vm=Mv*sqrt(gamma*p/rho);//max vortex angular velocity
                    if (r<=ra)
                    {
                        vx-=(pos(1)-yc)/r*vm*r/ra;
                        vy+=(pos(0)-xc)/r*vm*r/ra;
                        temper=p/(rho*temp_R)-(gamma-1)/(temp_R*gamma)*(pow(vm,2)/pow(ra,2)*0.5*(pow(ra,2)-pow(r,2))
                                +pow(vm,2)*pow(ra,2)/pow(pow(ra,2)-pow(rb,2),2)*(0.5*(pow(rb,2)-pow(ra,2))-0.5*pow(rb,4)*(1/pow(rb,2)-1/pow(ra,2))
                                        -2*pow(rb,2)*(log(rb/ra))));
                    }
                    else
                    {
                        vx-=(pos(1)-yc)/r*vm*ra/(pow(ra,2)-pow(rb,2))*(r-pow(rb,2)/r);
                        vy+=(pos(0)-xc)/r*vm*ra/(pow(ra,2)-pow(rb,2))*(r-pow(rb,2)/r);
                        temper=p/(rho*temp_R)-(gamma-1)/(temp_R*gamma)*pow(vm,2)*pow(ra,2)/pow(pow(ra,2)-pow(rb,2),2)*(0.5*(pow(rb,2)-pow(r,2))-0.5*pow(rb,4)*(1/(pow(rb,2))-1/(pow(r,2)))
                                -2*pow(rb,2)*(log(rb/r)));
                    }
                    rho_temp=rho;
                    rho=rho*pow(temper/(p/(rho*temp_R)),1/(gamma-1));
                    p=p*pow(temper/(p/(rho_temp*temp_R)),gamma/(gamma-1));
                    //copy solution back to solution hf_array
                    disu_upts(0)(j,i,0)=rho;
                    disu_upts(0)(j,i,1)=rho*vx;
                    disu_upts(0)(j,i,2)=rho*vy;
                    if(n_dims==2)
                    {
                        disu_upts(0)(j,i,3)=(p/(gamma-1.0))+(0.5*rho*((vx*vx)+(vy*vy)));
                    }
                    else if(n_dims==3)
                    {
                        disu_upts(0)(j,i,3)=rho*vz;
                        disu_upts(0)(j,i,4)=(p/(gamma-1.0))+(0.5*rho*((vx*vx)+(vy*vy)+(vz*vz)));
                    }
                    else
                    {
                        cout << "ERROR: Invalid number of dimensions ... " << endl;
                    }
                }
            }
            else if (run_input.patch_type==1)//uniform patch with ic for x greater than patch_x
            {
                if(pos(0)>=run_input.patch_x)
                {
                    rho=run_input.rho_c_ic;
                    vx=run_input.u_c_ic;
                    vy=run_input.v_c_ic;
                    vz=run_input.w_c_ic;
                    p=run_input.p_c_ic;
                        //copy solution back to solution hf_array
                    disu_upts(0)(j,i,0)=rho;
                    disu_upts(0)(j,i,1)=rho*vx;
                    disu_upts(0)(j,i,2)=rho*vy;
                    if(n_dims==2)
                    {
                        disu_upts(0)(j,i,3)=(p/(gamma-1.0))+(0.5*rho*((vx*vx)+(vy*vy)));
                    }
                    else if(n_dims==3)
                    {
                        disu_upts(0)(j,i,3)=rho*vz;
                        disu_upts(0)(j,i,4)=(p/(gamma-1.0))+(0.5*rho*((vx*vx)+(vy*vy)+(vz*vz)));
                    }
                    else
                    {
                        cout << "ERROR: Invalid number of dimensions ... " << endl;
                    }
                }
            }
            else
                FatalError("ERROR: Invalid form of patch ... ");
        }
    }
}

void eles::read_restart_data_ascii(ifstream& restart_file)
{

    int num_eles_to_read;
    string ele_name,str;
    if (ele_type==0) ele_name="TRIS";
    else if (ele_type==1) ele_name="QUADS";
    else if (ele_type==2) ele_name="TETS";
    else if (ele_type==3) ele_name="PRIS";
    else if (ele_type==4) ele_name="HEXAS";

    // Move cursor to correct element type

    restart_file.clear();
    restart_file.seekg(0,restart_file.beg);

    while(1)
    {
        getline(restart_file,str);
        if (str==ele_name) break;
        if (restart_file.eof()) return; // Restart file doesn't contain my elements
    }

    // Move cursor to n_eles
    while(1)
    {
        getline(restart_file,str);
        if (str=="n_eles") break;
    }

    // Read number of elements to read
    restart_file >> num_eles_to_read;
    getline(restart_file,str);

    //Skip ele2global_ele lines
    getline(restart_file,str);
    getline(restart_file,str);
    getline(restart_file,str);

    int ele,index;
    hf_array<double> disu_upts_rest;
    disu_upts_rest.setup(n_upts_per_ele_rest,n_fields);

    for (int i=0; i<num_eles_to_read; i++)
    {
        restart_file >> ele ;
        index = index_locate_int(ele,ele2global_ele.get_ptr_cpu(),n_eles);

        if (index!=-1) // Ele belongs to processor
        {
            for (int j=0; j<n_upts_per_ele_rest; j++)
                for (int k=0; k<n_fields; k++)
                    restart_file >> disu_upts_rest(j,k);

            // Now compute transformed solution at solution points using opp_r
            for (int m=0; m<n_fields; m++)
            {
                for (int j=0; j<n_upts_per_ele; j++)
                {
                    double value = 0.;
                    for (int k=0; k<n_upts_per_ele_rest; k++)
                        value += opp_r(j,k)*disu_upts_rest(k,m);

                    disu_upts(0)(j,index,m) = value;
                }
            }
        }
        else // Skip the data (doesn't belong to current processor)
        {
            // Skip rest of ele line
            getline(restart_file,str);
            for (int j=0; j<n_upts_per_ele_rest; j++)
                getline(restart_file,str);
        }
    }

    // If required, calculate element reference lengths
    if (run_input.dt_type > 0)
    {
        // Allocate hf_array
        h_ref.setup(n_eles);

        // Call element specific function to obtain length
        for (int i=0; i<n_eles; i++)
        {
            h_ref(i) = (*this).calc_h_ref_specific(i);
        }
    }
    else
    {
        h_ref.setup(1);
    }
#ifdef _GPU
    h_ref.cp_cpu_gpu();
#endif
}

#ifdef _HDF5
void eles::read_restart_data_hdf5(hid_t &restart_file)
{
    hid_t dataset_id, plist_id, memspace_id, dataspace_id;
    //dimensions for hyperslab
    hsize_t dim[3];
    hsize_t count[3];
    hsize_t offset[3];
    hf_array<double> disu_upts_rest;

    //open dataset
    dataset_id = H5Dopen2(restart_file, "data", H5P_DEFAULT);
    if (dataset_id < 0)
        FatalError("Failed to open restart data");

    plist_id = H5Pcreate(H5P_DATASET_XFER);
#ifdef _MPI
    //set collective read
    H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);
#endif

    if (n_eles)
    {
        dim[0] = n_fields;
        dim[1] = n_eles;
        dim[2] = n_upts_per_ele_rest;
        disu_upts_rest.setup(n_upts_per_ele_rest, n_eles, n_fields); 
        disu_upts(0).initialize_to_zero();//initialize solution array
        //set first subset to read
        memspace_id = H5Screate_simple(3, dim, NULL); 
        dataspace_id = H5Dget_space(dataset_id);
        offset[0] = 0;
        offset[1] = ele2global_ele(0); //based on ascending order of ele2global_ele
        offset[2] = 0;
        count[0] = dim[0];
        count[1] = 1; 
        count[2] = dim[2];
        if (H5Sselect_hyperslab(dataspace_id, H5S_SELECT_SET, offset, NULL, count, NULL) < 0)
            FatalError("Failed to find this element");
        for (int i = 1; i < n_eles; i++) //for other elements of this type
        {
            //add other subsets to read
            offset[1] = ele2global_ele(i);
            if (H5Sselect_hyperslab(dataspace_id, H5S_SELECT_OR, offset, NULL, count, NULL) < 0)
                FatalError("Failed to find this element");
        }

        H5Dread(dataset_id, H5T_NATIVE_DOUBLE, memspace_id, dataspace_id, plist_id, disu_upts_rest.get_ptr_cpu());
        //close ibjects
        H5Sclose(memspace_id);
        H5Sclose(dataspace_id);

        //extrapolate to solution points
#if defined _ACCELERATE_BLAS || defined _MKL_BLAS || defined _STANDARD_BLAS
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n_upts_per_ele, n_fields_mul_n_eles, n_upts_per_ele_rest, 1.0, opp_r.get_ptr_cpu(), n_upts_per_ele, disu_upts_rest.get_ptr_cpu(), n_upts_per_ele_rest, 0.0, disu_upts(0).get_ptr_cpu(), n_upts_per_ele);
#else
        dgemm(n_upts_per_ele, n_fields_mul_n_eles, n_upts_per_ele_rest, 1.0, 0.0, opp_r.get_ptr_cpu(), disu_upts_rest.get_ptr_cpu(), disu_upts(0).get_ptr_cpu());
#endif

        // If required, calculate element reference lengths
        if (run_input.dt_type > 0)
        {
            // Allocate hf_array
            h_ref.setup(n_eles);
            // Call element specific function to obtain length
            for (int i = 0; i < n_eles; i++)
                h_ref(i) = (*this).calc_h_ref_specific(i);
        }
        else
        {
            h_ref.setup(1);
        }
#ifdef _GPU
    h_ref.cp_cpu_gpu();
#endif
    }
#ifdef _MPI
    else//read empty
    {
        dataspace_id=H5Dget_space(dataset_id);
        H5Sselect_none(dataspace_id);
        H5Dread(dataset_id, H5T_NATIVE_DOUBLE, dataspace_id, dataspace_id, plist_id, NULL);
        H5Sclose(dataspace_id);
    }
#endif

    //close objects
    H5Pclose(plist_id);
    H5Dclose(dataset_id);
}
#endif

#ifndef _HDF5
void eles::write_restart_data_ascii(ofstream& restart_file)
{
    restart_file << "n_eles" << endl;
    restart_file << n_eles << endl;
    restart_file << "ele2global_ele hf_array" << endl;
    for (int i=0; i<n_eles; i++)
        restart_file << ele2global_ele(i) << " ";
    restart_file << endl;

    restart_file << "data" << endl;

    for (int i=0; i<n_eles; i++)
    {
        restart_file << ele2global_ele(i) << endl;
        for (int j=0; j<n_upts_per_ele; j++)
        {
            for (int k=0; k<n_fields; k++)
            {
                restart_file << disu_upts(0)(j,i,k) << " ";
            }
            restart_file << endl;
        }
    }
    restart_file << endl;
}
#endif

#ifdef _HDF5
void eles::write_restart_data_hdf5(hid_t &in_dataset_id)
{
    //dimensions for hyperslab
    hid_t plist_id, memspace_id, dataspace_id;
    hsize_t dim[3];
    hsize_t count[3];
    hsize_t offset[3];

    plist_id = H5Pcreate(H5P_DATASET_XFER);
#ifdef _MPI
    //set collective read
    H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);
#endif

    if (n_eles)
    {
        dim[0] = n_fields;
        dim[1] = n_eles;
        dim[2] = n_upts_per_ele;
        //set first subset to read
        memspace_id = H5Screate_simple(3, dim, NULL);
        dataspace_id = H5Dget_space(in_dataset_id);
        offset[0] = 0;
        offset[1] = ele2global_ele(0); //based on ascending order of ele2global_ele
        offset[2] = 0;
        count[0] = dim[0];
        count[1] = 1;
        count[2] = dim[2];
        if (H5Sselect_hyperslab(dataspace_id, H5S_SELECT_SET, offset, NULL, count, NULL) < 0)
            FatalError("Failed to find this element");
        for (int i = 1; i < n_eles; i++) //for other elements of this type
        {
            //add other subsets to read
            offset[1] = ele2global_ele(i);
            if (H5Sselect_hyperslab(dataspace_id, H5S_SELECT_OR, offset, NULL, count, NULL) < 0)
                FatalError("Failed to find this element");
        }
        H5Dwrite(in_dataset_id, H5T_NATIVE_DOUBLE, memspace_id, dataspace_id, plist_id, disu_upts(0).get_ptr_cpu());
        //close objects
        H5Sclose(memspace_id);
        H5Sclose(dataspace_id);
    }
#ifdef _MPI
    else
    {
        dataspace_id = H5Dget_space(in_dataset_id);
        H5Sselect_none(dataspace_id);
        H5Dwrite(in_dataset_id, H5T_NATIVE_DOUBLE, dataspace_id, dataspace_id, plist_id, NULL);
        H5Sclose(dataspace_id);
    }
#endif
    //close objects
    H5Pclose(plist_id);
}
#endif

#ifdef _GPU
// move all to from cpu to gpu
void eles::mv_all_cpu_gpu(void)
{
    if (n_eles!=0)
    {
        disu_upts(0).cp_cpu_gpu();
        div_tconf_upts(0).cp_cpu_gpu();
        src_upts.cp_cpu_gpu();

        for(int i=1; i<n_adv_levels; i++)
            disu_upts(i).cp_cpu_gpu();
        for (int i = 1; i < div_tconf_upts.get_dim(0); i++)
            div_tconf_upts(1).cp_cpu_gpu();
        disu_fpts.mv_cpu_gpu();
        tdisf_upts.mv_cpu_gpu();
        norm_tdisf_fpts.mv_cpu_gpu();
        norm_tconf_fpts.mv_cpu_gpu();

        //TODO: mv instead of cp
        if(viscous)
        {
            delta_disu_fpts.mv_cpu_gpu();
            grad_disu_upts.cp_cpu_gpu();
            grad_disu_fpts.mv_cpu_gpu();

            //tdisvisf_upts.mv_cpu_gpu();
            //norm_tdisvisf_fpts.mv_cpu_gpu();
            //norm_tconvisf_fpts.mv_cpu_gpu();
        }

        // LES and wall model arrays
        filter_upts.mv_cpu_gpu();
        disuf_upts.mv_cpu_gpu();
        sgsf_upts.mv_cpu_gpu();
        sgsf_fpts.mv_cpu_gpu();
        uu.mv_cpu_gpu();
        ue.mv_cpu_gpu();
        Lu.mv_cpu_gpu();
        Le.mv_cpu_gpu();
        twall.mv_cpu_gpu();

        if(run_input.shock_cap)
        {
            // Needed for shock capturing routines
            sensor.cp_cpu_gpu();
            inv_vandermonde.mv_cpu_gpu();

            if(run_input.shock_det == 1)
                concentration_array.mv_cpu_gpu();
                if(run_input.shock_cap==1)
                 sigma.mv_cpu_gpu();
        }
    }
}

// move wall distance hf_array to gpu
void eles::mv_wall_distance_cpu_gpu(void)
{
    wall_distance.mv_cpu_gpu();
}

// move wall distance magnitude hf_array to gpu
void eles::mv_wall_distance_mag_cpu_gpu(void)
{
    wall_distance_mag.mv_cpu_gpu();
}

// copy discontinuous solution at solution points to cpu
void eles::cp_disu_upts_gpu_cpu(void)
{
    if (n_eles!=0)
    {
        disu_upts(0).cp_gpu_cpu();
    }
}


// copy discontinuous solution at solution points to gpu
void eles::cp_disu_upts_cpu_gpu(void)
{
    if (n_eles!=0)
    {
        disu_upts(0).cp_cpu_gpu();
    }
}

// copy gradient of discontinuous solution at solution points to cpu
void eles::cp_grad_disu_upts_gpu_cpu(void)
{
    if (n_eles!=0)
    {
        grad_disu_upts.cp_gpu_cpu();
    }
}

// copy determinant of jacobian at solution points to cpu
void eles::cp_detjac_upts_gpu_cpu(void)
{
    detjac_upts.cp_gpu_cpu();
}

// copy divergence at solution points to cpu
void eles::cp_div_tconf_upts_gpu_cpu(void)
{
    if (n_eles!=0)
    {
        div_tconf_upts(0).cp_gpu_cpu();
    }
}

// copy local time stepping reference length at solution points to cpu
void eles::cp_h_ref_gpu_cpu(void)
{
    h_ref.cp_gpu_cpu();
}

// copy source term at solution points to cpu
void eles::cp_src_upts_gpu_cpu(void)
{
    if (n_eles!=0)
    {
        src_upts.cp_gpu_cpu();
    }
}

// copy sensor in each element to cpu
void eles::cp_sensor_gpu_cpu(void)
{
    if (n_eles!=0)
    {
        sensor.cp_gpu_cpu();
    }
}

// remove transformed discontinuous solution at solution points from cpu

void eles::rm_disu_upts_cpu(void)
{
    disu_upts(0).rm_cpu();
}

// remove determinant of jacobian at solution points from cpu

void eles::rm_detjac_upts_cpu(void)
{
    detjac_upts.rm_cpu();
}
#endif
// advance solution

void eles::AdvanceSolution(int in_step, int adv_type)
{

    if (n_eles!=0)
    {
        int i, ic, inp;
        double rhs;     
      
        /*! Time integration using a forwards Euler integration. */    

        if (adv_type == 0)
        {  

#ifdef _CPU

            for (i=0; i<n_fields; i++)
            {
                for (ic=0; ic<n_eles; ic++)
                {
                    for(inp=0; inp<n_upts_per_ele; inp++)
                    {
                        if (run_input.dt_type == 2)//local time step
                            disu_upts(0)(inp, ic, i) -= dt_local(ic) * (div_tconf_upts(0)(inp, ic, i) / detjac_upts(inp, ic) - src_upts(inp, ic, i));
                        else//global time step
                            disu_upts(0)(inp, ic, i) -= run_input.dt * (div_tconf_upts(0)(inp, ic, i) / detjac_upts(inp, ic) - src_upts(inp, ic, i));
                    }
                }
            }

#endif

#ifdef _GPU
            FatalError("GPU version of Euler method unavailable!");
            //RK11_update_kernel_wrapper(n_upts_per_ele,n_dims,n_fields,n_eles,disu_upts(0).get_ptr_gpu(),div_tconf_upts(0).get_ptr_gpu(),detjac_upts.get_ptr_gpu(),src_upts.get_ptr_gpu(),h_ref.get_ptr_gpu(),run_input.dt,run_input.const_src,run_input.CFL,run_input.gamma,run_input.mu_inf,run_input.order,viscous,run_input.dt_type);
#endif
        }

        /*!Time integration using a SSP-RK24(2N) method.
        RK24/RK34
        Ketcheson D I. 
        Highly efficient strong stability-preserving 
        Runge–Kutta methods with low-storage implementations
        SIAM Journal on Scientific Computing, 2008*/
        else if (adv_type==1)
        {
#ifdef _CPU

            if (in_step == 0)
                disu_upts(1) = disu_upts(0); //copy solution to register 2
            if (in_step < 3) //first 3 stages
            {
                //u=u+(-dt/3*F)
                for (i = 0; i < n_fields; i++)
                {
                    for (ic = 0; ic < n_eles; ic++)
                    {
                        for (inp = 0; inp < n_upts_per_ele; inp++)
                        {
                            if (run_input.dt_type == 2)//local time step
                                disu_upts(0)(inp, ic, i) -= dt_local(ic) / 3.0 * (div_tconf_upts(0)(inp, ic, i) / detjac_upts(inp, ic) - src_upts(inp, ic, i));
                            else//global time step
                                disu_upts(0)(inp, ic, i) -= run_input.dt / 3.0 * (div_tconf_upts(0)(inp, ic, i) / detjac_upts(inp, ic) - src_upts(inp, ic, i));
                        }
                    }
                }
            }
            else //the last stage
            {
                //u=3/4*u+u1/4+(-dt/4*F)
                for (i = 0; i < n_fields; i++)
                {
                    for (ic = 0; ic < n_eles; ic++)
                    {
                        for (inp = 0; inp < n_upts_per_ele; inp++)
                        {
                            rhs = -div_tconf_upts(0)(inp, ic, i) / detjac_upts(inp, ic) + src_upts(inp, ic, i); //function
                            if (run_input.dt_type == 2)//local time step
                                disu_upts(0)(inp, ic, i) = 3.0 / 4.0 * disu_upts(0)(inp, ic, i) + 1.0 / 4.0 * disu_upts(1)(inp, ic, i) + dt_local(ic) / 4.0 * rhs;
                            else//global time step
                                disu_upts(0)(inp, ic, i)= 3.0 / 4.0 * disu_upts(0)(inp, ic, i) + 1.0 / 4.0 * disu_upts(1)(inp, ic, i) + run_input.dt / 4.0 * rhs;
                        }
                    }
                }
            }

#endif

#ifdef _GPU
            FatalError("GPU version of SSP-RK24 unavailable!");
#endif
        }

        /*! Time integration using a RK34(2N) method. */
        else if (adv_type == 2)
        {
#ifdef _CPU

            if (in_step == 0)//first stage only
                disu_upts(1) = disu_upts(0); //copy to register 2
            if (in_step < 2||in_step==3)//stage 1 && 2 && 4
            {
                //u=u+(-dt/2*F)
                for (i = 0; i < n_fields; i++)
                {
                    for (ic = 0; ic < n_eles; ic++)
                    {
                        for (inp = 0; inp < n_upts_per_ele; inp++)
                        {
                            if (run_input.dt_type == 2)//local time step
                                disu_upts(0)(inp, ic, i) -= dt_local(ic) / 2.0 * (div_tconf_upts(0)(inp, ic, i) / detjac_upts(inp, ic) - src_upts(inp, ic, i));
                            else//global time step
                                disu_upts(0)(inp, ic, i) -= run_input.dt / 2.0 * (div_tconf_upts(0)(inp, ic, i) / detjac_upts(inp, ic) - src_upts(inp, ic, i));
                        }
                    }
                }
            }
            else if (in_step == 2) //stage 3
            {
                //u=1/3*u+2/3*u1+(-dt/6*F)
                for (i = 0; i < n_fields; i++)
                {
                    for (ic = 0; ic < n_eles; ic++)
                    {
                        for (inp = 0; inp < n_upts_per_ele; inp++)
                        {
                            rhs = -div_tconf_upts(0)(inp, ic, i) / detjac_upts(inp, ic) + src_upts(inp, ic, i); //function
                            if (run_input.dt_type == 2)//local time step
                                disu_upts(0)(inp, ic, i) = 1.0 / 3.0 * disu_upts(0)(inp, ic, i) + 2.0 / 3.0 * disu_upts(1)(inp, ic, i) + dt_local(ic) / 6.0 * rhs;
                            else//global time step
                                disu_upts(0)(inp, ic, i) = 1.0 / 3.0 * disu_upts(0)(inp, ic, i) + 2.0 / 3.0 * disu_upts(1)(inp, ic, i) + run_input.dt / 6.0 * rhs;
                        }
                    }
                }
            }

#endif

#ifdef _GPU
                    FatalError("GPU version of SSP-RK34 unavailable!");
#endif
        }
        /*!Time integration using a RK45(2N)/RK414(2N) method. 
        RK45:
        Carpenter M H, Kennedy C A. 
        Fourth-order 2N-storage Runge-Kutta schemes[J]. 1994.
        RK414
        Niegemann J, Diehl R, Busch K.
        Efficient low-storage Runge–Kutta schemes with optimized stability regions.
        Journal of Computational Physics, 2012*/
        else if (adv_type == 3||adv_type == 4)
        {
#ifdef _CPU

            for (i=0; i<n_fields; i++)
            {
                for (ic=0; ic<n_eles; ic++)
                {
                    for (inp=0; inp<n_upts_per_ele; inp++)
                    {
                        rhs = -div_tconf_upts(0)(inp, ic, i) / detjac_upts(inp, ic) + src_upts(inp, ic, i); //function
                        if (run_input.dt_type == 2)
                            disu_upts(1)(inp, ic, i) = run_input.RK_a(in_step) * disu_upts(1)(inp, ic, i) + dt_local(ic) * rhs; //new delta x
                        else
                            disu_upts(1)(inp, ic, i) = run_input.RK_a(in_step) * disu_upts(1)(inp, ic, i) + run_input.dt * rhs; //new delta x

                        disu_upts(0)(inp, ic, i) += run_input.RK_b(in_step) * disu_upts(1)(inp, ic, i); //new x
                    }
                }
            }

#endif

#ifdef _GPU
            FatalError("GPU version of RK45 unavailable!");
            //RK45_update_kernel_wrapper(n_upts_per_ele,n_dims,n_fields,n_eles,disu_upts(0).get_ptr_gpu(),disu_upts(1).get_ptr_gpu(),div_tconf_upts(0).get_ptr_gpu(),detjac_upts.get_ptr_gpu(),src_upts.get_ptr_gpu(),h_ref.get_ptr_gpu(),rk4a,rk4b,run_input.dt,run_input.const_src,run_input.CFL,run_input.gamma,run_input.mu_inf,run_input.order,viscous,run_input.dt_type,in_step);
#endif

        }

        /*! Time integration not implemented. */

        else
            FatalError("ERROR: Time integration type not recognised ... ");
    }
    
}

double eles::calc_dt_local(int in_ele)
{
    double lam_inv, lam_inv_new;
    double lam_visc, lam_visc_new;
    double out_dt_local;
    double dt_inv, dt_visc;
    double u,v,w,p,c;
    double mu,rt_ratio,inte;

    lam_inv = 0;
    lam_visc = 0;
    // 2-D Elements
    if (n_dims == 2)
    {

        // Calculate maximum internal wavespeed per element
        for (int i=0; i<n_upts_per_ele; i++)
        {
            u = disu_upts(0)(i,in_ele,1)/disu_upts(0)(i,in_ele,0);
            v = disu_upts(0)(i,in_ele,2)/disu_upts(0)(i,in_ele,0);
            p = (run_input.gamma - 1.0) * (disu_upts(0)(i,in_ele,3) - 0.5*disu_upts(0)(i,in_ele,0)*(u*u+v*v));
            c = sqrt(run_input.gamma * p/disu_upts(0)(i,in_ele,0));
            inte=p/((run_input.gamma-1.0)*disu_upts(0)(i,in_ele,0));

            rt_ratio = (run_input.gamma-1.0)*inte/(run_input.rt_inf);
            mu = (run_input.mu_inf)*pow(rt_ratio,1.5)*(1.+(run_input.c_sth))/(rt_ratio+(run_input.c_sth));
            mu = mu + run_input.fix_vis*(run_input.mu_inf - mu);

            lam_inv_new = sqrt(u*u + v*v) + c;
            lam_visc_new = max(4.0/3.0,run_input.gamma/run_input.prandtl)*mu/disu_upts(0)(i,in_ele,0);

            if (lam_inv < lam_inv_new)
                lam_inv = lam_inv_new;

            if (lam_visc < lam_visc_new)
                lam_visc = lam_visc_new;
        }

        if (viscous)
        {
            dt_visc = (run_input.CFL * 0.25 * h_ref(in_ele) * h_ref(in_ele))/(lam_visc) * 1.0/(2.0*run_input.order+1.0);
            dt_inv = run_input.CFL*h_ref(in_ele)/lam_inv*1.0/(2.0*run_input.order + 1.0);
        }
        else
        {
            dt_visc = 1e16;
            dt_inv = run_input.CFL*h_ref(in_ele)/lam_inv * 1.0/(2.0*run_input.order + 1.0);
        }
        out_dt_local = min(dt_visc,dt_inv);
    }

    //3D elements
    else if (n_dims == 3)
    {
        for (int i=0; i<n_upts_per_ele; i++)
        {
            u = disu_upts(0)(i,in_ele,1)/disu_upts(0)(i,in_ele,0);
            v = disu_upts(0)(i,in_ele,2)/disu_upts(0)(i,in_ele,0);
            w = disu_upts(0)(i,in_ele,3)/disu_upts(0)(i,in_ele,0);
            p = (run_input.gamma - 1.0) * (disu_upts(0)(i,in_ele,4) - 0.5*disu_upts(0)(i,in_ele,0)*(u*u+v*v+w*w));
            c = sqrt(run_input.gamma * p/disu_upts(0)(i,in_ele,0));
            inte=p/((run_input.gamma-1.0)*disu_upts(0)(i,in_ele,0));

            rt_ratio = (run_input.gamma-1.0)*inte/(run_input.rt_inf);
            mu = (run_input.mu_inf)*pow(rt_ratio,1.5)*(1.+(run_input.c_sth))/(rt_ratio+(run_input.c_sth));
            mu = mu + run_input.fix_vis*(run_input.mu_inf - mu);
            lam_inv_new = sqrt(u*u + v*v + w*w) + c;
            lam_visc_new = max(4.0/3.0,run_input.gamma/run_input.prandtl)*mu/disu_upts(0)(i,in_ele,0);

            if (lam_inv < lam_inv_new)
                lam_inv = lam_inv_new;
            if (lam_visc < lam_visc_new)
                lam_visc = lam_visc_new;
        }

        if (viscous)
        {
            dt_visc = (run_input.CFL * 0.25 * h_ref(in_ele) * h_ref(in_ele))/(lam_visc) * 1.0/(2.0*run_input.order+1.0);
            dt_inv = run_input.CFL*h_ref(in_ele)/lam_inv*1.0/(2.0*run_input.order + 1.0);
        }
        else
        {
            dt_visc = 1e16;
            dt_inv = run_input.CFL*h_ref(in_ele)/lam_inv * 1.0/(2.0*run_input.order + 1.0);
        }
        out_dt_local = min(dt_visc,dt_inv);
    }

    return out_dt_local;
}

// calculate the discontinuous solution at the flux points

void eles::extrapolate_solution(void)
{
    if (n_eles!=0)
    {

#ifdef _CPU

        if(opp_0_sparse==0) // dense
        {
#if defined _ACCELERATE_BLAS || defined _MKL_BLAS || defined _STANDARD_BLAS
            cblas_dgemm(CblasColMajor,CblasNoTrans,CblasNoTrans,n_fpts_per_ele,n_fields_mul_n_eles,n_upts_per_ele,1.0,opp_0.get_ptr_cpu(),n_fpts_per_ele,disu_upts(0).get_ptr_cpu(),n_upts_per_ele,0.0,disu_fpts.get_ptr_cpu(),n_fpts_per_ele);

#else
            dgemm(n_fpts_per_ele,n_fields_mul_n_eles,n_upts_per_ele,1.0,0.0,opp_0.get_ptr_cpu(),disu_upts(0).get_ptr_cpu(),disu_fpts.get_ptr_cpu());

#endif
        }
        else if(opp_0_sparse==1) // mkl blas four-hf_array coo format
        {
#if defined _MKL_BLAS
            mkl_sparse_d_mm(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, opp_0_mkl,
                            opp_0_descr, SPARSE_LAYOUT_COLUMN_MAJOR,
                            disu_upts(0).get_ptr_cpu(),
                            n_fields_mul_n_eles, n_upts_per_ele, 0.0,
                            disu_fpts.get_ptr_cpu(), n_fpts_per_ele);
#endif
        }
        else
        {
            cout << "ERROR: Unknown storage for opp_0 ... " << endl;
        }

#endif

#ifdef _GPU
        if(opp_0_sparse==0)
        {
            cublasDgemm('N','N',n_fpts_per_ele,n_fields_mul_n_eles,n_upts_per_ele,1.0,opp_0.get_ptr_gpu(),n_fpts_per_ele,disu_upts(0).get_ptr_gpu(),n_upts_per_ele,0.0,disu_fpts.get_ptr_gpu(),n_fpts_per_ele);
        }
        else if (opp_0_sparse==1)
        {
            bespoke_SPMV(n_fpts_per_ele,n_upts_per_ele,n_fields,n_eles,opp_0_ell_data.get_ptr_gpu(),opp_0_ell_indices.get_ptr_gpu(),opp_0_nnz_per_row,disu_upts(0).get_ptr_gpu(),disu_fpts.get_ptr_gpu(),ele_type,order,0);
        }
        else
        {
            cout << "ERROR: Unknown storage for opp_0 ... " << endl;
        }
#endif

    }

}

// calculate the transformed discontinuous inviscid flux at the solution points

void eles::evaluate_invFlux(void)
{
    if (n_eles!=0)
    {

#ifdef _CPU

        int i,j,k,l,m;
#if defined _ACCELERATE_BLAS || defined _MKL_BLAS || defined _STANDARD_BLAS //pre initialize array for blas
        hf_array<double> temp_tdisf_upts(n_fields, n_dims);
        temp_tdisf_upts.initialize_to_zero();
#endif
        for(i=0; i<n_eles; i++)
        {
            for(j=0; j<n_upts_per_ele; j++)
            {
                for(k=0; k<n_fields; k++)
                {
                    temp_u(k)=disu_upts(0)(j,i,k);
                }

                if(n_dims==2)
                {
                    calc_invf_2d(temp_u,temp_f);
                }
                else if(n_dims==3)
                {
                    calc_invf_3d(temp_u,temp_f);
                }
                else
                {
                    FatalError("Invalid number of dimensions!");
                }

                // Transform from static physical space to computational space

#if defined _ACCELERATE_BLAS || defined _MKL_BLAS || defined _STANDARD_BLAS
                cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, n_fields, n_dims, n_dims, 1.0, temp_f.get_ptr_cpu(), n_fields, JGinv_upts.get_ptr_cpu(0, 0, j, i), n_dims, 0.0, temp_tdisf_upts.get_ptr_cpu(), n_fields);
                for (k = 0; k < n_fields; k++)
                    for (l = 0; l < n_dims; l++)
                        tdisf_upts(j, i, k, l) = temp_tdisf_upts(k, l);
#else
                for(k=0; k<n_fields; k++)
                {
                    for(l=0; l<n_dims; l++)
                    {
                        tdisf_upts(j,i,k,l)=0.;
                        for(m=0; m<n_dims; m++)
                        {
                            tdisf_upts(j,i,k,l) += JGinv_upts(l,m,j,i)*temp_f(k,m);//JGinv_upts(j,i,l,m)*temp_f(k,m);
                        }
                    }
                }
#endif
            }
        }

#endif

#ifdef _GPU
        evaluate_invFlux_gpu_kernel_wrapper(n_upts_per_ele,n_dims,n_fields,n_eles,disu_upts(0).get_ptr_gpu(),tdisf_upts.get_ptr_gpu(),detjac_upts.get_ptr_gpu(),J_dyn_upts.get_ptr_gpu(),JGinv_upts.get_ptr_gpu(),JGinv_dyn_upts.get_ptr_gpu(),grid_vel_upts.get_ptr_gpu(),run_input.gamma,motion,run_input.equation,run_input.wave_speed(0),run_input.wave_speed(1),run_input.wave_speed(2),run_input.RANS);
#endif
    }
}

void eles::evaluate_invFlux_over_int(void)
{
    if (n_eles != 0)
    {
        int i, j, k, l, m;
        for (i = 0; i < n_eles; i++)
        {
            //interpolate the solution to over_int_cubpts
#if defined _ACCELERATE_BLAS || defined _MKL_BLAS || defined _STANDARD_BLAS
            for (k = 0; k < n_fields; k++)
                cblas_dgemv(CblasColMajor, CblasNoTrans, loc_over_int_cubpts.get_dim(1), n_upts_per_ele, 1.0, opp_over_int_cubpts.get_ptr_cpu(), loc_over_int_cubpts.get_dim(1), disu_upts(0).get_ptr_cpu(0, i, k), 1, 0.0, temp_u_over_int_cubpts.get_ptr_cpu(0, k), 1);
#else
            for (k = 0; k < n_fields; k++)
                dgemm(loc_over_int_cubpts.get_dim(1), 1, n_upts_per_ele, 1.0, 0.0, opp_over_int_cubpts.get_ptr_cpu(), disu_upts(0).get_ptr_cpu(0, i, k), temp_u_over_int_cubpts.get_ptr_cpu(0, k));
#endif
            for (j = 0; j < loc_over_int_cubpts.get_dim(1); j++) //loop over over_int_cubpts
            {
                for (k = 0; k < n_fields; k++)
                {
                    temp_u(k) = temp_u_over_int_cubpts(j, k);
                }

                if (n_dims == 2)
                {
                    calc_invf_2d(temp_u, temp_f);
                }
                else if (n_dims == 3)
                {
                    calc_invf_3d(temp_u, temp_f);
                }
                else
                {
                    FatalError("Invalid number of dimensions!");
                }

                // Transform from static physical space to computational space
                for (k = 0; k < n_fields; k++)
                {
                    for (l = 0; l < n_dims; l++)
                    {
                        temp_tdisf_over_int_cubpts(j, k, l) = 0.;
                        for (m = 0; m < n_dims; m++)
                        {
                            temp_tdisf_over_int_cubpts(j, k, l) += JGinv_over_int_cubpts(l, m, j, i) * temp_f(k, m); //JGinv_over_int_cubpts(j,i,l,m)*temp_f(k,m);
                        }
                    }
                }
            }
            //apply over integration filter, project transformd inviscid flux from over_int_cubpts back to upts
            for (j = 0; j < n_upts_per_ele; j++)
            {
                for (k = 0; k < n_fields; k++)
                {
                    for (l = 0; l < n_dims; l++)
                    {
                        tdisf_upts(j, i, k, l) = 0.;
                        for (m = 0; m < loc_over_int_cubpts.get_dim(1); m++)
                        {
                            tdisf_upts(j, i, k, l) += over_int_filter(j,m)*temp_tdisf_over_int_cubpts(m, k, l);
                        }
                    }
                }
            }
        }
    }
}

// calculate the normal transformed discontinuous flux at the flux points

void eles::extrapolate_totalFlux()
{
    if (n_eles!=0)
    {
#ifdef _CPU

        if(opp_1_sparse==0) // dense
        {
#if defined _ACCELERATE_BLAS || defined _MKL_BLAS || defined _STANDARD_BLAS

            cblas_dgemm(CblasColMajor,CblasNoTrans,CblasNoTrans,n_fpts_per_ele,n_fields_mul_n_eles,n_upts_per_ele,1.0,opp_1(0).get_ptr_cpu(),n_fpts_per_ele,tdisf_upts.get_ptr_cpu(0,0,0,0),n_upts_per_ele,0.0,norm_tdisf_fpts.get_ptr_cpu(),n_fpts_per_ele);
            for (int i=1; i<n_dims; i++)
            {
                cblas_dgemm(CblasColMajor,CblasNoTrans,CblasNoTrans,n_fpts_per_ele,n_fields_mul_n_eles,n_upts_per_ele,1.0,opp_1(i).get_ptr_cpu(),n_fpts_per_ele,tdisf_upts.get_ptr_cpu(0,0,0,i),n_upts_per_ele,1.0,norm_tdisf_fpts.get_ptr_cpu(),n_fpts_per_ele);
            }

#else
            dgemm(n_fpts_per_ele,n_fields_mul_n_eles,n_upts_per_ele,1.0,0.0,opp_1(0).get_ptr_cpu(),tdisf_upts.get_ptr_cpu(0,0,0,0),norm_tdisf_fpts.get_ptr_cpu());
            for (int i=1; i<n_dims; i++)
            {
                dgemm(n_fpts_per_ele,n_fields_mul_n_eles,n_upts_per_ele,1.0,1.0,opp_1(i).get_ptr_cpu(),tdisf_upts.get_ptr_cpu(0,0,0,i),norm_tdisf_fpts.get_ptr_cpu());
            }
#endif
        }
        else if(opp_1_sparse==1) // mkl blas four-hf_array coo format
        {
#if defined _MKL_BLAS

            mkl_sparse_d_mm(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, opp_1_mkl(0),
                            opp_1_descr(0), SPARSE_LAYOUT_COLUMN_MAJOR,
                            tdisf_upts.get_ptr_cpu(0, 0, 0, 0),
                            n_fields_mul_n_eles, n_upts_per_ele, 0.0,
                            norm_tdisf_fpts.get_ptr_cpu(), n_fpts_per_ele);
            for (int i = 1; i < n_dims; i++)
            {
                mkl_sparse_d_mm(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, opp_1_mkl(i),
                                opp_1_descr(i), SPARSE_LAYOUT_COLUMN_MAJOR,
                                tdisf_upts.get_ptr_cpu(0, 0, 0, i),
                                n_fields_mul_n_eles, n_upts_per_ele, 1.0,
                                norm_tdisf_fpts.get_ptr_cpu(), n_fpts_per_ele);
            }

#endif
        }
        else
        {
            cout << "ERROR: Unknown storage for opp_1 ... " << endl;
        }

#endif

#ifdef _GPU

        if (opp_1_sparse==0)
        {
            cublasDgemm('N','N',n_fpts_per_ele,n_fields_mul_n_eles,n_upts_per_ele,1.0,opp_1(0).get_ptr_gpu(),n_fpts_per_ele,tdisf_upts.get_ptr_gpu(0,0,0,0),n_upts_per_ele,0.0,norm_tdisf_fpts.get_ptr_gpu(),n_fpts_per_ele);
            for (int i=1; i<n_dims; i++)
            {
                cublasDgemm('N','N',n_fpts_per_ele,n_fields_mul_n_eles,n_upts_per_ele,1.0,opp_1(i).get_ptr_gpu(),n_fpts_per_ele,tdisf_upts.get_ptr_gpu(0,0,0,i),n_upts_per_ele,1.0,norm_tdisf_fpts.get_ptr_gpu(),n_fpts_per_ele);
            }
        }
        else if (opp_1_sparse==1)
        {
            bespoke_SPMV(n_fpts_per_ele,n_upts_per_ele,n_fields,n_eles,opp_1_ell_data(0).get_ptr_gpu(),opp_1_ell_indices(0).get_ptr_gpu(),opp_1_nnz_per_row(0),tdisf_upts.get_ptr_gpu(0,0,0,0),norm_tdisf_fpts.get_ptr_gpu(),ele_type,order,0);
            for (int i=1; i<n_dims; i++)
            {
                bespoke_SPMV(n_fpts_per_ele,n_upts_per_ele,n_fields,n_eles,opp_1_ell_data(i).get_ptr_gpu(),opp_1_ell_indices(i).get_ptr_gpu(),opp_1_nnz_per_row(i),tdisf_upts.get_ptr_gpu(0,0,0,i),norm_tdisf_fpts.get_ptr_gpu(),ele_type,order,1);
            }
        }
#endif

    }

    /*
     #ifdef _GPU
     tdisinvf_upts.cp_gpu_cpu();
     #endif

     cout << "Before" << endl;
     for (int i=0;i<n_fpts_per_ele;i++)
     for (int j=0;j<n_eles;j++)
     for (int k=0;k<n_fields;k++)
     for (int m=0;m<n_dims;m++)
     cout << setprecision(10)  << i << " " << j<< " " << k << " " << tdisinvf_upts(i,j,k,m) << endl;
     */

    /*
     cout << "After,ele_type =" << ele_type << endl;
     #ifdef _GPU
     norm_tdisinvf_fpts.cp_gpu_cpu();
     #endif

     for (int i=0;i<n_fpts_per_ele;i++)
     for (int j=0;j<n_eles;j++)
     for (int k=0;k<n_fields;k++)
     cout << setprecision(10)  << i << " " << j<< " " << k << " " << norm_tdisinvf_fpts(i,j,k) << endl;
     */
}


// calculate the divergence of the transformed discontinuous flux at the solution points

void eles::calculate_divergence(void)
{
    if (n_eles!=0)
    {
#ifdef _CPU

        if(opp_2_sparse==0) // dense
        {
#if defined _ACCELERATE_BLAS || defined _MKL_BLAS || defined _STANDARD_BLAS

            cblas_dgemm(CblasColMajor,CblasNoTrans,CblasNoTrans,n_upts_per_ele,n_fields_mul_n_eles,n_upts_per_ele,1.0,opp_2(0).get_ptr_cpu(),n_upts_per_ele,tdisf_upts.get_ptr_cpu(0,0,0,0),n_upts_per_ele,0.0,div_tconf_upts(0).get_ptr_cpu(),n_upts_per_ele);
            for (int i=1; i<n_dims; i++)
            {
                cblas_dgemm(CblasColMajor,CblasNoTrans,CblasNoTrans,n_upts_per_ele,n_fields_mul_n_eles,n_upts_per_ele,1.0,opp_2(i).get_ptr_cpu(),n_upts_per_ele,tdisf_upts.get_ptr_cpu(0,0,0,i),n_upts_per_ele,1.0,div_tconf_upts(0).get_ptr_cpu(),n_upts_per_ele);
            }

#else
            dgemm(n_upts_per_ele,n_fields_mul_n_eles,n_upts_per_ele,1.0,0.0,opp_2(0).get_ptr_cpu(),tdisf_upts.get_ptr_cpu(0,0,0,0),div_tconf_upts(0).get_ptr_cpu());
            for (int i=1; i<n_dims; i++)
            {
                dgemm(n_upts_per_ele,n_fields_mul_n_eles,n_upts_per_ele,1.0,1.0,opp_2(i).get_ptr_cpu(),tdisf_upts.get_ptr_cpu(0,0,0,i),div_tconf_upts(0).get_ptr_cpu());
            }

#endif
        }
        else if(opp_2_sparse==1) // mkl blas four-hf_array coo format
        {
#if defined _MKL_BLAS

            mkl_sparse_d_mm(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, opp_2_mkl(0),
                opp_2_descr(0), SPARSE_LAYOUT_COLUMN_MAJOR,
                tdisf_upts.get_ptr_cpu(0, 0, 0, 0),
                n_fields_mul_n_eles, n_upts_per_ele, 0.0,
                div_tconf_upts(0).get_ptr_cpu(), n_upts_per_ele);
            for (int i=1; i<n_dims; i++)
            {
                mkl_sparse_d_mm(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, opp_2_mkl(i),
                                opp_2_descr(i), SPARSE_LAYOUT_COLUMN_MAJOR,
                                tdisf_upts.get_ptr_cpu(0, 0, 0, i),
                                n_fields_mul_n_eles, n_upts_per_ele, 1.0,
                                div_tconf_upts(0).get_ptr_cpu(), n_upts_per_ele);
            }

#endif
        }
        else
        {
            cout << "ERROR: Unknown storage for opp_2 ... " << endl;
        }

#endif


#ifdef _GPU

        if (opp_2_sparse==0)
        {
            cublasDgemm('N','N',n_upts_per_ele,n_fields_mul_n_eles,n_upts_per_ele,1.0,opp_2(0).get_ptr_gpu(),n_upts_per_ele,tdisf_upts.get_ptr_gpu(0,0,0,0),n_upts_per_ele,0.0,div_tconf_upts(0).get_ptr_gpu(),n_upts_per_ele);
            for (int i=1; i<n_dims; i++)
            {
                cublasDgemm('N','N',n_upts_per_ele,n_fields_mul_n_eles,n_upts_per_ele,1.0,opp_2(i).get_ptr_gpu(),n_upts_per_ele,tdisf_upts.get_ptr_gpu(0,0,0,i),n_upts_per_ele,1.0,div_tconf_upts(0).get_ptr_gpu(),n_upts_per_ele);
            }
        }
        else if (opp_2_sparse==1)
        {
            bespoke_SPMV(n_upts_per_ele,n_upts_per_ele,n_fields,n_eles,opp_2_ell_data(0).get_ptr_gpu(),opp_2_ell_indices(0).get_ptr_gpu(),opp_2_nnz_per_row(0),tdisf_upts.get_ptr_gpu(0,0,0,0),div_tconf_upts(0).get_ptr_gpu(),ele_type,order,0);
            for (int i=1; i<n_dims; i++)
            {
                bespoke_SPMV(n_upts_per_ele,n_upts_per_ele,n_fields,n_eles,opp_2_ell_data(i).get_ptr_gpu(),opp_2_ell_indices(i).get_ptr_gpu(),opp_2_nnz_per_row(i),tdisf_upts.get_ptr_gpu(0,0,0,i),div_tconf_upts(0).get_ptr_gpu(),ele_type,order,1);
            }

        }
#endif

    }

    /*
     for (int j=0;j<n_eles;j++)
     for (int i=0;i<n_upts_per_ele;i++)
     //for (int k=0;k<n_fields;k++)
     cout << scientific << setw(16) << setprecision(12) << div_tconf_upts(0)(i,j,0) << endl;
     */
}


// calculate divergence of the transformed continuous flux at the solution points

void eles::calculate_corrected_divergence(void)
{
    if (n_eles!=0)
    {
#ifdef _CPU

#if defined _ACCELERATE_BLAS || defined _MKL_BLAS || defined _STANDARD_BLAS

        cblas_daxpy(n_eles*n_fields*n_fpts_per_ele,-1.0,norm_tdisf_fpts.get_ptr_cpu(),1,norm_tconf_fpts.get_ptr_cpu(),1);

#else

        daxpy_wrapper(n_eles*n_fields*n_fpts_per_ele,-1.0,norm_tdisf_fpts.get_ptr_cpu(),1,norm_tconf_fpts.get_ptr_cpu(),1);

#endif

        if(opp_3_sparse==0) // dense
        {
#if defined _ACCELERATE_BLAS || defined _MKL_BLAS || defined _STANDARD_BLAS

            cblas_dgemm(CblasColMajor,CblasNoTrans,CblasNoTrans,n_upts_per_ele,n_fields_mul_n_eles,n_fpts_per_ele,1.0,opp_3.get_ptr_cpu(),n_upts_per_ele,norm_tconf_fpts.get_ptr_cpu(),n_fpts_per_ele,1.0,div_tconf_upts(0).get_ptr_cpu(),n_upts_per_ele);

#else
            dgemm(n_upts_per_ele,n_fields_mul_n_eles,n_fpts_per_ele,1.0,1.0,opp_3.get_ptr_cpu(),norm_tconf_fpts.get_ptr_cpu(),div_tconf_upts(0).get_ptr_cpu());

#endif
        }
        else if(opp_3_sparse==1) // mkl blas four-hf_array coo format
        {
#if defined _MKL_BLAS

            mkl_sparse_d_mm(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, opp_3_mkl,
                            opp_3_descr, SPARSE_LAYOUT_COLUMN_MAJOR,
                            norm_tconf_fpts.get_ptr_cpu(),
                            n_fields_mul_n_eles, n_fpts_per_ele, 1.0,
                            div_tconf_upts(0).get_ptr_cpu(), n_upts_per_ele);
#endif
        }
        else
        {
            cout << "ERROR: Unknown storage for opp_3 ... " << endl;
        }

        int temp_size = n_eles * n_upts_per_ele * n_fields;
        for (int ct = 0; ct < temp_size; ct++)
        {
            if (std::isnan(div_tconf_upts(0)[ct]))
            {
                int i, j;
                j = (ct / n_upts_per_ele) % n_eles; //eles
                i = ct % n_upts_per_ele;              //upts
                cout<<"Residual is NaN at non-dimensionalized position:"<<endl;
                for (int intd=0; intd<n_dims; intd++)
                    cout<<pos_upts(i,j,intd);
                cout<<endl;
                FatalError("Aborting...");
            }
        }
#endif

#ifdef _GPU

        cublasDaxpy(n_eles*n_fields*n_fpts_per_ele,-1.0,norm_tdisf_fpts.get_ptr_gpu(),1,norm_tconf_fpts.get_ptr_gpu(),1);

        if (opp_3_sparse==0)
        {
            cublasDgemm('N','N',n_upts_per_ele,n_fields_mul_n_eles,n_fpts_per_ele,1.0,opp_3.get_ptr_gpu(),n_upts_per_ele,norm_tconf_fpts.get_ptr_gpu(),n_fpts_per_ele,1.0,div_tconf_upts(0).get_ptr_gpu(),n_upts_per_ele);
        }
        else if (opp_3_sparse==1)
        {
            bespoke_SPMV(n_upts_per_ele,n_fpts_per_ele,n_fields,n_eles,opp_3_ell_data.get_ptr_gpu(),opp_3_ell_indices.get_ptr_gpu(),opp_3_nnz_per_row,norm_tconf_fpts.get_ptr_gpu(),div_tconf_upts(0).get_ptr_gpu(),ele_type,order,1);
        }
        else
        {
            cout << "ERROR: Unknown storage for opp_3 ... " << endl;
        }
#endif

    }
}


// calculate uncorrected transformed gradient of the discontinuous solution at the solution points
// (mixed derivative)

void eles::calculate_gradient(void)
{
    if (n_eles!=0)
    {

#ifdef _CPU

        if(opp_4_sparse==0) // dense
        {
#if defined _ACCELERATE_BLAS || defined _MKL_BLAS || defined _STANDARD_BLAS
            for (int i=0; i<n_dims; i++)
            {
                cblas_dgemm(CblasColMajor,CblasNoTrans,CblasNoTrans,n_upts_per_ele,n_fields_mul_n_eles,n_upts_per_ele,1.0,opp_4(i).get_ptr_cpu(),n_upts_per_ele,disu_upts(0).get_ptr_cpu(),n_upts_per_ele,0.0,grad_disu_upts.get_ptr_cpu(0,0,0,i),n_upts_per_ele);
            }

#else
            for (int i=0; i<n_dims; i++)
            {
                dgemm(n_upts_per_ele,n_fields_mul_n_eles,n_upts_per_ele,1.0,0.0,opp_4(i).get_ptr_cpu(),disu_upts(0).get_ptr_cpu(),grad_disu_upts.get_ptr_cpu(0,0,0,i));
            }

#endif
        }
        else if(opp_4_sparse==1) // mkl blas four-hf_array coo format
        {
#if defined _MKL_BLAS

            for (int i=0; i<n_dims; i++)
            {
                mkl_sparse_d_mm(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, opp_4_mkl(i),
                                opp_4_descr(i), SPARSE_LAYOUT_COLUMN_MAJOR,
                                disu_upts(0).get_ptr_cpu(),
                                n_fields_mul_n_eles, n_upts_per_ele, 0.0,
                                grad_disu_upts.get_ptr_cpu(0,0,0,i), n_upts_per_ele);
            }

#endif
        }
        else
        {
            cout << "ERROR: Unknown storage for opp_4 ... " << endl;
        }

#endif

#ifdef _GPU

        if (opp_4_sparse==0)
        {
            for (int i=0; i<n_dims; i++)
            {
                cublasDgemm('N','N',n_upts_per_ele,n_fields_mul_n_eles,n_upts_per_ele,1.0,opp_4(i).get_ptr_gpu(),n_upts_per_ele,disu_upts(0).get_ptr_gpu(),n_upts_per_ele,0.0,grad_disu_upts.get_ptr_gpu(0,0,0,i),n_upts_per_ele);
            }
        }
        else if (opp_4_sparse==1)
        {
            for (int i=0; i<n_dims; i++)
            {
                bespoke_SPMV(n_upts_per_ele,n_upts_per_ele,n_fields,n_eles,opp_4_ell_data(i).get_ptr_gpu(),opp_4_ell_indices(i).get_ptr_gpu(),opp_4_nnz_per_row(i),disu_upts(0).get_ptr_gpu(),grad_disu_upts.get_ptr_gpu(0,0,0,i),ele_type,order,0);
            }
        }
#endif
    }
}

// calculate corrected gradient of the discontinuous solution at solution points and flux points

void eles::correct_gradient(void)
{
    if (n_eles!=0)
    {
#ifdef _CPU
        //correct gradient on solution points
        if(opp_5_sparse==0) // dense
        {
#if defined _ACCELERATE_BLAS || defined _MKL_BLAS || defined _STANDARD_BLAS
            for (int i=0; i<n_dims; i++)
                cblas_dgemm(CblasColMajor,CblasNoTrans,CblasNoTrans,n_upts_per_ele,n_fields_mul_n_eles,n_fpts_per_ele,1.0,opp_5(i).get_ptr_cpu(),n_upts_per_ele,delta_disu_fpts.get_ptr_cpu(),n_fpts_per_ele,1.0,grad_disu_upts.get_ptr_cpu(0,0,0,i),n_upts_per_ele);
#else
            for (int i=0; i<n_dims; i++)
                dgemm(n_upts_per_ele,n_fields_mul_n_eles,n_fpts_per_ele,1.0,1.0,opp_5(i).get_ptr_cpu(),delta_disu_fpts.get_ptr_cpu(),grad_disu_upts.get_ptr_cpu(0,0,0,i));
#endif
        }
        else if(opp_5_sparse==1) // mkl blas four-hf_array coo format
        {
#if defined _MKL_BLAS
            for (int i=0; i<n_dims; i++)
            {
                mkl_sparse_d_mm(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, opp_5_mkl(i),
                                opp_5_descr(i), SPARSE_LAYOUT_COLUMN_MAJOR,
                                delta_disu_fpts.get_ptr_cpu(),
                                n_fields_mul_n_eles, n_fpts_per_ele, 1.0,
                                grad_disu_upts.get_ptr_cpu(0,0,0,i), n_upts_per_ele);
            }

#endif
        }
        else
        {
            cout << "ERROR: Unknown storage for opp_5 ... " << endl;
        }

        //extrapolate transformed corrected gradients to flux points
        if (opp_6_sparse == 0) // dense
        {
#if defined _ACCELERATE_BLAS || defined _MKL_BLAS || defined _STANDARD_BLAS
            for (int i = 0; i < n_dims; i++)
                cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n_fpts_per_ele, n_fields_mul_n_eles, n_upts_per_ele, 1.0, opp_6.get_ptr_cpu(), n_fpts_per_ele, grad_disu_upts.get_ptr_cpu(0, 0, 0, i), n_upts_per_ele, 0.0, grad_disu_fpts.get_ptr_cpu(0, 0, 0, i), n_fpts_per_ele);
#else
            for (int i = 0; i < n_dims; i++)
                dgemm(n_fpts_per_ele, n_fields_mul_n_eles, n_upts_per_ele, 1.0, 0.0, opp_6.get_ptr_cpu(), grad_disu_upts.get_ptr_cpu(0, 0, 0, i), grad_disu_fpts.get_ptr_cpu(0, 0, 0, i));
#endif
        }
        else if (opp_6_sparse == 1) // mkl blas four-hf_array coo format
        {
#if defined _MKL_BLAS
            for (int i = 0; i < n_dims; i++)
            {
                mkl_sparse_d_mm(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, opp_6_mkl,
                                opp_6_descr, SPARSE_LAYOUT_COLUMN_MAJOR,
                                grad_disu_upts.get_ptr_cpu(0, 0, 0, i),
                                n_fields_mul_n_eles, n_upts_per_ele, 0.0,
                                grad_disu_fpts.get_ptr_cpu(0, 0, 0, i), n_fpts_per_ele);
            }
#endif
        }
        else
        {
            cout << "ERROR: Unknown storage for opp_6 ... " << endl;
        }

        // Transform to physical space
        double inv_detjac;
        hf_array<double> temp_cgradient(n_dims, n_fields);//temporary physical corrected gradients
        temp_cgradient.initialize_to_zero();
        hf_array<double> temp_tcgradient(n_dims, n_fields);//temporary transformed correct gradients

        for (int i = 0; i < n_eles; i++)
        {
            //for solution points
            for (int j = 0; j < n_upts_per_ele; j++)
            {
                inv_detjac = 1.0 / detjac_upts(j, i);

                //copy transformed gradients from array
                for (int k = 0; k < n_fields; k++)
                    for (int d = 0; d < n_dims; d++)
                        temp_tcgradient(d, k) = grad_disu_upts(j, i, k, d);

#if defined _ACCELERATE_BLAS || defined _MKL_BLAS || defined _STANDARD_BLAS
                cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, n_dims, n_fields, n_dims, inv_detjac, JGinv_upts.get_ptr_cpu(0, 0, j, i), n_dims, temp_tcgradient.get_ptr_cpu(), n_dims, 0.0, temp_cgradient.get_ptr_cpu(), n_dims);
#else
                hf_array<double> temp_JGinv(n_dims, n_dims);//transposed JGinv
                for (int k = 0; k < n_dims; k++)
                    for (int d = 0; d < n_dims; d++)
                        temp_JGinv(k, d) = JGinv_upts(d, k, j, i);
                dgemm(n_dims, n_fields, n_dims, inv_detjac, 0.0, temp_JGinv.get_ptr_cpu(), temp_tcgradient.get_ptr_cpu(), temp_cgradient.get_ptr_cpu());
#endif
                //copy physical gradient back to array
                for (int k = 0; k < n_fields; k++)
                    for (int d = 0; d < n_dims; d++)
                        grad_disu_upts(j, i, k, d) = temp_cgradient(d, k);
            }

            //for flux points
            for (int j = 0; j < n_fpts_per_ele; j++)
            {
                inv_detjac = 1.0 / detjac_fpts(j, i);

                //copy transformed gradients from array
                for (int k = 0; k < n_fields; k++)
                    for (int d = 0; d < n_dims; d++)
                        temp_tcgradient(d, k) = grad_disu_fpts(j, i, k, d);

#if defined _ACCELERATE_BLAS || defined _MKL_BLAS || defined _STANDARD_BLAS
                cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, n_dims, n_fields, n_dims, inv_detjac, JGinv_fpts.get_ptr_cpu(0, 0, j, i), n_dims, temp_tcgradient.get_ptr_cpu(), n_dims, 0.0, temp_cgradient.get_ptr_cpu(), n_dims);
#else
                hf_array<double> temp_JGinv(n_dims, n_dims);//transposed JGinv
                for (int k = 0; k < n_dims; k++)
                    for (int d = 0; d < n_dims; d++)
                        temp_JGinv(k, d) = JGinv_fpts(d, k, j, i);
                dgemm(n_dims, n_fields, n_dims, inv_detjac, 0.0, temp_JGinv.get_ptr_cpu(), temp_tcgradient.get_ptr_cpu(), temp_cgradient.get_ptr_cpu());
#endif
                //copy physical gradient back to array
                for (int k = 0; k < n_fields; k++)
                    for (int d = 0; d < n_dims; d++)
                        grad_disu_fpts(j, i, k, d) = temp_cgradient(d, k);
            }
        }

#endif

#ifdef _GPU

        if (opp_5_sparse == 0)
        {
            for (int i = 0; i < n_dims; i++)
            {
                cublasDgemm('N', 'N', n_upts_per_ele, n_fields_mul_n_eles, n_fpts_per_ele, 1.0, opp_5(i).get_ptr_gpu(), n_upts_per_ele, delta_disu_fpts.get_ptr_gpu(), n_fpts_per_ele, 1.0, grad_disu_upts.get_ptr_gpu(0, 0, 0, i), n_upts_per_ele);
            }
        }
        else if (opp_5_sparse == 1)
        {
            for (int i = 0; i < n_dims; i++)
            {
                bespoke_SPMV(n_upts_per_ele, n_fpts_per_ele, n_fields, n_eles, opp_5_ell_data(i).get_ptr_gpu(), opp_5_ell_indices(i).get_ptr_gpu(), opp_5_nnz_per_row(i), delta_disu_fpts.get_ptr_gpu(), grad_disu_upts.get_ptr_gpu(0, 0, 0, i), ele_type, order, 1);
            }
        }

        if (opp_6_sparse == 0)
        {
            for (int i = 0; i < n_dims; i++)
            {
                cublasDgemm('N', 'N', n_fpts_per_ele, n_fields_mul_n_eles, n_upts_per_ele, 1.0, opp_6.get_ptr_gpu(), n_fpts_per_ele, grad_disu_upts.get_ptr_gpu(0, 0, 0, i), n_upts_per_ele, 0.0, grad_disu_fpts.get_ptr_gpu(0, 0, 0, i), n_fpts_per_ele);
            }
        }
        else if (opp_6_sparse == 1)
        {
            for (int i = 0; i < n_dims; i++)
            {
                bespoke_SPMV(n_fpts_per_ele, n_upts_per_ele, n_fields, n_eles, opp_6_ell_data.get_ptr_gpu(), opp_6_ell_indices.get_ptr_gpu(), opp_6_nnz_per_row, grad_disu_upts.get_ptr_gpu(0, 0, 0, i), grad_disu_fpts.get_ptr_gpu(0, 0, 0, i), ele_type, order, 0);
            }
        }

        transform_grad_disu_upts_kernel_wrapper(n_upts_per_ele, n_dims, n_fields, n_eles, grad_disu_upts.get_ptr_gpu(), detjac_upts.get_ptr_gpu(), J_dyn_upts.get_ptr_gpu(), JGinv_upts.get_ptr_gpu(), JGinv_dyn_upts.get_ptr_gpu(), run_input.equation, motion);

#endif
    }

}

/*! If at first RK step and using certain LES models, compute some model-related quantities.
 If using similarity or WALE-similarity (WSM) models, compute filtered solution and Leonard tensors.
 If using spectral vanishing viscosity (SVV) model, compute filtered solution. */

void eles::calc_sgs_terms(void)
{
    if (n_eles!=0)
    {

        int i,j,k,l;
        int dim3;
        double diag, rsq;
        hf_array <double> utemp(n_fields);

        /*! Filter solution */

#ifdef _CPU

#if defined _ACCELERATE_BLAS || defined _MKL_BLAS || defined _STANDARD_BLAS

        cblas_dgemm(CblasColMajor,CblasNoTrans,CblasNoTrans,n_upts_per_ele,n_fields_mul_n_eles,n_upts_per_ele,1.0,filter_upts.get_ptr_cpu(),n_upts_per_ele,disu_upts(0).get_ptr_cpu(),n_upts_per_ele,0.0,disuf_upts.get_ptr_cpu(),n_upts_per_ele);

#else
        dgemm(n_upts_per_ele,n_fields_mul_n_eles,n_upts_per_ele,1.0,0.0,filter_upts.get_ptr_cpu(),disu_upts(0).get_ptr_cpu(),disuf_upts.get_ptr_cpu());

#endif

        /*! Check for NaNs */
        int temp_size = n_upts_per_ele * n_eles * n_fields;
        for (i = 0; i < temp_size; i++)
            if (std::isnan(disuf_upts[i]))
                FatalError("nan in filtered solution");

        /*! If SVV model, copy filtered solution back to solution */
        if(sgs_model==3)
            disu_upts(0) = disuf_upts;

        /*! If Similarity model, compute product terms and Leonard tensors */
        else if(sgs_model==2 || sgs_model==4)
        {

            /*! third dimension of Lu, uu arrays */
            if(n_dims==2)      dim3 = 3;
            else if(n_dims==3) dim3 = 6;

            /*! Calculate velocity and energy product arrays uu, ue */
            for(i=0; i<n_upts_per_ele; i++)
            {
                for(j=0; j<n_eles; j++)
                {
                    for(k=0; k<n_fields; k++)
                    {
                        utemp(k) = disu_upts(0)(i,j,k);
                    }

                    rsq = utemp(0)*utemp(0);

                    /*! note that product arrays are symmetric */
                    if(n_dims==2)
                    {
                        /*! velocity-velocity product */
                        uu(i,j,0) = utemp(1)*utemp(1)/rsq;
                        uu(i,j,1) = utemp(2)*utemp(2)/rsq;
                        uu(i,j,2) = utemp(1)*utemp(2)/rsq;

                        /*! velocity-energy product */
                        utemp(3) -= 0.5*(utemp(1)*utemp(1)+utemp(2)*utemp(2))/utemp(0); // internal energy*rho

                        ue(i,j,0) = utemp(1)*utemp(3)/rsq;
                        ue(i,j,1) = utemp(2)*utemp(3)/rsq;
                    }
                    else if(n_dims==3)
                    {
                        /*! velocity-velocity product */
                        uu(i,j,0) = utemp(1)*utemp(1)/rsq;
                        uu(i,j,1) = utemp(2)*utemp(2)/rsq;
                        uu(i,j,2) = utemp(3)*utemp(3)/rsq;
                        uu(i,j,3) = utemp(1)*utemp(2)/rsq;
                        uu(i,j,4) = utemp(1)*utemp(3)/rsq;
                        uu(i,j,5) = utemp(2)*utemp(3)/rsq;

                        /*! velocity-energy product */
                        utemp(4) -= 0.5*(utemp(1)*utemp(1)+utemp(2)*utemp(2)+utemp(3)*utemp(3))/utemp(0); // internal energy*rho

                        ue(i,j,0) = utemp(1)*utemp(4)/rsq;
                        ue(i,j,1) = utemp(2)*utemp(4)/rsq;
                        ue(i,j,2) = utemp(3)*utemp(4)/rsq;
                    }
                }
            }

            /*! Filter products uu and ue */

#if defined _ACCELERATE_BLAS || defined _MKL_BLAS || defined _STANDARD_BLAS


            cblas_dgemm(CblasColMajor,CblasNoTrans,CblasNoTrans,n_upts_per_ele,dim3*n_eles,n_upts_per_ele,1.0,filter_upts.get_ptr_cpu(),n_upts_per_ele,uu.get_ptr_cpu(),n_upts_per_ele,0.0,Lu.get_ptr_cpu(),n_upts_per_ele);


            cblas_dgemm(CblasColMajor,CblasNoTrans,CblasNoTrans,n_upts_per_ele,dim3*n_eles,n_upts_per_ele,1.0,filter_upts.get_ptr_cpu(),n_upts_per_ele,ue.get_ptr_cpu(),n_upts_per_ele,0.0,Le.get_ptr_cpu(),n_upts_per_ele);

#else


            dgemm(n_upts_per_ele,dim3*n_eles,n_upts_per_ele,1.0,0.0,filter_upts.get_ptr_cpu(),uu.get_ptr_cpu(),Lu.get_ptr_cpu());


            dgemm(n_upts_per_ele,dim3*n_eles,n_upts_per_ele,1.0,0.0,filter_upts.get_ptr_cpu(),ue.get_ptr_cpu(),Le.get_ptr_cpu());
#endif

            /*! Subtract product of unfiltered quantities from Leonard tensors */
            for(i=0; i<n_upts_per_ele; i++)
            {
                for(j=0; j<n_eles; j++)
                {

                    // filtered solution
                    for(k=0; k<n_fields; k++)
                        utemp(k) = disuf_upts(i,j,k);

                    rsq = utemp(0)*utemp(0);

                    if(n_dims==2)
                    {

                        Lu(i,j,0) -= (utemp(1)*utemp(1))/rsq;
                        Lu(i,j,1) -= (utemp(2)*utemp(2))/rsq;
                        Lu(i,j,2) -= (utemp(1)*utemp(2))/rsq;

                        diag = (Lu(i,j,0)+Lu(i,j,1))/3.0;

                        // internal energy*rho
                        utemp(3) -= 0.5*(utemp(1)*utemp(1)+utemp(2)*utemp(2))/utemp(0);

                        Le(i,j,0) = (Le(i,j,0) - utemp(1)*utemp(3))/rsq;
                        Le(i,j,1) = (Le(i,j,1) - utemp(2)*utemp(3))/rsq;

                    }
                    else if(n_dims==3)
                    {

                        Lu(i,j,0) -= (utemp(1)*utemp(1))/rsq;
                        Lu(i,j,1) -= (utemp(2)*utemp(2))/rsq;
                        Lu(i,j,2) -= (utemp(3)*utemp(3))/rsq;
                        Lu(i,j,3) -= (utemp(1)*utemp(2))/rsq;
                        Lu(i,j,4) -= (utemp(1)*utemp(3))/rsq;
                        Lu(i,j,5) -= (utemp(2)*utemp(3))/rsq;

                        diag = (Lu(i,j,0)+Lu(i,j,1)+Lu(i,j,2))/3.0;

                        // internal energy*rho
                        utemp(4) -= 0.5*(utemp(1)*utemp(1)+utemp(2)*utemp(2)+utemp(3)*utemp(3))/utemp(0);

                        Le(i,j,0) = (Le(i,j,0) - utemp(1)*utemp(4))/rsq;
                        Le(i,j,1) = (Le(i,j,1) - utemp(2)*utemp(4))/rsq;
                        Le(i,j,2) = (Le(i,j,2) - utemp(3)*utemp(4))/rsq;

                    }

                    /*! subtract diagonal from Lu */
                    for (k=0; k<n_dims; ++k) Lu(i,j,k) -= diag;

                }
            }
        }

#endif

        /*! GPU version of the above */
#ifdef _GPU

        /*! Filter solution (CUDA BLAS library) */
        cublasDgemm('N','N',n_upts_per_ele,dim3*n_eles,n_upts_per_ele,1.0,filter_upts.get_ptr_gpu(),n_upts_per_ele,disu_upts(0).get_ptr_gpu(),n_upts_per_ele,0.0,disuf_upts.get_ptr_gpu(),n_upts_per_ele);

        /*! Check for NaNs */
        disuf_upts.cp_gpu_cpu();

        for(i=0; i<n_upts_per_ele; i++)
            for(j=0; j<n_eles; j++)
                for(k=0; k<n_fields; k++)
                    if(std::isnan(disuf_upts(i,j,k)))
                        FatalError("nan in filtered solution");

        /*! If Similarity model */
        if(sgs_model==2 || sgs_model==4)
        {

            /*! compute product terms uu, ue (pass flag=0 to wrapper function) */
            calc_similarity_model_kernel_wrapper(0, n_fields, n_upts_per_ele, n_eles, n_dims, disu_upts(0).get_ptr_gpu(), disuf_upts.get_ptr_gpu(), uu.get_ptr_gpu(), ue.get_ptr_gpu(), Lu.get_ptr_gpu(), Le.get_ptr_gpu());

            /*! third dimension of Lu, uu arrays */
            if(n_dims==2)
                dim3 = 3;
            else if(n_dims==3)
                dim3 = 6;


            /*! Filter product terms uu and ue */
            cublasDgemm('N','N',n_upts_per_ele,dim3*n_eles,n_upts_per_ele,1.0,filter_upts.get_ptr_gpu(),n_upts_per_ele,uu.get_ptr_gpu(),n_upts_per_ele,0.0,Lu.get_ptr_gpu(),n_upts_per_ele);

            cublasDgemm('N','N',n_upts_per_ele,dim3*n_eles,n_upts_per_ele,1.0,filter_upts.get_ptr_gpu(),n_upts_per_ele,ue.get_ptr_gpu(),n_upts_per_ele,0.0,Le.get_ptr_gpu(),n_upts_per_ele);

            /*! compute Leonard tensors Lu, Le (pass flag=1 to wrapper function) */
            calc_similarity_model_kernel_wrapper(1, n_fields, n_upts_per_ele, n_eles, n_dims, disu_upts(0).get_ptr_gpu(), disuf_upts.get_ptr_gpu(), uu.get_ptr_gpu(), ue.get_ptr_gpu(), Lu.get_ptr_gpu(), Le.get_ptr_gpu());

        }

        /*! If SVV model, copy filtered solution back to original solution */
        else if(sgs_model==3)
        {
            for(i=0; i<n_upts_per_ele; i++)
            {
                for(j=0; j<n_eles; j++)
                {
                    for(k=0; k<n_fields; k++)
                    {
                        disu_upts(0)(i,j,k) = disuf_upts(i,j,k);
                    }
                }
            }
            /*! copy back to GPU */
            disu_upts(0).cp_cpu_gpu();
        }

#endif

    }
}

// calculate transformed discontinuous viscous flux at solution points

void eles::evaluate_viscFlux(void)
{
    if (n_eles!=0)
    {
#ifdef _CPU

        int i,j,k,l,m;
        double detjac;
#if defined _ACCELERATE_BLAS || defined _MKL_BLAS || defined _STANDARD_BLAS//pre initialize the array for blas
        hf_array<double> temp_tdisf_upts(n_fields, n_dims);
        temp_tdisf_upts.initialize_to_zero();
#endif
        for(i=0; i<n_eles; i++)
        {
            // Calculate viscous flux
            for(j=0; j<n_upts_per_ele; j++)
            {
                detjac = detjac_upts(j,i);

                // solution in static-physical domain
                for(k=0; k<n_fields; k++)
                {
                    temp_u(k)=disu_upts(0)(j,i,k);

                    // gradient in static-physical domain
                    for (m=0; m<n_dims; m++)
                    {
                        temp_grad_u(k,m) = grad_disu_upts(j,i,k,m);
                    }
                }


                if(n_dims==2)
                {
                    calc_visf_2d(temp_u,temp_grad_u,temp_f);
                }
                else if(n_dims==3)
                {
                    calc_visf_3d(temp_u,temp_grad_u,temp_f);
                }
                else
                {
                    cout << "ERROR: Invalid number of dimensions ... " << endl;
                }

                // If LES or wall model, calculate SGS viscous flux
                if(LES)
                {
                    calc_sgsf_upts(temp_u,temp_grad_u,detjac,i,j,temp_sgsf);

                    // Add SGS or wall flux to viscous flux
#if defined _ACCELERATE_BLAS || defined _MKL_BLAS || defined _STANDARD_BLAS
                    cblas_daxpy(n_fields * n_dims, 1.0, temp_sgsf.get_ptr_cpu(), 1, temp_f.get_ptr_cpu(), 1);
#else
                    daxpy_wrapper(n_fields * n_dims, 1.0, temp_sgsf.get_ptr_cpu(), 1, temp_f.get_ptr_cpu(), 1);
#endif

                    // Transform SGS flux back to computational domain F=|J|J^-1*f
#if defined _ACCELERATE_BLAS || defined _MKL_BLAS || defined _STANDARD_BLAS
                    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, n_fields, n_dims, n_dims, 1.0, temp_sgsf.get_ptr_cpu(), n_fields, JGinv_upts.get_ptr_cpu(0, 0, j, i), n_dims, 0.0, temp_tdisf_upts.get_ptr_cpu(), n_fields);
                    for (k = 0; k < n_fields; k++)
                        for (l = 0; l < n_dims; l++)
                            sgsf_upts(j, i, k, l) = temp_tdisf_upts(k, l);
#else
                    for(k=0; k<n_fields; k++)
                    {
                        for(l=0; l<n_dims; l++)
                        {
                            sgsf_upts(j,i,k,l) = 0.0;
                            for(m=0; m<n_dims; m++)
                            {
                                sgsf_upts(j,i,k,l)+=JGinv_upts(l,m,j,i)*temp_sgsf(k,m);
                            }
                        }
                    }
#endif
                }

// Transform viscous flux to reference domain, F_tot+=det(J)J^-1*f
#if defined _ACCELERATE_BLAS || defined _MKL_BLAS || defined _STANDARD_BLAS
                cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, n_fields, n_dims, n_dims, 1.0, temp_f.get_ptr_cpu(), n_fields, JGinv_upts.get_ptr_cpu(0, 0, j, i), n_dims, 0.0, temp_tdisf_upts.get_ptr_cpu(), n_fields);
                for (k = 0; k < n_fields; k++)
                    for (l = 0; l < n_dims; l++)
                        tdisf_upts(j, i, k, l) += temp_tdisf_upts(k, l);
#else
                for (k = 0; k < n_fields; k++)
                {
                    for (l = 0; l < n_dims; l++)
                    {
                        for (m = 0; m < n_dims; m++)
                        {
                            tdisf_upts(j, i, k, l) += JGinv_upts(l, m, j, i) * temp_f(k, m);
                        }
                    }
                }
#endif
            }
        }
#endif

#ifdef _GPU

        evaluate_viscFlux_gpu_kernel_wrapper(n_upts_per_ele, n_dims, n_fields, n_eles, ele_type, order, run_input.filter_ratio, LES, motion, sgs_model, wall_model, run_input.wall_layer_t, wall_distance.get_ptr_gpu(), twall.get_ptr_gpu(), Lu.get_ptr_gpu(), Le.get_ptr_gpu(), disu_upts(0).get_ptr_gpu(), tdisf_upts.get_ptr_gpu(), sgsf_upts.get_ptr_gpu(), grad_disu_upts.get_ptr_gpu(), detjac_upts.get_ptr_gpu(), J_dyn_upts.get_ptr_gpu(), JGinv_upts.get_ptr_gpu(), JGinv_dyn_upts.get_ptr_gpu(), run_input.gamma, run_input.prandtl, run_input.rt_inf, run_input.mu_inf, run_input.c_sth, run_input.fix_vis, run_input.equation, run_input.diff_coeff, run_input.RANS, run_input.c_v1, run_input.omega, run_input.prandtl_t);

#endif

    }
}

// Calculate SGS flux at solution points
void eles::calc_sgsf_upts(hf_array<double>& temp_u, hf_array<double>& temp_grad_u, double& detjac, int ele, int upt, hf_array<double>& temp_sgsf)
{
    int i,j,k;
    int eddy, sim;
    double diag;
    double ke;
    double delta, mu, mu_t, vol;
    double rho, inte, rt_ratio;
    hf_array<double> u(n_dims);//resolved velocity
    hf_array<double> drho(n_dims), dene(n_dims), dke(n_dims), de(n_dims);//gradients
    hf_array<double> dmom(n_dims,n_dims), du(n_dims,n_dims), S(n_dims,n_dims);

    // quantities for wall distance
    double y;

    // primitive variables
    rho = temp_u(0);
    ke=0.;
    for (i=0; i<n_dims; i++)
    {
        u(i) = temp_u(i+1)/rho;
        ke += 0.5*pow(u(i),2);
    }
    inte = temp_u(n_fields-1)/rho - ke;

    // fluid properties
    rt_ratio = (run_input.gamma-1.0)*inte/(run_input.rt_inf);
    mu = (run_input.mu_inf)*pow(rt_ratio,1.5)*(1+(run_input.c_sth))/(rt_ratio+(run_input.c_sth));
    mu = mu + run_input.fix_vis*(run_input.mu_inf - mu);

    if (sgs_model==0)//Smagorinsky
    {
        // Magnitude of wall distance vector
        y = 0.0;
        for (i = 0; i < n_dims; i++)
            y += wall_distance(upt, ele, i) * wall_distance(upt, ele, i);
        y = sqrt(y);
    }
    // Initialize SGS flux hf_array to zero
    temp_sgsf.initialize_to_zero();

        // 0: Smagorinsky, 1: WALE, 2: WALE-similarity, 3: SVV, 4: Similarity
        if(sgs_model==0)
        {
            eddy = 1;
            sim = 0;
        }
        else if(sgs_model==1)
        {
            eddy = 1;
            sim = 0;
        }
        else if(sgs_model==2)
        {
            eddy = 1;
            sim = 1;
        }
        else if(sgs_model==3)
        {
            eddy = 0;
            sim = 0;
        }
        else if(sgs_model==4)
        {
            eddy = 0;
            sim = 1;
        }
        else
        {
            FatalError("SGS model not implemented");
        }

        // The modelling term can be written as
        //tau_{ij}=-2\mu_t(S_{ij)-1/3*S_{kk}\delta_{ij})

        if(eddy==1)
        {

            // Delta is the cutoff length-scale representing local grid resolution.

            // OPTION 1. Approx resolution in 1D element. Interval is [-1:1]
            // Appropriate for quads, hexes and tris. Not sure about tets.
            //dlt = 2.0/order;

            // OPTION 2. Deardorff definition (Deardorff, JFM 1970)
            vol = (*this).calc_ele_vol(detjac);
            delta = run_input.filter_ratio*pow(vol,1./n_dims)/(order+1.);

            // OPTION 3. Suggested by Bardina, AIAA 1980:
            // delta = sqrt((dx^2+dy^2+dz^2)/3)

            // Filtered solution gradient
            for (i=0; i<n_dims; i++)
            {
                drho(i) = temp_grad_u(0,i); // drho/dx_i
                dene(i) = temp_grad_u(n_fields-1,i); // dE/dx_i

                for (j=1; j<n_fields-1; j++)
                {
                    dmom(i,j-1) = temp_grad_u(j,i); // drhou_j/dx_i
                }
            }

            // Velocity and energy gradients
            for (i=0; i<n_dims; i++)
            {
                dke(i) = ke*drho(i);//dke/dx_i=ke*drho/dx_i+rho*u_j*du_j/dx_i

                for (j=0; j<n_dims; j++)
                {
                    du(i,j) = (dmom(i,j)-u(j)*drho(i))/rho;//du_j/dx_i=(drhou_j/dx_i-u_j*drho/dx_i)/rho
                    dke(i) += rho*u(j)*du(i,j);
                }

                de(i) = (dene(i)-dke(i)-drho(i)*inte)/rho;//de/dx_i=(dE/dx_i-dke/dx_i-e*drho/dx_i)/rho
            }

            /*! calculate traceless strain rate */

            // Strain rate tensor
            for (i=0; i<n_dims; i++)
                for (j=0; j<n_dims; j++)
                    S(i,j) = (du(i,j)+du(j,i))/2.0;//S_ij=0.5*(du_j/dx_i+du_i/dx_j)

            /*!-----------------------------------------*/

            // Eddy viscosity

            // Smagorinsky model
            if(sgs_model==0)
            {
                double Smod = 0.0;
                // Strain modulus
                for (i = 0; i < n_dims; i++)
                    for (j = 0; j < n_dims; j++)
                        Smod += 2.0 * S(i, j) * S(i, j);

                Smod = sqrt(Smod);
                mu_t = rho * min(y * y * run_input.Kappa * run_input.Kappa, run_input.C_s * run_input.C_s * delta * delta) * Smod;
            }

            //  Wall-Adapting Local Eddy-viscosity (WALE) SGS Model
            //
            //  NICOUD F., DUCROS F.: "Subgrid-Scale Stress Modelling Based on the Square
            //                         of the Velocity Gradient Tensor"
            //  Flow, Turbulence and Combustion 62: 183-200, 1999.
            //
            //                                            (sqij*sqij)^3/2
            //  Output: mu_t = rho*C_s^2*delta^2 * -----------------------------
            //                                     (Sij*Sij)^5/2+(sqij*sqij)^5/4
            //
            //  Typically Cw = 0.325.

            else if(sgs_model==1 || sgs_model==2)
            {

                double num=0.0;
                double denom=0.0;
                double eps=1.e-12;
                hf_array<double> Sq(n_dims,n_dims);
                hf_array<double> g_bar;
                hf_array<double> g_bar_transpose(n_dims,n_dims);
                diag = 0.0;

                // Square of gradient tensor
                Sq.initialize_to_zero();
                g_bar_transpose.initialize_to_zero();
#if defined _ACCELERATE_BLAS || defined _MKL_BLAS || defined _STANDARD_BLAS
                cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n_dims, n_dims, n_dims, 1.0, du.get_ptr_cpu(), n_dims, du.get_ptr_cpu(), n_dims, 0.0, g_bar_transpose.get_ptr_cpu(), n_dims);
                g_bar = transpose_array(g_bar_transpose);
                cblas_daxpy(n_dims * n_dims, 0.5, g_bar.get_ptr_cpu(), 1, Sq.get_ptr_cpu(), 1);
                cblas_daxpy(n_dims * n_dims, 0.5, g_bar_transpose.get_ptr_cpu(), 1, Sq.get_ptr_cpu(), 1);
#else
                dgemm(n_dims, n_dims, n_dims, 1.0, 0.0, du.get_ptr_cpu(), du.get_ptr_cpu(), g_bar_transpose.get_ptr_cpu());
                g_bar = transpose_array(g_bar_transpose);
                daxpy_wrapper(n_dims * n_dims, 0.5, g_bar.get_ptr_cpu(), 1, Sq.get_ptr_cpu(), 1);
                daxpy_wrapper(n_dims * n_dims, 0.5, g_bar_transpose.get_ptr_cpu(), 1, Sq.get_ptr_cpu(), 1);
#endif
                //subract diagonal
                for (i = 0; i < n_dims; i++)
                    diag += g_bar(i, i) / 3.0;
                for (i = 0; i < n_dims; i++)
                    Sq(i, i) -= diag;

                // Numerator and denominator
                for (i=0; i<n_dims; i++)
                {
                    for (j=0; j<n_dims; j++)
                    {
                        num += Sq(i,j)*Sq(i,j);
                        denom += S(i,j)*S(i,j);
                    }
                }

                denom = pow(denom,2.5) + pow(num,1.25);
                num = pow(num,1.5);
                mu_t = rho*run_input.C_s*run_input.C_s*delta*delta*num/(denom+eps);
            }

            // Add eddy-viscosity term to SGS fluxes
            // Subtract diag
            diag = 0.;
            for (i = 0; i < n_dims; i++)
                diag += S(i, i) / 3.0;
            for (i = 0; i < n_dims; i++)
                S(i, i) -= diag;

            for (j=0; j<n_dims; j++)
            {
                temp_sgsf(0,j) = 0.0; // Density flux
                temp_sgsf(n_fields-1,j) = -1.0*run_input.gamma*mu_t/run_input.prandtl_t*de(j); // Energy flux
                for (k = 0; k < n_dims; k++)
                    temp_sgsf(n_fields - 1, j) -= u(k) * 2.0 * mu_t * S(k, j);
                for (i=1; i<n_fields-1; i++)
                {
                    temp_sgsf(i,j) = -2.0*mu_t*S(i-1,j); // Velocity flux
                }
            }
        }

        // Add similarity term to SGS fluxes if WSM or Similarity model
        if(sim==1)
        {
            for (j=0; j<n_dims; j++)
            {
                temp_sgsf(0,j) += 0.0; // Density flux
                temp_sgsf(n_fields-1,j) += run_input.gamma*rho*Le(upt,ele,j); // Energy flux
            }

            // Momentum fluxes
            if(n_dims==2)
            {
                temp_sgsf(1,0) += rho*Lu(upt,ele,0);
                temp_sgsf(1,1) += rho*Lu(upt,ele,2);
                temp_sgsf(2,0) += temp_sgsf(1,1);
                temp_sgsf(2,1) += rho*Lu(upt,ele,1);
            }
            else if(n_dims==3)
            {
                temp_sgsf(1,0) += rho*Lu(upt,ele,0);
                temp_sgsf(1,1) += rho*Lu(upt,ele,3);
                temp_sgsf(1,2) += rho*Lu(upt,ele,4);
                temp_sgsf(2,0) += temp_sgsf(1,1);
                temp_sgsf(2,1) += rho*Lu(upt,ele,1);
                temp_sgsf(2,2) += rho*Lu(upt,ele,5);
                temp_sgsf(3,0) += temp_sgsf(1,2);
                temp_sgsf(3,1) += temp_sgsf(2,2);
                temp_sgsf(3,2) += rho*Lu(upt,ele,2);
            }
        }
    
}


// calculate source term for SA turbulence model at solution points
void eles::calc_src_upts_SA(void)
{
    if (n_eles!=0)
    {
#ifdef _CPU

        int i,j,k,l,m;

        for(i=0; i<n_eles; i++)
        {
            for(j=0; j<n_upts_per_ele; j++)
            {

                // physical solution
                for(k=0; k<n_fields; k++)
                {
                    temp_u(k)=disu_upts(0)(j,i,k);
                }

                // physical gradient
                for(k=0; k<n_fields; k++)
                {
                    for (m=0; m<n_dims; m++)
                    {
                        temp_grad_u(k,m) = grad_disu_upts(j,i,k,m);
                    }
                }

                // source term
                if(n_dims==2)
                    calc_source_SA_2d(temp_u, temp_grad_u, wall_distance_mag(j,i), src_upts(j,i,n_fields-1));
                else if(n_dims==3)
                    calc_source_SA_3d(temp_u, temp_grad_u, wall_distance_mag(j,i), src_upts(j,i,n_fields-1));
                else
                    cout << "ERROR: Invalid number of dimensions ... " << endl;
            }
        }

#endif

#ifdef _GPU
        calc_src_upts_SA_gpu_kernel_wrapper(n_upts_per_ele, n_dims, n_fields, n_eles, disu_upts(0).get_ptr_gpu(), grad_disu_upts.get_ptr_gpu(), wall_distance_mag.get_ptr_gpu(), src_upts.get_ptr_gpu(), run_input.gamma, run_input.prandtl, run_input.rt_inf, run_input.mu_inf, run_input.c_sth, run_input.fix_vis, run_input.c_v1, run_input.c_v2, run_input.c_v3, run_input.c_b1, run_input.c_b2, run_input.c_w2, run_input.c_w3, run_input.omega, run_input.Kappa);
#endif

    }
}


/*! If using a RANS or LES near-wall model, calculate distance
 of each solution point to nearest no-slip wall by a brute-force method */

void eles::calc_wall_distance(const int n_seg_noslip_inters, const int n_tri_noslip_inters, const int n_quad_noslip_inters, hf_array<hf_array<double>> &loc_noslip_bdy)
{
    if(n_eles!=0)
    {
        int i,j,k,m,n;
        int n_fpts_per_inter_seg = order+1;
        int n_fpts_per_inter_tri = (order+2)*(order+1)/2;
        int n_fpts_per_inter_quad = (order+1)*(order+1);
        double dist;
        double distmin;
        hf_array<double> pos(n_dims);
        hf_array<double> vec(n_dims);
        hf_array<double> vecmin(n_dims);
        vecmin.initialize_to_value(1e20);

        // hold our breath and go round the brute-force loop...
        for (i=0; i<n_eles; ++i)
        {
            for (j=0; j<n_upts_per_ele; ++j)
            {

                // get coords of current solution point
                for(k=0; k<n_dims; k++)
                {
                    pos(k)=pos_upts(j,i,k);
                }

                // initialize wall distance
                distmin = 1e20;

                // line segment boundaries
                for (k=0; k<n_seg_noslip_inters; ++k)
                {

                    for (m=0; m<n_fpts_per_inter_seg; ++m)
                    {

                        dist = 0.0;
                        // get coords of boundary flux point
                        for (n=0; n<n_dims; ++n)
                        {
                            vec(n) = pos(n) - loc_noslip_bdy(0)(n,m,k);
                            dist += vec(n)*vec(n);
                        }
                        dist = sqrt(dist);

                        // update shortest vector
                        if (dist < distmin)
                        {
                            for (n=0; n<n_dims; ++n) vecmin(n) = vec(n);
                            distmin = dist;
                        }
                    }
                }

                // tri boundaries
                for (k=0; k<n_tri_noslip_inters; ++k)
                {

                    for (m=0; m<n_fpts_per_inter_tri; ++m)
                    {

                        dist = 0.0;
                        // get coords of boundary flux point
                        for (n=0; n<n_dims; ++n)
                        {
                            vec(n) = pos(n) - loc_noslip_bdy(1)(n,m,k);
                            dist += vec(n)*vec(n);
                        }
                        dist = sqrt(dist);

                        // update shortest vector
                        if (dist < distmin)
                        {
                            for (n=0; n<n_dims; ++n) vecmin(n) = vec(n);
                            distmin = dist;
                        }
                    }
                }

                // quad boundaries
                for (k=0; k<n_quad_noslip_inters; ++k)
                {

                    for (m=0; m<n_fpts_per_inter_quad; ++m)
                    {

                        dist = 0.0;
                        // get coords of boundary flux point
                        for (n=0; n<n_dims; ++n)
                        {
                            vec(n) = pos(n) - loc_noslip_bdy(2)(n,m,k);
                            dist += vec(n)*vec(n);
                        }
                        dist = sqrt(dist);

                        // update shortest vector
                        if (dist < distmin)
                        {
                            for (n=0; n<n_dims; ++n) vecmin(n) = vec(n);
                            distmin = dist;
                        }
                    }
                }
                for (n=0; n<n_dims; ++n) wall_distance(j,i,n) = vecmin(n);

                if (run_input.RANS > 0)
                {
                    wall_distance_mag(j,i) = distmin;
                }
            }
        }
    }
}

/*! Extrapolate transformed SGS flux to flux points and transform back to physical domain */
void eles::extrapolate_sgsFlux(void)
{
    if (n_eles!=0)
    {

#ifdef _CPU

        if(opp_0_sparse==0) // dense
        {
#if defined _ACCELERATE_BLAS || defined _MKL_BLAS || defined _STANDARD_BLAS

            for (int i=0; i<n_dims; i++)
            {
                cblas_dgemm(CblasColMajor,CblasNoTrans,CblasNoTrans,n_fpts_per_ele,n_fields_mul_n_eles,n_upts_per_ele,1.0,opp_0.get_ptr_cpu(),n_fpts_per_ele,sgsf_upts.get_ptr_cpu(0,0,0,i),n_upts_per_ele,0.0,sgsf_fpts.get_ptr_cpu(0,0,0,i),n_fpts_per_ele);
            }

#else
            for (int i=0; i<n_dims; i++)
            {
                dgemm(n_fpts_per_ele,n_fields_mul_n_eles,n_upts_per_ele,1.0,0.0,opp_0.get_ptr_cpu(),sgsf_upts.get_ptr_cpu(0,0,0,i),sgsf_fpts.get_ptr_cpu(0,0,0,i));
            }

#endif
        }
        else if(opp_0_sparse==1) // mkl blas four-hf_array coo format
        {
#if defined _MKL_BLAS

            for (int i = 0; i < n_dims; i++)
            {
                mkl_sparse_d_mm(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, opp_0_mkl,
                                opp_0_descr, SPARSE_LAYOUT_COLUMN_MAJOR,
                                sgsf_upts.get_ptr_cpu(0, 0, 0, i),
                                n_fields_mul_n_eles, n_upts_per_ele, 0.0,
                                sgsf_fpts.get_ptr_cpu(0, 0, 0, i), n_fpts_per_ele);
            }

#endif
        }
        else
        {
            cout << "ERROR: Unknown storage for opp_0 ... " << endl;
        }

#endif

//Transform back to physical domain

        hf_array<double> temp_psgsf(n_dims, n_fields); //temporary physical corrected gradients
        hf_array<double> temp_tsgsf(n_dims, n_fields); //temporary transformed correct gradients
        temp_psgsf.initialize_to_zero();
        double inv_detjac;
        for (int i = 0; i < n_eles; i++)
        {
            //at flux points
            for (int j = 0; j < n_fpts_per_ele; j++)
            {
                inv_detjac = 1.0 / detjac_fpts(j, i);
                //copy data from ref domain
                for (int k = 0; k < n_fields; k++)
                    for (int d = 0; d < n_dims; d++)
                        temp_tsgsf(d, k) = sgsf_fpts(j, i, k, d);

//f=|J|^-1*JF
#if defined _ACCELERATE_BLAS || defined _MKL_BLAS || defined _STANDARD_BLAS
                        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n_dims, n_fields, n_dims, inv_detjac, Jacobian_fpts.get_ptr_cpu(0,0,j,i), n_dims, temp_tsgsf.get_ptr_cpu(), n_dims, 0.0, temp_psgsf.get_ptr_cpu(), n_dims);
#else
                        dgemm(n_dims, n_fields, n_dims, inv_detjac, 0.0, Jacobian_fpts.get_ptr_cpu(0,0,j,i), temp_tsgsf.get_ptr_cpu(), temp_psgsf.get_ptr_cpu());
#endif
                        //copy physical sgs flux back to array
                        for (int k = 0; k < n_fields; k++)
                            for (int d = 0; d < n_dims; d++)
                                sgsf_fpts(j, i, k, d) = temp_psgsf(d, k);
            }
        }
#ifdef _GPU

        if(opp_0_sparse==0)
        {
            for (int i=0; i<n_dims; i++)
            {
                cublasDgemm('N','N',n_fpts_per_ele,n_fields_mul_n_eles,n_upts_per_ele,1.0,opp_0.get_ptr_gpu(),n_fpts_per_ele,sgsf_upts.get_ptr_gpu(0,0,0,i),n_upts_per_ele,0.0,sgsf_fpts.get_ptr_gpu(0,0,0,i),n_fpts_per_ele);
            }
        }
        else if (opp_0_sparse==1)
        {
            for (int i=0; i<n_dims; i++)
            {
                bespoke_SPMV(n_fpts_per_ele, n_upts_per_ele, n_fields, n_eles, opp_0_ell_data.get_ptr_gpu(), opp_0_ell_indices.get_ptr_gpu(), opp_0_nnz_per_row, sgsf_upts.get_ptr_gpu(0,0,0,i), sgsf_fpts.get_ptr_gpu(0,0,0,i), ele_type, order, 0);
            }
        }
        else
        {
            cout << "ERROR: Unknown storage for opp_0 ... " << endl;
        }
#endif
    }
}

// sense shock and filter (for concentration method) - only on GPUs

void eles::shock_capture(void)
{
    if (n_eles!=0)
    {
        //shock detection
        if(run_input.shock_det==0)//persson
            shock_det_persson();
       else
            FatalError("Shock detector not implemented.");

        //shock capturing
        if (run_input.shock_cap == 1) //exponential filter
        {
            hf_array<double> temp_sol(n_upts_per_ele, n_fields);
            hf_array<double> filt_sol(n_upts_per_ele, n_fields);
            filt_sol.initialize_to_zero();
            int i, j, k;
            for (i = 0; i < n_eles; i++)
            {
                if (sensor(i) >= run_input.s0)
                {
                    //copy solution to filt_sol
                    for (j = 0; j < n_upts_per_ele; j++)
                        for (k = 0; k < n_fields; k++)
                            temp_sol(j, k) = disu_upts(0)(j, i, k);

#if defined _ACCELERATE_BLAS || defined _MKL_BLAS || defined _STANDARD_BLAS
                    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n_upts_per_ele, n_fields, n_upts_per_ele, 1.0, exp_filter.get_ptr_cpu(), n_upts_per_ele, temp_sol.get_ptr_cpu(), n_upts_per_ele, 0.0, filt_sol.get_ptr_cpu(), n_upts_per_ele);
#else
                    dgemm(n_upts_per_ele, n_fields, n_upts_per_ele, 1.0, 0.0, exp_filter.get_ptr_cpu(), temp_sol.get_ptr_cpu(), filt_sol.get_ptr_cpu());
#endif
                    //copy filted solution back to disu_upts
                    for (j = 0; j < n_upts_per_ele; j++)
                        for (k = 0; k < n_fields; k++)
                            disu_upts(0)(j, i, k) = filt_sol(j, k);
                }
            }
        }
        else
            FatalError("Shock capturing method not implemented yet");
    }
}
// get the type of element

int eles::get_ele_type(void)
{
    return ele_type;
}

// get number of elements

int eles::get_n_eles(void)
{
    return n_eles;
}

// get number of ppts_per_ele
int eles::get_n_ppts_per_ele(void)
{
    return n_ppts_per_ele;
}

// get number of peles_per_ele
int eles::get_n_peles_per_ele(void)
{
    return n_peles_per_ele;
}

// get number of verts_per_ele
int eles::get_n_verts_per_ele(void)
{
    return n_verts_per_ele;
}

// get number of elements

int eles::get_n_dims(void)
{
    return n_dims;
}

// get number of solution fields

int eles::get_n_fields(void)
{
    return n_fields;
}

// get number of solutions points per element

int eles::get_n_upts_per_ele(void)
{
    return n_upts_per_ele;
}

// get number of shape points per element

int eles::get_n_spts_per_ele(int in_ele)
{
    return n_spts_per_ele(in_ele);
}


// set the shape hf_array
void eles::set_shape(int in_max_n_spts_per_ele)
{
    shape.setup(n_dims,in_max_n_spts_per_ele,n_eles);

    n_spts_per_ele.setup(n_eles);
}

// set a shape node

void eles::set_shape_node(int in_spt, int in_ele, hf_array<double>& in_pos)
{
    for(int i=0; i<n_dims; i++)
    {
        shape(i,in_spt,in_ele)=in_pos(i);
    }
}

// get shape point coordinates

double eles::get_shape(int in_dim, int in_spt, int in_ele)
{
    return shape(in_dim,in_spt,in_ele);
}

void eles::set_rank(int in_rank)
{
    rank = in_rank;
}

// set bc type
void eles::set_bcid(int in_ele,int in_inter, int in_bcid)
{
    bcid(in_ele, in_inter) = in_bcid;
}

// set number of shape points

void eles::set_n_spts(int in_ele, int in_n_spts)
{
    n_spts_per_ele(in_ele) = in_n_spts;
}

// set global element number

void eles::set_ele2global_ele(int in_ele, int in_global_ele)
{
    ele2global_ele(in_ele) = in_global_ele;
}


// set opp_0 (transformed discontinuous solution at solution points to transformed discontinuous solution at flux points)

void eles::set_opp_0(int in_sparse)
{
    int i,j,k;

    hf_array<double> loc(n_dims);

    opp_0.setup(n_fpts_per_ele,n_upts_per_ele);

    for(i=0; i<n_upts_per_ele; i++)
    {
        for(j=0; j<n_fpts_per_ele; j++)
        {
            for(k=0; k<n_dims; k++)
            {
                loc(k)=tloc_fpts(k,j);
            }

            opp_0(j,i)=eval_nodal_basis(i,loc);
        }
    }

#ifdef _GPU
    opp_0.cp_cpu_gpu();
#endif

    //cout << "opp_0" << endl;
    //cout << "ele_type=" << ele_type << endl;
    //opp_0.print();
    //cout << endl;

    if(in_sparse==0)
    {
        opp_0_sparse=0;
    }
    else if(in_sparse==1)
    {
        opp_0_sparse=1;

#ifdef _CPU
#ifdef _MKL_BLAS
        array_to_mklcoo(opp_0, opp_0_data, opp_0_rows, opp_0_cols);
        mkl_sparse_d_create_coo(&opp_0_mkl,
                                SPARSE_INDEX_BASE_ONE, n_fpts_per_ele, n_upts_per_ele,
                                opp_0_data.get_dim(0), opp_0_rows.get_ptr_cpu(),
                                opp_0_cols.get_ptr_cpu(), opp_0_data.get_ptr_cpu());
        opp_0_descr.type = SPARSE_MATRIX_TYPE_GENERAL;
#else // _MKL_BLAS
        FatalError("Sparse matrix operation support MKL BLAS only!");
#endif
#endif

#ifdef _GPU
        array_to_ellpack(opp_0, opp_0_ell_data, opp_0_ell_indices, opp_0_nnz_per_row);
        opp_0_ell_data.cp_cpu_gpu();
        opp_0_ell_indices.cp_cpu_gpu();
#endif

    }
    else
    {
        cout << "ERROR: Invalid sparse matrix form ... " << endl;
    }



}

// set opp_1 (transformed discontinuous flux at solution points to normal transformed discontinuous flux at flux points)

void eles::set_opp_1(int in_sparse)
{
    int i,j,k,l;
    hf_array<double> loc(n_dims);

    opp_1.setup(n_dims);
    for (i=0; i<n_dims; i++)
        opp_1(i).setup(n_fpts_per_ele,n_upts_per_ele);

    for(i=0; i<n_dims; i++)
    {
        for(j=0; j<n_upts_per_ele; j++)
        {
            for(k=0; k<n_fpts_per_ele; k++)
            {
                for(l=0; l<n_dims; l++)
                {
                    loc(l)=tloc_fpts(l,k);
                }

                opp_1(i)(k,j)=eval_nodal_basis(j,loc)*tnorm_fpts(i,k);
            }
        }
        //cout << "opp_1,i =" << i << endl;
        //cout << "ele_type=" << ele_type << endl;
        //opp_1(i).print();
        //cout << endl;
    }

#ifdef _GPU
    for (i=0; i<n_dims; i++)
        opp_1(i).cp_cpu_gpu();
#endif


    if(in_sparse==0)
    {
        opp_1_sparse=0;
    }
    else if(in_sparse==1)
    {
        opp_1_sparse=1;

#ifdef _CPU
#ifdef _MKL_BLAS
        opp_1_data.setup(n_dims);
        opp_1_rows.setup(n_dims);
        opp_1_cols.setup(n_dims);
        opp_1_mkl.setup(n_dims);
        opp_1_descr.setup(n_dims);
        for (i = 0; i < n_dims; i++)
        {
           array_to_mklcoo(opp_1(i), opp_1_data(i), opp_1_rows(i),opp_1_cols(i));
           mkl_sparse_d_create_coo(opp_1_mkl.get_ptr_cpu(i),
                                    SPARSE_INDEX_BASE_ONE, n_fpts_per_ele, n_upts_per_ele,
                                    opp_1_data(i).get_dim(0), opp_1_rows(i).get_ptr_cpu(),
                                    opp_1_cols(i).get_ptr_cpu(), opp_1_data(i).get_ptr_cpu());
            opp_1_descr(i).type = SPARSE_MATRIX_TYPE_GENERAL;
        }
#else
        FatalError("Sparse matrix operation support MKL BLAS only!");
#endif
#endif

#ifdef _GPU
        opp_1_ell_data.setup(n_dims);
        opp_1_ell_indices.setup(n_dims);
        opp_1_nnz_per_row.setup(n_dims);
        for (i=0; i<n_dims; i++)
        {
            array_to_ellpack(opp_1(i), opp_1_ell_data(i), opp_1_ell_indices(i), opp_1_nnz_per_row(i));
            opp_1_ell_data(i).cp_cpu_gpu();
            opp_1_ell_indices(i).cp_cpu_gpu();
        }
#endif

    }
    else
    {
        cout << "ERROR: Invalid sparse matrix form ... " << endl;
    }
}

// set opp_2 (transformed discontinuous flux at solution points to divergence of transformed discontinuous flux at solution points)

void eles::set_opp_2(int in_sparse)
{

    int i,j,k,l;

    hf_array<double> loc(n_dims);

    opp_2.setup(n_dims);
    for (i=0; i<n_dims; i++)
        opp_2(i).setup(n_upts_per_ele,n_upts_per_ele);

    for(i=0; i<n_dims; i++)
    {
        for(j=0; j<n_upts_per_ele; j++)
        {
            for(k=0; k<n_upts_per_ele; k++)
            {
                for(l=0; l<n_dims; l++)
                {
                    loc(l)=loc_upts(l,k);
                }

                opp_2(i)(k,j)=eval_d_nodal_basis(j,i,loc);
            }
        }

        //cout << "opp_2,i =" << i << endl;
        //cout << "ele_type=" << ele_type << endl;
        //opp_2(i).print();
        //cout << endl;

        //cout << "opp_2,i=" << i << endl;
        //opp_2(i).print();

    }

#ifdef _GPU
    for (i=0; i<n_dims; i++)
        opp_2(i).cp_cpu_gpu();
#endif

    //cout << "opp 2" << endl;
    //opp_2.print();

    if(in_sparse==0)
    {
        opp_2_sparse=0;
    }
    else if(in_sparse==1)
    {
        opp_2_sparse=1;

#ifdef _CPU
#ifdef _MKL_BLAS
        opp_2_data.setup(n_dims);
        opp_2_rows.setup(n_dims);
        opp_2_cols.setup(n_dims);
        opp_2_mkl.setup(n_dims);
        opp_2_descr.setup(n_dims);
        for (i = 0; i < n_dims; i++)
        {
           array_to_mklcoo(opp_2(i), opp_2_data(i), opp_2_rows(i),opp_2_cols(i));
           mkl_sparse_d_create_coo(opp_2_mkl.get_ptr_cpu(i),
                                    SPARSE_INDEX_BASE_ONE, n_upts_per_ele, n_upts_per_ele,
                                    opp_2_data(i).get_dim(0), opp_2_rows(i).get_ptr_cpu(),
                                    opp_2_cols(i).get_ptr_cpu(), opp_2_data(i).get_ptr_cpu());
            opp_2_descr(i).type = SPARSE_MATRIX_TYPE_GENERAL;
        }
#else
        FatalError("Sparse matrix operation support MKL BLAS only!");
#endif
#endif

#ifdef _GPU
        opp_2_ell_data.setup(n_dims);
        opp_2_ell_indices.setup(n_dims);
        opp_2_nnz_per_row.setup(n_dims);
        for (i=0; i<n_dims; i++)
        {
            array_to_ellpack(opp_2(i), opp_2_ell_data(i), opp_2_ell_indices(i), opp_2_nnz_per_row(i));
            opp_2_ell_data(i).cp_cpu_gpu();
            opp_2_ell_indices(i).cp_cpu_gpu();
        }
#endif
    }
    else
    {
        cout << "ERROR: Invalid sparse matrix form ... " << endl;
    }
}

// set opp_3 (normal transformed correction flux at edge flux points to divergence of transformed correction flux at solution points)

void eles::set_opp_3(int in_sparse)
{

    opp_3.setup(n_upts_per_ele,n_fpts_per_ele);
    (*this).fill_opp_3(opp_3);

    //cout << "OPP_3" << endl;
    //cout << "ele_type=" << ele_type << endl;
    //opp_3.print();
    //cout << endl;

#ifdef _GPU
    opp_3.cp_cpu_gpu();
#endif

    if(in_sparse==0)
    {
        opp_3_sparse=0;
    }
    else if(in_sparse==1)
    {
        opp_3_sparse=1;

#ifdef _CPU
#ifdef _MKL_BLAS
        array_to_mklcoo(opp_3, opp_3_data, opp_3_rows,opp_3_cols);
        mkl_sparse_d_create_coo(&opp_3_mkl,
                                SPARSE_INDEX_BASE_ONE, n_upts_per_ele, n_fpts_per_ele,
                                opp_3_data.get_dim(0), opp_3_rows.get_ptr_cpu(),
                                opp_3_cols.get_ptr_cpu(), opp_3_data.get_ptr_cpu());
        opp_3_descr.type = SPARSE_MATRIX_TYPE_GENERAL;
#else // _MKL_BLAS
        FatalError("Sparse matrix operation support MKL BLAS only!");
#endif
#endif

#ifdef _GPU
        array_to_ellpack(opp_3, opp_3_ell_data, opp_3_ell_indices, opp_3_nnz_per_row);
        opp_3_ell_data.cp_cpu_gpu();
        opp_3_ell_indices.cp_cpu_gpu();
#endif
    }
    else
    {
        cout << "ERROR: Invalid sparse matrix form ... " << endl;
    }
}

// set opp_4 (transformed solution at solution points to transformed gradient of transformed solution at solution points)

void eles::set_opp_4(int in_sparse)
{
    int i,j,k,l;

    hf_array<double> loc(n_dims);

    opp_4.setup(n_dims);
    for (i=0; i<n_dims; i++)
        opp_4(i).setup(n_upts_per_ele, n_upts_per_ele);

    for(i=0; i<n_dims; i++)
    {
        for(j=0; j<n_upts_per_ele; j++)
        {
            for(k=0; k<n_upts_per_ele; k++)
            {
                for(l=0; l<n_dims; l++)
                {
                    loc(l)=loc_upts(l,k);
                }

                opp_4(i)(k,j) = eval_d_nodal_basis(j,i,loc);
            }
        }
    }

#ifdef _GPU
    for (i=0; i<n_dims; i++)
        opp_4(i).cp_cpu_gpu();
#endif

    if(in_sparse==0)
    {
        opp_4_sparse=0;
    }
    else if(in_sparse==1)
    {
        opp_4_sparse=1;

#ifdef _CPU
#ifdef _MKL_BLAS
        opp_4_data.setup(n_dims);
        opp_4_rows.setup(n_dims);
        opp_4_cols.setup(n_dims);
        opp_4_mkl.setup(n_dims);
        opp_4_descr.setup(n_dims);
        for (i = 0; i < n_dims; i++)
        {
            array_to_mklcoo(opp_4(i), opp_4_data(i), opp_4_rows(i),opp_4_cols(i));
            mkl_sparse_d_create_coo(opp_4_mkl.get_ptr_cpu(i),
                                   SPARSE_INDEX_BASE_ONE, n_upts_per_ele, n_upts_per_ele,
                                   opp_4_data(i).get_dim(0), opp_4_rows(i).get_ptr_cpu(),
                                   opp_4_cols(i).get_ptr_cpu(), opp_4_data(i).get_ptr_cpu());
            opp_4_descr(i).type = SPARSE_MATRIX_TYPE_GENERAL;
        }
#else
        FatalError("Sparse matrix operation support MKL BLAS only!");
#endif
#endif

#ifdef _GPU
        opp_4_ell_data.setup(n_dims);
        opp_4_ell_indices.setup(n_dims);
        opp_4_nnz_per_row.setup(n_dims);
        for (i=0; i<n_dims; i++)
        {
            array_to_ellpack(opp_4(i), opp_4_ell_data(i), opp_4_ell_indices(i), opp_4_nnz_per_row(i));
            opp_4_ell_data(i).cp_cpu_gpu();
            opp_4_ell_indices(i).cp_cpu_gpu();
        }
#endif
    }
    else
    {
        cout << "ERROR: Invalid sparse matrix form ... " << endl;
    }
}

// transformed solution correction at flux points to transformed gradient correction at solution points

void eles::set_opp_5(int in_sparse)
{
    int i,j,k,l;

    hf_array<double> loc(n_dims);

    opp_5.setup(n_dims);
    for (i=0; i<n_dims; i++)
        opp_5(i).setup(n_upts_per_ele, n_fpts_per_ele);

    for(i=0; i<n_dims; i++)
    {
        for(j=0; j<n_fpts_per_ele; j++)
        {
            for(k=0; k<n_upts_per_ele; k++)
            {
                /*
                 for(l=0;l<n_dims;l++)
                 {
                 loc(l)=loc_upts(l,k);
                 }
                 */

                //opp_5(i)(k,j) = eval_div_vcjh_basis(j,loc)*tnorm_fpts(i,j);
                opp_5(i)(k,j) = opp_3(k,j)*tnorm_fpts(i,j);
            }
        }
    }

#ifdef _GPU
    for (i=0; i<n_dims; i++)
        opp_5(i).cp_cpu_gpu();
#endif

    //cout << "opp_5" << endl;
    //opp_5.print();

    if(in_sparse==0)
    {
        opp_5_sparse=0;
    }
    else if(in_sparse==1)
    {
        opp_5_sparse=1;

#ifdef _CPU
#ifdef _MKL_BLAS
        opp_5_data.setup(n_dims);
        opp_5_rows.setup(n_dims);
        opp_5_cols.setup(n_dims);
        opp_5_mkl.setup(n_dims);
        opp_5_descr.setup(n_dims);
        for (i = 0; i < n_dims; i++)
        {
            array_to_mklcoo(opp_5(i), opp_5_data(i), opp_5_rows(i),opp_5_cols(i));
            mkl_sparse_d_create_coo(opp_5_mkl.get_ptr_cpu(i),
                                    SPARSE_INDEX_BASE_ONE, n_upts_per_ele, n_fpts_per_ele,
                                    opp_5_data(i).get_dim(0), opp_5_rows(i).get_ptr_cpu(),
                                    opp_5_cols(i).get_ptr_cpu(), opp_5_data(i).get_ptr_cpu());
            opp_5_descr(i).type = SPARSE_MATRIX_TYPE_GENERAL;
        }
#else
        FatalError("Sparse matrix operation support MKL BLAS only!");
#endif
#endif

#ifdef _GPU
        opp_5_ell_data.setup(n_dims);
        opp_5_ell_indices.setup(n_dims);
        opp_5_nnz_per_row.setup(n_dims);
        for (i=0; i<n_dims; i++)
        {
            array_to_ellpack(opp_5(i), opp_5_ell_data(i), opp_5_ell_indices(i), opp_5_nnz_per_row(i));
            opp_5_ell_data(i).cp_cpu_gpu();
            opp_5_ell_indices(i).cp_cpu_gpu();
        }
#endif
    }
    else
    {
        cout << "ERROR: Invalid sparse matrix form ... " << endl;
    }
}

// transformed gradient at solution points to transformed gradient at flux points

void eles::set_opp_6(int in_sparse)
{
    int j,l,m;

    hf_array<double> loc(n_dims);

    opp_6.setup(n_fpts_per_ele, n_upts_per_ele);

    for(j=0; j<n_upts_per_ele; j++)
    {
        for(l=0; l<n_fpts_per_ele; l++)
        {
            for(m=0; m<n_dims; m++)
            {
                loc(m) = tloc_fpts(m,l);
            }
            opp_6(l,j) = eval_nodal_basis(j,loc);
        }
    }

    //cout << "opp_6" << endl;
    //opp_6.print();

#ifdef _GPU
    opp_6.cp_cpu_gpu();
#endif

    if(in_sparse==0)
    {
        opp_6_sparse=0;
    }
    else if(in_sparse==1)
    {
        opp_6_sparse=1;

#ifdef _CPU
#ifdef _MKL_BLAS
        array_to_mklcoo(opp_6,opp_6_data,opp_6_rows,opp_6_cols);
        mkl_sparse_d_create_coo(&opp_6_mkl,
                        SPARSE_INDEX_BASE_ONE, n_fpts_per_ele, n_upts_per_ele,
                        opp_6_data.get_dim(0), opp_6_rows.get_ptr_cpu(),
                        opp_6_cols.get_ptr_cpu(), opp_6_data.get_ptr_cpu());
            opp_6_descr.type = SPARSE_MATRIX_TYPE_GENERAL;
#else // _MKL_BLAS
FatalError("Sparse matrix operation support MKL BLAS only!");
#endif
#endif

#ifdef _GPU
        array_to_ellpack(opp_6, opp_6_ell_data, opp_6_ell_indices, opp_6_nnz_per_row);
        opp_6_ell_data.cp_cpu_gpu();
        opp_6_ell_indices.cp_cpu_gpu();
#endif

    }
    else
    {
        cout << "ERROR: Invalid sparse matrix form ... " << endl;
    }
}

// set opp_p (solution at solution points to solution at plot points)

void eles::set_opp_p(void)
{
    int i,j,k;

    hf_array<double> loc(n_dims);

    opp_p.setup(n_ppts_per_ele,n_upts_per_ele);

    for(i=0; i<n_upts_per_ele; i++)
    {
        for(j=0; j<n_ppts_per_ele; j++)
        {
            for(k=0; k<n_dims; k++)
            {
                loc(k)=loc_ppts(k,j);
            }

            opp_p(j,i)=eval_nodal_basis(i,loc);
        }
    }

}

// set opp_probe (solution at solution points to solution at probe points)

void eles::set_opp_probe(hf_array<double>& in_loc)
{
    int i;
    for(i=0; i<n_upts_per_ele; i++)
    {
        opp_probe(i)=eval_nodal_basis(i,in_loc);
    }

}

void eles::set_opp_inters_cubpts(void)
{

    int i,j,k,l;

    hf_array<double> loc(n_dims);

    opp_inters_cubpts.setup(n_inters_per_ele);

    for (i=0; i<n_inters_per_ele; i++)
    {
        opp_inters_cubpts(i).setup(n_cubpts_per_inter(i),n_upts_per_ele);
    }

    for(l=0; l<n_inters_per_ele; l++)
    {
        for(i=0; i<n_upts_per_ele; i++)
        {
            for(j=0; j<n_cubpts_per_inter(l); j++)
            {
                for(k=0; k<n_dims; k++)
                {
                    loc(k)=loc_inters_cubpts(l)(k,j);
                }

                opp_inters_cubpts(l)(j,i)=eval_nodal_basis(i,loc);
            }
        }
    }

}

void eles::set_opp_volume_cubpts(hf_array<double> &in_loc_volume_cubpts, hf_array<double> &out_opp_volume_cubpts)
{

    int i,j,k,l;
    hf_array<double> loc(n_dims);
    int n_cubpts_per_ele = in_loc_volume_cubpts.get_dim(1);
    out_opp_volume_cubpts.setup(n_cubpts_per_ele, n_upts_per_ele);

    for(i=0; i<n_upts_per_ele; i++)
    {
        for(j=0; j<n_cubpts_per_ele; j++)
        {
            for(k=0; k<n_dims; k++)
            {
                loc(k)=in_loc_volume_cubpts(k,j);
            }

            out_opp_volume_cubpts(j,i)=eval_nodal_basis(i,loc);
        }
    }
}


// set opp_r (solution at restart points to solution at solution points)

void eles::set_opp_r(void)
{
    int i,j,k;

    hf_array<double> loc(n_dims);

    opp_r.setup(n_upts_per_ele,n_upts_per_ele_rest);

    for(i=0; i<n_upts_per_ele_rest; i++)
    {
        for(j=0; j<n_upts_per_ele; j++)
        {
            for(k=0; k<n_dims; k++)
                loc(k)=loc_upts(k,j);

            opp_r(j,i)=eval_nodal_basis_restart(i,loc);
        }
    }
}

// calculate position of the plot points

void eles::calc_pos_ppts(int in_ele, hf_array<double>& out_pos_ppts)
{
    int i,j;

    hf_array<double> loc(n_dims);
    hf_array<double> pos(n_dims);

    for(i=0; i<n_ppts_per_ele; i++)
    {
            for(j=0; j<n_dims; j++)
            {
                loc(j)=loc_ppts(j,i);
            }
            calc_pos(loc,in_ele,pos);

        for(j=0; j<n_dims; j++)
        {
            out_pos_ppts(i,j)=pos(j);
        }
    }
}
// calculate solution at the probe points
void eles::calc_disu_probepoints(int in_ele, hf_array<double>& out_disu_probepoints)
{
    if (n_eles!=0)
    {
        hf_array<double> disu_upts_probe(n_upts_per_ele,n_fields);

        for(int i=0; i<n_fields; i++)
            for(int j=0; j<n_upts_per_ele; j++)
                    disu_upts_probe(j,i)=disu_upts(0)(j,in_ele,i);

#if defined _ACCELERATE_BLAS || defined _MKL_BLAS || defined _STANDARD_BLAS

        cblas_dgemm(CblasColMajor,CblasNoTrans,CblasNoTrans,1,n_fields,n_upts_per_ele,1.0,opp_probe.get_ptr_cpu(),1,disu_upts_probe.get_ptr_cpu(),n_upts_per_ele,0.0,out_disu_probepoints.get_ptr_cpu(),1);
#else
        dgemm(1,n_fields,n_upts_per_ele,1.0,0.0,opp_probe.get_ptr_cpu(),disu_upts_probe.get_ptr_cpu(),out_disu_probepoints.get_ptr_cpu());
#endif

    }
}

// calculate solution at the plot points
void eles::calc_disu_ppts(int in_ele, hf_array<double>& out_disu_ppts)
{
    if (n_eles!=0)
    {

        int i,j,k;

        hf_array<double> disu_upts_plot(n_upts_per_ele,n_fields);

        for(i=0; i<n_fields; i++)
            for(j=0; j<n_upts_per_ele; j++)
                    disu_upts_plot(j,i)=disu_upts(0)(j,in_ele,i);

#if defined _ACCELERATE_BLAS || defined _MKL_BLAS || defined _STANDARD_BLAS

        cblas_dgemm(CblasColMajor,CblasNoTrans,CblasNoTrans,n_ppts_per_ele,n_fields,n_upts_per_ele,1.0,opp_p.get_ptr_cpu(),n_ppts_per_ele,disu_upts_plot.get_ptr_cpu(),n_upts_per_ele,0.0,out_disu_ppts.get_ptr_cpu(),n_ppts_per_ele);
#else
        dgemm(n_ppts_per_ele,n_fields,n_upts_per_ele,1.0,0.0,opp_p.get_ptr_cpu(),disu_upts_plot.get_ptr_cpu(),out_disu_ppts.get_ptr_cpu());
#endif

    }
}

// calculate gradient of solution at the plot points
void eles::calc_grad_disu_ppts(int in_ele, hf_array<double>& out_grad_disu_ppts)
{
    if (n_eles!=0)
    {

        int i,j,k,l;

        hf_array<double> grad_disu_upts_temp(n_upts_per_ele,n_fields,n_dims);

        for(i=0; i<n_fields; i++)
        {
            for(j=0; j<n_upts_per_ele; j++)
            {
                for(k=0; k<n_dims; k++)
                {
                    grad_disu_upts_temp(j,i,k)=grad_disu_upts(j,in_ele,i,k);
                }
            }
        }

#if defined _ACCELERATE_BLAS || defined _MKL_BLAS || defined _STANDARD_BLAS

        for (i=0; i<n_dims; i++)
        {
            cblas_dgemm(CblasColMajor,CblasNoTrans,CblasNoTrans,n_ppts_per_ele,n_fields,n_upts_per_ele,1.0,opp_p.get_ptr_cpu(),n_ppts_per_ele,grad_disu_upts_temp.get_ptr_cpu(0,0,i),n_upts_per_ele,0.0,out_grad_disu_ppts.get_ptr_cpu(0,0,i),n_ppts_per_ele);
        }

#else

        for (i=0; i<n_dims; i++)
        {
            dgemm(n_ppts_per_ele,n_fields,n_upts_per_ele,1.0,0.0,opp_p.get_ptr_cpu(),grad_disu_upts_temp.get_ptr_cpu(0,0,i),out_grad_disu_ppts.get_ptr_cpu(0,0,i));
        }
#endif

    }
}

// calculate the time averaged field values at plot points
void eles::calc_time_average_ppts(int in_ele, hf_array<double>& out_disu_average_ppts)
{
    if (n_eles!=0)
    {

        int i,j,k;

        hf_array<double> disu_average_upts_plot(n_upts_per_ele,n_average_fields);

        for(i=0; i<n_average_fields; i++)
        {
            for(j=0; j<n_upts_per_ele; j++)
            {
                disu_average_upts_plot(j,i)=disu_average_upts(j,in_ele,i);
            }
        }

#if defined _ACCELERATE_BLAS || defined _MKL_BLAS || defined _STANDARD_BLAS

        cblas_dgemm(CblasColMajor,CblasNoTrans,CblasNoTrans,n_ppts_per_ele,n_average_fields,n_upts_per_ele,1.0,opp_p.get_ptr_cpu(),n_ppts_per_ele,disu_average_upts_plot.get_ptr_cpu(),n_upts_per_ele,0.0,out_disu_average_ppts.get_ptr_cpu(),n_ppts_per_ele);
#else
        dgemm(n_ppts_per_ele,n_average_fields,n_upts_per_ele,1.0,0.0,opp_p.get_ptr_cpu(),disu_average_upts_plot.get_ptr_cpu(),out_disu_average_ppts.get_ptr_cpu());
#endif

    }
}

// calculate the sensor values at plot points
void eles::calc_sensor_ppts(int in_ele, hf_array<double>& out_sensor_ppts)
{
    if (n_eles!=0)
    {
        for(int i=0; i<n_ppts_per_ele; i++)
            out_sensor_ppts(i) = sensor(in_ele);
    }
}

// calculate diagnostic fields at the plot points
void eles::calc_diagnostic_fields_ppts(int in_ele, hf_array<double>& in_disu_ppts, hf_array<double>& in_grad_disu_ppts, hf_array<double>& in_sensor_ppts, hf_array<double>& out_diag_field_ppts, double& time)
{
    int j,k,m;
    double diagfield_upt;
    double u,v,w;
    double irho,pressure,v_sq;
    double wx,wy,wz;
    double dudx, dudy, dudz;
    double dvdx, dvdy, dvdz;
    double dwdx, dwdy, dwdz;

    for(j=0; j<n_ppts_per_ele; j++)
    {
        // Compute velocity square
        v_sq = 0.;
        for (m=0; m<n_dims; m++)
            v_sq += (in_disu_ppts(j,m+1)*in_disu_ppts(j,m+1));
        v_sq /= in_disu_ppts(j,0)*in_disu_ppts(j,0);

        // Compute pressure
        pressure = (run_input.gamma-1.0)*( in_disu_ppts(j,n_dims+1) - 0.5*in_disu_ppts(j,0)*v_sq);

        // compute diagnostic fields
        for (k=0; k<n_diagnostic_fields; k++)
        {
            irho = 1./in_disu_ppts(j,0);

            if (run_input.diagnostic_fields(k)=="u")
                diagfield_upt = in_disu_ppts(j,1)*irho;
            else if (run_input.diagnostic_fields(k)=="v")
                diagfield_upt = in_disu_ppts(j,2)*irho;
            else if (run_input.diagnostic_fields(k)=="w")
            {
                if (n_dims==2)
                    diagfield_upt = 0.;
                else if (n_dims==3)
                    diagfield_upt = in_disu_ppts(j,3)*irho;
            }
            else if (run_input.diagnostic_fields(k)=="energy")
            {
                if (n_dims==2)
                    diagfield_upt = in_disu_ppts(j,3);
                else if (n_dims==3)
                    diagfield_upt = in_disu_ppts(j,4);
            }
            // flow properties
            else if (run_input.diagnostic_fields(k)=="mach")
            {
                diagfield_upt = sqrt( v_sq / (run_input.gamma*pressure/in_disu_ppts(j,0)) );
            }
            else if (run_input.diagnostic_fields(k)=="pressure")
            {
                diagfield_upt = pressure;
            }
            // turbulence metrics
            else if (run_input.diagnostic_fields(k)=="vorticity" || run_input.diagnostic_fields(k)=="q_criterion"||run_input.diagnostic_fields(k)=="scaled_q_criterion")
            {
                if (!viscous)
                    FatalError("Trying to calculate diagnostic field only supported by viscous simualtion");
                u = in_disu_ppts(j,1)*irho;
                v = in_disu_ppts(j,2)*irho;

                dudx = irho*(in_grad_disu_ppts(j,1,0) - u*in_grad_disu_ppts(j,0,0));
                dudy = irho*(in_grad_disu_ppts(j,1,1) - u*in_grad_disu_ppts(j,0,1));
                dvdx = irho*(in_grad_disu_ppts(j,2,0) - v*in_grad_disu_ppts(j,0,0));
                dvdy = irho*(in_grad_disu_ppts(j,2,1) - v*in_grad_disu_ppts(j,0,1));

                if (n_dims==2)
                {
                    if (run_input.diagnostic_fields(k) == "vorticity")
                    {
                        diagfield_upt = abs(dvdx-dudy);
                    }
                    else
                    {
                        FatalError("Q criterion Not implemented in 2D");
                    }
                }
                else if (n_dims==3)
                {
                    w = in_disu_ppts(j,3)*irho;

                    dudz = irho*(in_grad_disu_ppts(j,1,2) - u*in_grad_disu_ppts(j,0,2));
                    dvdz = irho*(in_grad_disu_ppts(j,2,2) - v*in_grad_disu_ppts(j,0,2));

                    dwdx = irho*(in_grad_disu_ppts(j,3,0) - w*in_grad_disu_ppts(j,0,0));
                    dwdy = irho*(in_grad_disu_ppts(j,3,1) - w*in_grad_disu_ppts(j,0,1));
                    dwdz = irho*(in_grad_disu_ppts(j,3,2) - w*in_grad_disu_ppts(j,0,2));

                    wx = dwdy - dvdz;
                    wy = dudz - dwdx;
                    wz = dvdx - dudy;

                    if (run_input.diagnostic_fields(k) == "vorticity")
                    {
                        diagfield_upt = sqrt(wx*wx+wy*wy+wz*wz);
                    }
                    else
                    {

                        wx *= 0.5;
                        wy *= 0.5;
                        wz *= 0.5;

                        double Sxx,Syy,Szz,Sxy,Sxz,Syz,SS,OO;
                        Sxx = dudx;
                        Syy = dvdy;
                        Szz = dwdz;
                        Sxy = 0.5*(dudy+dvdx);
                        Sxz = 0.5*(dudz+dwdx);
                        Syz = 0.5*(dvdz+dwdy);

                        SS = Sxx*Sxx + Syy*Syy + Szz*Szz + 2*Sxy*Sxy + 2*Sxz*Sxz + 2*Syz*Syz;
                        OO = 2*wx*wx + 2*wy*wy + 2*wz*wz;
                        if (run_input.diagnostic_fields(k) == "q_criterion")
                            diagfield_upt = 0.5 * (OO - SS);
                        else
                        {
                            double eps = 1.e-24;
                            diagfield_upt = 0.5 * (OO - SS) / (SS + eps); //normalized
                        }
                    }
                }
            }
            // Artificial Viscosity diagnostics
            else if (run_input.diagnostic_fields(k)=="sensor")
            {
                if (run_input.shock_cap)
                    diagfield_upt = in_sensor_ppts(j);
                else
                    FatalError("Sensor unavailable");
            }

            else
            {
                cout << "plot_quantity = " << run_input.diagnostic_fields(k) << ": " << flush;
                FatalError("plot_quantity not recognized");
            }
            if (std::isnan(diagfield_upt))
            {
                cout << "In calculation of plot_quantitiy " << run_input.diagnostic_fields(k) << ": " << flush;
                FatalError("NaN");
            }

            // set hf_array with solution point value
            out_diag_field_ppts(j,k) = diagfield_upt;
        }
    }
}

double eles::get_loc_upt(int in_upt, int in_dim)
{
    return loc_upts(in_dim,in_upt);
}

// set transforms

void eles::set_transforms(void)
{
    if (n_eles != 0)
    {
        set_transforms_upts();
        set_transforms_fpts();

        // Set metrics at interface cubpts
        if (run_input.calc_force != 0 || run_input.forcing != 0) //only calculated when need to calculate surface/body force
            set_transforms_inters_cubpts(); //calculate static mesh element interface cubature transform

        // Set metrics at volume cubpts. Only needed for computing error and integral diagnostic quantities.
        if (run_input.test_case != 0 || run_input.n_integral_quantities != 0)
            set_transforms_vol_cubpts(); //calculate static mesh element volume cubature transform
        
        if(run_input.over_int)
            set_transforms_over_int_cubtps();
    } // if n_eles!=0
}

void eles::set_transforms_upts(void)
{
        int i,j,k;

        hf_array<double> loc(n_dims);
        hf_array<double> pos(n_dims);
        hf_array<double> d_pos(n_dims,n_dims);

        double xr, xs, xt;
        double yr, ys, yt;
        double zr, zs, zt;

        // Determinant of Jacobian (transformation matrix) (J = |G|)
        detjac_upts.setup(n_upts_per_ele,n_eles);
        // Determinant of Jacobian times inverse of Jacobian (Full vector transform from physcial->reference frame)
        JGinv_upts.setup(n_dims,n_dims,n_upts_per_ele,n_eles);
        // Static-Physical position of solution points
        pos_upts.setup(n_upts_per_ele,n_eles,n_dims);

        if (rank==0)
        {
            cout << " at solution points" << endl;
        }

        for(i=0; i<n_eles; i++)
        {
            if ((i%(max(n_eles,10)/10))==0 && rank==0)
                cout << fixed << setprecision(2) <<  (i*1.0/n_eles)*100 << "% " << flush;

            for(j=0; j<n_upts_per_ele; j++)
            {
                // get coordinates of the solution point

                for(k=0;k<n_dims;k++)
                {
                    loc(k)=loc_upts(k,j);
                }
                calc_pos(loc,i,pos);


                for(k=0; k<n_dims; k++)
                {
                    pos_upts(j,i,k)=pos(k);
                }

                // calculate first derivatives of shape functions at the solution point
                calc_d_pos(loc,i,d_pos);


                // store quantities at the solution point

                if(n_dims==2)
                {
                    xr = d_pos(0,0);
                    xs = d_pos(0,1);

                    yr = d_pos(1,0);
                    ys = d_pos(1,1);

                    // store determinant of jacobian at solution point
                    detjac_upts(j,i)= xr*ys - xs*yr;

                    if (detjac_upts(j,i) < 0)
                    {
                        FatalError("Negative Jacobian at solution points");
                    }

                    // store determinant of jacobian multiplied by inverse of jacobian at the solution point
                    JGinv_upts(0,0,j,i)= ys;
                    JGinv_upts(0,1,j,i)= -xs;
                    JGinv_upts(1,0,j,i)= -yr;
                    JGinv_upts(1,1,j,i)= xr;
                }
                else if(n_dims==3)
                {
                    xr = d_pos(0,0);
                    xs = d_pos(0,1);
                    xt = d_pos(0,2);

                    yr = d_pos(1,0);
                    ys = d_pos(1,1);
                    yt = d_pos(1,2);

                    zr = d_pos(2,0);
                    zs = d_pos(2,1);
                    zt = d_pos(2,2);

                    // store determinant of jacobian at solution point

                    detjac_upts(j,i) = xr*(ys*zt - yt*zs) - xs*(yr*zt - yt*zr) + xt*(yr*zs - ys*zr);

                    // store determinant of jacobian multiplied by inverse of jacobian at the solution point
                    JGinv_upts(0,0,j,i) = ys*zt - yt*zs;
                    JGinv_upts(0,1,j,i) = xt*zs - xs*zt;
                    JGinv_upts(0,2,j,i) = xs*yt - xt*ys;
                    JGinv_upts(1,0,j,i) = yt*zr - yr*zt;
                    JGinv_upts(1,1,j,i) = xr*zt - xt*zr;
                    JGinv_upts(1,2,j,i) = xt*yr - xr*yt;
                    JGinv_upts(2,0,j,i) = yr*zs - ys*zr;
                    JGinv_upts(2,1,j,i) = xs*zr - xr*zs;
                    JGinv_upts(2,2,j,i) = xr*ys - xs*yr;
                }
                else
                {
                    cout << "ERROR: Invalid number of dimensions ... " << endl;
                }
            }
        }
        if (rank == 0)
            cout << endl;
#ifdef _GPU
        detjac_upts.cp_cpu_gpu(); // Copy since need in write_tec
        JGinv_upts.cp_cpu_gpu(); // Copy since needed for calc_d_pos_dyn
#endif
}

void eles::set_transforms_over_int_cubtps(void)
{
        int i,j,k;
        hf_array<double> loc(n_dims);
        hf_array<double> d_pos(n_dims,n_dims);
        double xr, xs, xt;
        double yr, ys, yt;
        double zr, zs, zt;
        JGinv_over_int_cubpts.setup(n_dims, n_dims, loc_over_int_cubpts.get_dim(1), n_eles);

        for(i=0; i<n_eles; i++)
        {
            for(j=0; j<loc_over_int_cubpts.get_dim(1); j++)
            {
                for(k=0;k<n_dims;k++)
                {
                    loc(k)=loc_over_int_cubpts(k,j);
                }
                calc_d_pos(loc,i,d_pos);
                if(n_dims==2)
                {
                    xr = d_pos(0,0);
                    xs = d_pos(0,1);
                    yr = d_pos(1,0);
                    ys = d_pos(1,1);

                    // store determinant of jacobian multiplied by inverse of jacobian at the solution point
                    JGinv_over_int_cubpts(0,0,j,i)= ys;
                    JGinv_over_int_cubpts(0,1,j,i)= -xs;
                    JGinv_over_int_cubpts(1,0,j,i)= -yr;
                    JGinv_over_int_cubpts(1,1,j,i)= xr;
                }
                else if(n_dims==3)
                {
                    xr = d_pos(0,0);
                    xs = d_pos(0,1);
                    xt = d_pos(0,2);

                    yr = d_pos(1,0);
                    ys = d_pos(1,1);
                    yt = d_pos(1,2);

                    zr = d_pos(2,0);
                    zs = d_pos(2,1);
                    zt = d_pos(2,2);

                    JGinv_over_int_cubpts(0,0,j,i) = ys*zt - yt*zs;
                    JGinv_over_int_cubpts(0,1,j,i) = xt*zs - xs*zt;
                    JGinv_over_int_cubpts(0,2,j,i) = xs*yt - xt*ys;
                    JGinv_over_int_cubpts(1,0,j,i) = yt*zr - yr*zt;
                    JGinv_over_int_cubpts(1,1,j,i) = xr*zt - xt*zr;
                    JGinv_over_int_cubpts(1,2,j,i) = xt*yr - xr*yt;
                    JGinv_over_int_cubpts(2,0,j,i) = yr*zs - ys*zr;
                    JGinv_over_int_cubpts(2,1,j,i) = xs*zr - xr*zs;
                    JGinv_over_int_cubpts(2,2,j,i) = xr*ys - xs*yr;
                }
                else
                {
                    cout << "ERROR: Invalid number of dimensions ... " << endl;
                }
            }
        }
}

void eles::set_transforms_fpts(void)
{
    int i, j, k;

    hf_array<double> loc(n_dims);
    hf_array<double> pos(n_dims);
    hf_array<double> d_pos(n_dims, n_dims);
    hf_array<double> tnorm_dot_inv_detjac_mul_jac(n_dims);

    double xr, xs, xt;
    double yr, ys, yt;
    double zr, zs, zt;

        // Compute metrics term at flux points
        /// Determinant of Jacobian (transformation matrix)
        detjac_fpts.setup(n_fpts_per_ele,n_eles);
        Jacobian_fpts.setup(n_dims, n_dims, n_fpts_per_ele, n_eles);
        /// Determinant of Jacobian times inverse of Jacobian (Full vector transform from physcial->reference frame)
        JGinv_fpts.setup(n_dims,n_dims,n_fpts_per_ele,n_eles);
        tdA_fpts.setup(n_fpts_per_ele,n_eles);
        norm_fpts.setup(n_fpts_per_ele,n_eles,n_dims);
        // Static-Physical position of solution points
        pos_fpts.setup(n_fpts_per_ele,n_eles,n_dims);

        if (rank==0)
            cout << " at flux points"  << endl;

        for(i=0; i<n_eles; i++)
        {
            if ((i%(max(n_eles,10)/10))==0 && rank==0)
                cout << fixed << setprecision(2) <<  (i*1.0/n_eles)*100 << "% " << flush;

            for(j=0; j<n_fpts_per_ele; j++)
            {
                // get coordinates of the flux point

                for(k=0;k<n_dims;k++)
                {
                    loc(k)=tloc_fpts(k,j);
                }
                calc_pos(loc,i,pos);


                for(k=0; k<n_dims; k++)
                {
                    pos_fpts(j,i,k)=pos(k);
                }

                // calculate first derivatives of shape functions at the flux points
                    calc_d_pos(loc,i,d_pos);

                    
                // store quantities at the flux point

                if(n_dims==2)
                {
                    xr = d_pos(0,0);
                    xs = d_pos(0,1);

                    yr = d_pos(1,0);
                    ys = d_pos(1,1);

                    Jacobian_fpts(0, 0, j, i) = xr;
                    Jacobian_fpts(0, 1, j, i) = xs;
                    Jacobian_fpts(1, 0, j, i) = yr;
                    Jacobian_fpts(1, 1, j, i) = ys;

                    // store determinant of jacobian at flux point

                    detjac_fpts(j,i)= xr*ys - xs*yr;

                    if (detjac_fpts(j,i) < 0)
                    {
                        FatalError("Negative Jacobian at flux points");
                    }

                    // store determinant of jacobian multiplied by inverse of jacobian at the flux point

                    JGinv_fpts(0,0,j,i)= ys;
                    JGinv_fpts(0,1,j,i)= -xs;
                    JGinv_fpts(1,0,j,i)= -yr;
                    JGinv_fpts(1,1,j,i)= xr;

                    // temporarily store transformed normal dot inverse of determinant of jacobian multiplied by jacobian at the flux point

                    tnorm_dot_inv_detjac_mul_jac(0)=(tnorm_fpts(0,j)*d_pos(1,1))-(tnorm_fpts(1,j)*d_pos(1,0));
                    tnorm_dot_inv_detjac_mul_jac(1)=-(tnorm_fpts(0,j)*d_pos(0,1))+(tnorm_fpts(1,j)*d_pos(0,0));

                    // store magnitude of transformed normal dot inverse of determinant of jacobian multiplied by jacobian at the flux point

                    tdA_fpts(j,i)=sqrt(tnorm_dot_inv_detjac_mul_jac(0)*tnorm_dot_inv_detjac_mul_jac(0)+
                                       tnorm_dot_inv_detjac_mul_jac(1)*tnorm_dot_inv_detjac_mul_jac(1));


                    // store normal at flux point

                    norm_fpts(j,i,0)=tnorm_dot_inv_detjac_mul_jac(0)/tdA_fpts(j,i);
                    norm_fpts(j,i,1)=tnorm_dot_inv_detjac_mul_jac(1)/tdA_fpts(j,i);
                }
                else if(n_dims==3)
                {
                    xr = d_pos(0,0);
                    xs = d_pos(0,1);
                    xt = d_pos(0,2);

                    yr = d_pos(1,0);
                    ys = d_pos(1,1);
                    yt = d_pos(1,2);

                    zr = d_pos(2,0);
                    zs = d_pos(2,1);
                    zt = d_pos(2,2);

                    Jacobian_fpts(0, 0, j, i) = xr;
                    Jacobian_fpts(0, 1, j, i) = xs;
                    Jacobian_fpts(0, 2, j, i) = xt;
                    Jacobian_fpts(1, 0, j, i) = yr;
                    Jacobian_fpts(1, 1, j, i) = ys;
                    Jacobian_fpts(1, 2, j, i) = yt;
                    Jacobian_fpts(2, 0, j, i) = zr;
                    Jacobian_fpts(2, 1, j, i) = zs;
                    Jacobian_fpts(2, 2, j, i) = zt;
                    // store determinant of jacobian at flux point

                    detjac_fpts(j,i) = xr*(ys*zt - yt*zs) - xs*(yr*zt - yt*zr) + xt*(yr*zs - ys*zr);

                    // store inverse of determinant of jacobian multiplied by jacobian at the flux point

                    JGinv_fpts(0,0,j,i) = ys*zt - yt*zs;
                    JGinv_fpts(0,1,j,i) = xt*zs - xs*zt;
                    JGinv_fpts(0,2,j,i) = xs*yt - xt*ys;
                    JGinv_fpts(1,0,j,i) = yt*zr - yr*zt;
                    JGinv_fpts(1,1,j,i) = xr*zt - xt*zr;
                    JGinv_fpts(1,2,j,i) = xt*yr - xr*yt;
                    JGinv_fpts(2,0,j,i) = yr*zs - ys*zr;
                    JGinv_fpts(2,1,j,i) = xs*zr - xr*zs;
                    JGinv_fpts(2,2,j,i) = xr*ys - xs*yr;

                    // temporarily store transformed normal dot inverse of determinant of jacobian multiplied by jacobian at the flux point

                    tnorm_dot_inv_detjac_mul_jac(0)=((tnorm_fpts(0,j)*(d_pos(1,1)*d_pos(2,2)-d_pos(1,2)*d_pos(2,1)))+(tnorm_fpts(1,j)*(d_pos(1,2)*d_pos(2,0)-d_pos(1,0)*d_pos(2,2)))+(tnorm_fpts(2,j)*(d_pos(1,0)*d_pos(2,1)-d_pos(1,1)*d_pos(2,0))));
                    tnorm_dot_inv_detjac_mul_jac(1)=((tnorm_fpts(0,j)*(d_pos(0,2)*d_pos(2,1)-d_pos(0,1)*d_pos(2,2)))+(tnorm_fpts(1,j)*(d_pos(0,0)*d_pos(2,2)-d_pos(0,2)*d_pos(2,0)))+(tnorm_fpts(2,j)*(d_pos(0,1)*d_pos(2,0)-d_pos(0,0)*d_pos(2,1))));
                    tnorm_dot_inv_detjac_mul_jac(2)=((tnorm_fpts(0,j)*(d_pos(0,1)*d_pos(1,2)-d_pos(0,2)*d_pos(1,1)))+(tnorm_fpts(1,j)*(d_pos(0,2)*d_pos(1,0)-d_pos(0,0)*d_pos(1,2)))+(tnorm_fpts(2,j)*(d_pos(0,0)*d_pos(1,1)-d_pos(0,1)*d_pos(1,0))));

                    // store magnitude of transformed normal dot inverse of determinant of jacobian multiplied by jacobian at the flux point

                    tdA_fpts(j,i)=sqrt(tnorm_dot_inv_detjac_mul_jac(0)*tnorm_dot_inv_detjac_mul_jac(0)+
                                       tnorm_dot_inv_detjac_mul_jac(1)*tnorm_dot_inv_detjac_mul_jac(1)+
                                       tnorm_dot_inv_detjac_mul_jac(2)*tnorm_dot_inv_detjac_mul_jac(2));

                    // store normal at flux point

                    norm_fpts(j,i,0)=tnorm_dot_inv_detjac_mul_jac(0)/tdA_fpts(j,i);
                    norm_fpts(j,i,1)=tnorm_dot_inv_detjac_mul_jac(1)/tdA_fpts(j,i);
                    norm_fpts(j,i,2)=tnorm_dot_inv_detjac_mul_jac(2)/tdA_fpts(j,i);
                }
                else
                {
                    cout << "ERROR: Invalid number of dimensions ... " << endl;
                }
            }
        }


        if (rank == 0)
            cout << endl;
#ifdef _GPU
        tdA_fpts.mv_cpu_gpu();
        pos_fpts.cp_cpu_gpu();

        JGinv_fpts.cp_cpu_gpu();
        detjac_fpts.cp_cpu_gpu();


            norm_fpts.mv_cpu_gpu();
            // move the dummy dynamic-transform pointers to GPUs
            cp_transforms_cpu_gpu();
#endif
}


void eles::set_bdy_ele2ele(void)
{

    n_bdy_eles=0;
    // Count the number of bdy_eles
    for (int i=0; i<n_eles; i++)
    {
        for (int j=0; j<n_inters_per_ele; j++)
        {
            if (bcid(i,j) != -1)
            {
                n_bdy_eles++;
                break;
            }
        }
    }

    if (n_bdy_eles!=0)
    {

        bdy_ele2ele.setup(n_bdy_eles);

        n_bdy_eles=0;
        for (int i=0; i<n_eles; i++)
        {
            for (int j=0; j<n_inters_per_ele; j++)
            {
                if (bcid(i,j) != -1)
                {
                    bdy_ele2ele(n_bdy_eles++) = i;
                    break;
                }
            }
        }

    }

}

void eles::set_ele2bdy_ele(void)
{

    n_bdy_eles=0;
    // Count the number of bdy_eles
    for (int i=0; i<n_eles; i++)
    {
        for (int j=0; j<n_inters_per_ele; j++)
        {
            if (bcid(i,j) != -1)
            {
                n_bdy_eles++;
                break;
            }
        }
    }

    if (n_bdy_eles!=0)
    {

        ele2bdy_ele.setup(n_eles);
        ele2bdy_ele.initialize_to_value(-1);

        n_bdy_eles=0;
        for (int i=0; i<n_eles; i++)
        {
            for (int j=0; j<n_inters_per_ele; j++)
            {
                if (bcid(i,j) != -1)
                {
                    
                    ele2bdy_ele(i)=n_bdy_eles++;
                    break;
                }
            }
        }

    }

}



// set transforms

void eles::set_transforms_inters_cubpts(void)
{

        int i,j,k;

        double xr, xs, xt;
        double yr, ys, yt;
        double zr, zs, zt;

        // Initialize bdy_ele2ele hf_array
        (*this).set_bdy_ele2ele();
        (*this).set_ele2bdy_ele();

        double mag_tnorm;

        hf_array<double> loc(n_dims);
        hf_array<double> d_pos(n_dims,n_dims);
        hf_array<double> tnorm_dot_inv_detjac_mul_jac(n_dims);

        inter_detjac_inters_cubpts.setup(n_inters_per_ele);
        norm_inters_cubpts.setup(n_inters_per_ele);
        vol_detjac_inters_cubpts.setup(n_inters_per_ele);

        for (i=0; i<n_inters_per_ele; i++)
        {
            inter_detjac_inters_cubpts(i).setup(n_cubpts_per_inter(i),n_bdy_eles);
            norm_inters_cubpts(i).setup(n_cubpts_per_inter(i),n_bdy_eles,n_dims);
            vol_detjac_inters_cubpts(i).setup(n_cubpts_per_inter(i),n_bdy_eles);
        }

        for(i=0; i<n_bdy_eles; i++)
        {
            for (int l=0; l<n_inters_per_ele; l++)
            {
                for(j=0; j<n_cubpts_per_inter(l); j++)
                {
                    // get coordinates of the cubature point

                    for(k=0; k<n_dims; k++)
                    {
                        loc(k)=loc_inters_cubpts(l)(k,j);
                    }

                    // calculate first derivatives of shape functions at the cubature points

                    calc_d_pos(loc,bdy_ele2ele(i),d_pos);

                    // store quantities at the cubature point

                    if(n_dims==2)
                    {

                        xr = d_pos(0,0);
                        xs = d_pos(0,1);

                        yr = d_pos(1,0);
                        ys = d_pos(1,1);

                        // store determinant of jacobian at cubature point. 
                        vol_detjac_inters_cubpts(l)(j,i)= xr*ys - xs*yr;

                        // temporarily store transformed normal dot inverse of determinant of jacobian multiplied by jacobian at the cubature point
                        tnorm_dot_inv_detjac_mul_jac(0)=(tnorm_inters_cubpts(l)(0,j)*d_pos(1,1))-(tnorm_inters_cubpts(l)(1,j)*d_pos(1,0));
                        tnorm_dot_inv_detjac_mul_jac(1)=-(tnorm_inters_cubpts(l)(0,j)*d_pos(0,1))+(tnorm_inters_cubpts(l)(1,j)*d_pos(0,0));

                        // calculate interface area
                        mag_tnorm = sqrt(tnorm_dot_inv_detjac_mul_jac(0)*tnorm_dot_inv_detjac_mul_jac(0)+
                                         tnorm_dot_inv_detjac_mul_jac(1)*tnorm_dot_inv_detjac_mul_jac(1));

                        // store normal at cubature point
                        norm_inters_cubpts(l)(j,i,0)=tnorm_dot_inv_detjac_mul_jac(0)/mag_tnorm;
                        norm_inters_cubpts(l)(j,i,1)=tnorm_dot_inv_detjac_mul_jac(1)/mag_tnorm;

                        inter_detjac_inters_cubpts(l)(j,i) = compute_inter_detjac_inters_cubpts(l,d_pos);
                    }
                    else if(n_dims==3)
                    {

                        xr = d_pos(0,0);
                        xs = d_pos(0,1);
                        xt = d_pos(0,2);

                        yr = d_pos(1,0);
                        ys = d_pos(1,1);
                        yt = d_pos(1,2);

                        zr = d_pos(2,0);
                        zs = d_pos(2,1);
                        zt = d_pos(2,2);

                        // store determinant of jacobian at cubature point
                        vol_detjac_inters_cubpts(l)(j,i) = xr*(ys*zt - yt*zs) - xs*(yr*zt - yt*zr) + xt*(yr*zs - ys*zr);

                        // temporarily store transformed normal dot inverse of determinant of jacobian multiplied by jacobian at the cubature point
                        tnorm_dot_inv_detjac_mul_jac(0)=((tnorm_inters_cubpts(l)(0,j)*(d_pos(1,1)*d_pos(2,2)-d_pos(1,2)*d_pos(2,1)))+(tnorm_inters_cubpts(l)(1,j)*(d_pos(1,2)*d_pos(2,0)-d_pos(1,0)*d_pos(2,2)))+(tnorm_inters_cubpts(l)(2,j)*(d_pos(1,0)*d_pos(2,1)-d_pos(1,1)*d_pos(2,0))));
                        tnorm_dot_inv_detjac_mul_jac(1)=((tnorm_inters_cubpts(l)(0,j)*(d_pos(0,2)*d_pos(2,1)-d_pos(0,1)*d_pos(2,2)))+(tnorm_inters_cubpts(l)(1,j)*(d_pos(0,0)*d_pos(2,2)-d_pos(0,2)*d_pos(2,0)))+(tnorm_inters_cubpts(l)(2,j)*(d_pos(0,1)*d_pos(2,0)-d_pos(0,0)*d_pos(2,1))));
                        tnorm_dot_inv_detjac_mul_jac(2)=((tnorm_inters_cubpts(l)(0,j)*(d_pos(0,1)*d_pos(1,2)-d_pos(0,2)*d_pos(1,1)))+(tnorm_inters_cubpts(l)(1,j)*(d_pos(0,2)*d_pos(1,0)-d_pos(0,0)*d_pos(1,2)))+(tnorm_inters_cubpts(l)(2,j)*(d_pos(0,0)*d_pos(1,1)-d_pos(0,1)*d_pos(1,0))));

                        // calculate interface area
                        mag_tnorm=sqrt(tnorm_dot_inv_detjac_mul_jac(0)*tnorm_dot_inv_detjac_mul_jac(0)+
                                       tnorm_dot_inv_detjac_mul_jac(1)*tnorm_dot_inv_detjac_mul_jac(1)+
                                       tnorm_dot_inv_detjac_mul_jac(2)*tnorm_dot_inv_detjac_mul_jac(2));

                        // store normal at cubature point
                        norm_inters_cubpts(l)(j,i,0)=tnorm_dot_inv_detjac_mul_jac(0)/mag_tnorm;
                        norm_inters_cubpts(l)(j,i,1)=tnorm_dot_inv_detjac_mul_jac(1)/mag_tnorm;
                        norm_inters_cubpts(l)(j,i,2)=tnorm_dot_inv_detjac_mul_jac(2)/mag_tnorm;

                        inter_detjac_inters_cubpts(l)(j,i) = compute_inter_detjac_inters_cubpts(l,d_pos);
                    }
                    else
                    {
                        FatalError("ERROR: Invalid number of dimensions ... ");
                    }
                }
            }
        }
}

// Set transforms at volume cubature points
void eles::set_transforms_vol_cubpts(void)
{
        int i,j,m;
        hf_array<double> d_pos(n_dims,n_dims);
        hf_array<double> loc(n_dims);
        int n_cubpts_per_ele = loc_volume_cubpts.get_dim(1);

        vol_detjac_vol_cubpts.setup(n_cubpts_per_ele);

        for (i=0; i<n_cubpts_per_ele; i++)
            vol_detjac_vol_cubpts(i).setup(n_eles);

        for (i=0; i<n_eles; i++)
        {
            for (j=0; j<n_cubpts_per_ele; j++)
            {
                // Get jacobian determinant at cubpts
                for (m=0; m<n_dims; m++)
                    loc(m) = loc_volume_cubpts(m,j);

                calc_d_pos(loc,i,d_pos);

                if (n_dims==2)
                {
                    vol_detjac_vol_cubpts(j)(i) = d_pos(0,0)*d_pos(1,1) - d_pos(0,1)*d_pos(1,0);
                }
                else if (n_dims==3)
                {
                    vol_detjac_vol_cubpts(j)(i) = d_pos(0,0)*(d_pos(1,1)*d_pos(2,2) - d_pos(1,2)*d_pos(2,1))
                                                  - d_pos(0,1)*(d_pos(1,0)*d_pos(2,2) - d_pos(1,2)*d_pos(2,0))
                                                  + d_pos(0,2)*(d_pos(1,0)*d_pos(2,1) - d_pos(1,1)*d_pos(2,0));
                }
            }
        }
}

// get a pointer to the transformed discontinuous solution at a flux point

double* eles::get_disu_fpts_ptr(int in_inter_local_fpt, int in_ele_local_inter, int in_field, int in_ele)
{
    int i;

    int fpt;

    fpt=in_inter_local_fpt;

    for(i=0; i<in_ele_local_inter; i++)
    {
        fpt+=n_fpts_per_inter(i);
    }

#ifdef _GPU
    return disu_fpts.get_ptr_gpu(fpt,in_ele,in_field);
#else
    return disu_fpts.get_ptr_cpu(fpt,in_ele,in_field);
#endif
}

// get a pointer to the normal transformed continuous inviscid flux at a flux point

double* eles::get_norm_tconf_fpts_ptr(int in_inter_local_fpt, int in_ele_local_inter, int in_field, int in_ele)
{
    int i;

    int fpt;

    fpt=in_inter_local_fpt;

    for(i=0; i<in_ele_local_inter; i++)
    {
        fpt+=n_fpts_per_inter(i);
    }

#ifdef _GPU
    return norm_tconf_fpts.get_ptr_gpu(fpt,in_ele,in_field);
#else
    return norm_tconf_fpts.get_ptr_cpu(fpt,in_ele,in_field);
#endif

}

// get a pointer to the determinant of the jacobian at a flux point

double* eles::get_detjac_fpts_ptr(int in_inter_local_fpt, int in_ele_local_inter, int in_ele)
{
    int i;

    int fpt;

    fpt=in_inter_local_fpt;

    for(i=0; i<in_ele_local_inter; i++)
    {
        fpt+=n_fpts_per_inter(i);
    }

#ifdef _GPU
    return detjac_fpts.get_ptr_gpu(fpt,in_ele);
#else
    return detjac_fpts.get_ptr_cpu(fpt,in_ele);
#endif
}

// get a pointer to the magnitude of normal dot inverse of (determinant of jacobian multiplied by jacobian) at flux points

double* eles::get_tdA_fpts_ptr(int in_inter_local_fpt, int in_ele_local_inter, int in_ele)
{
    int i;

    int fpt;

    fpt=in_inter_local_fpt;

    for(i=0; i<in_ele_local_inter; i++)
    {
        fpt+=n_fpts_per_inter(i);
    }

#ifdef _GPU
    return tdA_fpts.get_ptr_gpu(fpt,in_ele);
#else
    return tdA_fpts.get_ptr_cpu(fpt,in_ele);
#endif
}

// get a pointer to weight at flux points

double* eles::get_weight_fpts_ptr(int in_inter_local_fpt, int in_ele_local_inter)
{
    int fpt;

    fpt=in_inter_local_fpt;


#ifdef _GPU
    return weight_inters_cubpts(in_ele_local_inter).get_ptr_gpu(fpt);
#else
    return weight_inters_cubpts(in_ele_local_inter).get_ptr_cpu(fpt);
#endif
}

// get a pointer to inter_detjac_inters_cubpts at flux points

double* eles::get_inter_detjac_inters_cubpts_ptr(int in_inter_local_fpt, int in_ele_local_inter, int in_ele)
{
    int fpt,bdy_ele;

    fpt=in_inter_local_fpt;

    bdy_ele=ele2bdy_ele(in_ele);


#ifdef _GPU
    return inter_detjac_inters_cubpts(in_ele_local_inter).get_ptr_gpu(fpt,bdy_ele);
#else
    return inter_detjac_inters_cubpts(in_ele_local_inter).get_ptr_cpu(fpt,bdy_ele);
#endif
}


// get a pointer to the normal at a flux point

double* eles::get_norm_fpts_ptr(int in_inter_local_fpt, int in_ele_local_inter, int in_dim, int in_ele)
{
    int i;

    int fpt;

    fpt=in_inter_local_fpt;

    for(i=0; i<in_ele_local_inter; i++)
    {
        fpt+=n_fpts_per_inter(i);
    }

#ifdef _GPU
    return norm_fpts.get_ptr_gpu(fpt,in_ele,in_dim);
#else
    return norm_fpts.get_ptr_cpu(fpt,in_ele,in_dim);
#endif
}

// get a CPU pointer to the coordinates at a flux point

double* eles::get_loc_fpts_ptr_cpu(int in_inter_local_fpt, int in_ele_local_inter, int in_dim, int in_ele)
{
    int i;

    int fpt;

    fpt=in_inter_local_fpt;

    for(i=0; i<in_ele_local_inter; i++)
    {
        fpt+=n_fpts_per_inter(i);
    }

    return pos_fpts.get_ptr_cpu(fpt,in_ele,in_dim);
}

double *eles::get_wm_disu_ptr(int in_ele, int in_upt, int in_field)
{
    return disu_upts(0).get_ptr_cpu(in_upt, in_ele, in_field);
}

// get a GPU pointer to the coordinates at a flux point
#ifdef _GPU
double* eles::get_loc_fpts_ptr_gpu(int in_inter_local_fpt, int in_ele_local_inter, int in_dim, int in_ele)
{
    int i;

    int fpt;

    fpt=in_inter_local_fpt;

    for(i=0; i<in_ele_local_inter; i++)
    {
        fpt+=n_fpts_per_inter(i);
    }

    return pos_fpts.get_ptr_gpu(fpt,in_ele,in_dim);
}
#endif
// get a pointer to delta of the transformed discontinuous solution at a flux point

double* eles::get_delta_disu_fpts_ptr(int in_inter_local_fpt, int in_ele_local_inter, int in_field, int in_ele)
{
    int i;

    int fpt;

    fpt=in_inter_local_fpt;

    //if (ele2global_ele(in_ele)==53)
    //{
    //  cout << "HERE" << endl;
    //  cout << "local_face=" << in_ele_local_inter << endl;
    //}

    for(i=0; i<in_ele_local_inter; i++)
    {
        fpt+=n_fpts_per_inter(i);
    }

#ifdef _GPU
    return delta_disu_fpts.get_ptr_gpu(fpt,in_ele,in_field);
#else
    return delta_disu_fpts.get_ptr_cpu(fpt,in_ele,in_field);
#endif
}

// get a pointer to gradient of discontinuous solution at a flux point

double* eles::get_grad_disu_fpts_ptr(int in_inter_local_fpt, int in_ele_local_inter, int in_dim, int in_field, int in_ele)
{
    int i;

    int fpt;

    fpt=in_inter_local_fpt;

    for(i=0; i<in_ele_local_inter; i++)
    {
        fpt+=n_fpts_per_inter(i);
    }

#ifdef _GPU
    return grad_disu_fpts.get_ptr_gpu(fpt,in_ele,in_field,in_dim);
#else
    return grad_disu_fpts.get_ptr_cpu(fpt,in_ele,in_field,in_dim);
#endif
}

double eles::calc_wm_upts_dist(int in_ele, int in_local_inter, int &out_upt)
{
    double dist_min, temp_dist, dist_max = 0.;
    //step 1. find out the distance for each solution point to the interface
    //step 2. find out the largest distance among the solution points
    for (int i = 0; i < n_upts_per_ele; i++)
    {
        for (int j = 0; j < n_fpts_per_inter(in_local_inter); j++)
        {
            temp_dist = 0.;
            for (int k = 0; k < n_dims; k++)
            {
                temp_dist += ((*get_loc_fpts_ptr_cpu(j, in_local_inter, k, in_ele)) - pos_upts(i, in_ele, k)) * (*get_norm_fpts_ptr(j, in_local_inter, k, in_ele));
            }
            if (j == 0)
                dist_min = temp_dist;
            else
            {
                if (temp_dist < dist_min)
                    dist_min = temp_dist;
            }
        }
        if (dist_min > dist_max)
        {
            dist_max = dist_min;
            out_upt = i;
        }
    }

    return dist_max;
}

// get a pointer to the normal transformed continuous viscous flux at a flux point
/*
 double* eles::get_norm_tconvisf_fpts_ptr(int in_inter_local_fpt, int in_ele_local_inter, int in_field, int in_ele)
 {
 int i;

 int fpt;

 fpt=in_inter_local_fpt;

 for(i=0;i<in_ele_local_inter;i++)
 {
 fpt+=n_fpts_per_inter(i);
 }

 #ifdef _GPU
 return norm_tconvisf_fpts.get_ptr_gpu(fpt,in_ele,in_field);
 #else
 return norm_tconvisf_fpts.get_ptr_cpu(fpt,in_ele,in_field);
 #endif
 }
 */



// get a pointer to the subgrid-scale flux at a flux point
double* eles::get_sgsf_fpts_ptr(int in_inter_local_fpt, int in_ele_local_inter, int in_field, int in_dim, int in_ele)
{
    int i;

    int fpt;

    fpt=in_inter_local_fpt;

    for(i=0; i<in_ele_local_inter; i++)
    {
        fpt+=n_fpts_per_inter(i);
    }

#ifdef _GPU
    return sgsf_fpts.get_ptr_gpu(fpt,in_ele,in_field,in_dim);
#else
    return sgsf_fpts.get_ptr_cpu(fpt,in_ele,in_field,in_dim);
#endif
}

//#### helper methods ####

// calculate position

void eles::calc_pos(hf_array<double> in_loc, int in_ele, hf_array<double>& out_pos)
{
    int i,j;

    for(i=0; i<n_dims; i++)
    {
        out_pos(i)=0.0;

        for(j=0; j<n_spts_per_ele(in_ele); j++)
        {
            out_pos(i)+=eval_nodal_s_basis(j,in_loc,n_spts_per_ele(in_ele))*shape(i,j,in_ele);
        }
    }

}

// calculate derivative of position - NEEDS TO BE OPTIMIZED
/** Calculate derivative of position wrt computational space (dx/dr, dx/ds, etc.) */
void eles::calc_d_pos(hf_array<double> in_loc, int in_ele, hf_array<double>& out_d_pos)
{
    int i,j,k;

    eval_d_nodal_s_basis(d_nodal_s_basis,in_loc,n_spts_per_ele(in_ele));

    for(j=0; j<n_dims; j++)
    {
        for(k=0; k<n_dims; k++)
        {
            out_d_pos(j,k)=0.0;
            for(i=0; i<n_spts_per_ele(in_ele); i++)
            {
                out_d_pos(j,k)+=d_nodal_s_basis(i,k)*shape(j,i,in_ele);
            }
        }
    }
}

/**
 * Calculate derivative of static position wrt computational-space position at upt
 * Uses pre-computed nodal shape basis derivatives for efficiency
 * \param[in] in_upt - ID of solution point within element to evaluate at
 * \param[in] in_ele - local element ID
 * \param[out] out_d_pos - hf_array of size (n_dims,n_dims); (i,j) = dx_i / dxi_j
 */
void eles::calc_d_pos_upt(int in_upt, int in_ele, hf_array<double>& out_d_pos)
{
    int i,j,k;

    // Calculate dx/d<c>
    out_d_pos.initialize_to_zero();
    for(j=0; j<n_dims; j++)
    {
        for(k=0; k<n_dims; k++)
        {
            for(i=0; i<n_spts_per_ele(in_ele); i++)
            {
                out_d_pos(j,k)+=d_nodal_s_basis_upts(k,i,in_upt,in_ele)*shape(j,i,in_ele);
                //out_d_pos(j,k)+=d_nodal_s_basis_upts(in_upt,in_ele,k,i)*shape(j,i,in_ele);
            }
        }
    }
}

/**
 * Calculate derivative of static position wrt computational-space position at fpt
 * Uses pre-computed nodal shape basis derivatives for efficiency
 * \param[in] in_fpt - ID of flux point within element to evaluate at
 * \param[in] in_ele - local element ID
 * \param[out] out_d_pos - hf_array of size (n_dims,n_dims); (i,j) = dx_i / dxi_j
 */
void eles::calc_d_pos_fpt(int in_fpt, int in_ele, hf_array<double>& out_d_pos)
{
    int i,j,k;

    // Calculate dx/d<c>
    out_d_pos.initialize_to_zero();
    for(j=0; j<n_dims; j++)
    {
        for(k=0; k<n_dims; k++)
        {
            for(i=0; i<n_spts_per_ele(in_ele); i++)
            {
                out_d_pos(j,k)+=d_nodal_s_basis_fpts(k,i,in_fpt,in_ele)*shape(j,i,in_ele);
                //out_d_pos(j,k)+=d_nodal_s_basis_fpts(in_fpt,in_ele,k,i)*shape(j,i,in_ele);
            }
        }
    }
}

/*! Calculate residual sum for monitoring purposes */
double eles::compute_res_upts(int in_norm_type, int in_field)
{

    int i, j;
    double sum = 0.;

    // NOTE: div_tconf_upts must be on CPU

    for (i=0; i<n_eles; i++)
    {
        for (j=0; j<n_upts_per_ele; j++)
        {
            if (in_norm_type == 0)
            {
                sum = max(sum, abs(div_tconf_upts(0)(j, i, in_field)/detjac_upts(j, i)-src_upts(j,i,in_field)));
            }
            if (in_norm_type == 1)
            {
                sum += abs(div_tconf_upts(0)(j, i, in_field)/detjac_upts(j, i)-src_upts(j,i,in_field));
            }
            else if (in_norm_type == 2)
            {
                sum += (div_tconf_upts(0)(j, i, in_field)/detjac_upts(j,i)-src_upts(j,i,in_field))*(div_tconf_upts(0)(j, i, in_field)/detjac_upts(j, i)-src_upts(j,i,in_field));
            }
        }
    }

    return sum;

}

hf_array<double> eles::compute_error(int in_norm_type, double &time)
{
    hf_array<double> disu_cubpt(n_fields);
    hf_array<double> grad_disu_cubpt(n_fields, n_dims);
    double detjac;
    hf_array<double> pos(n_dims);
    hf_array<double> temp_loc(n_dims);
    hf_array<double> error(2, n_fields);     //storage
    hf_array<double> error_sum(2, n_fields); //output
    error_sum.initialize_to_zero();
    int n_cubpts_per_ele = loc_volume_cubpts.get_dim(1);

    for (int i = 0; i < n_eles; i++)
    {
        for (int j = 0; j < n_cubpts_per_ele; j++)
        {
            // Get jacobian determinant at cubpts
            detjac = vol_detjac_vol_cubpts(j)(i);

            for (int k = 0; k < n_dims; k++)
                temp_loc(k) = loc_volume_cubpts(k, j);
            calc_pos(temp_loc, i, pos);

            //Get the solution at cubature point
            for (int m = 0; m < n_fields; m++)
            {
                disu_cubpt(m) = 0.;
                for (int k = 0; k < n_upts_per_ele; k++)
                {
                    disu_cubpt(m) += opp_volume_cubpts(j, k) * disu_upts(0)(k, i, m);
                }
            }

            // Get the gradient at cubature point
            if (viscous == 1)
            {
                for (int m = 0; m < n_fields; m++)
                {
                    for (int n = 0; n < n_dims; n++)
                    {
                        grad_disu_cubpt(m, n) = 0;
                        for (int k = 0; k < n_upts_per_ele; k++)
                        {
                            grad_disu_cubpt(m, n) += opp_volume_cubpts(j, k) * grad_disu_upts(k, i, m, n);
                        }
                    }
                }
            }

            error = get_pointwise_error(disu_cubpt, grad_disu_cubpt, pos, time, in_norm_type);

            for (int m = 0; m < n_fields; m++)
            {
                error_sum(0, m) += error(0, m) * weight_volume_cubpts(j) * detjac;
                error_sum(1, m) += error(1, m) * weight_volume_cubpts(j) * detjac;
            }
        }
    }

    return error_sum;
}

hf_array<double> eles::get_pointwise_error(hf_array<double>& sol, hf_array<double>& grad_sol, hf_array<double>& loc, double& time, int in_norm_type)
{
    hf_array<double> error(2,n_fields);  //output

    hf_array<double> error_sol(n_fields);
    hf_array<double> error_grad_sol(n_fields,n_dims);

    error.initialize_to_zero();
    error_sol.initialize_to_zero();
    error_grad_sol.initialize_to_zero();

    if (run_input.test_case==1) // Isentropic vortex
    {
        // Computing error in all quantities
        double rho, vx, vy, vz, p;
        eval_isentropic_vortex(loc, time, rho, vx, vy, vz, p, n_dims);

        error_sol(0) = sol(0) - rho;
        error_sol(1) = sol(1) - rho * vx;
        error_sol(2) = sol(2) - rho * vy;
        if (n_dims == 2)
            error_sol(3) = sol(3) - (p / (run_input.gamma - 1) + 0.5 * rho * (vx * vx + vy * vy));
        else
        {
            error_sol(3) = sol(3) - rho * vz;
            error_sol(4) = sol(4) - (p / (run_input.gamma - 1) + 0.5 * rho * (vx * vx + vy * vy + vz * vz));
        }
    }
    else if (run_input.test_case==2) // Sine Wave (single)
    {
        double rho;
        hf_array<double> grad_rho(n_dims);

        if(viscous)
        {
            eval_sine_wave_single(loc,run_input.wave_speed,run_input.diff_coeff,time,rho,grad_rho,n_dims);
        }
        else
        {
            eval_sine_wave_single(loc,run_input.wave_speed,0.,time,rho,grad_rho,n_dims);
        }

        error_sol(0) = sol(0) - rho;

        for (int j=0; j<n_dims; j++)
        {
            error_grad_sol(0,j) = grad_sol(0,j) - grad_rho(j);
        }

    }
    else if (run_input.test_case==3) // Sine Wave (group)
    {
        double rho;
        hf_array<double> grad_rho(n_dims);

        if(viscous)
        {
            eval_sine_wave_group(loc,run_input.wave_speed,run_input.diff_coeff,time,rho,grad_rho,n_dims);
        }
        else
        {
            eval_sine_wave_group(loc,run_input.wave_speed,0.,time,rho,grad_rho,n_dims);
        }

        error_sol(0) = sol(0) - rho;

        for (int j=0; j<n_dims; j++)
        {
            error_grad_sol(0,j) = grad_sol(0,j) - grad_rho(j);
        }
    }
    else if (run_input.test_case==4) // Sphere Wave
    {
        double rho;
        eval_sphere_wave(loc,run_input.wave_speed,time,rho,n_dims);
        error_sol(0) = sol(0) - rho;
    }
    else if (run_input.test_case==5) // Couette flow
    {
        double u_wall,T_wall;

        for (int j = 0; j < run_input.bc_list.get_dim(0); j++)
        {
            if (run_input.bc_list(j).get_bc_flag() == ISOTHERM_WALL)
            {
                if (run_input.bc_list(j).velocity(0) != 0)//moving wall
                {
                    u_wall = run_input.bc_list(j).velocity(0);
                }
                else//fixed wall
                    T_wall = run_input.bc_list(j).T_static;
            }
        }

        hf_array<double> temp_sol_couette(n_fields),temp_grad_couette(n_fields,n_dims);
        eval_couette_flow(loc,run_input.gamma, run_input.R_ref, u_wall, T_wall, run_input.p_c_ic, run_input.prandtl, time, temp_sol_couette, temp_grad_couette, n_dims);

        for (int j = 0; j < n_fields; j++)
        {
            error_sol(j) = sol(j) - temp_sol_couette(j);

            for (int k = 0; k < n_dims; k++)
            {
                error_grad_sol(j, k) = grad_sol(j, k) - temp_grad_couette(j, k);
            }
        }
    }
    else
    {
        FatalError("Test case not recognized in compute error, exiting");
    }

    if (in_norm_type==1)
    {
        for (int m=0; m<n_fields; m++)
        {
            error(0,m) += abs(error_sol(m));

            for(int n=0; n<n_dims; n++)
            {
                error(1,m) += abs(error_grad_sol(m,n)); //might be incorrect
            }
        }
    }

    if (in_norm_type==2)
    {
        for (int m=0; m<n_fields; m++)
        {
            error(0,m) += error_sol(m)*error_sol(m);

            for(int n=0; n<n_dims; n++)
            {
                error(1,m) += error_grad_sol(m,n)*error_grad_sol(m,n);
            }
        }
    }

    return error;
}

// Calculate body forcing term for periodic channel flow. HARDCODED FOR THE CHANNEL AND PERIODIC HILL!

void eles::evaluate_body_force(int in_file_num)
{
//#ifdef _CPU

    if (n_eles!=0)
    {
        int i,j,k,l,m,ele;
        double area, vol, detjac, ubulk, wgt;
        double mdot0, mdot_old, alpha, dt;
        hf_array <int> inflowinters(n_bdy_eles,n_inters_per_ele);
        hf_array <double> body_force(n_fields);
        hf_array <double> disu_cubpt(4);
        hf_array <double> integral(4);
        hf_array <double> norm(n_dims), flow(n_dims), loc(n_dims), pos(n_dims);
        ofstream write_mdot;
        bool open_mdot;

        for (i=0; i<4; i++)
        {
            integral(i)=0.0;
        }

        // zero the interface flags
        for (i=0; i<n_bdy_eles; i++)
        {
            for (l=0; l<n_inters_per_ele; l++)
            {
                inflowinters(i,l)=0;
            }
        }

        // Mass flux on inflow boundary
        // Integrate density and x-velocity over inflow area
        for (i=0; i<n_bdy_eles; i++)
        {
            ele = bdy_ele2ele(i);
            for (l=0; l<n_inters_per_ele; l++)
            {
                if(inflowinters(i,l)!=1) // only unflagged inters
                {
                    // HACK: Inlet is always a Cyclic (9) BC
                    if(run_input.bc_list(bcid(ele,l)).get_bc_flag()==CYCLIC)
                    {
                        // Get the normal
                        for (m=0; m<n_dims; m++)
                        {
                            norm(m) = norm_inters_cubpts(l)(0,i,m);
                        }

                        // HACK: inflow plane normal direction is -x
                        if(norm(0)==-1)
                        {
                            inflowinters(i,l)=1; // Flag this interface
                        }
                    }
                }
            }
        }

        // Now loop over flagged inters
        for (i=0; i<n_bdy_eles; i++)
        {
            ele = bdy_ele2ele(i);
            for (l=0; l<n_inters_per_ele; l++)
            {
                if(inflowinters(i,l)==1)
                {
                    for (j=0; j<n_cubpts_per_inter(l); j++)
                    {
                        wgt = weight_inters_cubpts(l)(j);
                        detjac = inter_detjac_inters_cubpts(l)(j,i);

                        for (m=0; m<4; m++)
                        {
                            disu_cubpt(m) = 0.;
                        }

                        // Get the solution at cubature point
                        for (k=0; k<n_upts_per_ele; k++)
                        {
                            for (m=0; m<4; m++)
                            {
                                disu_cubpt(m) += opp_inters_cubpts(l)(j,k)*disu_upts(0)(k,ele,m);
                            }
                        }
                        for (m=0; m<4; m++)
                        {
                            integral(m) += wgt*disu_cubpt(m)*detjac;
                        }
                    }
                }
            }
        }

#ifdef _MPI

        hf_array<double> integral_global(4);
        for (m=0; m<4; m++)
        {
            integral_global(m) = 0.;
            MPI_Allreduce(&integral(m), &integral_global(m), 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            integral(m) = integral_global(m);
        }

#endif

        // case-specific parameters
        // periodic channel:
        //area = 2*pi;
        //vol = 4*pi**2;

        // periodic hill (HIOCFD3 Case 3.4):
        area = 9.162;
        vol = 114.34;
        mdot0 = 9.162; // initial mass flux

        // get old mass flux
        if(run_input.restart_flag==0 and in_file_num == 0)
            mdot_old = mdot0;
        else if(run_input.restart_flag==1 and in_file_num == run_input.restart_iter)
            mdot_old = mdot0;
        else
            mdot_old = mass_flux;

        // get timestep
        if (run_input.dt_type == 0||run_input.dt_type==1)
            dt = run_input.dt;
        else if (run_input.dt_type == 2)
            FatalError("Not sure what value of timestep to use in body force term when using local timestepping.");

        // bulk velocity
        if(integral(0)==0)
            ubulk = 0.0;
        else
            ubulk = integral(1)/integral(0);

        // compute new mass flux
        mass_flux = ubulk*integral(0);

        //alpha = 1; // relaxation parameter

        // set body force for streamwise momentum and energy
        body_force(0) = 0.;
        //body_force(1) = alpha/area/dt*(mdot0 - mass_flux); // modified SD3D version
        body_force(1) = 1.0/area/dt*(mdot0 - 2.0*mass_flux + mdot_old); // HIOCFD C3.4 version
        body_force(2) = 0.;
        body_force(3) = 0.;
        body_force(4) = body_force(1)*ubulk; // energy forcing

        if (rank == 0) cout << "iter, mdot0, mdot_old, mass_flux, body_force(1): " << in_file_num << ", " << setprecision(8) << mdot0 << ", " << mdot_old << ", " << mass_flux << ", " << body_force(1) << endl;

        // write out mass flux to file
        if (rank == 0)
        {
            if (run_input.restart_flag==0 and in_file_num == 1)
            {
                // write file header
                write_mdot.open("massflux.dat", ios::out);
                write_mdot << "Iteration, massflux, Ubulk, bodyforce(x)" << endl;
                write_mdot.close();
            }
            else
            {
                // append subsequent dqata
                write_mdot.open("massflux.dat", ios::app);
                write_mdot.precision(15);
                write_mdot << in_file_num;
                write_mdot << ", " << mass_flux;
                write_mdot << ", " << ubulk;
                write_mdot << ", " << body_force(1) << endl;
                write_mdot.close();
            }
        }
        // error checking
        if(std::isnan(body_force(1)))
        {
            FatalError("ERROR: NaN body force, exiting");
        }

//#endif

//TODO: GPU version of above?
//#ifdef _GPU

//#endif

#ifdef _CPU

        // Add to source term at solution points
        for (i=0; i<n_eles; i++)
            for (j=0; j<n_upts_per_ele; j++)
                for(k=0; k<n_fields; k++)
                    src_upts(j,i,k) += body_force(k);

#endif

#ifdef _GPU
        body_force.cp_cpu_gpu();
        evaluate_body_force_gpu_kernel_wrapper(n_upts_per_ele,n_dims,n_fields,n_eles,src_upts.get_ptr_gpu(),body_force.get_ptr_gpu());
#endif
    }
}

// Compute integral quantities
void eles::CalcIntegralQuantities(int n_integral_quantities, hf_array <double>& integral_quantities)
{
    hf_array<double> disu_cubpt(n_fields);
    hf_array<double> grad_disu_cubpt(n_fields,n_dims);
    hf_array<double> S(n_dims,n_dims);
    double wx, wy, wz;
    double dudx, dudy, dudz;
    double dvdx, dvdy, dvdz;
    double dwdx, dwdy, dwdz;
    double diagnostic, tke, pressure, diag, irho, detjac;
    int n_cubpts_per_ele = loc_volume_cubpts.get_dim(1);

    // Sum over elements
    for (int i=0; i<n_eles; i++)
    {
        for (int j=0; j<n_cubpts_per_ele; j++)
        {
            // Get jacobian determinant at cubpts
            detjac = vol_detjac_vol_cubpts(j)(i);

            // Get the solution at cubature point
            for (int m=0; m<n_fields; m++)
            {
                disu_cubpt(m) = 0.;
                for (int k=0; k<n_upts_per_ele; k++)
                {
                    disu_cubpt(m) += opp_volume_cubpts(j,k)*disu_upts(0)(k,i,m);
                }
            }
            // Get the solution gradient at cubature point
            for (int m=0; m<n_fields; m++)
            {
                for (int n=0; n<n_dims; n++)
                {
                    grad_disu_cubpt(m,n)=0.;
                    for (int k=0; k<n_upts_per_ele; k++)
                    {
                        grad_disu_cubpt(m,n) += opp_volume_cubpts(j,k)*grad_disu_upts(k,i,m,n);
                    }
                }
            }
            irho = 1./disu_cubpt(0);
            dudx = irho*(grad_disu_cubpt(1,0) - disu_cubpt(1)*irho*grad_disu_cubpt(0,0));
            dudy = irho*(grad_disu_cubpt(1,1) - disu_cubpt(1)*irho*grad_disu_cubpt(0,1));
            dvdx = irho*(grad_disu_cubpt(2,0) - disu_cubpt(2)*irho*grad_disu_cubpt(0,0));
            dvdy = irho*(grad_disu_cubpt(2,1) - disu_cubpt(2)*irho*grad_disu_cubpt(0,1));

            if (n_dims==3)
            {
                dudz = irho*(grad_disu_cubpt(1,2) - disu_cubpt(1)*irho*grad_disu_cubpt(0,2));
                dvdz = irho*(grad_disu_cubpt(2,2) - disu_cubpt(2)*irho*grad_disu_cubpt(0,2));
                dwdx = irho*(grad_disu_cubpt(3,0) - disu_cubpt(3)*irho*grad_disu_cubpt(0,0));
                dwdy = irho*(grad_disu_cubpt(3,1) - disu_cubpt(3)*irho*grad_disu_cubpt(0,1));
                dwdz = irho*(grad_disu_cubpt(3,2) - disu_cubpt(3)*irho*grad_disu_cubpt(0,2));
            }

            // Now calculate integral quantities
            for (int m=0; m<n_integral_quantities; ++m)
            {
                diagnostic = 0.0;
                if (run_input.integral_quantities(m)=="kineticenergy")
                {
                    // Compute kinetic energy
                    tke = 0.0;
                    for (int n=1; n<n_dims+1; n++)
                        tke += 0.5*disu_cubpt(n)*disu_cubpt(n);

                    diagnostic = irho*tke;
                }
                else if (run_input.integral_quantities(m)=="enstropy")
                {
                    // Compute vorticity squared
                    wz = dvdx - dudy;
                    diagnostic = wz*wz;
                    if (n_dims==3)
                    {
                        wx = dwdy - dvdz;
                        wy = dudz - dwdx;
                        diagnostic += wx*wx+wy*wy;
                    }
                    diagnostic *= 0.5/irho;
                }
                else if (run_input.integral_quantities(m)=="pressuredilatation")
                {
                    // Kinetic energy
                    tke = 0.0;
                    for (int n=1; n<n_dims+1; n++)
                        tke += 0.5*disu_cubpt(n)*disu_cubpt(n);

                    // Compute pressure
                    pressure = (run_input.gamma-1.0)*(disu_cubpt(n_dims+1) - irho*tke);

                    // Multiply pressure by divergence of velocity
                    if (n_dims==2)
                    {
                        diagnostic = pressure*(dudx+dvdy);
                    }
                    else if (n_dims==3)
                    {
                        diagnostic = pressure*(dudx+dvdy+dwdz);
                    }
                }
                else if (run_input.integral_quantities(m)=="straincolonproduct" || run_input.integral_quantities(m)=="devstraincolonproduct")
                {
                    // Rate of strain tensor
                    S(0,0) = dudx;
                    S(0,1) = (dudy+dvdx)/2.0;
                    S(1,0) = S(0,1);
                    S(1,1) = dvdy;
                    diag = (S(0,0)+S(1,1))/3.0;

                    if (n_dims==3)
                    {
                        S(0,2) = (dudz+dwdx)/2.0;
                        S(1,2) = (dvdz+dwdy)/2.0;
                        S(2,0) = S(0,2);
                        S(2,1) = S(1,2);
                        S(2,2) = dwdz;
                        diag += S(2,2)/3.0;
                    }

                    // Subtract diag if deviatoric strain
                    if (run_input.integral_quantities(m)=="devstraincolonproduct")
                    {
                        for (int i=0; i<n_dims; i++)
                            S(i,i) -= diag;
                    }

                    for (int i=0; i<n_dims; i++)
                        for (int j=0; j<n_dims; j++)
                            diagnostic += S(i,j)*S(i,j);

                }
                else
                {
                    FatalError("integral diagnostic quantity not recognized");
                }
                // Add contribution to global integral
                integral_quantities(m) += diagnostic*weight_volume_cubpts(j)*detjac;
            }
        }
    }
}

// Compute time-averaged quantities
void eles::CalcTimeAverageQuantities(double& time)
{
    double current_value, average_value;
    double a, b, dt;
    double spinup_time = run_input.spinup_time;
    double rho;
    int i, j, k;


    for(k=0; k<n_eles; k++)
    {
        for(j=0; j<n_upts_per_ele; j++)
        {
            for(i=0; i<n_average_fields; ++i)
            {

                rho = disu_upts(0)(j,k,0);
                average_value = disu_average_upts(j,k,i);
                if(run_input.average_fields(i)=="rho_average")
                {
                    current_value = rho;
                }
                else if(run_input.average_fields(i)=="u_average")
                {
                    current_value = disu_upts(0)(j,k,1)/rho;
                }
                else if(run_input.average_fields(i)=="v_average")
                {
                    current_value = disu_upts(0)(j,k,2)/rho;
                }
                else if(run_input.average_fields(i)=="w_average")
                {
                    current_value = disu_upts(0)(j,k,3)/rho;
                }
                else if(run_input.average_fields(i)=="e_average")//e only
                {
                    if(n_dims==2)
                    {
                        current_value = disu_upts(0)(j,k,3)/rho;
                    }
                    else
                    {
                        current_value = disu_upts(0)(j,k,4)/rho;
                    }
                }

                // get timestep
                if (run_input.dt_type == 0||run_input.dt_type == 1)
                    dt = run_input.dt;
                else if (run_input.dt_type == 2)
                    dt=dt_local(k);

                // set average value to current value if before spinup time
                // and prevent division by a very small number if time = spinup time
                if(time==spinup_time)
                {
                    a = 0.0;
                    b = 1.0;
                }
                // calculate running average
                else
                {
                    a = (time-spinup_time-dt)/(time-spinup_time);
                    b = dt/(time-spinup_time);
                }

                // Set new average value for next timestep
                disu_average_upts(j,k,i) = a*average_value + b*current_value;
                if(std::isnan(disu_average_upts(j,k,i))) FatalError("NaN in average value, exiting...")
            }
        }
    }
}

void eles::compute_wall_forces( hf_array<double>& inv_force, hf_array<double>& vis_force,  double& temp_cl, double& temp_cd, ofstream& coeff_file, bool write_forces)
{

    hf_array<double> u_l(n_fields),norm(n_dims);
    double p_l,v_sq,vn_l;
    hf_array<double> grad_u_l(n_fields,n_dims);
    hf_array<double> dv(n_dims,n_dims);
    hf_array<double> dE(n_dims);
    hf_array<double> drho(n_dims);
    hf_array<double> taun(n_dims);
    hf_array<double> tautan(n_dims);
    hf_array<double> Finv(n_dims);
    hf_array<double> Fvis(n_dims);
    hf_array<double> loc(n_dims);
    hf_array<double> pos(n_dims);
    double inte, mu, rt_ratio, gamma=run_input.gamma;
    double diag, tauw, taundotn, wgt, detjac;
    double factor, aoa, aos, cp, cf, cl, cd;

    double area_ref = run_input.area_ref;

    //initialize forces
    Finv.initialize_to_zero();
    Fvis.initialize_to_zero();
    inv_force.initialize_to_zero();
    vis_force.initialize_to_zero();
    temp_cd = 0.0;
    temp_cl = 0.0;

    // angle of attack
    aoa = atan2(run_input.v_c_ic, run_input.u_c_ic);

    // angle of side slip
    if (n_dims == 3)
    {
        aos = atan2(run_input.w_c_ic, run_input.u_c_ic);
    }

    // one over the dynamic pressure - 1/(0.5rho*u^2)
    factor = 1.0 / (0.5*run_input.rho_c_ic*(run_input.u_c_ic*run_input.u_c_ic+run_input.v_c_ic*run_input.v_c_ic+run_input.w_c_ic*run_input.w_c_ic));

    // loop over the boundary elements
    for (int i=0; i<n_bdy_eles; i++)
    {

        int ele = bdy_ele2ele(i);

        // loop over the interfaces of the element
        for (int l=0; l<n_inters_per_ele; l++)
        {

            if (run_input.bc_list(bcid(ele,l)).get_bc_flag() == SLIP_WALL){
                coeff_file<<"SLIP_WALL"<<endl;
            }
            else if(run_input.bc_list(bcid(ele,l)).get_bc_flag() == ISOTHERM_WALL){
                coeff_file<<"ISOTHERM_WALL"<<endl;
            }
            else if(run_input.bc_list(bcid(ele,l)).get_bc_flag() == ADIABAT_WALL){
                coeff_file<<"ADIABAT_WALL"<<endl;
            }
            else if(run_input.bc_list(bcid(ele,l)).get_bc_flag() == SLIP_WALL_DUAL){
                coeff_file<<"SLIP_WALL_DUAL"<<endl;
            }

            if (run_input.bc_list(bcid(ele,l)).get_bc_flag() == SLIP_WALL || run_input.bc_list(bcid(ele,l)).get_bc_flag() == ISOTHERM_WALL ||
             run_input.bc_list(bcid(ele,l)).get_bc_flag() == ADIABAT_WALL || run_input.bc_list(bcid(ele,l)).get_bc_flag()==SLIP_WALL_DUAL)
            {

                // Compute force on this interface
                for (int j=n_cubpts_per_inter(l)-1; j>=0; j--)
                {
                    // Get determinant of Jacobian (=area of interface)
                    detjac = inter_detjac_inters_cubpts(l)(j,i);

                    // Get cubature weight
                    wgt = weight_inters_cubpts(l)(j);

                    // Get position of cubature point
                    for (int m=0; m<n_dims; m++)
                        loc(m) = loc_inters_cubpts(l)(m,j);

                    calc_pos(loc,ele,pos);

                    // Compute solution at current cubature point
                    for (int m=0; m<n_fields; m++)
                    {
                        double value = 0.;
                        for (int k=0; k<n_upts_per_ele; k++)
                        {
                            value += opp_inters_cubpts(l)(j,k)*disu_upts(0)(k,ele,m);
                        }
                        u_l(m) = value;
                    }

                    // If viscous, extrapolate the gradient at the cubature points, TODO: this is only exact for linear transform
                    if (viscous==1)
                    {
                        for (int m=0; m<n_fields; m++)
                        {
                            for (int n=0; n<n_dims; n++)
                            {
                                double value=0.;
                                for (int k=0; k<n_upts_per_ele; k++)
                                {
                                    value += opp_inters_cubpts(l)(j,k)*grad_disu_upts(k,ele,m,n);
                                }
                                grad_u_l(m,n) = value;
                            }
                        }
                    }

                    // Get the normal
                    for (int m=0; m<n_dims; m++)
                    {
                        norm(m) = norm_inters_cubpts(l)(j,i,m);
                    }

                    // Get pressure

                    // Not dual consistent
                    if (run_input.bc_list(bcid(ele,l)).get_bc_flag()!=SLIP_WALL_DUAL)
                    {
                        v_sq = 0.;
                        for (int m=0; m<n_dims; m++)
                            v_sq += (u_l(m+1)*u_l(m+1));
                        p_l   = (gamma-1.0)*( u_l(n_dims+1) - 0.5*v_sq/u_l(0));
                    }
                    else
                    {
                        //Dual consistent approach
                        vn_l = 0.;
                        for (int m=0; m<n_dims; m++)
                            vn_l += u_l(m+1)*norm(m);
                        vn_l /= u_l(0);

                        for (int m=0; m<n_dims; m++)
                            u_l(m+1) = u_l(m+1)-(vn_l)*norm(m);

                        v_sq = 0.;
                        for (int m=0; m<n_dims; m++)
                            v_sq += (u_l(m+1)*u_l(m+1));
                        p_l   = (gamma-1.0)*( u_l(n_dims+1) - 0.5*v_sq/u_l(0));
                    }

                    // calculate pressure coefficient at current point on the surface
                    cp = (p_l-run_input.p_c_ic)*factor;

                    // Inviscid force coefficient, F/(0.5*rho*u^2*A)
                    for (int m=0; m<n_dims; m++)
                    {
                        Finv(m) = wgt * (p_l - run_input.p_c_ic) * norm(m) * detjac * factor / area_ref;
                    }

                    // inviscid component of the lift and drag coefficients

                    if (n_dims==2)
                    {
                        cl = -Finv(0)*sin(aoa) + Finv(1)*cos(aoa);
                        cd = Finv(0)*cos(aoa) + Finv(1)*sin(aoa);
                    }
                    else if (n_dims==3)
                    {
                        cl = -Finv(0)*sin(aoa) + Finv(1)*cos(aoa);
                        cd = Finv(0)*cos(aoa)*cos(aos) + Finv(1)*sin(aoa) + Finv(2)*sin(aoa)*cos(aos);
                    }

                    // write to file
                    if (write_forces){
                        if (n_dims==2)
                    {
                        coeff_file << scientific << setw(18) << setprecision(12) << pos(0) << " "  << setw(18) << setprecision(12) << pos(1) << " "<< setw(18) << setprecision(12) << cp;
                    }
                    else if (n_dims==3)
                    {
                        coeff_file << scientific << setw(18) << setprecision(12) << pos(0) << " "  << setw(18) << setprecision(12) << pos(1) << " " << setw(18) << setprecision(12) << pos(2) << " "<< setw(18) << setprecision(12) << cp;
                    }

                    }
                        

                    if (viscous)
                    {
                        // TODO: Have a function that returns tau given u and grad_u
                        // Computing the n_dims derivatives of rho,u,v,w and ene
                        for (int m=0; m<n_dims; m++)
                        {
                            drho(m) = grad_u_l(0,m);
                            for (int n=0; n<n_dims; n++)
                            {
                                dv(n,m) = (grad_u_l(n+1,m)-drho(m)*u_l(n+1)/u_l(0))/u_l(0);//dv_n/dx_m=(drhou_n/dx_m-u_n*drho/dx_m)/rho
                            }
                            dE(m) = (grad_u_l(n_dims+1,m)-drho(m)*u_l(n_dims+1))/u_l(0);//dE/dx_m=(drhoE/dx_m-E*drho/dx_m)/rho
                        }

                        // trace of stress tensor
                        diag = 0.;
                        for (int m=0; m<n_dims; m++)
                        {
                            diag += dv(m,m);
                        }
                        diag /= 3.0;

                        // internal energy
                        inte = u_l(n_dims+1)/u_l(0);
                        for (int m=0; m<n_dims; m++)
                        {
                            inte -= 0.5*u_l(m+1)*u_l(m+1)/u_l(0)/u_l(0);
                        }

                        // get viscosity
                        rt_ratio = (run_input.gamma-1.0)*inte/(run_input.rt_inf);
                        mu = (run_input.mu_inf)*pow(rt_ratio,1.5)*(1+(run_input.c_sth))/(rt_ratio+(run_input.c_sth));
                        mu = mu + run_input.fix_vis*(run_input.mu_inf - mu);

                        // Compute the coefficient of friction and wall shear stress
                        hf_array<double> S(n_dims,n_dims);
                        taun.initialize_to_zero();
                        for (int n = 0; n < n_dims; n++)
                        {
                            for (int m = 0; m < n_dims; m++)
                                S(m, n) = 0.5 * (dv(m, n) + dv(n, m));
                            S(n, n) -= diag;
                        }
#if defined _ACCELERATE_BLAS || defined _MKL_BLAS || defined _STANDARD_BLAS
                        cblas_dgemv(CblasColMajor, CblasNoTrans, n_dims, n_dims, 2.*mu, S.get_ptr_cpu(), n_dims, norm.get_ptr_cpu(), 1, 0.0, taun.get_ptr_cpu(), 1);
#else
                        for (int m = 0; m < n_dims; m++)
                            for (int n = 0; n < n_dims; n++)
                                taun(m) += 2.*mu*S(m, n) * norm(n);
#endif

                        // take dot product with normal
                        taundotn = 0.;
                        for (int m = 0; m < n_dims; m++)
                            taundotn += taun(m) * norm(m);

                        // stresses tangent to wall
                        for (int m = 0; m < n_dims; m++)
                            tautan(m) = taun(m) - taundotn * norm(m);

                        // wall shear stress
                        tauw = 0.;
                        for (int m = 0; m < n_dims; m++)
                            tauw += pow(tautan(m), 2);
                        tauw = sqrt(tauw);

                        // coefficient of friction
                        cf = tauw * factor;
                        if (write_forces)
                            coeff_file << " " << setw(18) << setprecision(12) << cf;

                        // viscous force coefficient
                        for (int m = 0; m < n_dims; m++)
                        {
                            Fvis(m) = -wgt * taun(m) * detjac * factor / area_ref;
                        }

                        // viscous component of the lift and drag coefficients
                        if (n_dims == 2)
                        {
                            cl += -Fvis(0) * sin(aoa) + Fvis(1) * cos(aoa);
                            cd += Fvis(0) * cos(aoa) + Fvis(1) * sin(aoa);
                        }
                        else if (n_dims == 3)
                        {
                            // viscous component of the lift and drag coefficients
                            cl += -Fvis(0) * sin(aoa) + Fvis(1) * cos(aoa);
                            cd += Fvis(0) * cos(aoa) * cos(aos) + Fvis(1) * sin(aoa) + Fvis(2) * sin(aoa) * cos(aos);
                        }
                    } // End of if viscous

                    if (write_forces)
                        coeff_file << endl;

                    // Add force coefficient and lift/drag coefficient contributions from current face
                    for (int m=0; m<n_dims; m++)
                    {
                        inv_force(m) += Finv(m);
                        vis_force(m) += Fvis(m);
                    }
                    temp_cl += cl;
                    temp_cd += cd;
                }
            }
        }
    }
}

void eles::pos_to_loc(hf_array<double>& in_pos,int in_ele,hf_array<double>& out_loc)
{
    //use newton's method to solve non-linear system
    //dx=J^-1*(-f(xn))
    //set initial values
    hf_array<double> temp_d_pos(n_dims,n_dims);
    hf_array<double> fx_n(n_dims);
    hf_array<double> dx(n_dims);
    out_loc.initialize_to_zero();
    do
    {
        //calculate jacobian matrix
        calc_d_pos(out_loc,in_ele,temp_d_pos);
        temp_d_pos=inv_array(temp_d_pos);
        //calculate position based on last step
        calc_pos(out_loc,in_ele,fx_n);

        //setup rhs of equation
        for (int i=0;i<n_dims;i++)
            fx_n(i)=-fx_n(i)+in_pos(i);

        dx=mult_arrays(temp_d_pos,fx_n);

        for(int i=0;i<n_dims;i++)
        {
            out_loc(i)+=dx(i);
        }
    }while(sqrt(inner_product(dx.get_ptr_cpu(), dx.get_ptr_cpu(n_dims), dx.get_ptr_cpu(), 0.)) > 1.e-6);
}


double eles::calc_inlet_length_scale(){
    int i,j,l,ele;
    double detjac;
    double max_detjac=-100000;
    double vol;
    double delta;
    hf_array <int> infloweles(n_bdy_eles);

    infloweles.initialize_to_zero();
    for (i=0; i<n_bdy_eles; i++)
    {
        ele = bdy_ele2ele(i);
        for(l=0; l<n_inters_per_ele; l++)
        {
            int temp_bc_flag=run_input.bc_list(bcid(ele,l)).get_bc_flag();
            if(temp_bc_flag==SUB_IN_SIMP||temp_bc_flag==SUB_IN_CHAR||temp_bc_flag==SUP_IN)            
            {                        
                infloweles(i)=1; // Flag this interface                        
            }
        }

    }
    // get the element max detjac
    for (i=0; i<n_bdy_eles; i++)
    {
        ele = bdy_ele2ele(i);
        if(infloweles(i)==1){
            for(j=0; j<n_upts_per_ele; j++)
            {            
                detjac = detjac_upts(j,ele);
                max_detjac=max(detjac,max_detjac);
            }
        }

    }
#ifdef _MPI
    double max_detjac_global;
    MPI_Allreduce(&max_detjac, &max_detjac_global, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    max_detjac=max_detjac_global;
#endif
    if(max_detjac<0){
        max_detjac=0;
    }
    vol = (*this).calc_ele_vol(max_detjac);
    delta = run_input.filter_ratio*pow(vol,1./n_dims)/(order+1.);
    return delta;
}

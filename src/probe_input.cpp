/*!
 * \file probe_input.cpp
 * \author - Original code: SD++ developed by Patrice Castonguay, Antony Jameson,
 *                          Peter Vincent, David Williams (alphabetical by surname).
  *         - Current development: Weiqi Shen
                                  University of Florida
 * \version 0.1.0
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
#include "../include/probe_input.h"
#include "../include/array.h"
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <sstream>
#include <algorithm>
#include <math.h>
using namespace std;
probe_input::probe_input()
{
    //ctor
}

probe_input::~probe_input()
{
    //dtor
}
void probe_input::setup(string filenameS,int in_dim, int rank)
{
    n_dims=in_dim;
    read_probe_input(filenameS,rank);

}
void probe_input::read_probe_input(string filename, int rank)

{
    fileReader probf(filename);
    if (rank==0)
        cout << endl << "---------------------- Setting up probes ---------------------" << endl;


    /*!----------read probe input parameters ------------*/
    probf.getVectorValueOptional("probe_fields",probe_fields);//name of probe fields
    probf.getScalarValue("probe_layout",probe_layout,0);
    n_probe_fields=probe_fields.get_dim(0);
    for (int i=0; i<n_probe_fields; i++)
    {
        std::transform(probe_fields(i).begin(), probe_fields(i).end(),
                       probe_fields(i).begin(), ::tolower);
    }
    probf.getScalarValue("prob_freq",prob_freq);

    /*!----------calculate probes coordinates ------------*/
    if(probe_layout==0)//point sources
    {
        probf.getVectorValueOptional("probe_x",probe_x);
        probf.getVectorValueOptional("probe_y",probe_y);
        n_probe=probe_x.get_dim(0);//set number of probes
        if (n_probe!=probe_y.get_dim(0))
            FatalError("Probe coordinate data don't agree!\n");
        pos_probe.setup(n_dims,n_probe);
        for(int i=0; i<n_probe; i++)
        {
            pos_probe(0,i)=probe_x(i);
            pos_probe(1,i)=probe_y(i);
        }
        if(n_dims==3)
        {
            probf.getVectorValueOptional("probe_z",probe_z);
            if (n_probe!=probe_z.get_dim(0))
                FatalError("Probe coordinate data don't agree!\n");
            for(int i=0; i<n_probe; i++)
            {
                pos_probe(2,i)=probe_z(i);
            }
        }
    }
    else if(probe_layout==1)//line source input
    {
        double growth_rate_old=1.e6;
        probf.getVectorValueOptional("p_0",p_0);
        probf.getVectorValueOptional("p_1",p_1);
        probf.getScalarValue("init_incre",init_incre);
        probf.getScalarValue("n_probe",n_probe);
        if (p_0.get_dim(0)!=n_dims||p_1.get_dim(0)!=n_dims)
            FatalError("Inappropriate dimension!");
        //calculate growth rate
        l_length=0.;
        for (int i=0; i<n_dims; i++)
        {
            l_length+=pow(p_1(i)-p_0(i),2);
        }
        l_length=sqrt(l_length);

        if ((l_length/init_incre)!=(n_probe-1))
        {
            if (l_length/init_incre<(n_probe-1))
                growth_rate=0.1;
            else
                growth_rate=5.;//initialze to be greater than 2
            //find growth rate use fixed point method
            while (fabs(growth_rate-growth_rate_old)<=1.e-10)
            {
                growth_rate_old=growth_rate;
                if (l_length/init_incre<(n_probe-1))
                    growth_rate=init_incre/l_length*(pow(growth_rate,n_probe-1)-1.)+1.;
                else
                    growth_rate=pow((growth_rate-1.)*(l_length/init_incre+1.),1./(double)(n_probe-1));
            }

            if(isnan(growth_rate))
                FatalError("growth rate NaN!");
            //calculate probe coordiantes
            pos_probe.setup(n_dims,n_probe);
            for (int i=0; i<n_probe; i++)
            {
                for (int j=0; j<n_dims; j++)
                {
                    if(i==0)
                        pos_probe(j,i)=p_0(j);
                    else if (i==(n_probe-1))
                        pos_probe(j,i)=p_1(j);
                    else
                    {
                        pos_probe(j,i)=p_0(j)+init_incre*(pow(growth_rate,(double)i)-1.)/(growth_rate-1.)/l_length*(p_1(j)-p_0(j));
                    }

                }
            }
        }
        else
        {
            growth_rate=1.0;
            for (int i=0; i<n_probe; i++)
                for (int j=0; j<n_dims; j++)
                    pos_probe(j,i)=p_0(j)+i*init_incre/l_length*(p_1(j)-p_0(j));
        }
    }

    if (rank==0)
    {
        cout<<"Number of probe points: "<<n_probe<<endl;
        if(probe_layout==1)
            cout<<"Growth rate: "<<growth_rate<<endl;
    }
}

void probe_input::set_probe_connection(struct solution* FlowSol,int rank)
{
    p2c.setup(n_probe);
    p2t.setup(n_probe);
    p2c.initialize_to_value(-1);
    p2t.initialize_to_value(-1);

    if(rank ==0)
        cout<<"Setting up probe points connectivity.."<<endl;

    for (int i=0; i<FlowSol->n_ele_types; i++)//for each element type
    {
        if (FlowSol->mesh_eles(i)->get_n_eles()!=0)
        {
            for (int j=0; j<n_probe; j++)//for each probe
            {
                if(p2c(j)==-1)//if not inside another type of elements
                {
                    int temp_p2c;
                    array<double>temp_pos(n_dims);
                    for (int k=0; k<n_dims; k++)
                        temp_pos(k)=pos_probe(k,j);
                    temp_p2c=FlowSol->mesh_eles(i)->calc_p2c(temp_pos);

                    if (temp_p2c!=-1)//if inside this type of elements
                    {
                        p2c(j)=temp_p2c;
                        p2t(j)=i;
                    }
                }
            }
        }
    }

#ifdef _MPI
    //MPI_Barrier(MPI_COMM_WORLD);
    array<int> p2cglobe(n_probe,FlowSol->nproc);
    MPI_Allgather(p2c.get_ptr_cpu(),n_probe,MPI_INT,p2cglobe.get_ptr_cpu(),n_probe,MPI_INT,MPI_COMM_WORLD);
    //MPI_Barrier(MPI_COMM_WORLD);
    for (int i=0; i<n_probe; i++)//for each probe
    {
        for(int j=0; j<rank; j++)//loop over all processors before this one
        {
            if(p2c(i)!=-1&&p2cglobe(i,j)!=-1)//there's a conflict
            {
                p2c(i)=-1;
                p2t(i)=-1;
                break;
            }
        }
    }
#endif

/*
    for(int i=0; i<n_probe; i++)
        if(p2c(i)!=-1)
            cout<<"probe "<<i<<" is found in local element No."<<p2c(i)<<
                ", element type: "<<p2t(i)<<", rank: "<<rank<<endl;
    FatalError("Test end!")
    */
}

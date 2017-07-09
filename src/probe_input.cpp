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
#include "../include/input.h"
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
    if (n_dims >=2)
    {
        probf.getVectorValueOptional("probe_fields",probe_fields);//name of probe fields
        probf.getScalarValue("probe_layout",probe_layout,0);
        n_probe_fields=probe_fields.get_dim(0);
        for (int i=0; i<n_probe_fields; i++)
        {
            std::transform(probe_fields(i).begin(), probe_fields(i).end(),
            probe_fields(i).begin(), ::tolower);
        }
        probf.getScalarValue("prob_freq",prob_freq);
        if(probe_layout==0)//manual points input
        {
            probf.getVectorValueOptional("probe_x",probe_x);
            probf.getVectorValueOptional("probe_y",probe_y);
            n_probe=probe_x.get_dim(0);
            cout<<"n_probe: "<<n_probe<<endl;
            if (n_probe!=probe_y.get_dim(0))
                FatalError("Probe coordinate data don't agree!\n");
            probe_pos.setup(n_dims,n_probe);
            for(int i=0; i<n_probe; i++)
            {
                probe_pos(0,i)=probe_x(i);
                probe_pos(1,i)=probe_y(i);
            }
            if(n_dims==3)
            {
                probf.getVectorValueOptional("probe_z",probe_z);
                if (n_probe!=probe_z.get_dim(0))
                    FatalError("Probe coordinate data don't agree!\n");
                for(int i=0; i<n_probe; i++)
                {
                    probe_pos(2,i)=probe_z(i);
                }
            }
        }
        else if(probe_layout==1)//formatted input
        {
            probf.getVectorValueOptional("probe_init_cord",probe_init_cord);
            probf.getVectorValueOptional("growth_rate",growth_rate);
            probf.getVectorValueOptional("init_incre",init_incre);
            if (probe_init_cord.get_dim(0)!=n_dims||growth_rate.get_dim(0)!=n_dims||init_incre.get_dim(0)!=n_dims)
                FatalError("input error!");
            probf.getScalarValue("probe_dim_x",probe_dim_x);
            probf.getScalarValue("probe_dim_y",probe_dim_y);
            probf.getScalarValue("probe_dim_z",probe_dim_z,1);
            n_probe=probe_dim_x*probe_dim_y*probe_dim_z;
            probe_pos.setup(n_dims,n_probe);
            array<int> counter(3);
            for(counter(0)=0; counter(0)<probe_dim_x; counter(0)++)
            {
                for(counter(1)=0; counter(1)<probe_dim_y; counter(1)++)
                {
                    for(counter(2)=0; counter(2)<probe_dim_z; counter(2)++)
                    {
                        int index;
                        index=counter(2)+counter(1)*probe_dim_z+counter(0)*probe_dim_y*probe_dim_z;
                        for(int l=0; l<n_dims; l++)
                        {
                            if(growth_rate(l)!=1)
                                probe_pos(l,index)=probe_init_cord(l)+init_incre(l)*(pow(growth_rate(l),((double)counter(l)))-1.)/(growth_rate(l)-1.);
                            else
                                probe_pos(l,index)=probe_init_cord(l)+((double)counter(l))*init_incre(l);
                        }
                    }
                }
            }
        }
    }
    else
    {
        FatalError("Dimension must be greater than 1\n");
    }
    if(rank==0)
    probe_pos.print();
}

void probe_input::set_probe_connection(struct solution* FlowSol,int rank)
{
    array<double> v1;
    array<double> v2;
    p2c.setup(n_probe);
    p2t.setup(n_probe);
    p2e.setup(n_probe);
    array<int> indicator(n_probe);
    p2c.initialize_to_value(-1);
    p2t.initialize_to_value(-1);
    p2e.initialize_to_value(-1);
    int s2v[2][4]= {{0,1,2,0},{0,1,3,2}};
    if(rank ==0)
        cout<<"setting probe points connection.."<<endl;
#ifdef _MPI
    int nproc;
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);
#endif

    if(n_dims==2)
    {
        v1.setup(2);
        v2.setup(2);
        for(int i=0; i<2; i++)//element type i(0 is tri,1 is quad)
        {
            if (FlowSol->mesh_eles(i)->get_n_eles()!=0)//if have ele
            {
                for (int j=0; j<FlowSol->mesh_eles(i)->get_n_eles(); j++) //element j
                {
                    indicator.initialize_to_zero();
                    //indicator.print();
                    //cout<<"element "<<j<<endl;
                    int n_spts_per_ele=FlowSol->mesh_eles(i)->get_n_spts_per_ele(j);//get number of shape points in element
                    if (n_spts_per_ele>4) FatalError("quadrilateral elements no implemented!")//if is simple shape
                        for(int k=0; k<n_spts_per_ele; k++) //shape point k
                        {
                            v2(0)=FlowSol->mesh_eles(i)->get_shape(0,(k+1)<n_spts_per_ele?s2v[i][k+1]:s2v[i][0],j)-FlowSol->mesh_eles(i)->get_shape(0,s2v[i][k],j);//A in AXB
                            v2(1)=FlowSol->mesh_eles(i)->get_shape(1,(k+1)<n_spts_per_ele?s2v[i][k+1]:s2v[i][0],j)-FlowSol->mesh_eles(i)->get_shape(1,s2v[i][k],j);
                            for(int l=0; l<n_probe; l++)
                            {
                                v1(0)=probe_pos(0,l) - FlowSol->mesh_eles(i)->get_shape(0,s2v[i][k],j);//B in AXB
                                v1(1)=probe_pos(1,l) - FlowSol->mesh_eles(i)->get_shape(1,s2v[i][k],j);
                                double vv=0;
                                vv=v2(0)*v1(1)-v1(0)*v2(1);
                                //cout<<"v1"<<endl;
                                //v1.print();cout<<"v2"<<endl;v2.print();
                                //cout<<"vv"<<setprecision(5)<<vv<<endl;
                                if(vv<0)//RHS of the edge vector
                                    indicator(l)=-1;
                                else if(vv==0)//on edge of the cell
                                    indicator(l)=k+1;
                            }
                        }
                    for (int l=0; l<n_probe; l++)
                    {
                        if(indicator(l)>=0)//in the cell or on the cell
                        {
                            p2c(l)=j;
                            p2t(l)=i;
                            p2e(l)=indicator(l);
                            // cout<<"p2e"<<p2e(l)<<endl;
                            //for(int k=0;k<n_spts_per_ele;k++)
                            //{
                            // cout<<"x: "<<setprecision(5)<<FlowSol->mesh_eles(i)->get_shape(0,s2v[i][k],j)<<" y: "<<setprecision(5)<<FlowSol->mesh_eles(i)->get_shape(1,s2v[i][k],j)<<endl;
                            //}
                        }

                    }
                }

            }

        }
    }
    else if(n_dims==3)
    {
        FatalError("3D not implemented yet!");
    }
#ifdef _MPI
    MPI_Barrier(MPI_COMM_WORLD);
    array<int> p2cglobe(nproc,n_probe);
    MPI_Allgather(p2c.get_ptr_cpu(),n_probe,MPI_INT,p2cglobe.get_ptr_cpu(),n_probe,MPI_INT,MPI_COMM_WORLD);
    //MPI_Barrier(MPI_COMM_WORLD);
    for (int i=0; i<n_probe; i++)
    {
        for(int j=0; j<rank; j++)
        {
                if(p2c(i)!=-1&&p2cglobe(j,i)!=-1)//there's a conflict
                {
                    p2c(i)=-1;
                    p2t(i)=-1;
                    p2e(i)=-1;
                    break;
                }
        }
    }
#endif

    for(int i=0; i<n_probe; i++)
    {
        cout<<"probe "<<i<<" is found in "<<"local element No."<<p2c(i)<<", element type: ";
        switch(p2t(i))
        {
        case 0:
            cout<<"Tri";
        case 1:
            cout<<"Quad";
        }
        cout<<", rank: "<<rank<<endl;
    }
}

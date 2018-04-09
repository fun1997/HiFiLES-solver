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
#include "../include/global.h"
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
void probe_input::setup(string filenameS,struct solution* FlowSol, int rank)
{
    n_dims=FlowSol->n_dims;
    read_probe_input(filenameS,rank);
    set_probe_connectivity(FlowSol,rank);
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
        double l_length;
        probf.getVectorValueOptional("p_0",p_0);
        probf.getVectorValueOptional("p_1",p_1);
        probf.getScalarValue("init_incre",init_incre);
        probf.getScalarValue("n_probe",n_probe);
        if (p_0.get_dim(0)!=n_dims||p_1.get_dim(0)!=n_dims)
            FatalError("Dimension of input start/end point must agree with simulation dimension!");

        //calculate growth rate
        l_length=0.;
        for (int i=0; i<n_dims; i++)
        {
            l_length+=pow(p_1(i)-p_0(i),2);
        }
        l_length=sqrt(l_length);


        if ((l_length/init_incre)!=(n_probe-1))//need iteratively solve for growth rate
        {
            double growth_rate_old=10.;
            double jacob;
            double fx_n;
            double tol=1e-10;
            if (l_length/init_incre<(n_probe-1))
                growth_rate=0.1;
            else
                growth_rate=5;//initialze to be greater than 2
            //find growth rate use newton's method
            while (fabs(growth_rate-growth_rate_old)>tol)
            {
                growth_rate_old=growth_rate;
                jacob=init_incre*((n_probe-2.)*pow(growth_rate,n_probe)-(n_probe-1.)*pow(growth_rate,n_probe-1)+growth_rate)
                                  /(pow(growth_rate-1.,2)*growth_rate);
                fx_n=l_length-init_incre*(pow(growth_rate,n_probe-1)-1.)/(growth_rate-1.);
                growth_rate+=fx_n/jacob;;
            }

            if(isnan(growth_rate))
                FatalError("Growth rate NaN!");
            //calculate probe coordiantes
            pos_probe.setup(n_dims,n_probe);
            for (int i=0; i<n_probe; i++)
                for (int j=0; j<n_dims; j++)
                    pos_probe(j,i)=p_0(j)+init_incre*(pow(growth_rate,(double)i)-1.)/(growth_rate-1.)/l_length*(p_1(j)-p_0(j));
        }
        else
        {
            growth_rate=1.0;
            for (int i=0; i<n_probe; i++)
                for (int j=0; j<n_dims; j++)
                    pos_probe(j,i)=p_0(j)+i*init_incre/l_length*(p_1(j)-p_0(j));
        }

    }
    else if(probe_layout==2)
    {
        probf.getScalarValue("neu_file",neu_file);
        set_probe_gambit(neu_file);
    }
    else
        FatalError("Probe layout not implemented")

    if (rank==0)
    {
        if (probe_layout!=2)
        {
            cout<<"Number of probe points: "<<n_probe<<endl;
            if(probe_layout==1)
                cout<<"Growth rate: "<<growth_rate<<endl;
        }
        else
        cout<<"Number of probe elements: "<<n_probe<<endl;
    }
}

void probe_input::set_probe_connectivity(struct solution* FlowSol,int rank)
{
    p2c.setup(n_probe);
    p2t.setup(n_probe);
    p2c.initialize_to_value(-1);
    p2t.initialize_to_value(-1);

    if(rank ==0)
        cout<<"Setting up probe points connectivity.."<<flush;

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

    if (rank==0)
        cout<<"done"<<endl;
    //setup probe location in reference domain.
    if (rank==0)
        cout<<"Calculating location of probes in reference domain.."<<flush;
    set_loc_probepts(FlowSol);

    /*
    for(int i=0; i<n_probe; i++)
       if(p2c(i)!=-1)
           cout<<"probe "<<i<<" is found in local element No."<<p2c(i)<<
               ", element type: "<<p2t(i)<<", rank: "<<rank<<endl;
    FatalError("Test end!")
    */
    if(rank==0)
        cout<<"done"<<endl;
}

void probe_input::set_probe_gambit(string filename)
{
    if (filename.compare(filename.size()-3,3,"neu"))
        FatalError("Only gambit neutral file format is supported!");
    int n_verts_global;
    int dummy,dummy2;
    int gambit_ndims;
    char buf[BUFSIZ]= {""};
    array<int> probe_c2n_v;//element to number of vertex
    array<int> probe_c2v;//elements to vertex index
    array<double> probe_xv;//vertex coordinates
    array<int> probe_ctype;//type of element
    ifstream f_neu;

    /*! read points and elements */
    f_neu.open(filename.c_str());
    if(!f_neu)
        FatalError("Unable to open file");
    // Skip 6-line header
    for (int i=0; i<6; i++)
        f_neu.getline(buf,BUFSIZ);

    // Find number of vertices and number of cells
    f_neu >> n_verts_global   // num vertices in mesh
          >> n_probe     // num elements
          >> dummy              // num material groups
          >> dummy              // num boundary groups
          >> dummy2  // num space dimensions of mesh element
          >> gambit_ndims;//num dimension of components

if(n_dims<dummy2||n_dims<gambit_ndims)
    FatalError("Dimension in gambit file must be less/equal to that of simulation");

    f_neu.getline(buf,BUFSIZ);  // clear rest of line
    f_neu.getline(buf,BUFSIZ);  // Skip 2 lines
    f_neu.getline(buf,BUFSIZ);

    probe_c2n_v.setup(n_probe);
    probe_c2v.setup(n_probe,MAX_V_PER_C);
    probe_xv.setup(n_verts_global,n_dims);
    probe_ctype.setup(n_probe);

    probe_c2n_v.initialize_to_value(-1);
    probe_c2v.initialize_to_value(-1);
    probe_xv.initialize_to_value(-1.);
    probe_ctype.initialize_to_zero();

    for (int i=0; i<n_verts_global; i++)
    {
        f_neu >> dummy;//read id
        for (int j=0; j<gambit_ndims; j++)
            f_neu>>probe_xv(i,j);
        if (gambit_ndims<n_dims)//3D simulation but 2D mesh component
            probe_xv(i,2)=0.0;//HACK: if specified a plane then have to be at z=0
        f_neu.getline(buf,BUFSIZ);
    }

    f_neu.getline(buf,BUFSIZ); // Skip "ENDOFSECTION"
    f_neu.getline(buf,BUFSIZ); // Skip "ELEMENTS/CELLS"

    // Start reading elements
    int eleType;
    for (int i=0; i<n_probe; i++)
    {
        //  ctype is the element type:  1=edge, 2=quad, 3=tri, 4=brick, 5=wedge, 6=tet, 7=pyramid
        f_neu >> dummy >> eleType >> probe_c2n_v(i);

        if (eleType==3)
            probe_ctype(i)=TRI;
        else if (eleType==2)
            probe_ctype(i)=QUAD;
        else if (eleType==6)
            probe_ctype(i)=TET;
        else if (eleType==5)
            probe_ctype(i)=PRISM;
        else if (eleType==4)
            probe_ctype(i)=HEX;
        // triangle
        if (probe_ctype(i)==TRI)
        {
            if (probe_c2n_v(i)==3) // linear triangle
                f_neu >> probe_c2v(i,0) >> probe_c2v(i,1) >> probe_c2v(i,2);
            else
                FatalError("triangle element type not implemented");
        }
        // quad
        else if (probe_ctype(i)==QUAD)
        {
            if (probe_c2n_v(i)==4) // linear quadrangle
                f_neu >> probe_c2v(i,0) >> probe_c2v(i,1) >> probe_c2v(i,2) >> probe_c2v(i,3);
            else
                FatalError("quad element type not implemented");
        }
        // tet
        else if (probe_ctype(i)==TET)
        {
            if (probe_c2n_v(i)==4) // linear tets
            {
                f_neu >> probe_c2v(i,0) >> probe_c2v(i,1) >> probe_c2v(i,2) >> probe_c2v(i,3);
            }
            else
                FatalError("tet element type not implemented");
        }
        // prisms
        else if (probe_ctype(i)==PRISM)
        {
            if (probe_c2n_v(i)==6) // linear prism
                f_neu >> probe_c2v(i,0) >> probe_c2v(i,1) >> probe_c2v(i,2) >> probe_c2v(i,3) >> probe_c2v(i,4) >> probe_c2v(i,5);
            else
                FatalError("Prism element type not implemented");
        }
        // hexa
        else if (probe_ctype(i)==HEX)
        {
            if (probe_c2n_v(i)==8) // linear hexas
                f_neu >> probe_c2v(i,0) >> probe_c2v(i,1) >> probe_c2v(i,2) >> probe_c2v(i,3) >> probe_c2v(i,4) >> probe_c2v(i,5) >> probe_c2v(i,6) >> probe_c2v(i,7);
            else
                FatalError("Hexa element type not implemented");
        }
        else
        {
            cout << "Element Type = " << probe_ctype(i) << endl;
            FatalError("Haven't implemented this element type in gambit_meshreader, exiting ");
        }

        f_neu.getline(buf,BUFSIZ); // skip end of line

        // Shift every values of c2v by -1
        for(int k=0; k<probe_c2n_v(i); k++)
            if(probe_c2v(i,k)!=0)
                probe_c2v(i,k)--;
    }

    f_neu.close();

    /*! calculate face centroid and copy it to pos_probe*/
    pos_probe.setup(n_dims,n_probe);
    for (int i=0; i<n_probe; i++)
    {
        //store coordinates of all vertex belongs to this cell
        array<double> temp_pos(n_dims,probe_c2n_v(i));
        for (int j=0; j<probe_c2n_v(i); j++)
            for (int k=0; k<n_dims; k++)
                temp_pos(k,j)=probe_xv(probe_c2v(i,j),k);
        array<double> temp_pos_centroid;
        temp_pos_centroid=calc_centroid(temp_pos);
        for (int j=0; j<n_dims; j++)
            pos_probe(j,i)=temp_pos_centroid(j);
    }

    /*! calculate face normals and area*/
    if (dummy2==2&&n_dims==3)//if 2D mesh and 3D simulation
    {
        //calculate face normal
        output_normal=true;
        surf_normal.setup(n_dims,n_probe);
        for (int i=0; i<n_probe; i++)
        {
            array<double> temp_vec(n_dims);
            array<double> temp_vec2(n_dims);
            array<double> temp_normal;
            for (int k=0; k<n_dims; k++) //every dimension
            {
                temp_vec(k)=probe_xv(probe_c2v(i,1),k)-probe_xv(probe_c2v(i,0),k);
                temp_vec2(k)=probe_xv(probe_c2v(i,2),k)-probe_xv(probe_c2v(i,1),k);
            }
            temp_normal=cross_prod_3d(temp_vec,temp_vec2);

            //normalize the normal vector
            double temp_length=0.0;
            for (int j=0; j<n_dims; j++)
                temp_length+=temp_normal(j)*temp_normal(j);
            temp_length=sqrt(temp_length);
            for (int j=0; j<n_dims; j++)
                surf_normal(j,i)=temp_normal(j)/temp_length;
        }

        //calculate face area
        surf_area.setup(n_probe);
        surf_area.initialize_to_zero();
        for (int i =0; i<n_probe; i++)
        {
            array<double> temp_vec(n_dims);
            array<double> temp_vec2(n_dims);
            array<double> temp_normal;
            for (int j=0; j<probe_ctype(i)+1;j++) //1->2 part
            {
                for (int k=0; k<n_dims; k++) //every dimension
                {
                    temp_vec(k)=probe_xv(probe_c2v(i,1+2*j),k)-probe_xv(probe_c2v(i,2*j),k);
                    temp_vec2(k)=probe_xv(((2+2*j)<probe_c2n_v(i))?probe_c2v(i,2+2*j):probe_c2v(i,0),k)
                                -probe_xv(probe_c2v(i,1+2*j),k);
                }
                temp_normal=cross_prod_3d(temp_vec,temp_vec2);
                double temp_area=0.0;
                for (int k=0; k<n_dims; k++)
                    temp_area+=temp_normal(k)*temp_normal(k);
                temp_area=0.5*sqrt(temp_area);
                surf_area(i)+=temp_area;
            }
        }
    }
    else
        output_normal=false;
}

void probe_input::set_loc_probepts(struct solution* FlowSol)
{
    loc_probe.setup(n_dims,n_probe);
    array<double> temp_pos(n_dims);
    array<double> temp_loc(n_dims);
    for (int i=0; i<n_probe; i++)//loop over all probes
    {
        if(p2c(i)!=-1)//if probe belongs to this processor
        {
            for (int j=0; j<n_dims; j++)
                temp_pos(j)=pos_probe(j,i);
            FlowSol->mesh_eles(p2t(i))->pos_to_loc(temp_pos,p2c(i),temp_loc);
            //copy to loc_probe
            for (int j=0; j<n_dims; j++)
                loc_probe(j,i)=temp_loc(j);
        }
        else
        {
            for (int j=0; j<n_dims; j++)
                loc_probe(j,i)=0;
        }
    }
}
/*! END */

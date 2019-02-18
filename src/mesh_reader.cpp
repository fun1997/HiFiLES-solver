#include "../include/mesh_reader.h"
#include "../include/global.h"
#include <string.h>
#include <string>
#include <vector>
#include <algorithm>
using namespace std;

mesh_reader::mesh_reader(string in_fileName, mesh *in_mesh)
{
    fname = in_fileName;
    mesh_ptr = in_mesh;
    if (!fname.compare(fname.size() - 3, 3, "neu"))
        mesh_format = 0;
    else if (!fname.compare(fname.size() - 3, 3, "msh"))
        mesh_format = 1;
    else
        FatalError("Mesh format not recognized");

    read_header();
}

mesh_reader::~mesh_reader()
{
}

void mesh_reader::read_header(void)
{

    if (mesh_format == 0) //gambit
    {
        read_header_gambit();
    }
    else if (mesh_format == 1) //gmsh
    {
        read_header_gmsh();
    }
}

void mesh_reader::partial_read_connectivity(int kstart, int in_num_cells)
{
    if (kstart >= mesh_ptr->num_cells_global || in_num_cells > (mesh_ptr->num_cells_global - kstart))
        FatalError("Illegal block of elements to read");

    mesh_ptr->num_cells = in_num_cells; //store number of cells read by this processor in mesh obj

    if (mesh_format == 0) //gambit
    {
        partial_read_connectivity_gambit(kstart, in_num_cells);
    }
    else if (mesh_format == 1) //gmsh
    {
        partial_read_connectivity_gmsh(kstart, in_num_cells);
    }
}

void mesh_reader::read_vertices(void)
{
    if (mesh_format == 0) //gambit
    {
        read_vertices_gambit();
    }
    else if (mesh_format == 1) //gmsh
    {
        read_vertices_gmsh();
    }
}

void mesh_reader::read_boundary(void)
{
    if (mesh_format == 0) //gambit
    {
        read_boundary_gambit();
    }
    else if (mesh_format == 1) //gmsh
    {
        read_boundary_gmsh();
    }
}
/*------------------------------gambit readers----------------------------*/
void mesh_reader::read_header_gambit(void)
{
    int dummy;
    char buf[BUFSIZ] = {""};

    mesh_file.open(fname.c_str());
    if (!mesh_file)
        FatalError("Unable to open mesh file");

    // Skip 6-line header
    for (int i = 0; i < 6; i++)
        mesh_file.getline(buf, BUFSIZ);

    // Find number of vertices and number of cells
    mesh_file >> mesh_ptr->num_verts_global // num vertices in mesh
        >> mesh_ptr->num_cells_global       // num elements
        >> dummy                            // num material groups
        >> mesh_ptr->n_bdy                  // num boundary groups
        >> mesh_ptr->n_ele_dims             // num ele dimensions(surf/vol)
        >> mesh_ptr->n_dims;                // num coordinate dimensions
    if (mesh_ptr->n_dims != 2 && mesh_ptr->n_dims != 3)
    {
        FatalError("Invalid mesh dimensionality. Expected 2D or 3D.");
    }
    mesh_file.close();
}

void mesh_reader::partial_read_connectivity_gambit(int kstart, int in_num_cells)
{
    int dummy;
    char buf[BUFSIZ] = {""};

    mesh_file.open(fname.c_str());
    if (!mesh_file)
        FatalError("Unable to open mesh file");

    while (1) //skip to element section
    {
        mesh_file.getline(buf, BUFSIZ);
        if (strstr(buf, "ELEMENTS/CELLS") != NULL)
            break;
    }

    //allocate memory
    mesh_ptr->c2v.setup(in_num_cells, MAX_V_PER_C); // stores the vertices making that cell
    mesh_ptr->c2n_v.setup(in_num_cells);            // stores the number of nodes making that cell
    mesh_ptr->ctype.setup(in_num_cells);            // stores the type of cell
    mesh_ptr->ic2icg.setup(in_num_cells);

    // Initialize arrays to -1
    mesh_ptr->c2v.initialize_to_value(-1);

    // Skip elements being read by other processors

    for (int i = 0; i < kstart; i++)
    {
        mesh_file >> dummy >> dummy >> dummy;
        mesh_file.getline(buf, BUFSIZ); // skip end of line
        if (dummy > 7)
            mesh_file.getline(buf, BUFSIZ); // skip another line
        if (dummy > 14)
            mesh_file.getline(buf, BUFSIZ); // skip another line
        if (dummy > 21)
            mesh_file.getline(buf, BUFSIZ); // skip another line
    }

    // Read a block of elements

    int eleType;//general type of element 

    for (int i = 0; i < in_num_cells; i++)
    {
        //  ctype is the element type:  1=edge, 2=quad, 3=tri, 4=brick, 5=wedge, 6=tet, 7=pyramid
        mesh_file >> mesh_ptr->ic2icg(i) >> eleType >> mesh_ptr->c2n_v(i);

        //identify type of the element
        if (eleType == 3)
            mesh_ptr->ctype(i) = TRI;
        else if (eleType == 2)
            mesh_ptr->ctype(i) = QUAD;
        else if (eleType == 6)
            mesh_ptr->ctype(i) = TET;
        else if (eleType == 5)
            mesh_ptr->ctype(i) = PRISM;
        else if (eleType == 4)
            mesh_ptr->ctype(i) = HEX;

        // triangle
        if (mesh_ptr->ctype(i) == TRI)
        {
            if (mesh_ptr->c2n_v(i) == 3) // linear triangle
                mesh_file >> mesh_ptr->c2v(i, 0) >> mesh_ptr->c2v(i, 1) >> mesh_ptr->c2v(i, 2);
            else if (mesh_ptr->c2n_v(i) == 6) // quadratic triangle
                mesh_file >> mesh_ptr->c2v(i, 0) >> mesh_ptr->c2v(i, 3) >> mesh_ptr->c2v(i, 1) >> mesh_ptr->c2v(i, 4) >> mesh_ptr->c2v(i, 2) >> mesh_ptr->c2v(i, 5);
            else
                FatalError("triangle element type not implemented");
        }
        // quad
        else if (mesh_ptr->ctype(i) == QUAD)
        {
            if (mesh_ptr->c2n_v(i) == 4) // linear quadrangle
                mesh_file >> mesh_ptr->c2v(i, 0) >> mesh_ptr->c2v(i, 1) >> mesh_ptr->c2v(i, 3) >> mesh_ptr->c2v(i, 2);
            else if (mesh_ptr->c2n_v(i) == 8) // quadratic quad
                mesh_file >> mesh_ptr->c2v(i, 0) >> mesh_ptr->c2v(i, 4) >> mesh_ptr->c2v(i, 1) >> mesh_ptr->c2v(i, 5) >> mesh_ptr->c2v(i, 2) >> mesh_ptr->c2v(i, 6) >> mesh_ptr->c2v(i, 3) >> mesh_ptr->c2v(i, 7);
            else
                FatalError("quad element type not implemented");
        }
        // tet
        else if (mesh_ptr->ctype(i) == TET)
        {
            if (mesh_ptr->c2n_v(i) == 4) // linear tets
            {
                mesh_file >> mesh_ptr->c2v(i, 0) >> mesh_ptr->c2v(i, 1) >> mesh_ptr->c2v(i, 2) >> mesh_ptr->c2v(i, 3);
            }
            else if (mesh_ptr->c2n_v(i) == 10) // quadratic tet
            {
                mesh_file >> mesh_ptr->c2v(i, 0) >> mesh_ptr->c2v(i, 4) >> mesh_ptr->c2v(i, 1) >> mesh_ptr->c2v(i, 5) >> mesh_ptr->c2v(i, 7);
                mesh_file >> mesh_ptr->c2v(i, 2) >> mesh_ptr->c2v(i, 6) >> mesh_ptr->c2v(i, 9) >> mesh_ptr->c2v(i, 8) >> mesh_ptr->c2v(i, 3);
            }
            else
                FatalError("tet element type not implemented");
        }
        // prisms
        else if (mesh_ptr->ctype(i) == PRISM)
        {
            if (mesh_ptr->c2n_v(i) == 6) // linear prism
                mesh_file >> mesh_ptr->c2v(i, 0) >> mesh_ptr->c2v(i, 1) >> mesh_ptr->c2v(i, 2) >> mesh_ptr->c2v(i, 3) >> mesh_ptr->c2v(i, 4) >> mesh_ptr->c2v(i, 5);
            else if (mesh_ptr->c2n_v(i) == 15) // quadratic prism
                mesh_file >> mesh_ptr->c2v(i, 0) >> mesh_ptr->c2v(i, 6) >> mesh_ptr->c2v(i, 1) >> mesh_ptr->c2v(i, 8) >> mesh_ptr->c2v(i, 7) >> mesh_ptr->c2v(i, 2) >> mesh_ptr->c2v(i, 9) >> mesh_ptr->c2v(i, 10) >> mesh_ptr->c2v(i, 11) >> mesh_ptr->c2v(i, 3) >> mesh_ptr->c2v(i, 12) >> mesh_ptr->c2v(i, 4) >> mesh_ptr->c2v(i, 14) >> mesh_ptr->c2v(i, 13) >> mesh_ptr->c2v(i, 5);
            else
                FatalError("Prism element type not implemented");
        }
        // hexa
        else if (mesh_ptr->ctype(i) == HEX)
        {
            if (mesh_ptr->c2n_v(i) == 8) // linear hexas
                mesh_file >> mesh_ptr->c2v(i, 0) >> mesh_ptr->c2v(i, 2) >> mesh_ptr->c2v(i, 4) >> mesh_ptr->c2v(i, 6) >> mesh_ptr->c2v(i, 1) >> mesh_ptr->c2v(i, 3) >> mesh_ptr->c2v(i, 5) >> mesh_ptr->c2v(i, 7);
            else if (mesh_ptr->c2n_v(i) == 20) // quadratic hexas
                mesh_file >> mesh_ptr->c2v(i, 0) >> mesh_ptr->c2v(i, 11) >> mesh_ptr->c2v(i, 3) >> mesh_ptr->c2v(i, 12) >> mesh_ptr->c2v(i, 15) >> mesh_ptr->c2v(i, 4) >> mesh_ptr->c2v(i, 19) >> mesh_ptr->c2v(i, 7) >> mesh_ptr->c2v(i, 8) >> mesh_ptr->c2v(i, 10) >> mesh_ptr->c2v(i, 16) >> mesh_ptr->c2v(i, 18) >> mesh_ptr->c2v(i, 1) >> mesh_ptr->c2v(i, 9) >> mesh_ptr->c2v(i, 2) >> mesh_ptr->c2v(i, 13) >> mesh_ptr->c2v(i, 14) >> mesh_ptr->c2v(i, 5) >> mesh_ptr->c2v(i, 17) >> mesh_ptr->c2v(i, 6);
            else
                FatalError("Hexa element type not implemented");
        }
        else
        {
            cout << "Element Type = " << mesh_ptr->ctype(i) << endl;
            FatalError("Haven't implemented this element type in gambit_meshreader3, exiting ");
        }
        mesh_file.getline(buf, BUFSIZ); // skip end of line

        // Shift every values of c2v by -1 to be 0 based, rest of it to be -1
        for (int k = 0; k < mesh_ptr->c2n_v(i); k++)
                mesh_ptr->c2v(i, k)--;

        // Also shift every value of ic2icg to be 0 based
        mesh_ptr->ic2icg(i)--;
    }
    mesh_file.close();
}

void mesh_reader::read_vertices_gambit(void)
{
    // Now open gambit file and read the vertices
    char buf[BUFSIZ] = {""};

    mesh_file.open(fname.c_str());

    if (!mesh_file)
        FatalError("Could not open mesh file");

    while (1) //skip to vertex section
    {
        mesh_file.getline(buf, BUFSIZ);
        if (strstr(buf, "NODAL COORDINATES") != NULL)
            break;
    }

    // Read the location of vertices
    mesh_ptr->xv.setup(mesh_ptr->num_verts, mesh_ptr->n_dims);
    int id, index;
    for (int i = 0; i < mesh_ptr->num_verts_global; i++)
    {
        mesh_file >> id;//global id
        index = index_locate_int(id - 1, mesh_ptr->iv2ivg.get_ptr_cpu(), mesh_ptr->num_verts);//find local index

        if (index != -1) // Vertex belongs to this processor
        {
            for (int m = 0; m < mesh_ptr->n_dims; m++)
                mesh_file >> mesh_ptr->xv(index, m);
        }
        mesh_file.getline(buf, BUFSIZ); //clear the line
    }
    mesh_file.close();
}

void mesh_reader::read_boundary_gambit()
{
    char buf[BUFSIZ] = {""};
    mesh_file.open(fname.c_str());
    if (!mesh_file)
        FatalError("Unable to open mesh file");

    mesh_ptr->bc_id.setup(mesh_ptr->num_cells, MAX_F_PER_C);//array that hold the index of bc_objects in bc_list
    mesh_ptr->bc_id.initialize_to_value(-1);//-1 as default internal face
    run_input.bc_list.setup(mesh_ptr->n_bdy);//list that hold bc objects

    for (int i = 0; i < mesh_ptr->n_bdy; i++)
    {
        // Move cursor to the next boundary
        while (1)
        {
            mesh_file.getline(buf, BUFSIZ);
            if (strstr(buf, "BOUNDARY CONDITIONS") != NULL)
                break;
        }

        int bcNF, dummy, icg, k, real_face;
        string bcname;
        mesh_file >> bcname >> dummy >> bcNF;
        run_input.bc_list(i).setup(bcname); //setup bcname

        mesh_file.getline(buf, BUFSIZ);//skip rest of line

        int eleType;
        for (int bf = 0; bf < bcNF; bf++)
        {
            mesh_file >> icg >> eleType >> k;
            icg--; // 1-indexed -> 0-indexed
            // Matching Gambit faces with face convention in code
            if (eleType == 2 || eleType == 3)
                real_face = k - 1;
            // Hex
            else if (eleType == 4)
            {
                if (k == 1)
                    real_face = 0;
                else if (k == 2)
                    real_face = 3;
                else if (k == 3)
                    real_face = 5;
                else if (k == 4)
                    real_face = 1;
                else if (k == 5)
                    real_face = 4;
                else if (k == 6)
                    real_face = 2;
            }
            // Tet
            else if (eleType == 6)
            {
                if (k == 1)
                    real_face = 3;
                else if (k == 2)
                    real_face = 2;
                else if (k == 3)
                    real_face = 0;
                else if (k == 4)
                    real_face = 1;
            }
            else if (eleType == 5)
            {
                if (k == 1)
                    real_face = 2;
                else if (k == 2)
                    real_face = 3;
                else if (k == 3)
                    real_face = 4;
                else if (k == 4)
                    real_face = 0;
                else if (k == 5)
                    real_face = 1;
            }
            else
            {
                cout << "Element Type = " << eleType << endl;
                FatalError("Cannot handle other element type in readbnd");
            }

            // Check if cell icg belongs to processor, cellid is reordered to be the idth large cell
            int cellID = index_locate_int(icg, mesh_ptr->ic2icg.get_ptr_cpu(), mesh_ptr->num_cells);

            // If it does, find local cell ic corresponding to icg
            if (cellID != -1)
                mesh_ptr->bc_id(cellID, real_face) = i;

        }
    }

    mesh_file.close();
}

/*------------------------------gmsh readers----------------------------*/

void mesh_reader::read_header_gmsh(void)
{
    int dummy;
    char buf[BUFSIZ] = {""};
    string str, bc_txt_temp;
    int bcid;

    //open file
    mesh_file.open(fname.c_str());
    if (!mesh_file)
        FatalError("Unable to open mesh file");

    // Move cursor to $PhysicalNames
    while (1)
    {
        getline(mesh_file, str);
        if (str.find("$PhysicalNames") != string::npos)
            break;
        if (mesh_file.eof())
            FatalError("$PhysicalNames tag not found!");
    }

    // Read number of physical groups
    mesh_file >> mesh_ptr->n_bdy;
    mesh_ptr->n_bdy--;              //substract FLUID group
    mesh_file.getline(buf, BUFSIZ); // clear rest of line
    for (int i = 0; i < mesh_ptr->n_bdy + 1; i++)
    {
        mesh_file >> mesh_ptr->n_dims >> bcid >> bc_txt_temp;
        bc_txt_temp.erase(bc_txt_temp.find_last_not_of(" \n\r\t") + 1);
        bc_txt_temp.erase(bc_txt_temp.find_last_not_of("\"") + 1);
        if (bc_txt_temp.find_first_not_of("\"") != 0)
            bc_txt_temp.erase(bc_txt_temp.find_first_not_of("\"") - 1, 1);
        if (bc_txt_temp == "FLUID")
            break;
        if (i == mesh_ptr->n_bdy)
            FatalError("Cant find fluid group in mesh file");
        mesh_file.getline(buf, BUFSIZ); // clear rest of line
    }

    if (mesh_ptr->n_dims != 2 && mesh_ptr->n_dims != 3)
        FatalError("Invalid mesh dimensionality. Expected 2D or 3D.");

    mesh_file.clear();
    mesh_file.seekg(mesh_file.beg);

    // Move cursor to $Nodes
    while (1)
    {
        getline(mesh_file, str);
        if (str.find("$Nodes") != string::npos)
            break;
        if (mesh_file.eof())
            FatalError("$Nodes tag not found!");
    }

    mesh_file >> mesh_ptr->num_verts_global; // total num vertices in mesh

    mesh_file.clear();
    mesh_file.seekg(mesh_file.beg);

    // Move cursor to $Elements
    while (1)
    {
        getline(mesh_file, str);
        if (str.find("$Elements") != string::npos)
            break;
        if (mesh_file.eof())
            FatalError("$Elements tag not found!");
    }

    // Each processor first reads number of global cells
    int n_entities, bcid2;
    // Read number of elements and bdys
    mesh_file >> n_entities;        // num entities in mesh
    mesh_file.getline(buf, BUFSIZ); // clear rest of line

    int icount = 0;

    for (int i = 0; i < n_entities; i++)
    {
        mesh_file >> dummy >> dummy >> dummy;
        mesh_file >> bcid2;

        if (bcid2 == bcid) //if the element belongs to fluid physical group
            icount++;

        mesh_file.getline(buf, BUFSIZ); // clear rest of line
    }
    mesh_ptr->num_cells_global = icount;
    mesh_file.close();
}

void mesh_reader::partial_read_connectivity_gmsh(int kstart, int in_num_cells)
{

    //allocate memory
    mesh_ptr->c2v.setup(in_num_cells, MAX_V_PER_C); // stores the vertices making that cell
    mesh_ptr->c2n_v.setup(in_num_cells);            // stores the number of nodes making that cell
    mesh_ptr->ctype.setup(in_num_cells);            // stores the type of cell
    mesh_ptr->ic2icg.setup(in_num_cells);

    // Initialize arrays to -1
    mesh_ptr->c2v.initialize_to_value(-1);

    int ntags, dummy, bcid; //bcid:id number that hold "FLUID";
    char buf[BUFSIZ] = {""};
    string str, bc_txt_temp;

    //open file
    mesh_file.open(fname.c_str());
    if (!mesh_file)
        FatalError("Unable to open mesh file");

    // Move cursor to $PhysicalNames
    while (1)
    {
        getline(mesh_file, str);
        if (str.find("$PhysicalNames") != string::npos)
            break;
        if (mesh_file.eof())
            FatalError("$PhysicalNames tag not found!");
    }

    // Read number of boundaries and fields defined
    mesh_file >> dummy;
    mesh_file.getline(buf, BUFSIZ); // clear rest of line
    for (int i = 0; i < mesh_ptr->n_bdy + 1; i++)
    {
        mesh_file >> dummy >> bcid >> bc_txt_temp;
        bc_txt_temp.erase(bc_txt_temp.find_last_not_of(" \n\r\t") + 1);
        bc_txt_temp.erase(bc_txt_temp.find_last_not_of("\"") + 1);
        if (bc_txt_temp.find_first_not_of("\"") != 0)
            bc_txt_temp.erase(bc_txt_temp.find_first_not_of("\"") - 1, 1);
        if (bc_txt_temp == "FLUID")
            break;
        mesh_file.getline(buf, BUFSIZ); // clear rest of line
    }

    // Move cursor to $Elements
    mesh_file.clear();
    mesh_file.seekg(0, ios::beg);

    while (1)
    {
        getline(mesh_file, str);
        if (str.find("$Elements") != string::npos)
            break;
        if (mesh_file.eof())
            FatalError("$Elements tag not found!");
    }

    int n_entities, elmtype, bcid2;

    mesh_file >> n_entities;        // num entities in mesh
    mesh_file.getline(buf, BUFSIZ); // clear rest of line

    // Skip elements being read by other processors
    int icount = 0; //index of fluid cell(global index)
    int i = 0;      //index of cell to read(local index)

    // ctype is the element type:  for HiFiLES: 0=tri, 1=quad, 2=tet, 3=prism, 4=hex
    // For Gmsh node ordering, see: http://geuz.org/gmsh/doc/texinfo/gmsh.html#Node-ordering

    for (int k = 0; k < n_entities; k++)
    {
        mesh_file >> dummy >> elmtype >> ntags;
        mesh_file >> bcid2;
        for (int tag = 0; tag < ntags - 1; tag++)//skip tags
            mesh_file >> dummy;

        if (bcid2 == bcid) //if belong to fluid group
        {
            if (icount >= kstart && i < in_num_cells) // if belong to the block to read
            {
                mesh_ptr->ic2icg(i) = icount;
                if (elmtype == 2 || elmtype == 9) // Triangle
                {
                    mesh_ptr->ctype(i) = TRI;
                    if (elmtype == 2) // linear triangle
                    {
                        mesh_ptr->c2n_v(i) = 3;
                        mesh_file >> mesh_ptr->c2v(i, 0) >> mesh_ptr->c2v(i, 1) >> mesh_ptr->c2v(i, 2);
                    }
                    else if (elmtype == 9) // quadratic triangle
                    {
                        mesh_ptr->c2n_v(i) = 6;
                        mesh_file >> mesh_ptr->c2v(i, 0) >> mesh_ptr->c2v(i, 1) >> mesh_ptr->c2v(i, 2) >> mesh_ptr->c2v(i, 3) >> mesh_ptr->c2v(i, 4) >> mesh_ptr->c2v(i, 5);
                    }
                }
                else if (elmtype == 3 || elmtype == 16) // Quad
                {
                    mesh_ptr->ctype(i) = QUAD;
                    if (elmtype == 3) // linear quadrangle
                    {
                        mesh_ptr->c2n_v(i) = 4;
                        mesh_file >> mesh_ptr->c2v(i, 0) >> mesh_ptr->c2v(i, 1) >> mesh_ptr->c2v(i, 3) >> mesh_ptr->c2v(i, 2);
                    }
                    else if (elmtype == 16) // quadratic quadrangle
                    {
                        mesh_ptr->c2n_v(i) = 8;
                        mesh_file >> mesh_ptr->c2v(i, 0) >> mesh_ptr->c2v(i, 1) >> mesh_ptr->c2v(i, 2) >> mesh_ptr->c2v(i, 3) >> mesh_ptr->c2v(i, 4) >> mesh_ptr->c2v(i, 5) >> mesh_ptr->c2v(i, 6) >> mesh_ptr->c2v(i, 7);
                    }
                }
                else if (elmtype == 4 || elmtype == 11) // Tetrahedron
                {
                    mesh_ptr->ctype(i) = TET;
                    if (elmtype == 4) // Linear tet
                    {
                        mesh_ptr->c2n_v(i) = 4;
                        mesh_file >> mesh_ptr->c2v(i, 0) >> mesh_ptr->c2v(i, 1) >> mesh_ptr->c2v(i, 2) >> mesh_ptr->c2v(i, 3);
                    }
                    else if (elmtype == 11) // Quadratic tet
                    {
                        mesh_ptr->c2n_v(i) = 10;
                        mesh_file >> mesh_ptr->c2v(i, 0) >> mesh_ptr->c2v(i, 1) >> mesh_ptr->c2v(i, 2) >> mesh_ptr->c2v(i, 3) >> mesh_ptr->c2v(i, 4);
                        mesh_file >> mesh_ptr->c2v(i, 7) >> mesh_ptr->c2v(i, 5) >> mesh_ptr->c2v(i, 6) >> mesh_ptr->c2v(i, 8) >> mesh_ptr->c2v(i, 9);
                    }
                }
                else if (elmtype == 6 || elmtype == 18) // prisms
                {
                    mesh_ptr->ctype(i) = PRISM;
                    if (elmtype == 6) //linear prism
                    {
                        mesh_ptr->c2n_v(i) = 6; // linear prism
                        mesh_file >> mesh_ptr->c2v(i, 0) >> mesh_ptr->c2v(i, 1) >> mesh_ptr->c2v(i, 2) >> mesh_ptr->c2v(i, 3) >> mesh_ptr->c2v(i, 4) >> mesh_ptr->c2v(i, 5);
                    }
                    else if (elmtype == 18) // 15 points prism
                    {
                        mesh_ptr->c2n_v(i) = 15;
                        mesh_file >> mesh_ptr->c2v(i, 0) >> mesh_ptr->c2v(i, 1) >> mesh_ptr->c2v(i, 2) >> mesh_ptr->c2v(i, 3) >> mesh_ptr->c2v(i, 4) >> mesh_ptr->c2v(i, 5) >> mesh_ptr->c2v(i, 6) >> mesh_ptr->c2v(i, 8) >> mesh_ptr->c2v(i, 9) >> mesh_ptr->c2v(i, 7) >> mesh_ptr->c2v(i, 10) >> mesh_ptr->c2v(i, 11) >> mesh_ptr->c2v(i, 12) >> mesh_ptr->c2v(i, 14) >> mesh_ptr->c2v(i, 13);
                    }
                }
                else if (elmtype == 5 || elmtype == 12) // Hexahedron
                {
                    mesh_ptr->ctype(i) = HEX;
                    if (elmtype == 5) // linear quadrangle
                    {
                        mesh_ptr->c2n_v(i) = 8;
                        mesh_file >> mesh_ptr->c2v(i, 0) >> mesh_ptr->c2v(i, 1) >> mesh_ptr->c2v(i, 3) >> mesh_ptr->c2v(i, 2);
                        mesh_file >> mesh_ptr->c2v(i, 4) >> mesh_ptr->c2v(i, 5) >> mesh_ptr->c2v(i, 7) >> mesh_ptr->c2v(i, 6);
                    }
                    else if (elmtype == 17) // 20-node quadratic hexahedron
                    {
                        mesh_ptr->c2n_v(i) = 20;
                        mesh_file >> mesh_ptr->c2v(i, 0) >> mesh_ptr->c2v(i, 1) >> mesh_ptr->c2v(i, 2) >> mesh_ptr->c2v(i, 3) >> mesh_ptr->c2v(i, 4) >> mesh_ptr->c2v(i, 5) >> mesh_ptr->c2v(i, 6) >> mesh_ptr->c2v(i, 7) >> mesh_ptr->c2v(i, 8) >> mesh_ptr->c2v(i, 11) >> mesh_ptr->c2v(i, 12) >> mesh_ptr->c2v(i, 9) >> mesh_ptr->c2v(i, 13) >> mesh_ptr->c2v(i, 10) >> mesh_ptr->c2v(i, 14) >> mesh_ptr->c2v(i, 15) >> mesh_ptr->c2v(i, 16) >> mesh_ptr->c2v(i, 19) >> mesh_ptr->c2v(i, 17) >> mesh_ptr->c2v(i, 18);
                    }
                }
                else
                {
                    cout << "elmtype=" << elmtype << endl;
                    FatalError("element type not recognized");
                }

                // Shift every values of c2v by -1 to be 0 based with -1 as nan
                for (int k = 0; k < mesh_ptr->c2n_v(i); k++)
                        mesh_ptr->c2v(i, k)--;

                i++;
                mesh_file.getline(buf, BUFSIZ); // skip end of line
            }
            else //
            {
                mesh_file.getline(buf, BUFSIZ); // skip line, cell doesn't belong to this processor
            }
            icount++; // FLUID cell, increase icount
        }
        else // Not FLUID cell, skip line
        {
            mesh_file.getline(buf, BUFSIZ); // skip line, cell doesn't belong to this processor
        }

    } // End of loop over entities

    mesh_file.close();
}

void mesh_reader::read_vertices_gmsh()
{
    string str;
    char buf[BUFSIZ] = {""};

    mesh_file.open(fname.c_str());

    if (!mesh_file)
        FatalError("Could not open mesh file");

    // Move cursor to $Nodes
    while (1)
    {
        getline(mesh_file, str);
        if (str.find("$Nodes") != string::npos)
            break;
        if (mesh_file.eof())
            FatalError("$Nodes tag not found!");
    }

    mesh_file.getline(buf, BUFSIZ); //skip total vertex number

    int id;
    int index;
    mesh_ptr->xv.setup(mesh_ptr->num_verts, mesh_ptr->n_dims);

    for (int i = 0; i < mesh_ptr->num_verts_global; i++)
    {
        mesh_file >> id;//global id
        index = index_locate_int(id - 1, mesh_ptr->iv2ivg.get_ptr_cpu(), mesh_ptr->num_verts);//local index

        if (index != -1) // Vertex belongs to this processor
            for (int m = 0; m < mesh_ptr->n_dims; m++)
                mesh_file >> mesh_ptr->xv(index, m);
        mesh_file.getline(buf, BUFSIZ);
    }

    mesh_file.close();
}

void mesh_reader::read_boundary_gmsh(void)
{
    string str;

    mesh_file.open(fname.c_str());
    if (!mesh_file)
        FatalError("Unable to open mesh file");

    // Move cursor to $PhysicalNames
    while (1)
    {
        getline(mesh_file, str);
        if (str.find("$PhysicalNames") != string::npos)
            break;
        if (mesh_file.eof())
            FatalError("$PhysicalNames tag not found!");
    }

    // Read number of boundaries and fields defined
    int id, dummy, n_bcs;
    int elmtype, ntags, bcid, bcdim;
    char buf[BUFSIZ] = {""};
    string bc_txt_temp;
    int fluid_id; //id of fluid group

    mesh_ptr->bc_id.setup(mesh_ptr->num_cells, MAX_F_PER_C);
    mesh_ptr->bc_id.initialize_to_value(-1);//-1 as default internal face
    run_input.bc_list.setup(mesh_ptr->n_bdy);

    mesh_file >> n_bcs;
    map<int,int> temp_bcid;
    mesh_file.getline(buf, BUFSIZ); // clear rest of line
    int bc_counter = 0;

    for (int i = 0; i < n_bcs; i++)//read physical groups
    {
        mesh_file >> bcdim >> bcid >> bc_txt_temp;//bcid is 1 based
        mesh_file.getline(buf, BUFSIZ); //clear the rest of line
        bc_txt_temp.erase(bc_txt_temp.find_last_not_of(" \n\r\t") + 1);
        bc_txt_temp.erase(bc_txt_temp.find_last_not_of("\"") + 1);
        if(bc_txt_temp.find_first_not_of("\"")!=0)
                bc_txt_temp.erase(bc_txt_temp.find_first_not_of("\"")-1,1);
        if (bc_txt_temp != "FLUID")
        {
            run_input.bc_list(bc_counter).setup(bc_txt_temp);
            temp_bcid[bcid] = bc_counter;
            bc_counter++;
        }
        else
        {
            fluid_id = bcid;
        }
    }

    mesh_file.clear();
    mesh_file.seekg(mesh_file.beg);

    // Move cursor to $Elements
    while (1)
    {
        getline(mesh_file, str);
        if (str.find("$Elements") != string::npos)
            break;
        if (mesh_file.eof())
            FatalError("$Elements tag not found!");
    }

    // Each processor reads number of entities
    int n_entities;
    // Read number of elements and bdys
    mesh_file >> n_entities;        // num cells in mesh
    mesh_file.getline(buf, BUFSIZ); // clear rest of line
    
    int num_v_per_f;
    int num_face_vert;
    hf_array<int> vlist_bound,vlist_cell;

    for (int i = 0; i < n_entities; i++) //bc_face list
    {
        mesh_file >> id >> elmtype >> ntags;
        mesh_file >> bcid;
        for (int tag = 0; tag < ntags - 1; tag++)//skip tags
            mesh_file >> dummy;

        if (bcid == fluid_id)//belong to fluid,skip
        {
            mesh_file.getline(buf, BUFSIZ); // skip line
            continue;
        }

        if (elmtype == 1 || elmtype == 8) // first and second order Edge
        {
            num_face_vert = 2;
            vlist_bound.setup(num_face_vert);
            // Read the two vertices
            mesh_file >> vlist_bound(0) >> vlist_bound(1);
        }
        else if (elmtype == 3 || elmtype == 16) // first and second order Quad face
        {
            num_face_vert = 4;
            vlist_bound.setup(num_face_vert);

            mesh_file >> vlist_bound(0) >> vlist_bound(1) >> vlist_bound(2) >> vlist_bound(3);
        }
        else if (elmtype == 2 || elmtype == 9) // first and second order tri face
        {
            num_face_vert = 3;
            vlist_bound.setup(num_face_vert);

            mesh_file >> vlist_bound(0) >> vlist_bound(1) >> vlist_bound(2);
        }
        else
        {
            cout << "Gmsh boundary element type: " << elmtype << endl;
            FatalError("Boundary elmtype not recognized");
        }

        // Shift by -1 (1-indexed -> 0-indexed)
        for (int j = 0; j < num_face_vert; j++)
        {
            vlist_bound(j)--;
        }

        mesh_file.getline(buf, BUFSIZ); // Get rest of line

        // Check if all vertices belong to processor
        bool belong_to_proc = true;
        for (int j = 0; j < num_face_vert; j++)
        {
            vlist_bound(j) = index_locate_int(vlist_bound(j), mesh_ptr->iv2ivg.get_ptr_cpu(), mesh_ptr->num_verts);
            if (vlist_bound(j) == -1)
            {
                belong_to_proc = false;
                break;
            }
        }

        if (belong_to_proc)
        {
            // All vertices on face belong to processor
            // Try to find the cell that they belong to
            // loop over the vertex on the face to find out the common cell they share
            vector<int> intersection = mesh_ptr->v2c(vlist_bound(0));
            std::vector<int>::iterator it_intersect;
            for (int i = 1; i < num_face_vert; i++)
            {
                it_intersect = set_intersection(mesh_ptr->v2c(vlist_bound(i)).begin(), mesh_ptr->v2c(vlist_bound(i)).end(), intersection.begin(), intersection.end(), intersection.begin()); //get the intersection of 2 sorted data
                intersection.resize(it_intersect - intersection.begin());
            }
            if (intersection.size() == 1) //only cell
            {
                for (int k = 0; k < mesh_ptr->num_f_per_c(mesh_ptr->ctype(intersection[0])); k++)
                {
                    // Get local vertices of local face k of cell ic
                    num_v_per_f = mesh_ptr->get_corner_vlist_face(intersection[0], k, vlist_cell);

                    if (num_v_per_f == num_face_vert)
                    {
                        if (mesh_ptr->compare_faces_boundary(vlist_bound, vlist_cell))
                        {
                            mesh_ptr->bc_id(intersection[0], k) = temp_bcid.find(bcid)->second;
                            break;
                        }
                    }
                }
            }
            else if (intersection.size() > 1)
            {
                FatalError("2 cell sharing a boundary face");
            }
            else
            {
                cout << "vlist_bound(2)=" << vlist_bound(2) << " vlist_bound(3)=" << vlist_bound(3) << endl;
                FatalError("All nodes of boundary face belong to processor but could not find the coresponding faces");
            }

            } // If all vertices belong to processor

        } // Loop over entities

        mesh_file.close();
    }
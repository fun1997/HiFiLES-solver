/*! \class param_reader
 *  \brief Simple, robust method for reading input files
 *  \author Jacob Crabill
 *  \date 4/30/2015
 */
#pragma once
#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include "hf_array.h"
#include "error.h"

class param_reader
{
public:
    /*! Default constructor */

    param_reader();

    param_reader(string fileName);

    /*! Default destructor */
    ~param_reader();

    /*! Set the file to be read from */
    void setFile(string fileName);

    /*! Open the file to prepare for reading simulation parameters */
    void openFile(void);

    /*! Close the file & clean up */
    void closeFile(void);

    /* === Functions to read paramters from input file === */

    /*! Read a single value from the input file; if not found, apply a default value */
    template <typename T>
    void getScalarValue(string optName, T &opt, T defaultVal);

    /*! Read a single value from the input file; if not found, throw an error and exit */
    template <typename T>
    void getScalarValue(string optName, T &opt);

    /*! Read a vector of values from the input file; if not found, apply the default value to all elements */
    template <typename T>
    void getVectorValue(string optName, vector<T> &opt, T defaultVal);

    /*! Read a vector of values from the input file; if not found, throw an error and exit */
    template <typename T>
    void getVectorValue(string optName, vector<T> &opt);

    template <typename T>
    void getVectorValue(string optName, hf_array<T> &opt);

    /*! Read a vector of values from the input file; if not found, setup vector to size 0 and continue */
    template <typename T>
    void getVectorValueOptional(string optName, vector<T> &opt);

    template <typename T>
    void getVectorValueOptional(string optName, hf_array<T> &opt);

    /*! Read in a map of type <T,U> from input file; each entry prefaced by optName */
    template <typename T, typename U>
    void getMap(string optName, map<T, U> &opt);

    
private:
    ifstream optFile;
    string fileName;

};

template<typename T>
void param_reader::getScalarValue(string optName, T &opt, T defaultVal)
{
    string str, optKey;

    if (!optFile.is_open() || !getline(optFile,str))
    {
        optFile.open(fileName.c_str());
        if (!optFile.is_open())
            FatalError("Cannont open input file for reading.");
    }

    // Rewind to the start of the file
    optFile.clear();
    optFile.seekg(0,optFile.beg);

    // Search for the given option string
    while (getline(optFile,str))
    {
        // Remove any leading whitespace & see if first word is the input option
        stringstream ss;
        ss.str(str);
        ss >> optKey;
        if (optKey.compare(optName)==0)
        {
            if (!(ss >> opt))
            {
                // This could happen if, for example, trying to assign a string to a double
                cout << "WARNING: Unable to assign value to option " << optName << endl;
                cout << "Using default value of " << defaultVal << " instead." << endl;
                opt = defaultVal;
            }

            return;
        }
    }

    opt = defaultVal;
}

template<typename T>
void param_reader::getScalarValue(string optName, T &opt)
{
    string str, optKey;

    if (!optFile.is_open())
    {
        optFile.open(fileName.c_str());
        if (!optFile.is_open())
            FatalError("Cannont open input file for reading.");
    }

    // Rewind to the start of the file
    optFile.clear();
    optFile.seekg(0,optFile.beg);

    // Search for the given option string
    while (getline(optFile,str))
    {
        // Remove any leading whitespace & see if first word is the input option
        stringstream ss;
        ss.str(str);
        ss >> optKey;

        if (optKey.compare(optName)==0)
        {
            if (!(ss >> opt))
            {
                // This could happen if, for example, trying to assign a string to a double
                cerr << "WARNING: Unable to assign value to option " << optName << endl;
                string errMsg = "Required option not set: " + optName;
                FatalError(errMsg.c_str())
            }

            return;
        }
    }

    // Option was not found; throw error & exit
    string errMsg = "Required option not found: " + optName;
    FatalError(errMsg.c_str())
}

template<typename T, typename U>
void param_reader::getMap(string optName, map<T,U> &opt)
{
    string str, optKey;
    T tmpT;
    U tmpU;
    bool found;


    if (!optFile.is_open())
    {
        optFile.open(fileName.c_str());
        if (!optFile.is_open())
            FatalError("Cannont open input file for reading.");
    }

    // Rewind to the start of the file
    optFile.clear();
    optFile.seekg(0,optFile.beg);

    // Search for the given option string
    while (getline(optFile,str))
    {
        // Remove any leading whitespace & see if first word is the input option
        stringstream ss;
        ss.str(str);
        ss >> optKey;
        if (optKey.compare(optName)==0)
        {
            found = true;
            if (!(ss >> tmpT >> tmpU))
            {
                // This could happen if, for example, trying to assign a string to a double
                cerr << "WARNING: Unable to assign value to option " << optName << endl;
                string errMsg = "Required option not set: " + optName;
                FatalError(errMsg.c_str())
            }

            opt[tmpT] = tmpU;
            optKey = "";
        }
    }

    if (!found)
    {
        // Option was not found; throw error & exit
        string errMsg = "Required option not found: " + optName;
        FatalError(errMsg.c_str())
    }

}

template<typename T>
void param_reader::getVectorValue(string optName, vector<T> &opt)
{
    string str, optKey;

    if (!optFile.is_open())
    {
        optFile.open(fileName.c_str());
        if (!optFile.is_open())
            FatalError("Cannont open input file for reading.");
    }

    // Rewind to the start of the file
    optFile.clear();
    optFile.seekg(0,optFile.beg);

    // Search for the given option string
    while (getline(optFile,str))
    {
        // Remove any leading whitespace & see if first word is the input option
        stringstream ss;
        ss.str(str);
        ss >> optKey;
        if (optKey.compare(optName)==0)
        {
            int nVals;
            if (!(ss >> nVals))
            {
                // This could happen if, for example, trying to assign a string to a double
                cerr << "WARNING: Unable to read number of entries for vector option " << optName << endl;
                string errMsg = "Required option not set: " + optName;
                FatalError(errMsg.c_str());
            }

            opt.resize(nVals);
            for (int i=0; i<nVals; i++)
            {
                if (!ss >> opt[i])
                {
                    cerr << "WARNING: Unable to assign all values to vector option " << optName << endl;
                    string errMsg = "Required option not set: " + optName;
                    FatalError(errMsg.c_str())
                }
            }

            return;
        }
    }

    // Option was not found; throw error & exit
    string errMsg = "Required option not found: " + optName;
    FatalError(errMsg.c_str())
}

template<typename T>
void param_reader::getVectorValue(string optName, hf_array<T> &opt)
{
    string str, optKey;

    if (!optFile.is_open())
    {
        optFile.open(fileName.c_str());
        if (!optFile.is_open())
            FatalError("Cannont open input file for reading.");
    }

    // Rewind to the start of the file
    optFile.clear();
    optFile.seekg(0,optFile.beg);

    // Search for the given option string
    while (getline(optFile,str))
    {
        // Remove any leading whitespace & see if first word is the input option
        stringstream ss;
        ss.str(str);
        ss >> optKey;
        if (optKey.compare(optName)==0)
        {
            int nVals;
            if (!(ss >> nVals))
            {
                // This could happen if, for example, trying to assign a string to a double
                cerr << "WARNING: Unable to read number of entries for vector option " << optName << endl;
                string errMsg = "Required option not set: " + optName;
                FatalError(errMsg.c_str());
            }

            opt.setup(nVals);
            for (int i=0; i<nVals; i++)
            {
                if (!(ss >> opt(i)))
                {
                    cerr << "WARNING: Unable to assign all values to vector option " << optName << endl;
                    string errMsg = "Required option not set: " + optName;
                    FatalError(errMsg.c_str());
                }
            }

            return;
        }
    }

    // Option was not found; throw error & exit
    string errMsg = "Required option not found: " + optName;
    FatalError(errMsg.c_str())
}

template<typename T>
void param_reader::getVectorValueOptional(string optName, hf_array<T> &opt)
{
    string str, optKey;

    if (!optFile.is_open())
    {
        optFile.open(fileName.c_str());
        if (!optFile.is_open())
            FatalError("Cannont open input file for reading.");
    }

    // Rewind to the start of the file
    optFile.clear();
    optFile.seekg(0,optFile.beg);

    // Search for the given option string
    while (getline(optFile,str))
    {
        // Remove any leading whitespace & see if first word is the input option
        stringstream ss;
        ss.str(str);
        ss >> optKey;
        if (optKey.compare(optName)==0)
        {
            int nVals;
            if (!(ss >> nVals))
            {
                // This could happen if, for example, trying to assign a string to a double
                cerr << "WARNING: Unable to read number of entries for vector option " << optName << endl;
                cerr << "Option not set: " << optName << endl;
                opt.setup(0);
                return;
            }

            opt.setup(nVals);
            for (int i=0; i<nVals; i++)
            {
                if (!(ss >> opt(i)))
                {
                    cerr << "WARNING: Unable to assign all values to vector option " << optName << endl;
                    cerr << "Option not set: " << optName << endl;
                    opt.setup(0);
                    return;
                }
            }

            return;
        }
    }

    // Option was not found; setup hf_array to size 0
    opt.setup(0);
}

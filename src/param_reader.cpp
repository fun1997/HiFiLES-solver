#include "../include/param_reader.h"

using namespace std;

param_reader::param_reader()
{
    
}

param_reader::param_reader(string fileName)
{
    this->fileName = fileName;
}

param_reader::~param_reader()
{
    if (optFile.is_open()) optFile.close();
}

void param_reader::setFile(string fileName)
{
    this->fileName = fileName;
}

void param_reader::openFile(void)
{
    optFile.open(fileName.c_str(), ifstream::in);
}

void param_reader::closeFile()
{
    optFile.close();
}

# HiFiLES

This repository is an inhouse version of HiFiLES (see https://github.com/HiFiLES/HiFiLES-solver/) by [Theoretical Fluid Dynamics and Turbulence Group](https://faculty.eng.ufl.edu/fluids/) at University of Florida. This code implements some new features and fixes critical bugs in addition to the original code. The purpose of this code is to create a suitable tool for supersonic jet aeroacoustics simulations on mixed element unstructured mesh. The new features include:

- New boundary condition implementation (support multiple BCs of same type with different parameters)
- New initial conditions (patch, shock, complex vortex)
- Second and third order Low storage SSP-RK
- Polynomial de-aliasing
- Shock capturing
- HLLC and RoeM Riemann Solvers
- New wall models
- Parallel Data I/O based on HDF5 and CGNS

For more information, go to [Wiki](https://github.com/weiqishen/HiFiLES-solver/wiki).

## Getting Started
 	
### Dependencies

This code has been tested on WSL in Windows 10, Ubuntu, and RedHat Enterprise. Minimum serial build does not require any external dependencies. However, we suggest build the code with the following dependencies to get the maximum performance

```
ParMETIS
HDF5
CGNS
BLAS
OpenMPI
```

### Build

```
git clone https://github.com/weiqishen/HiFiLES-solver.git
cd HiFiLES-solver
cmake .
ccmake ..
make
```

## Testcases

The testcases are in the folder ```HiFiLES-solver/testcases```. References of the input file options can be found in [Wiki](https://github.com/weiqishen/HiFiLES-solver/wiki).

## Author

Weiqi Shen
Email: weiqishen1994@ufl.edu
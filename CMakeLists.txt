cmake_minimum_required(VERSION 3.10)
project(HiFiLES CXX)

include_directories("${PROJECT_BINARY_DIR}/include")
SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

## Build type ##
if(NOT CMAKE_BUILD_TYPE) # force set build type in cache to debug
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING
      "Choose the type of build, options are: Debug Release
RelWithDebInfo MinSizeRel."
      FORCE)
endif(NOT CMAKE_BUILD_TYPE)
SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g -ggdb -D_DEBUG" CACHE STRING "Flags used by the linker during debug builds" FORCE)
SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3" CACHE STRING "Flags used by the linker during release builds" FORCE)

## Building options ##
set(PARALLEL OFF CACHE BOOL "Build with MPI support")
set(BLAS "MKL" CACHE STRING "Build with external BLAS support, 'MKL', 'CBLAS', 'ATLAS','ACCELERATE' or 'NO'")
set(USE_CGNS ON CACHE BOOL "Build with CGNS support")
set(USE_HDF5 ON CACHE BOOL "Build with HDF5 support")
if(${USE_HDF5})
      set(USE_ZLIB ON CACHE BOOL "Use Zlib with HDF5")
endif()
## Dependencies ##

#BLAS
if (NOT(${BLAS} STREQUAL "NO"))# if use blas
      set(BLAS_INCLUDE "/opt" CACHE PATH "path to BLAS include")
      set(BLAS_LD "/opt" CACHE PATH "path to BLAS library")
      if (${BLAS} STREQUAL "MKL")
            set(BLAS_LIB mkl_intel_lp64 mkl_sequential mkl_core pthread m dl)
            add_definitions(-D_MKL_BLAS)
      elseif(${BLAS} STREQUAL "CBLAS")
            set(BLAS_LIB cblas blas gfortran)
            add_definitions(-D_STANDARD_BLAS)
      elseif(${BLAS} STREQUAL "ATLAS")
            set(BLAS_LIB f77blas cblas atlas)
            add_definitions(-D_STANDARD_BLAS)
      elseif(${BLAS} STREQUAL "ACCELERATE")
            set(BLAS_LIB -framework Accelerate)
            add_definitions(-flax-vector-conversions -D_ACCELERATE_BLAS)
      endif()
      include_directories(${BLAS_INCLUDE})
      set(CXX_LD ${CXX_LD} ${BLAS_LD})
      set(CXX_LIB ${CXX_LIB} ${BLAS_LIB})
endif()

#CGNS
if (${USE_CGNS})
      set(CGNS_INCLUDE "/opt" CACHE PATH "path to cgns include")
      set(CGNS_LD "/opt" CACHE PATH "path to cgns library")
      include_directories(${CGNS_INCLUDE})
      add_definitions(-D_CGNS)
      set(CXX_LD ${CXX_LD} ${CGNS_LD})
      set(CXX_LIB ${CXX_LIB} cgns)
endif()

#HDF5
if (${USE_HDF5})
      set(HDF5_INCLUDE "/opt" CACHE PATH "path to hdf5 include")
      set(HDF5_LD "/opt" CACHE PATH "path to hdf5 library")
      include_directories(${HDF5_INCLUDE})
      add_definitions(-D_HDF5)
      set(CXX_LD ${CXX_LD} ${HDF5_LD})
      set(CXX_LIB ${CXX_LIB} hdf5 m dl)
      if(${USE_ZLIB})
            find_package(ZLIB)
            if(${ZLIB_FOUND})
                  set(CXX_LIB ${CXX_LIB} ${ZLIB_LIBRARIES}) 
            else()
                  message(SEND_ERROR "Cannot find zlib.")
            endif()
      endif()
endif()

#source file list
set(SRCLIST 
./src/global.cpp 
./src/param_reader.cpp 
./src/input.cpp 
./src/bc.cpp 
./src/mesh_reader.cpp 
./src/probe_input.cpp 
./src/flux.cpp 
./src/source.cpp 
./src/cubature_tet.cpp 
./src/cubature_hexa.cpp
./src/cubature_quad.cpp
./src/cubature_tri.cpp 
./src/cubature_1d.cpp 
./src/cubature_pris.cpp
./src/funcs.cpp 
./src/wall_model_funcs.cpp
./src/inters.cpp 
./src/bdy_inters.cpp 
./src/int_inters.cpp 
./src/eles.cpp 
./src/eles_tris.cpp 
./src/eles_quads.cpp 
./src/eles_tets.cpp 
./src/eles_hexas.cpp 
./src/eles_pris.cpp 
./src/output.cpp 
./src/geometry.cpp 
./src/solver.cpp 
./src/mesh.cpp 
./src/HiFiLES.cpp)


#MPI
if(${PARALLEL})
      find_package(MPI) 
      add_definitions(-D_MPI)
      set(METIS_INCLUDE "/opt" CACHE PATH "path to metis include")
      set(METIS_LD "/opt" CACHE PATH "path to metis library")
      set(PARMETIS_INCLUDE "/opt" CACHE PATH "path to parmetis include")
      set(PARMETIS_LD "/opt" CACHE PATH "path to parmetis library")
      if(${MPI_FOUND})
            include_directories(${MPI_CXX_INCLUDE_PATH} ${METIS_INCLUDE} ${PARMETIS_INCLUDE})
            set(CXX_LD ${CXX_LD} ${PARMETIS_LD} ${METIS_LD})
            set(CXX_LIB ${CXX_LIB} ${MPI_CXX_LIBRARIES} parmetis metis)
            set (SRCLIST ${SRCLIST} ./src/mpi_inters.cpp)
      else()
            message(SEND_ERROR  "Cannot find MPI library, please specify manually.")
      endif()
endif()

# use C++14 and cpu option
add_definitions(-std=c++14 -D_CPU)

# Output building strings
message("Build summary:")
message("Build type: ${CMAKE_BUILD_TYPE}")
message("BLAS support: ${BLAS}")
if(${PARALLEL})
      message("Compiler: ${MPI_CXX_COMPILER}")
else()
      message("Compiler: ${CMAKE_CXX_COMPILER}")
endif()
message("Linker flags: ${CXX_LD}")
message("Libraries: ${CXX_LIB}")

#build
LINK_DIRECTORIES(${CXX_LD})
add_executable(HiFiLES ${SRCLIST})
target_link_libraries(HiFiLES PRIVATE ${CXX_LIB})

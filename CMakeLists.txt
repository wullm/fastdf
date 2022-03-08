cmake_minimum_required(VERSION 3.11)
project(fastdf VERSION 1.0.0 DESCRIPTION "Code for neutrino particle initial conditions")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# The root directory of the CLASS installation.
# The default path should work when CLASS and FastDF are both loaded as
# submodules for monofonIC. If FastDF is used as a stand-alone code, use
# the more general autotools configuration script on the main branch.
set(SEARCH_CLASS_PATH "../class-src/" CACHE STRING "Root directory of CLASS installation")
message(STATUS "SEARCH_CLASS_PATH = ${SEARCH_CLASS_PATH}")

# The CLASS include directories (as of CLASS v3.1.2)
set(CLASS_INCLUDES )
list(APPEND CLASS_INCLUDES ${SEARCH_CLASS_PATH}/include)
list(APPEND CLASS_INCLUDES ${SEARCH_CLASS_PATH}/external/heating)
list(APPEND CLASS_INCLUDES ${SEARCH_CLASS_PATH}/external/HyRec2020)
list(APPEND CLASS_INCLUDES ${SEARCH_CLASS_PATH}/external/RecfastCLASS)

# OpenMP
find_package(OpenMP REQUIRED)
if(OPENMP_FOUND)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

# mpi support
option(ENABLE_MPI "Enable MPI support" ON)
if(ENABLE_MPI)
  find_package(MPI)
  if(MPI_C_FOUND)
    message(STATUS "MPI_C_COMPILER = ${MPI_C_COMPILER}")
    message(STATUS "MPI_C_INCLUDE_PATH = ${MPI_C_INCLUDE_PATH}")
    message(STATUS "MPI_C_LIBRARIES = ${MPI_C_LIBRARIES}")
  endif(MPI_C_FOUND)
endif(ENABLE_MPI)

find_package(GSL REQUIRED)
mark_as_advanced(pkgcfg_lib_GSL_gsl pkgcfg_lib_GSL_gslcblas pkgcfg_lib_GSL_m)

set(HDF5_PREFER_PARALLEL ON)
find_package(HDF5 REQUIRED)
message(STATUS "HDF5_LIBRARIES = ${HDF5_LIBRARIES}")
message(STATUS "HDF5_INCLUDE_DIRS = ${HDF5_INCLUDE_DIRS}")
message(STATUS "HDF5_IS_PARALLEL = ${HDF5_IS_PARALLEL}")

file(GLOB fastdf_SOURCES src/*.c parser/*.c)

# Static library
add_library(fastdf_static STATIC ${fastdf_SOURCES})
add_library(fastdf::fastdf_static ALIAS fastdf_static)
set_target_properties(fastdf_static PROPERTIES OUTPUT_NAME fastdf)
set_target_properties(fastdf_static PROPERTIES C_STANDARD 11)
set_property(TARGET fastdf_static PROPERTY POSITION_INDEPENDENT_CODE ON)
target_link_libraries(fastdf_static PRIVATE m)
target_link_libraries(fastdf_static PRIVATE GSL::gsl)
target_include_directories(fastdf_static PUBLIC include)
target_link_libraries(fastdf_static PRIVATE ${HDF5_LIBRARIES})
target_include_directories(fastdf_static PRIVATE ${HDF5_INCLUDE_DIRS})
target_compile_definitions(fastdf_static PRIVATE "USE_HDF5")
target_include_directories(fastdf_static PRIVATE ${CLASS_INCLUDES})
target_compile_definitions(fastdf_static PRIVATE "WITH_CLASS")
target_compile_definitions(fastdf_static PRIVATE _SVID_SOURCE PORTABLE_STRNICMP)

# mpi flags
if(MPI_C_FOUND)
  target_link_libraries(fastdf_static PRIVATE MPI::MPI_C)
  target_compile_definitions(fastdf_static PRIVATE "USE_MPI")
  target_include_directories(fastdf_static PRIVATE ${MPI_C_INCLUDE_PATH})
endif(MPI_C_FOUND)
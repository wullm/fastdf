cmake_minimum_required(VERSION 3.11)
project(fastdf VERSION 1.0.0 DESCRIPTION "Code for neutrino particle initial conditions")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

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

file(GLOB fastdf_SOURCES src/*.c)

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

# Shared library
add_library(fastdf_shared SHARED ${fastdf_SOURCES})
add_library(fastdf::fastdf_shared ALIAS fastdf_shared)
set_target_properties(fastdf_shared PROPERTIES OUTPUT_NAME fastdf)
set_target_properties(fastdf_shared PROPERTIES C_STANDARD 11)
set_property(TARGET fastdf_shared PROPERTY POSITION_INDEPENDENT_CODE ON)
target_link_libraries(fastdf_shared PRIVATE m)
target_link_libraries(fastdf_shared PRIVATE GSL::gsl)
target_include_directories(fastdf_shared PUBLIC include)
target_link_libraries(fastdf_shared PRIVATE ${HDF5_LIBRARIES})
target_include_directories(fastdf_shared PRIVATE ${HDF5_INCLUDE_DIRS})
target_compile_definitions(fastdf_shared PRIVATE "USE_HDF5")


# mpi flags
if(MPI_C_FOUND)
  if(CODE_PRECISION STREQUAL "FLOAT")
    if(FFTW3_SINGLE_MPI_FOUND)
      target_link_libraries(fastdf_static PRIVATE FFTW3::FFTW3_SINGLE_MPI)
      target_compile_definitions(fastdf_static PRIVATE "USE_FFTW_MPI")
      target_link_libraries(fastdf_shared PRIVATE FFTW3::FFTW3_SINGLE_MPI)
      target_compile_definitions(fastdf_shared PRIVATE "USE_FFTW_MPI")
    else()
      message(SEND_ERROR "MPI enabled but FFTW3 library not found with MPI support for single precision!")
    endif()
  elseif(CODE_PRECISION STREQUAL "DOUBLE")
    if(FFTW3_DOUBLE_MPI_FOUND)
      target_link_libraries(fastdf_static PRIVATE FFTW3::FFTW3_DOUBLE_MPI)
      target_compile_definitions(fastdf_static PRIVATE "USE_FFTW_MPI")
      target_link_libraries(fastdf_shared PRIVATE FFTW3::FFTW3_DOUBLE_MPI)
      target_compile_definitions(fastdf_shared PRIVATE "USE_FFTW_MPI")
    else()
      message(SEND_ERROR "MPI enabled but FFTW3 library not found with MPI support for double precision!")
    endif()
  elseif(CODE_PRECISION STREQUAL "LONGDOUBLE")
    if(FFTW3_LONGDOUBLE_MPI_FOUND)
      target_link_libraries(fastdf_static PRIVATE FFTW3::FFTW3_LONGDOUBLE_MPI)
      target_compile_definitions(fastdf_static PRIVATE "USE_FFTW_MPI")
      target_link_libraries(fastdf_shared PRIVATE FFTW3::FFTW3_LONGDOUBLE_MPI)
      target_compile_definitions(fastdf_shared PRIVATE "USE_FFTW_MPI")
    else()
      message(SEND_ERROR "MPI enabled but FFTW3 library not found with MPI support for long double precision!")
    endif()
  endif()

  target_link_libraries(fastdf_static PRIVATE MPI::MPI_C)
  target_compile_definitions(fastdf_static PRIVATE "USE_MPI")
  target_include_directories(fastdf_static PRIVATE ${MPI_C_INCLUDE_PATH})
  target_link_libraries(fastdf_shared PRIVATE MPI::MPI_C)
  target_compile_definitions(fastdf_shared PRIVATE "USE_MPI")
  target_include_directories(fastdf_shared PRIVATE ${MPI_C_INCLUDE_PATH})
  
  message(STATUS "HERE")
endif(MPI_C_FOUND)


#add_executable(fastdf ${fastdf_SOURCES})
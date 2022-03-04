AC_INIT([fastdf], [2022.1.1], [whe@willemelbers.com])
AM_INIT_AUTOMAKE([subdir-objects])
AC_CONFIG_FILES([Makefile src/Makefile])
AC_PROG_CC

# Add local macro collection.
AC_CONFIG_MACRO_DIR([m4])

# Check for MPI
AX_MPI([CC="$MPICC" AC_DEFINE(HAVE_MPI, 1, [Define if you have the MPI library.]) ])

# Check for FFTW
AC_CHECK_LIB([fftw3], [fftw_plan_dft])
AC_CHECK_LIB([fftw3_mpi], [fftw_mpi_init])
AC_CHECK_LIB([gsl], [gsl_integration_qag])
AC_SEARCH_LIBS([sqrt], [m])

# Check for HDF5
AX_LIB_HDF5([parallel])
# Make sure to use it
LIBS="$LIBS -lhdf5"

AC_OUTPUT

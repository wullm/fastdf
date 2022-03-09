AC_INIT([fastdf], [2022.1.1], [whe@willemelbers.com])
AM_INIT_AUTOMAKE([subdir-objects])
AC_CONFIG_FILES([Makefile class/Makefile])
AC_PROG_CC
AM_PROG_CC_C_O

LT_INIT

# Add local macro collection.
AC_CONFIG_MACRO_DIR([m4])

# Check for MPI
AX_MPI([CC="$MPICC" AC_DEFINE(HAVE_MPI, 1, [Define if you have the MPI library.]) ])

# Check for FFTW
AC_CHECK_LIB([fftw3], [fftw_plan_dft])
AC_CHECK_LIB([fftw3_mpi], [fftw_mpi_init])
AC_CHECK_LIB([gslcblas], [cblas_dgemm])
AC_CHECK_LIB([gsl], [gsl_integration_qag])
AC_SEARCH_LIBS([sqrt], [m])

# Check for HDF5
AX_LIB_HDF5([parallel])
# Make sure to use it
LIBS="$LIBS -lhdf5"

# Check for CLASS.
have_class="no"
AC_ARG_WITH([class],
    [AS_HELP_STRING([--with-class=PATH],
       [root directory where CLASS is installed @<:@path/no@:>@]
    )],
    [with_class="$withval"],
    [with_class="no"]
)

AM_CONDITIONAL([WITH_CLASS], [test "$with_class" != "no"])

if test "$with_class" != "no"; then
    CLASS_PATH=$with_class
    CLASS_LIBS="class/libclass.so"
    CLASS_OBJS="$CLASS_PATH/build"
    CLASS_INCS="-I$CLASS_PATH/include -I$CLASS_PATH/external/HyRec2020 -I$CLASS_PATH/external/RecfastCLASS -I$CLASS_PATH/external/heating"
    AC_SUBST([CLASS_PATH]) # Subsitute Makefile definition
    AC_SUBST([CLASS_OBJS]) # Subsitute Makefile definition
    AC_SUBST([CLASS_LIBS]) # Subsitute Makefile definition
    AC_SUBST([CLASS_INCS]) # Subsitute Makefile definition
    AC_MSG_RESULT([
    Compiling with CLASS at: $CLASS_PATH
    ])
    AC_DEFINE(WITH_CLASS, 1) # C definition
fi

AC_OUTPUT

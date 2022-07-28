# Automake stuff
ACLOCAL_AMFLAGS = -I m4

# The main libraries
AM_CFLAGS = $(HDF5_CPPFLAGS)  $(FFTW_INCS) $(GSL_INCS) $(CLASS_INCS) -fopenmp
AM_LDFLAGS = $(HDF5_LDFLAGS) $(FFTW_LIBS) $(CLASS_LIBS)

# The main program
bin_PROGRAMS = fastdf
fastdf_SOURCES = src/mesh_grav.c src/fft.c src/cosmology.c src/header.c src/input.c src/output.c src/perturb_data.c src/perturb_spline.c src/titles.c parser/minIni.c src/runner.c src/fastdf.c
fastdf_CFLAGS = $(AM_CFLAGS)

if WITH_CLASS
fastdf_SOURCES += src/classex.c
endif

SUBDIRS = class

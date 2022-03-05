# Automake stuff
ACLOCAL_AMFLAGS = -I m4

# The main libraries
AM_CFLAGS = $(HDF5_CPPFLAGS)  $(FFTW_INCS) $(GSL_INCS) $(CLASS_INCS) -fopenmp
AM_LDFLAGS = $(HDF5_LDFLAGS) $(FFTW_LIBS) $(CLASS_LIBS)

# The main program
bin_PROGRAMS = fastdf
fastdf_SOURCES = src/mesh_grav.c src/fft.c src/cosmology.c src/header.c src/input.c src/output.c src/perturb_data.c src/perturb_spline.c src/titles.c parser/minIni.c src/fastdf.c
fastdf_CFLAGS = $(AM_CFLAGS)

# The main library
lib_LTLIBRARIES = libfastdf.la
libfastdf_la_SOURCES = src/mesh_grav.c src/fft.c src/cosmology.c src/header.c src/input.c src/output.c src/perturb_data.c src/perturb_spline.c src/titles.c parser/minIni.c src/libfastdf.c
libfastdf_la_LDFLAGS = -module -avoid-version -shared -version-info 1 $(HDF5_LDFLAGS) $(FFTW_LIBS) $(CLASS_LIBS)

if WITH_CLASS
fastdf_SOURCES += src/classex_legacy.c
libfastdf_la_SOURCES += src/classex_legacy.c
endif

SUBDIRS = class
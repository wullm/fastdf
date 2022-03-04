
# The minIni parser
#PARSER = ../parser/minIni.o

# The main libraries
AM_CFLAGS = $(HDF5_CPPFLAGS)  $(FFTW_INCS) $(GSL_INCS)
AM_LDFLAGS = $(HDF5_LDFLAGS) $(FFTW_LIBS) $(PARSER)

# The main programs
bin_PROGRAMS = fastdf
fastdf_SOURCES = mesh_grav.c fft.c cosmology.c header.c input.c output.c perturb_data.c perturb_spline.c titles.c fastdf.c ../parser/minIni.c
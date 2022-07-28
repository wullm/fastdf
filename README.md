FastDF
======

FastDF is a code to set up 1%-accurate particle realisations of the
perturbed phase-space distribution of cosmological relic neutrinos.

The code initially samples particles from the perturbed phase-space
distribution at high redshift and then evolves them forward by
solving the geodesic equation in linear theory. This requires
pre-computed transfer functions for the first-order metric perturbations,
as well as the random phases field used to set up initial conditions.

The code can be compiled with CLASS, which then automatically provides
the required transfer functions. It can also be run together with
monofonIC to set up dark matter, baryon, and neutrino ICs with a single
configuration file and command. See [monofonIC](https://github.com/wullm/monofonic).

Quick Installation
------------------

The code can be configured and compiled with

```
./autogen.sh
./configure
make
```

Then check out the examples.

To compile with CLASS, configure with the option --with-class=/path/to/class/.
Alternatively, an HDF5 file with perturbation vectors (transfer functions)
can be supplied. These can be pre-generated with [classex](https://github.com/wullm/classex)).

Requirements
------------
+ GSL
+ FFTW
+ HDF5
+ MPI
+ CLASS (optional)

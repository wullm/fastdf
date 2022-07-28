#!/bin/bash

# Fetch an example random phases field & perturbations vector
if { [ ! -e fastdf_example.tar.gz ] && [ ! -e gaussian_pure.hdf5 ] && [ ! -e perturb_DES3_300mev.hdf5 ]; };
then
    echo "Fetching random phases and perturbations file for the neutrino example..."
    wget http://willemelbers.com/files/fastdf_example.tar.gz
    tar -zxvf fastdf_example.tar.gz
fi


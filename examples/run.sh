#!/bin/bash

./fetch_files.sh
rm -f particles.hdf5
../fastdf example_pars.ini

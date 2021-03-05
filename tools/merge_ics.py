import h5py
import numpy as np
import sys;

fname1 = sys.argv[1];
fname2 = sys.argv[2];

print("Dark matter ics ", fname1);
print("Neutrino ics ", fname2);

f1 = h5py.File(fname1, mode="r+")
f2 = h5py.File(fname2, mode="r")

Nnupart = f2["Header"].attrs["NumPart_Total"][6]

print(Nnupart, "neutrino particles")

Npart_arr = f1["Header"].attrs["NumPart_Total"]
Npart_arr[6] = Nnupart
del f1["Header"].attrs["NumPart_Total"]
f1["Header"].attrs["NumPart_Total"] = Npart_arr

print(f1.keys)

coords = f2["PartType6/Coordinates"][:]
f1["PartType6/Coordinates"] = coords
del coords

vels = f2["PartType6/Velocities"][:]
f1["PartType6/Velocities"] = vels
del vels

masses = f2["PartType6/Masses"][:]
f1["PartType6/Masses"] = masses
del masses

ids = f2["PartType6/ParticleIDs"][:]
f1["PartType6/ParticleIDs"] = ids
del ids

# f1.close()
# f2.close()

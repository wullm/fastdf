Installation from a clean repository:

```
./autogen.sh
mkdir build && cd build
../configure
make
```

FastDF can be run without CLASS, but then a perturbation vector file needs
to be supplied.

To compile with CLASS:

```
./autogen.sh
mkdir build && cd build
../configure --with-class=/your/class/
make
```
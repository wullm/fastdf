/*******************************************************************************
 * This file is part of FastDF.
 * Copyright (c) 2020 Willem Elbers (whe@willemelbers.com)
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published
 * by the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 ******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <hdf5.h>
#include <assert.h>
#include <math.h>
#include <sys/time.h>

#include "../include/fastdf.h"

// Returns the number of local particles generated
long long run_fastdf(struct params *pars, struct units *us) {
    
    /* Initialize MPI for distributed memory parallelization */
    // MPI_Init(&argc, &argv);
    fftw_mpi_init();

    /* Get the dimensions of the cluster */
    int rank, MPI_Rank_Count;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &MPI_Rank_Count);

    /* Read options */
    // const char *fname = "/home/willem/fastdf5/fastdf/default.ini";
    header(rank, "FastDF Neutrino Initial Condition Generator");

    struct perturb_data ptdat;
    struct perturb_spline spline;
    struct perturb_params ptpars;

    /* Store the MPI rank */
    pars->rank = rank;

    /* The base output file name */
    char out_fname[200];
    sprintf(out_fname, "%s", pars->OutputFilename);

    /* The file name for the current rank */
    char out_fname_local[220];
    if (pars->DistributedFiles && MPI_Rank_Count > 1) {
        sprintf(out_fname_local, "%s.%d", out_fname, pars->rank);
    } else {
        sprintf(out_fname_local, "%s", out_fname);
    }

    /* Check if this file already exists */
    int file_exists = fileExists(out_fname_local);

    /* The ExportName for the neutrino particles */
    const char *ExportName = pars->ExportName;
    int group_exists = groupExists(out_fname_local, ExportName);

    if (file_exists && group_exists) {
        printf("Error: file '%s' already exists and has a neutrino particle group.\n", out_fname_local);
        exit(1);
    } else if (file_exists && !group_exists) {
        message(rank, "Appending neutrino particles to '%s'.\n", out_fname_local);
    }

    /* Check if the user specified a perturbation data file or if CLASS
     * is to be run. If so, FastDF must be compiled with CLASS. */
    if (pars->PerturbFile[0] != '\0') {
        /* Read the perturbation data file */
        readPerturb(pars, us, &ptdat);
        readPerturbParams(pars, us, &ptpars);
    } else {
        /* Run CLASS */
        #ifdef WITH_CLASS
        run_class(&ptdat, us, &ptpars, pars->ClassIniFile, /* verbose = */ pars->rank == 0);
        #else
        printf("\n");
        printf("Error: Not compiled with CLASS.\n");
        printf("You could reconfigure with \"./configure --with-class=/your/class\" and then \n");
        printf("\"make clean && make\". Alternatively, you could provide a perturbation vector file.\n");
        return 1;
        #endif
    }

    /* Integration limits */
    double a_begin = pars->ScaleFactorBegin;
    double a_end = pars->ScaleFactorEnd;
    double a_factor = 1.0 + pars->ScaleFactorStep;

    const double h = ptpars.h;
    const double H_0 = h * 100 * KM_METRES / MPC_METRES * us->UnitTimeSeconds;
    const double rho_crit = 3.0 * H_0 * H_0 / (8. * M_PI * us->GravityG);

    /* Package physical constants */
    const double m_eV = ptpars.M_ncdm_eV[0];
    const double T_nu = ptpars.T_ncdm[0] * ptpars.T_CMB;
    const double T_eV = T_nu * us->kBoltzmann / us->ElectronVolt;

    /* Retrieve further physical constant */
    const double c = us->SpeedOfLight;
    const double inv_c = 1.0 / c;
    const double inv_c2 = inv_c * inv_c;

    /* Initialize the interpolation spline for the perturbation data */
    initPerturbSpline(&spline, DEFAULT_K_ACC_TABLE_SIZE, &ptdat);

    header(rank, "Simulation parameters");
    message(rank, "We want %lld (%d^3) particles\n", pars->NumPartGenerate, pars->CubeRootNumber);

    /* Check if we recognize the output gauge */
    int gauge_Nbody = 0;
    if (strcmp(pars->Gauge, "Newtonian") == 0 ||
        strcmp(pars->Gauge, "newtonian") == 0) {
        message(rank, "Output gauge: Newtonian\n");
    } else if (strcmp(pars->Gauge, "N-body") == 0 ||
               strcmp(pars->Gauge, "n-body") == 0 ||
               strcmp(pars->Gauge, "nbody") == 0) {
        message(rank, "Output gauge: N-body\n");
        gauge_Nbody = 1;
    } else {
        printf("Error: unknown output gauge '%s'.\n", pars->Gauge);
        exit(1);
    }

    /* Check if we recognize the output velocity type */
    int velocity_type = 0;
    if (strcmp(pars->VelocityType, "peculiar") == 0 ||
        strcmp(pars->VelocityType, "Peculiar") == 0) {
        message(rank, "Output velocities: peculiar (a*dx/dt)\n");
    } else if (strcmp(pars->VelocityType, "Gadget") == 0 ||
               strcmp(pars->VelocityType, "gadget") == 0) {
        message(rank, "Output velocities: Gadget (a^.5*dx/dt)\n");
        velocity_type = 1;
    } else {
        printf("Error: unknown output velocity type '%s'.\n", pars->VelocityType);
        exit(1);
    }

    if (pars->IncludeHubbleFactors) {
        printf("Output coordinates in U_L h^-1.\n");
        printf("Output masses in U_M h^-1.\n");
    }

    message(rank, "a_begin = %.3e (z = %.2f)\n", a_begin, 1./a_begin - 1);
    message(rank, "a_end = %.3e (z = %.2f)\n", a_end, 1./a_end - 1);

    const char use_alternative_eom = pars->AlternativeEquations;
    if (use_alternative_eom) {
        message(rank, "\n\n");
        message(rank, "WARNING: Using alternative equations of motion!");
        message(rank, "\n\n");
    }

    /* Read the Gaussian random field on each MPI rank */
    double BoxLen = 0;

    /* Override the box length? */
    if (BoxLen > 0.) {
        pars->BoxLen = BoxLen;
    } else {
        /* Otherwise, look for a user-specified value */
        BoxLen = pars->BoxLen;
    }

    message(rank, "BoxLen = %.2f U_L\n", BoxLen);

    /* The index of the present day, corresponds to the last index in the array */
    int today_index = ptdat.tau_size - 1;

    /* Find the relevant density title among the transfer functions */
    char *title = pars->TransferFunctionDensity;
    int index_src = findTitle(ptdat.titles, title, ptdat.n_functions);
    if (index_src < 0) {
        printf("Error: transfer function '%s' not found (%d).\n", title, index_src);
        return 1;
    }

    /* Find the present-day density, as fraction of the critical density */
    const double box_vol = BoxLen * BoxLen * BoxLen;
    const double Omega = ptdat.Omega[ptdat.tau_size * index_src + today_index];
    const double rho = Omega * rho_crit;
    const double particle_mass = rho * box_vol / pars->NumPartGenerate;

    header(rank, "Mass factors");
    message(rank, "Neutrino mass is %f eV\n", m_eV);
    message(rank, "Particle mass is %f U_M\n", particle_mass);

    /* Store the Box Length */
    pars->BoxLen = BoxLen;

    /* Determine the number of particle to be generated on each rank */
    header(rank, "Particle distribution");

    /* The particle number is M^3 */
    long long M = pars->CubeRootNumber;

    /* Determine what particles belong to this slice */
    double fac = (double) M / MPI_Rank_Count;
    long long X_min = ceil(rank * fac);
    long long X_max = ceil((rank + 1) * fac);
    long long MX = X_max - X_min;
    long long localParticleNumber = MX * M * M;
    long long localFirstNumber = X_min * M * M;

    /* Check that all particles have been assigned to a node */
    long long totalParticlesAssigned;
    MPI_Allreduce(&localParticleNumber, &totalParticlesAssigned, 1,
                   MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    assert(totalParticlesAssigned == M * M * M);
    printf("%03d: Local particles [%04lld, %04lld], first = %lld, last = %lld, total = %lld\n", rank, X_min, X_max, localFirstNumber, localFirstNumber + localParticleNumber - 1, totalParticlesAssigned);

    /* Check that we are not exceeding the HDF5 parallel write limit */
    if (localParticleNumber * 3 * sizeof(double) > HDF5_PARALLEL_LIMIT) {
        printf("\nError: Exceeding 2GB limit on parallel HDF5 writes. Increase the number of MPI tasks.\n");
        exit(1);
    }

    /* The particles to be generated on this node */
    struct particle_ext *genparts = calloc(localParticleNumber,
                                           sizeof(struct particle_ext));

    /* ID of the first particle on this node */
    long long firstID = pars->FirstID + localFirstNumber;

    MPI_Barrier(MPI_COMM_WORLD);

    double z_begin = 1./a_begin - 1;
    double log_tau_begin = perturbLogTauAtRedshift(&spline, z_begin);

    header(rank, "Generating pre-initial conditions");
    message(rank, "ID of first particle = %lld\n", firstID);
    message(rank, "T_nu = %e eV\n", T_eV);

    /* Generate random neutrino particles */
    for (long long i=0; i<localParticleNumber; i++) {
        struct particle_ext *p = &genparts[i];

        /* Set the ID of the particle */
        uint64_t id = i + firstID;

        /* Generate random particle velocity and position */
        init_neutrino_particle(id, m_eV, p->v, p->x, &p->mass, BoxLen, us, T_eV);

        /* Compute the momentum in eV */
        const double p_eV = fermi_dirac_momentum(p->v, m_eV, us->SpeedOfLight);
        const double f_i = fermi_dirac_density(p_eV, T_eV);

        if (i==0)
        message(rank, "First random momentum = %e eV\n", p_eV);

        /* The current energy */
        double eps_eV = hypot(p_eV/a_begin, m_eV);
        
        /* Place all neutrinos at the centre */
        p->x[0] = BoxLen * 0.5;
        p->x[1] = BoxLen * 0.5;
        p->x[2] = BoxLen * 0.5;
        
        /* Compute initial phase space density */
        p->f_i = f_i;

        /* Compute the magnitude of the initial velocity */
        p->v_i[0] = p->v[0];
        p->v_i[1] = p->v[1];
        p->v_i[2] = p->v[2];
        p->v_i_mag = hypot3(p->v[0], p->v[1], p->v[2]);
    }

    message(rank, "Done with pre-initial conditions.\n");

    /* Compute energies, masses, and weights and store them in contiguous arrays */
    double *energies = malloc(1 * localParticleNumber * sizeof(double));
    double *masses = malloc(1 * localParticleNumber * sizeof(double));
    double *weights = malloc(1 * localParticleNumber * sizeof(double));
    double *phaseDensities = malloc(1 * localParticleNumber * sizeof(double));
    for (long long i=0; i<localParticleNumber; i++) {
        struct particle_ext *p = &genparts[i];

        /* Compute the energy & weight (needs to happen before converting the velocities!)*/
        double p_eV = fermi_dirac_momentum(p->v, m_eV, us->SpeedOfLight);
        double eps_eV = hypot(p_eV/a_end, m_eV);
        double eps = particle_mass / m_eV * eps_eV;
        double f = fermi_dirac_density(p_eV, T_eV);
        double w = (p->f_i - f)/p->f_i;
        // p->mass = particle_mass * w;

        energies[i] = eps;
        masses[i] = particle_mass;
        weights[i] = w;
        phaseDensities[i] = p->f_i;

        /* Include Hubble factors for Gadget ICs? */
        if (pars->IncludeHubbleFactors) {
            energies[i] *= h;
            masses[i] *= h;
        }
    }

    /* Final operations before writing the particles to disk */
    #pragma omp parallel for
    for (long long i=0; i<localParticleNumber; i++) {
        struct particle_ext *p = &genparts[i];

        /* Ensure that particles wrap */
        p->x[0] = fwrap(p->x[0], BoxLen);
        p->x[1] = fwrap(p->x[1], BoxLen);
        p->x[2] = fwrap(p->x[2], BoxLen);

        /* Convert momenta to velocities */
        p->v[0] *= c / m_eV;
        p->v[1] *= c / m_eV;
        p->v[2] *= c / m_eV;

        /* Convert to peculiar velocities */
        p->v[0] /= a_end;
        p->v[1] /= a_end;
        p->v[2] /= a_end;

        /* Possibly convert to Gadget velocities */
        if (velocity_type == 1) {
            p->v[0] /= sqrt(a_end);
            p->v[1] /= sqrt(a_end);
            p->v[2] /= sqrt(a_end);
        }

        /* Include Hubble factors for Gadget ICs? */
        if (pars->IncludeHubbleFactors) {
            p->x[0] *= h;
            p->x[1] *= h;
            p->x[2] *= h;
        }
    }

    header(rank, "Prepare output");

    if (rank == 0 || pars->DistributedFiles) {
        /* The number of particles in the file */
        long long parts_in_file = pars->DistributedFiles ? localParticleNumber : pars->NumPartGenerate;

        /* Create the output file if it does not exist */
        hid_t h_out_file;
        if (!fileExists(out_fname_local)) {
            /* Create the file */
            h_out_file = createFile(out_fname_local);

            /* Writing attributes into the Header & Cosmology groups */
            int err = writeHeaderAttributes(pars, us, parts_in_file, pars->NumPartGenerate, h_out_file);
            if (err > 0) exit(1);
        } else {
            /* Otherwise, open the file in read & write mode */
            h_out_file = H5Fopen(out_fname_local, H5F_ACC_RDWR , H5P_DEFAULT);
        }

        /* The ExportName */
        const char *ExportName = pars->ExportName;

        /* The particle group in the output file */
        hid_t h_grp;

        /* Datsets */
        hid_t h_data;

        /* Vector dataspace (e.g. positions, velocities) */
        const hsize_t vrank = 2;
        const hsize_t vdims[2] = {parts_in_file, 3};
        hid_t h_vspace = H5Screate_simple(vrank, vdims, NULL);

        /* Scalar dataspace (e.g. masses, particle ids) */
        const hsize_t srank = 1;
        const hsize_t sdims[1] = {parts_in_file};
        hid_t h_sspace = H5Screate_simple(srank, sdims, NULL);

        /* Set chunking for vectors */
        hid_t h_prop_vec = H5Pcreate(H5P_DATASET_CREATE);
        const hsize_t vchunk[2] = {HDF5_CHUNK_SIZE, 3};
        H5Pset_chunk(h_prop_vec, vrank, vchunk);

        /* Set chunking for scalars */
        hid_t h_prop_sca = H5Pcreate(H5P_DATASET_CREATE);
        const hsize_t schunk[1] = {HDF5_CHUNK_SIZE};
        H5Pset_chunk(h_prop_sca, srank, schunk);

        /* Create the particle group in the output file */
        message(rank, "Creating Group '%s' with %lld particles.\n", ExportName, pars->NumPartGenerate);
        h_grp = H5Gcreate(h_out_file, ExportName, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

        /* Coordinates (use vector space) */
        h_data = H5Dcreate(h_grp, "Coordinates", H5T_NATIVE_DOUBLE, h_vspace, H5P_DEFAULT, h_prop_vec, H5P_DEFAULT);
        H5Dclose(h_data);

        /* Velocities (use vector space) */
        h_data = H5Dcreate(h_grp, "Velocities", H5T_NATIVE_DOUBLE, h_vspace, H5P_DEFAULT, h_prop_vec, H5P_DEFAULT);
        H5Dclose(h_data);

        /* Particle IDs (use scalar space) */
        h_data = H5Dcreate(h_grp, "ParticleIDs", H5T_NATIVE_LLONG, h_sspace, H5P_DEFAULT, h_prop_sca, H5P_DEFAULT);
        H5Dclose(h_data);

        /* Masses (use scalar space) */
        h_data = H5Dcreate(h_grp, "Masses", H5T_NATIVE_DOUBLE, h_sspace, H5P_DEFAULT, h_prop_sca, H5P_DEFAULT);
        H5Dclose(h_data);

        /* Neutrino (delta-f) weights (use scalar space) */
        h_data = H5Dcreate(h_grp, "Weights", H5T_NATIVE_DOUBLE, h_sspace, H5P_DEFAULT, h_prop_sca, H5P_DEFAULT);
        H5Dclose(h_data);
        
        /* Initial unperturbed phase-space densities (use scalar space) */
        h_data = H5Dcreate(h_grp, "PhaseSpaceDensities", H5T_NATIVE_DOUBLE, h_sspace, H5P_DEFAULT, h_prop_sca, H5P_DEFAULT);
        H5Dclose(h_data);

        /* Energies (use scalar space) */
        h_data = H5Dcreate(h_grp, "Energies", H5T_NATIVE_DOUBLE, h_sspace, H5P_DEFAULT, h_prop_sca, H5P_DEFAULT);
        H5Dclose(h_data);

        /* Close the group */
        H5Gclose(h_grp);

        /* Close the file */
        H5Fclose(h_out_file);
    }

    message(rank, "Writing output to %s.\n", out_fname);

    /* Now open the file in parallel mode if possible */
#ifdef H5_HAVE_PARALLEL
    hid_t h_out_file;
    if (pars->DistributedFiles) {
        h_out_file = openFile(out_fname_local);
    } else {
        h_out_file = openFile_MPI(MPI_COMM_WORLD, out_fname_local);
    }
#else
    hid_t h_out_file = openFile(out_fname_local);
#endif

    /* The particle group in the output file */
    hid_t h_grp = H5Gopen(h_out_file, pars->ExportName, H5P_DEFAULT);

    /* Datsets */
    hid_t h_data;

    /* Vector dataspace (e.g. positions, velocities) */
    const hsize_t vrank = 2;
    h_data = H5Dopen2(h_grp, "Velocities", H5P_DEFAULT);
    hid_t h_vspace = H5Dget_space(h_data);
    H5Dclose(h_data);

    /* Scalar dataspace (e.g. masses, particle ids) */
    const hsize_t srank = 1;
    h_data = H5Dopen2(h_grp, "ParticleIDs", H5P_DEFAULT);
    hid_t h_sspace = H5Dget_space(h_data);
    H5Dclose(h_data);

    /* Create vector & scalar datapsace for smaller chunks of data */
    const hsize_t ch_vdims[2] = {localParticleNumber, 3};
    const hsize_t ch_sdims[1] = {localParticleNumber};
    hid_t h_ch_vspace = H5Screate_simple(vrank, ch_vdims, NULL);
    hid_t h_ch_sspace = H5Screate_simple(srank, ch_sdims, NULL);

    /* The start of this chunk, in the overall vector & scalar spaces */
    const hsize_t start_in_group = pars->DistributedFiles ? 0 : localFirstNumber;
    const hsize_t vstart[2] = {start_in_group, 0}; //always with the "x" coordinate
    const hsize_t sstart[1] = {start_in_group};

    /* Choose the corresponding hyperslabs inside the overall spaces */
    H5Sselect_hyperslab(h_vspace, H5S_SELECT_SET, vstart, NULL, ch_vdims, NULL);
    H5Sselect_hyperslab(h_sspace, H5S_SELECT_SET, sstart, NULL, ch_sdims, NULL);

    /* Unpack the remaining particle data into contiguous arrays */
    double *coords = malloc(3 * localParticleNumber * sizeof(double));
    double *vels = malloc(3 * localParticleNumber * sizeof(double));
    long long *ids = malloc(1 * localParticleNumber * sizeof(long long));
    for (long long i=0; i<localParticleNumber; i++) {
        coords[i * 3 + 0] = genparts[i].x[0];
        coords[i * 3 + 1] = genparts[i].x[1];
        coords[i * 3 + 2] = genparts[i].x[2];
        vels[i * 3 + 0] = genparts[i].v[0];
        vels[i * 3 + 1] = genparts[i].v[1];
        vels[i * 3 + 2] = genparts[i].v[2];
        ids[i] = firstID + i;
    }

    message(rank, "Writing Coordinates.\n");

    /* Write coordinate data (vector) */
    h_data = H5Dopen(h_grp, "Coordinates", H5P_DEFAULT);
    H5Dwrite(h_data, H5T_NATIVE_DOUBLE, h_ch_vspace, h_vspace, H5P_DEFAULT, coords);
    H5Dclose(h_data);
    free(coords);

    message(rank, "Writing Velocities.\n");

    /* Write velocity data (vector) */
    h_data = H5Dopen(h_grp, "Velocities", H5P_DEFAULT);
    H5Dwrite(h_data, H5T_NATIVE_DOUBLE, h_ch_vspace, h_vspace, H5P_DEFAULT, vels);
    H5Dclose(h_data);
    free(vels);

    message(rank, "Writing ParticleIDs.\n");

    /* Write particle id data (scalar) */
    h_data = H5Dopen(h_grp, "ParticleIDs", H5P_DEFAULT);
    H5Dwrite(h_data, H5T_NATIVE_LLONG, h_ch_sspace, h_sspace, H5P_DEFAULT, ids);
    H5Dclose(h_data);
    free(ids);

    message(rank, "Writing Masses.\n");

    /* Write mass data (scalar) */
    h_data = H5Dopen(h_grp, "Masses", H5P_DEFAULT);
    H5Dwrite(h_data, H5T_NATIVE_DOUBLE, h_ch_sspace, h_sspace, H5P_DEFAULT, masses);
    H5Dclose(h_data);
    free(masses);

    message(rank, "Writing Weights.\n");

    /* Write delta-f weight data (scalar) */
    h_data = H5Dopen(h_grp, "Weights", H5P_DEFAULT);
    H5Dwrite(h_data, H5T_NATIVE_DOUBLE, h_ch_sspace, h_sspace, H5P_DEFAULT, weights);
    H5Dclose(h_data);
    free(weights);

    message(rank, "Writing PhaseSpaceDensities.\n");

    /* Write initial unperturbed phase-space density data (scalar) */
    h_data = H5Dopen(h_grp, "PhaseSpaceDensities", H5P_DEFAULT);
    H5Dwrite(h_data, H5T_NATIVE_DOUBLE, h_ch_sspace, h_sspace, H5P_DEFAULT, phaseDensities);
    H5Dclose(h_data);
    free(phaseDensities);

    message(rank, "Writing Energies.\n");

    /* Write energy data (scalar) */
    h_data = H5Dopen(h_grp, "Energies", H5P_DEFAULT);
    H5Dwrite(h_data, H5T_NATIVE_DOUBLE, h_ch_sspace, h_sspace, H5P_DEFAULT, energies);
    H5Dclose(h_data);
    free(energies);

    message(rank, "Done with writing on rank 0.\n");

    /* Close the chunk-sized scalar and vector dataspaces */
    H5Sclose(h_ch_vspace);
    H5Sclose(h_ch_sspace);

    /* Close the group */
    H5Gclose(h_grp);

    /* Close the file */
    H5Fclose(h_out_file);

    /* Free the particles array */
    free(genparts);

    /* Done with MPI parallelization */
    MPI_Barrier(MPI_COMM_WORLD);

    /* Clean up */
    cleanParams(pars);
    cleanPerturb(&ptdat);
    cleanPerturbParams(&ptpars);

    /* Release the interpolation splines */
    cleanPerturbSpline(&spline);
    
    return localParticleNumber;
}

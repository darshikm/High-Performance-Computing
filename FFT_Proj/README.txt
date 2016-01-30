This code has been written as a part of HPC Project aimed at better understanding the performance enhancements/optimizations of a complicated algorithm like Fast Fourier Transform

It is a very naive implementation of radix-2 algorithm to identify the parallelization points and see it scales on the Cluster.

sequential_comp.cc: Contains the reference sequential methods for evaluating FFT for Decimation in Time and Decimation in Frequency. At the end we compare the execution times in terms of the two algorithms.

OpenMP_fft.cc: Contains the parallel OpenMP versions of the sequential program described above. To enable multiple threads executing the program,
it needs to be compiled with '-fopenmp' flag. Either changing the Macro directive OMP_NT value or explicitly set omp_num_threads.
One can also set GOMP_CPU_AFFINITY to bind the statically scheduled threads to the cores executing the program on the server (Multicore Platform)

mpi_dft.cc: Base/reference MPI code to compute a Discrete Fourier Transform (simple DFT) to check how the problem of any given size.. can scale on any specified number of process nodes participating in MPI based communication.

mpi_dit.cc: Naive implementation of FFT algorithm for decimation in time.
mpi_dif.cc: Naive implementation of FFT algorithm for decimation in frequency.

============================================
Commands to compile MPI programs:
mpiCC or mpic++ (flags/optimization) -c (c_file) -o (exe_output)

Command to execute MPI program:
mpirun -np 1 ./exe 

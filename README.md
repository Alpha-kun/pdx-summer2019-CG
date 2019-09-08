# pdx-summer2019-CG
A repository which contains the code of the summer 2019 conjugate gradient / parallel computing project at PSU

"conjugate_gradient.hpp" contains the implementation of the plain conjugate gradient algorithm, the precondtioned conjugate gradient with Jacobi and Gauss-Seidel preconditioner, and two-level and multi-level algorithm.
"lubys_partition.hpp" contains the implementation of the Luby's algorithm, including the single-level and multi-level version of it.
"parallel_utility.hpp" contains OpenMP parallelized matrix algorithms, which are dense matrix-vector multiplicaion, sparse matrix-vector multiplicaion, and sparse matrix-matrix multiplicaion.
"randomGen.hpp" provides functions to generate random double-precision decimals, vectors and SPD matrix.
"partition_example.cpp" contains the main function which runs the test program of code mentioned above.

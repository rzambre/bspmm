# BSPMM

Block sparse matrix multiplication (BSPMM) is the dominant
cost in the CCSD and CCSD(T) quantum chemical many-body
methods of NWChem, a prominent quantum chemistry application
suite for large-scale simulations of chemical and biological
systems. NWChem implements its BSPMM with dense matrix operations
using a get-compute-update pattern: each worker (processing
entity) uses MPI_Get to retrieve the submatrices it needs,
and after the multiplication it uses an MPI_Accumulate to update
the memory at the target location.

This repository contains mini-apps that implement a 2D version
of BSPMM to perform A × B = C, wherein the input matrices A and
B are composed of tiles. The nonzero tiles are evenly distributed
among the ranks in a round-robin fashion. Each rank maintains a
work-unit table that lists all the multiplication operations that
workers need in order to cooperatively execute. Rank 0 hosts a
global counter, which the workers fetch and add atomically
(MPI_Fetch_and_op). The fetched counter serves as an index to the
work-unit table. Each worker locally accumulates its C tiles until
the next fetched work unit corresponds to a different C tile, in
which case the worker uses an MPI_Accumulate to update the C tile.
A worker is a process in MPI everywhere and a thread in MPI+threads.

This mini-app was written after discussions with Pavan Balaji, Min Si,
and Shintaro Iwasaki from Argonne National Laboratory, and with
Jeff Hammond from Intel.

# Versions

bspmm_single.c
- MPI everywhere version (flat MPI)

bspmm_multiple.c
- MPI+OpenMP version using MPI_THREAD_MULTIPLE
- Expressing no logical parallelism exposed to the MPI library

bspmm_multiple_nwins.c
- MPI+OpenMP version using MPI_THREAD_MULTIPLE
- Expressing logical parallelism to the MPI library
  - The MPI_Get operations of each thread are independent. Hence, each thread uses their own window.
  - All threads must use the same window for MPI_Accumulate operations since atomicity across windows
  for the same memory location is undefined. However, the issue of MPI_Accumulate operations from
  multiple threads could still occur independently since BSPMM does not require ordering of these
  operations. Hence, we hint `accumulate_ordering=none` to window for MPI_Accumulate operations.

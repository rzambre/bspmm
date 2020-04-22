mpiexec -n 4 -ppn 1 -bind-to hfi1_0 -f hostfile_4 -env HFI_NO_CPUAFFINITY 1 -env OMP_NUM_THREADS 1 -env OMP_PLACES cores -env OMP_PROC_BIND close ./bspmm_multiple 1 4 2 2 1

mpiexec -n 4 -ppn 1 -bind-to hfi1_0 -env HFI_NO_CPUAFFINITY 1 -f hostfile_4 ./bspmm_single 1 16 2 2 1

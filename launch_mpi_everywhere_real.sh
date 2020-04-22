mpiexec -n 64 -ppn 16 -bind-to hfi1_0 -env HFI_NO_CPUAFFINITY 1 -f hostfile_4 ./bspmm_single 2 60 8 2 16

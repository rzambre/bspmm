# -*- Mode: Makefile; -*-
CC=icc
CFLAGS= -g3 -O3 -Wall -lmpi
OPENMPFLAGS=-fopenmp
BSPMM_COMMON_SRC=bspmm_common.c
BINS=bspmm_single
BINS+=bspmm_multiple
BINS+=bspmm_multiple_nwins

all: $(BINS)

bspmm_single: bspmm_single.c $(BSPMM_COMMON_SRC)
	$(CC) $(CFLAGS) $^ -o $@

bspmm_multiple: bspmm_multiple.c $(BSPMM_COMMON_SRC)
	$(CC) $(CFLAGS) $(OPENMPFLAGS) $^ -o $@

bspmm_multiple_nwins: bspmm_multiple_nwins.c $(BSPMM_COMMON_SRC)
	$(CC) $(CFLAGS) $(OPENMPFLAGS) $^ -o $@

clean:
	rm -f $(BINS)

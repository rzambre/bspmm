/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 * See COPYRIGHT in top-level directory.
 */

#ifndef BSPMM_H_INCLUDED
#define BSPMM_H_INCLUDED

#include <mpi.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <sys/types.h>
#include <time.h>

#define SPARSITY_A 0.8
#define SPARSITY_B 0.8
#define RAND_RANGE 10
#define PAGE_SIZE 4096

int setup(int rank, int nprocs, int argc, char **argv, int *tile_dim_ptr, int *tile_num_ptr, int *p_dim_ptr, int *node_dim_ptr, int *ppn_ptr);
void init_tile_map(int *tile_map, int tile_num, int *non_zero_tile_num_ptr);
void init_work_unit_table(int *tile_map_a, int *tile_map_b, int *tile_map_c, int tile_num, int **work_unit_table, int *work_units);
void init_mats(int mat_dim, int tile_dim, double *mat_a, double *mat_b, double *mat_c);
void init_sub_mats(double *sub_mat_a, double *sub_mat_b, double *sub_mat_c, size_t sub_mat_elements);
void init_mat_according_to_map(double *mat, size_t mat_dim);
void check_mats(double *mat_a, double *mat_b, double *mat_c, int mat_dim);

/*
 * global_tile_id -- ID of the tile
 * tot_ranks -- Size of comm world
 */
static inline int target_rank_of_tile(int global_tile_id, int tot_ranks)
{
    return global_tile_id % tot_ranks;
}

/*
 * global_tile_id -- ID of the tile
 * tot_ranks -- Size of comm world
 * tile_dim -- Number of elements in one dimension of the tile
 */
static inline MPI_Aint offset_of_tile(int global_tile_id, int tot_ranks, int tile_dim)
{
    MPI_Aint target_offset;

    target_offset = (global_tile_id / tot_ranks) * tile_dim * tile_dim;

    return target_offset;
}

#endif /* BSPMM_H_INCLUDED */

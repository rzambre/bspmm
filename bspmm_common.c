/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 * See COPYRIGHT in top-level directory.
 */

#include "bspmm.h"

int setup(int rank, int nprocs, int argc, char **argv, int *tile_dim_ptr, int *tile_num_ptr, int *p_dim_ptr, int *node_dim_ptr, int *ppn_ptr)
{
    int tile_dim, tile_num, p_dim, node_dim, ppn;

    if (argc != 6) {
        if (!rank) {
            printf("usage: bspmm_mpi <tile-dim> <tile-num> <p-dim> <node-dim> <ppn>\n");
            printf("\n");
            printf("tile-dim:   Number of elements (double) in one dimension of a tile.\n");
            printf("tile-num:   Number of tiles in one dimension of the global matrix.\n");
            printf("p-dim:     Number of processes in one dimension of the the process matrix.\n");
            printf("node-dim:     Number of nodes in one dimension of the the node matrix.\n");
            printf("ppn:     Number of processes per node.\n");
        }
        return 1;
    }

    tile_dim = atoi(argv[1]);   /* number of elements in one dimension of a tile */
    tile_num = atoi(argv[2]);   /* number of tiles in one dimension */
    p_dim = atoi(argv[3]);      /* number of processes in one dimension */
    node_dim = atoi(argv[4]);   /* number of nodes in one dimension */
    ppn = atoi(argv[5]);        /* number of processes per node */

    if (tile_num % 4 != 0) {
        if (!rank)
            printf("Please keep number of blocks in one dimension to be a multiple for 4.\n");
    }

    (*tile_dim_ptr) = tile_dim;
    (*tile_num_ptr) = tile_num;
    (*p_dim_ptr) = p_dim;
    (*node_dim_ptr) = node_dim;
    (*ppn_ptr) = ppn;

    return 0;
}

void init_tile_map(int *tile_map, int tile_num, int *non_zero_tile_num_ptr)
{
    int i, j;
    int non_zero_tile_num;

    /* Initialize map to 0 */
    for (i = 0; i < tile_num; i++)
        for (j = 0; j < tile_num; j++)
            tile_map[i*tile_num + j] = -1;
    
    non_zero_tile_num = 0;

    /* First 2 quarters in both dimensions are dense */
    for (i = 0; i < (tile_num / 2); i++)
        for (j = 0; j < (tile_num / 2); j++) {
            tile_map[i*tile_num + j] = non_zero_tile_num;
            non_zero_tile_num++;
        }

    /* Intersection of the 3rd quarter in both dimensions is dense */
    for (i = (tile_num / 2); i < (3 * tile_num / 4); i++)
        for (j = (tile_num / 2); j < (3 * tile_num / 4); j++) {
            tile_map[i*tile_num + j] = non_zero_tile_num;
            non_zero_tile_num++;
        } 
    /* Intersection of the 4th quarter in both dimensions is dense */
    for (i = (3 * tile_num / 4); i < tile_num ; i++)
        for (j = (3 * tile_num / 4); j < tile_num ; j++) {
            tile_map[i*tile_num + j] = non_zero_tile_num;
            non_zero_tile_num++;
        }
    
#if DEBUG
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) {
        for (i = 0; i < tile_num; i++) {
            for (j = 0; j < tile_num; j++) 
                printf("%d ", tile_map[i*tile_num + j]);
            printf("\n");
        }
    }
#endif

    *(non_zero_tile_num_ptr) = non_zero_tile_num;

}

void init_mat_according_to_map(double *mat, size_t mat_dim)
{
    size_t i, j;

    /* First 2 quarters in both dimensions are dense */
    for (i = 0; i < (mat_dim / 2); i++)
        for (j = 0; j < (mat_dim / 2); j++)
            mat[i*mat_dim + j] = 1;

    /* Intersection of the 3rd quarter in both dimensions is dense */
    for (i = (mat_dim / 2); i < (3 * mat_dim / 4); i++)
        for (j = (mat_dim / 2); j < (3 * mat_dim / 4); j++)
            mat[i*mat_dim + j] = 1;
    
    /* Intersection of the 4th quarter in both dimensions is dense */
    for (i = (3 * mat_dim / 4); i < mat_dim ; i++)
        for (j = (3 * mat_dim / 4); j < mat_dim ; j++)
            mat[i*mat_dim + j] = 1;
}

void init_work_unit_table(int *tile_map_a, int *tile_map_b, int *tile_map_c, int tile_num, int **work_unit_table_ptr, int *work_units_ptr)
{
    int *tmp_table, *work_unit_table;
    size_t max_work_units;
    int work_unit;
    int i, j, k;

    max_work_units = (size_t) tile_num * tile_num * tile_num;
    
    /* A | B | C 
     * - | - | - 
     * indexed by work unit
     */
    tmp_table = calloc(max_work_units * 3, sizeof(int));
    
    work_unit = 0;
    for (i = 0; i < tile_num; i++) {
        for (j = 0; j < tile_num; j++) {
            for (k = 0; k < tile_num; k++) {
                if ((tile_map_a[i*tile_num + k] != -1) && (tile_map_b[k*tile_num + j] != -1)) {
                    tmp_table[3*work_unit + 0] = tile_map_a[i*tile_num + k] /* A */;
                    tmp_table[3*work_unit + 1] = tile_map_b[k*tile_num + j] /* B */;
                    tmp_table[3*work_unit + 2] = tile_map_c[i*tile_num + j] /* C */;
                    work_unit++;
                }
            }
        }
    }

    work_unit_table = calloc(work_unit * 3, sizeof(int));
    /* Copy from tmp table into real table */
#if DEBUG
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank ==0) printf("A\tB\tC\n");
#endif
    for (i = 0; i < work_unit; i++) {
        work_unit_table[3*i + 0] = tmp_table[3*i + 0];
        work_unit_table[3*i + 1] = tmp_table[3*i + 1];
        work_unit_table[3*i + 2] = tmp_table[3*i + 2];
#if DEBUG
        if (rank == 0) {
            printf("%d\t", work_unit_table[3*i + 0]);
            printf("%d\t", work_unit_table[3*i + 1]);
            printf("%d\t", work_unit_table[3*i + 2]);
            printf("\n");
        }
#endif
    }

    *(work_units_ptr) = work_unit;
    *(work_unit_table_ptr) = work_unit_table;

    free(tmp_table);
}

void init_sub_mats(double *sub_mat_a, double *sub_mat_b, double *sub_mat_c, size_t sub_mat_elements)
{
    int element_i;

    for (element_i = 0; element_i < sub_mat_elements; element_i++) {
        sub_mat_a[element_i] = 1;
        sub_mat_b[element_i] = 1;
        sub_mat_c[element_i] = 0;
    }
}

void init_mats(int mat_dim, int tile_dim, double *mat_a, double *mat_b, double *mat_c)
{
    int i, j, bi, bj;

    srand(0);

    for (bj = 0; bj < mat_dim; bj += tile_dim) {
        for (bi = 0; bi < mat_dim; bi += tile_dim) {
            /* initialize mat_a */
            if (rand() < SPARSITY_A * RAND_MAX) {
                for (j = bj; j < bj + tile_dim; j++)
                    for (i = bi; i < bi + tile_dim; i++)
                        mat_a[j + i * mat_dim] = 0.0;
            } else {
                for (j = bj; j < bj + tile_dim; j++)
                    for (i = bi; i < bi + tile_dim; i++)
                        mat_a[j + i * mat_dim] = (double) rand() / (RAND_MAX / RAND_RANGE + 1);
            }
            /* initialize mat_b */
            if (rand() < SPARSITY_B * RAND_MAX) {
                for (j = bj; j < bj + tile_dim; j++)
                    for (i = bi; i < bi + tile_dim; i++)
                        mat_b[j + i * mat_dim] = 0.0;
            } else {
                for (j = bj; j < bj + tile_dim; j++)
                    for (i = bi; i < bi + tile_dim; i++)
                        mat_b[j + i * mat_dim] = (double) rand() / (RAND_MAX / RAND_RANGE + 1);
            }
        }
    }
    /* reset mat_c */
    memset(mat_c, 0, sizeof(double) * mat_dim * mat_dim);
}

void check_mats(double *mat_a, double *mat_b, double *mat_c, int mat_dim)
{
    int i, j, k, r;
    int bogus = 0;
    double temp_c;
    double diff, max_diff = 0.0;

    /* pick up 1000 values to check correctness */
    for (r = 0; r < 1000; r++) {
        i = rand() % mat_dim;
        j = rand() % mat_dim;
        temp_c = 0.0;
        for (k = 0; k < mat_dim; k++)
            temp_c += mat_a[k + i * mat_dim] * mat_b[j + k * mat_dim];
        diff = mat_c[j + i * mat_dim] - temp_c;
        if (fabs(diff) > 0.00001) {
            bogus = 1;
            if (fabs(diff) > fabs(max_diff))
                max_diff = diff;
        }
    }

    if (bogus)
        printf("\nTEST FAILED: (%.5f MAX diff)\n\n", max_diff);
    else
        printf("\nTEST PASSED\n\n");
}

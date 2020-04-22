/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */

#include <omp.h>
#include "bspmm.h"

/*
 * Block sparse matrix multiplication using RMA operations, a global counter for workload
 * distribution, MPI_THREAD_MULTIPLE mode using a separate window per thread for the A
 * and B submatrices. Each thread needs to share one window to accumulate C tiles.
 *
 * A, B, and C denote submatrices (tile_dim x tile_dim) and n is tile_num
 *
 * | C11 ... C1n |   | A11 ... A1n |    | B11 ... B1n |
 * |  . .     .  |   |  . .     .  |    |  . .     .  |
 * |  .  Cij  .  | = |  .  Aik  .  | *  |  .  Bkj  .  |
 * |  .     . .  |   |  .     . .  |    |  .     . .  |
 * | Cn1 ... Cnn |   | An1 ... Ann |    | Bn1 ... Bnn |
 *
 * bspmm parallelizes all indpendent relevant work units. It maitains a table of
 * work units. Each work unit corresponds to 1 DGEMM of tiles (one A tile,
 * one B tile, and one C tile). The non-zero tiles of A, B, and C matrices are
 * evenly distributed amongst the ranks. Each thread will locally accumulate C until
 * its next work unit corresponds to a different C tile.
 *
 * The distribution of work between the threads of all the ranks is dynamic:
 * each thread reads a counter to obtain its work id. The counter is updated
 * atomically each time it is read.
 */

#define OFI_WINDOW_HINTS 0
#define COMPUTE 1
#define FINE_TIME 1
#define WARMUP  1
#define CHECK_FOR_ERRORS 0
#define SHOW_WORKLOAD_DIST 0

#if COMPUTE
void dgemm(double *local_a, double *local_b, double *local_c, int tile_dim);
#endif

int main(int argc, char **argv)
{
    int rank, nprocs, provided;
    int thread_i, num_threads;
    int num_workers;
    int tile_dim, tile_num, *tile_map;
    size_t elements_in_tile, tile_size;
    size_t sub_mat_elements;
    int tot_non_zero_tiles, tiles_per_rank;
    int p_dim, node_dim;
    int ppn;
    int *work_unit_table, work_units;
    double *sub_mats_ab, *sub_mat_a, *sub_mat_b, *sub_mat_c;
    MPI_Info win_info;
   
    double *win_c_mem;
    int *counter_win_mem;
    MPI_Win *wins_ab, win_c, win_counter;

#if SHOW_WORKLOAD_DIST
    int *threads_work_counter;
    int *all_worker_counter;
#endif

#if FINE_TIME
    double *t_get_threads, *t_accum_threads;
    double *t_get_flush_threads, *t_accum_flush_threads;
    double *t_get_workers, *t_accum_workers;
    double *t_get_flush_workers, *t_accum_flush_workers;
    double min_t_get, max_t_get, mean_t_get;
    double min_t_get_flush, max_t_get_flush, mean_t_get_flush;
    double min_t_accum, max_t_accum, mean_t_accum;
    double min_t_accum_flush, max_t_accum_flush, mean_t_accum_flush;
    int *threads_get_count, *threads_accum_count;
    int *workers_get_count, *workers_accum_count;
    int tot_get_count, tot_accum_count;
#else
    double t1, t2;
#endif

    /* initialize MPI environment */
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    if (provided < MPI_THREAD_MULTIPLE)
        MPI_Abort(MPI_COMM_WORLD, 1);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    num_threads = omp_get_max_threads();
    num_workers = nprocs * num_threads;

    /* argument checking and setting */
    if (setup(rank, nprocs, argc, argv, &tile_dim, &tile_num, &p_dim, &node_dim, &ppn)) {
        MPI_Finalize();
        exit(0);
    }
#if DEBUG
    if (rank == 0) {
        printf("tile_dim %d\n", tile_dim);
        printf("tile_num %d\n", tile_num);
        printf("p_dim %d\n", p_dim);
    }
#endif
    elements_in_tile = tile_dim * tile_dim;
    tile_size = elements_in_tile * sizeof(double);

    /* Create a map of non-sparse tiles in the whole matrix */
    tile_map = calloc(tile_num * tile_num, sizeof(int));
    init_tile_map(tile_map, tile_num, &tot_non_zero_tiles);
    /* For now, this map is the same for matrices A and B */
 
    /* The non-zero tiles are distributed amongst the ranks in a round-robin fashion */
    tiles_per_rank = tot_non_zero_tiles / nprocs;
    if (tot_non_zero_tiles % nprocs) {
        /* Although only some of the ranks need extra tiles, we allocate the extra tiles
         * to all the ranks so that the displacement from the start of the window's memory
         * to the A (disp_a), B (disp_b) or C (disp_c) submatrix is the same for all ranks.
         */
        tiles_per_rank++;
    }
 #if DEBUG
    printf("Rank %d: Non-zero tiles with me %d\n", rank, tiles_per_rank);
#endif

    /* init work unit table */
    init_work_unit_table(tile_map, tile_map, tile_map, tile_num, &work_unit_table, &work_units);
#if DEBUG
    if (rank == 0) printf("work_units %d\n", work_units);
#endif
 
    MPI_Info_create(&win_info);
#if OFI_WINDOW_HINTS
    MPI_Info_set(win_info, "which_accumulate_ops", "sum");
    MPI_Info_set(win_info, "disable_shm_accumulate", "true");
#endif
    /* Allocate and create RMA windows for the tiles in submatrices A, B, and C */
    sub_mat_elements = elements_in_tile * tiles_per_rank;
    
    sub_mats_ab = (double *) malloc(2 * sub_mat_elements * sizeof(double));
    sub_mat_a = sub_mats_ab;
    sub_mat_b = sub_mats_ab + sub_mat_elements;

    /* We are algorithmically constrained to use one window per rank for accumulates
     * on C. However, we do not need ordering for the accumulates. So, the MPI library
     * can use this information to issue operations on this window in parallel.
     */
    MPI_Info_set(win_info, "accumulate_ordering", "none");
    MPI_Win_allocate(sub_mat_elements * sizeof(double), sizeof(double),
                    win_info,
                    MPI_COMM_WORLD, &win_c_mem, &win_c);
    sub_mat_c = win_c_mem;

    MPI_Win_lock(MPI_LOCK_EXCLUSIVE, rank, MPI_MODE_NOCHECK, win_c);
    init_sub_mats(sub_mat_a, sub_mat_b, sub_mat_c, sub_mat_elements);
    MPI_Win_unlock(rank, win_c);
   
    wins_ab = (MPI_Win *) malloc(sizeof(MPI_Win) * num_threads);
    /* We can use separate windows for the Gets. Hence, we can expose parallelism
     * to the MPI library by creating multiple windows and use multiple VCIs that way */
    for (thread_i = 0; thread_i < num_threads; thread_i++) {
        MPI_Win_create(sub_mats_ab, 2 * sub_mat_elements * sizeof(double), sizeof(double),
                MPI_INFO_NULL, MPI_COMM_WORLD, &wins_ab[thread_i]);
    }
 
    /* Allocate RMA window for the counter that allows for load balancing */
    /* Since we don't measure the performance of fetch and ops, we don't need
     * multiple VCIs for this step. But, if needed, we can allocate multiple
     * VCIs the same way as we did for window of tile C since we are still
     * constrained to use a single window per rank. */
    if (rank == 0) {
        MPI_Win_allocate(sizeof(int), sizeof(int),
                    MPI_INFO_NULL,
                    MPI_COMM_WORLD, &counter_win_mem, &win_counter);
        
        /* initialize global counter */
        MPI_Win_lock(MPI_LOCK_EXCLUSIVE, 0, MPI_MODE_NOCHECK, win_counter);
        *counter_win_mem = 0;
        MPI_Win_unlock(0, win_counter); /* MEM_MODE: update to my private window becomes
                                         * visible in public window */
    } else {
        MPI_Win_allocate(0, sizeof(int),
                        MPI_INFO_NULL,
                        MPI_COMM_WORLD, &counter_win_mem,
                        &win_counter);
    }

#if SHOW_WORKLOAD_DIST
    threads_work_counter = calloc(num_threads, sizeof(int));
#endif

#if FINE_TIME
    t_get_threads = calloc(num_threads, sizeof(double));
    t_get_flush_threads = calloc(num_threads, sizeof(double));
    t_accum_threads = calloc(num_threads, sizeof(double));
    t_accum_flush_threads = calloc(num_threads, sizeof(double));
    threads_get_count = calloc(num_threads, sizeof(int));
    threads_accum_count = calloc(num_threads, sizeof(int));
    workers_get_count = calloc(num_workers, sizeof(int));
    workers_accum_count = calloc(num_workers, sizeof(int));
#endif
 
    MPI_Barrier(MPI_COMM_WORLD);

    for (thread_i = 0; thread_i < num_threads; thread_i++)
        MPI_Win_lock_all(MPI_MODE_NOCHECK, wins_ab[thread_i]);
    MPI_Win_lock_all(MPI_MODE_NOCHECK, win_c);
    MPI_Win_lock_all(MPI_MODE_NOCHECK, win_counter);

#if WARMUP
    /* Warmup */
#pragma omp parallel
    {
        int tid, work_id;
        int i, k, j;
        int prev_tile_c;
        double *target_tile;
        double *local_ta, *local_tb, *local_tc;
        const int one = 1;
        int global_tile_a, global_tile_b, global_tile_c;
        int target_rank_a, target_rank_b, target_rank_c;
        MPI_Aint target_offset_a, target_offset_b, target_offset_c;
        MPI_Aint disp_a, disp_b;

        tid = omp_get_thread_num();
 
        posix_memalign((void**)&local_ta, PAGE_SIZE, tile_size);
        posix_memalign((void**)&local_tb, PAGE_SIZE, tile_size);
        posix_memalign((void**)&local_tc, PAGE_SIZE, tile_size);
        memset(local_ta, 0, tile_size);
        memset(local_tb, 0, tile_size);
        memset(local_tc, 0, tile_size);

        disp_a = 0;
        disp_b = disp_a + sub_mat_elements;
 
        prev_tile_c = -1;
        
#pragma omp master
        {
            MPI_Barrier(MPI_COMM_WORLD);
        }
#pragma omp barrier
        
        do {
            /* read and increment global counter atomically */
            MPI_Fetch_and_op(&one, &work_id, MPI_INT, 0, 0, MPI_SUM, win_counter);
            MPI_Win_flush(0, win_counter);
            if (work_id >= work_units)
                break;
 
            global_tile_c = work_unit_table[work_id * 3 + 2];
            if (global_tile_c != prev_tile_c && prev_tile_c >= 0) {
                /* MPI_Accumulate locally accumulated C before proceeding */
                target_rank_c = target_rank_of_tile(prev_tile_c, nprocs);
                target_offset_c = offset_of_tile(prev_tile_c, nprocs, tile_dim);
                
                /* accumulate tile C (always use MPI since we need to ensure atomicity during accumulation) */
                MPI_Accumulate(local_tc, elements_in_tile, MPI_DOUBLE, target_rank_c, target_offset_c, elements_in_tile,
                               MPI_DOUBLE, MPI_SUM, win_c);
                MPI_Win_flush(target_rank_c, win_c);
                
                /* Reset the local C tile for local accumulation */
                memset(local_tc, 0, tile_size);
            }
            prev_tile_c = global_tile_c;

            /* calculate target rank from the work_id for A */
            global_tile_a = work_unit_table[work_id * 3 + 0];
            target_rank_a = target_rank_of_tile(global_tile_a, nprocs);
            target_offset_a = offset_of_tile(global_tile_a, nprocs, tile_dim); 

            /* Obtain tile A */
            if (target_rank_a == rank) {
                /* Copy tile A from local memory */
                target_tile = &sub_mat_a[(int) target_offset_a];
                for (i = 0; i < tile_dim; i++)
                    for (k = 0; k < tile_dim; k++)
                        local_ta[i*tile_dim + k] = target_tile[i*tile_dim + k];
            } else {
                /* get tile A */
                MPI_Get(local_ta, elements_in_tile, MPI_DOUBLE, target_rank_a, disp_a + target_offset_a, elements_in_tile, MPI_DOUBLE, wins_ab[tid]);
                MPI_Win_flush(target_rank_a, wins_ab[tid]);
            }

            /* calculate target rank from the work_id for B */
            global_tile_b = work_unit_table[work_id * 3 + 1];
            target_rank_b = target_rank_of_tile(global_tile_b, nprocs);
            target_offset_b = offset_of_tile(global_tile_b, nprocs, tile_dim); 

            /* Obtain tile B */
            if (target_rank_b == rank) {
                /* Copy tile B from local memory */
                target_tile = &sub_mat_b[(int) target_offset_b];
                for (k = 0; k < tile_dim; k++)
                    for (j = 0; j < tile_dim; j++)
                        local_tb[k*tile_dim + j] = target_tile[k*tile_dim + j];
            } else {
                /* get tile B */
                MPI_Get(local_tb, elements_in_tile, MPI_DOUBLE, target_rank_b, disp_b + target_offset_b, elements_in_tile, MPI_DOUBLE, wins_ab[tid]);
                MPI_Win_flush(target_rank_b, wins_ab[tid]);
            }
            
            /* compute Cij += Aik * Bkj */
            dgemm(local_ta, local_tb, local_tc, tile_dim);
        
        } while (work_id < work_units);
        
        if (prev_tile_c >= 0) {
            /* MPI_Accumulate locally accumulated C before finishing */
            target_rank_c = target_rank_of_tile(prev_tile_c, nprocs);
            target_offset_c = offset_of_tile(prev_tile_c, nprocs, tile_dim);
            
            /* accumulate tile C (always use MPI since we need to ensure atomicity during accumulation) */
            MPI_Accumulate(local_tc, elements_in_tile, MPI_DOUBLE, target_rank_c, target_offset_c, elements_in_tile,
                       MPI_DOUBLE, MPI_SUM, win_c);
            MPI_Win_flush(target_rank_c, win_c);
        }

        free(local_ta);
        free(local_tb);
        free(local_tc);
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    for (thread_i = 0; thread_i < num_threads; thread_i++)
        MPI_Win_unlock_all(wins_ab[thread_i]);
    MPI_Win_unlock_all(win_c);
    MPI_Win_unlock_all(win_counter); 
    //printf("Done with warmup!\n");

    MPI_Win_lock(MPI_LOCK_EXCLUSIVE, rank, MPI_MODE_NOCHECK, win_c);
    init_sub_mats(sub_mat_a, sub_mat_b, sub_mat_c, sub_mat_elements);
    MPI_Win_unlock(rank, win_c);
   
    if (rank == 0 ) {
        /* re-initialize global counter */
        MPI_Win_lock(MPI_LOCK_EXCLUSIVE, 0, MPI_MODE_NOCHECK, win_counter);
        *counter_win_mem = 0;
        MPI_Win_unlock(0, win_counter);
    }
    
    for (thread_i = 0; thread_i < num_threads; thread_i++)
        MPI_Win_lock_all(MPI_MODE_NOCHECK, wins_ab[thread_i]);
    MPI_Win_lock_all(MPI_MODE_NOCHECK, win_c);
    MPI_Win_lock_all(MPI_MODE_NOCHECK, win_counter);
#endif

    MPI_Barrier(MPI_COMM_WORLD);
    /* Benchmark */
#pragma omp parallel
    {
        int tid, work_id;
        int i, k, j;
        int prev_tile_c;
        double *target_tile;
        double *local_ta, *local_tb, *local_tc;
        const int one = 1;
        int global_tile_a, global_tile_b, global_tile_c;
        int target_rank_a, target_rank_b, target_rank_c;
        MPI_Aint target_offset_a, target_offset_b, target_offset_c;
        MPI_Aint disp_a, disp_b;

#if SHOW_WORKLOAD_DIST
        int my_work_counter = 0;
#endif

#if FINE_TIME
        int get_counter, accum_counter;
        double t_start;
        double t_get, t_accum;
        double t_get_flush, t_accum_flush;
#endif

        tid = omp_get_thread_num();
 
        posix_memalign((void**)&local_ta, PAGE_SIZE, tile_size);
        posix_memalign((void**)&local_tb, PAGE_SIZE, tile_size);
        posix_memalign((void**)&local_tc, PAGE_SIZE, tile_size);
        memset(local_ta, 0, tile_size);
        memset(local_tb, 0, tile_size);
        memset(local_tc, 0, tile_size);
        /*local_ta = calloc(elements_in_tile, sizeof(double));
        local_tb = calloc(elements_in_tile, sizeof(double));
        local_tc = calloc(elements_in_tile, sizeof(double));
        for (i = 0; i < tile_dim; i++) {
            for (j = 0; j < tile_dim; j++) {
                local_ta[i*tile_dim + j] = 7;
                local_tb[i*tile_dim + j] = 7;
            }
        }*/

        disp_a = 0;
        disp_b = disp_a + sub_mat_elements;
 
        prev_tile_c = -1;
        
#pragma omp master
        {
            MPI_Barrier(MPI_COMM_WORLD);
        }
#pragma omp barrier
#if FINE_TIME
        t_get = t_accum = 0;
        t_get_flush = t_accum_flush = 0;
        get_counter = accum_counter = 0;
#else
#pragma omp master
        {
            t1 = MPI_Wtime();
        }
#endif
        do {
            /* read and increment global counter atomically */
            MPI_Fetch_and_op(&one, &work_id, MPI_INT, 0, 0, MPI_SUM, win_counter);
            MPI_Win_flush(0, win_counter);
#if DEBUG
            printf("Worker %d\t%d\n", rank*num_threads + tid, work_id);
            fflush(stdout);
#endif
            if (work_id >= work_units)
                break;
 
#if SHOW_WORKLOAD_DIST
            my_work_counter++;
#endif 
            global_tile_c = work_unit_table[work_id * 3 + 2];
            if (global_tile_c != prev_tile_c && prev_tile_c >= 0) {
                /* MPI_Accumulate locally accumulated C before proceeding */
                target_rank_c = target_rank_of_tile(prev_tile_c, nprocs);
                target_offset_c = offset_of_tile(prev_tile_c, nprocs, tile_dim);
#if DEBUG
                double tile_sum = 0;
                int tile_i, tile_j;
                for (tile_i = 0; tile_i < tile_dim; tile_i++) {
                    for (tile_j = 0; tile_j < tile_dim; tile_j++) {
                        tile_sum += local_tc[tile_i*tile_dim + tile_j];
                    }
                }
                printf("Worker %d accumulating (0) tile %d with value %.1f on rank %d using offset %d\n", rank*num_threads + tid, prev_tile_c, tile_sum, target_rank_c, target_offset_c); 
                fflush(stdout);
#endif
                /* accumulate tile C (always use MPI since we need to ensure atomicity during accumulation) */
#if FINE_TIME
                accum_counter++;
                t_start = MPI_Wtime();
#endif
                MPI_Accumulate(local_tc, elements_in_tile, MPI_DOUBLE, target_rank_c, target_offset_c, elements_in_tile,
                               MPI_DOUBLE, MPI_SUM, win_c);
#if FINE_TIME
                t_accum += (MPI_Wtime() - t_start);
                t_start = MPI_Wtime();
#endif
                MPI_Win_flush(target_rank_c, win_c);
#if FINE_TIME
                t_accum_flush += (MPI_Wtime() - t_start);
#endif
#if DEBUG
                printf("Worker %d accumulated (0) tile %d on rank %d using offset %d\n", rank*num_threads + tid, prev_tile_c, target_rank_c, target_offset_c); 
                fflush(stdout);
#endif
                /* Reset the local C tile for local accumulation */
                memset(local_tc, 0, tile_size);
            }
            prev_tile_c = global_tile_c;

            /* calculate target rank from the work_id for A */
            global_tile_a = work_unit_table[work_id * 3 + 0];
            target_rank_a = target_rank_of_tile(global_tile_a, nprocs);
            target_offset_a = offset_of_tile(global_tile_a, nprocs, tile_dim); 

            /* Obtain tile A */
            if (target_rank_a == rank) {
                /* Copy tile A from local memory */
                target_tile = &sub_mat_a[(int) target_offset_a];
                for (i = 0; i < tile_dim; i++)
                    for (k = 0; k < tile_dim; k++)
                        local_ta[i*tile_dim + k] = target_tile[i*tile_dim + k];
            } else {
                /* get tile A */
#if DEBUG
                printf("Worker %d trying to get A tile %d from rank %d using offset %d (old value %.1f)\n", rank*num_threads + tid, global_tile_a, target_rank_a, disp_a + target_offset_a, local_ta[0]);
                fflush(stdout);
#endif
#if FINE_TIME
                get_counter++;
                t_start = MPI_Wtime();
#endif
                MPI_Get(local_ta, elements_in_tile, MPI_DOUBLE, target_rank_a, disp_a + target_offset_a, elements_in_tile, MPI_DOUBLE, wins_ab[tid]);
#if FINE_TIME
                t_get += (MPI_Wtime() - t_start);
                t_start = MPI_Wtime();
#endif
                MPI_Win_flush(target_rank_a, wins_ab[tid]);
#if FINE_TIME
                t_get_flush += (MPI_Wtime() - t_start);
#endif      
                //if (local_ta[0] != 1)
                //    printf("WRONG: Worker %d got wrong value of tile A\n", rank*num_threads + tid);
            }
#if DEBUG
                printf("Worker %d got A tile %d with value %.1f from rank %d using offset %d\n", rank*num_threads + tid, global_tile_a, local_ta[0], target_rank_a, disp_a + target_offset_a);
                fflush(stdout);
#endif

            /* calculate target rank from the work_id for B */
            global_tile_b = work_unit_table[work_id * 3 + 1];
            target_rank_b = target_rank_of_tile(global_tile_b, nprocs);
            target_offset_b = offset_of_tile(global_tile_b, nprocs, tile_dim); 

            /* Obtain tile B */
            if (target_rank_b == rank) {
                /* Copy tile B from local memory */
                target_tile = &sub_mat_b[(int) target_offset_b];
                for (k = 0; k < tile_dim; k++)
                    for (j = 0; j < tile_dim; j++)
                        local_tb[k*tile_dim + j] = target_tile[k*tile_dim + j];
            } else {
                /* get tile B */
#if DEBUG
                printf("Worker %d trying to get B tile %d from rank %d using offset %d (old value %.1f)\n", rank*num_threads + tid, global_tile_b, target_rank_b, disp_b + target_offset_b, local_tb[0]);
                fflush(stdout);
#endif
#if FINE_TIME
                get_counter++;
                t_start = MPI_Wtime();
#endif
                MPI_Get(local_tb, elements_in_tile, MPI_DOUBLE, target_rank_b, disp_b + target_offset_b, elements_in_tile, MPI_DOUBLE, wins_ab[tid]);
#if FINE_TIME
                t_get += (MPI_Wtime() - t_start);
                t_start = MPI_Wtime();
#endif
                MPI_Win_flush(target_rank_b, wins_ab[tid]);
#if FINE_TIME
                t_get_flush += (MPI_Wtime() - t_start);
#endif
                //if (local_tb[0] != 1)
                //    printf("WRONG: Worker %d got wrong value of tile B\n", rank*num_threads + tid);
            }
#if DEBUG
                printf("Worker %d got B tile %d with value %.1f from rank %d using offset %d\n", rank*num_threads + tid, global_tile_b, local_tb[0], target_rank_b, disp_b + target_offset_b);
                fflush(stdout);
#endif
#if COMPUTE
            /* compute Cij += Aik * Bkj */
            dgemm(local_ta, local_tb, local_tc, tile_dim);
#endif
#if DEBUG
                printf("Worker %d computed C tile %d with value %.1f\n", rank*num_threads + tid, global_tile_c, local_tc[0]);
                fflush(stdout);
#endif
        } while (work_id < work_units);
        
        if (prev_tile_c >= 0) {
            /* MPI_Accumulate locally accumulated C before finishing */
            target_rank_c = target_rank_of_tile(prev_tile_c, nprocs);
            target_offset_c = offset_of_tile(prev_tile_c, nprocs, tile_dim);
            
            /* accumulate tile C (always use MPI since we need to ensure atomicity during accumulation) */
#if DEBUG
            double tile_sum = 0;
            int tile_i, tile_j;
            for (tile_i = 0; tile_i < tile_dim; tile_i++) {
                for (tile_j = 0; tile_j < tile_dim; tile_j++) {
                    tile_sum += local_tc[tile_i*tile_dim + tile_j];
                }
            }
            printf("Worker %d accumulating (1) tile %d with value %.1f on rank %d using offset %d\n", rank*num_threads + tid, prev_tile_c, tile_sum, target_rank_c, target_offset_c); 
            fflush(stdout);
#endif
#if FINE_TIME
            accum_counter++;
            t_start = MPI_Wtime();
#endif
            MPI_Accumulate(local_tc, elements_in_tile, MPI_DOUBLE, target_rank_c, target_offset_c, elements_in_tile,
                       MPI_DOUBLE, MPI_SUM, win_c);
#if FINE_TIME
            t_accum += (MPI_Wtime() - t_start);
            t_start = MPI_Wtime();
#endif
            MPI_Win_flush(target_rank_c, win_c);
#if FINE_TIME
            t_accum_flush += (MPI_Wtime() - t_start);
#endif
#if DEBUG
        printf("Worker %d accumulated (1) tile %d on rank %d using offset %d\n", rank*num_threads + tid, prev_tile_c, target_rank_c, target_offset_c);
        fflush(stdout);
#endif
        }

#if SHOW_WORKLOAD_DIST
        threads_work_counter[tid] = my_work_counter;
#endif

#if FINE_TIME
        threads_get_count[tid] = get_counter;
        threads_accum_count[tid] = accum_counter;
        if (get_counter > 0) {
            t_get_threads[tid] = t_get / get_counter;
            t_get_flush_threads[tid] = t_get_flush / get_counter;
        }
        if (accum_counter > 0) {
            t_accum_threads[tid] = t_accum / accum_counter;
            t_accum_flush_threads[tid] = t_accum_flush / accum_counter;
        }
#endif

        free(local_ta);
        free(local_tb);
        free(local_tc);
#if DEBUG
        printf("Worker %d done (before parallel region)\n", rank*num_threads + tid);
#endif
    }
#if DEBUG
        printf("Rank %d waiting on barrier\n", rank);
#endif
    MPI_Barrier(MPI_COMM_WORLD);
#if DEBUG
        printf("Rank %d done with barrier\n", rank);
#endif
    //printf("Rank %d: sub_mat_c[0] is %.1f\n", rank, sub_mat_c[0]);
    //printf("Rank %d: sub_mat_c[1] is %.1f\n", rank, sub_mat_c[1]);
    //printf("Rank %d done!\n", rank);

#if SHOW_WORKLOAD_DIST
    if (rank == 0)
        all_worker_counter = calloc(num_workers, sizeof(int));
    else
        all_worker_counter = NULL;
    MPI_Gather(threads_work_counter, num_threads, MPI_INT, all_worker_counter, num_threads, MPI_INT, 0, MPI_COMM_WORLD);
#endif

#if FINE_TIME
    if (rank == 0) {
        t_get_workers = calloc(nprocs*num_threads, sizeof(double));
        t_get_flush_workers = calloc(nprocs*num_threads, sizeof(double));
        t_accum_workers = calloc(nprocs*num_threads, sizeof(double));
        t_accum_flush_workers = calloc(nprocs*num_threads, sizeof(double));
        if (!t_get_workers || !t_accum_workers || !t_get_flush_workers || !t_accum_flush_workers) {
            fprintf(stderr, "Unable to allocate memory for t_get_workers, t_get_flush_workers, t_accum_workers, or t_accum_flush_workers\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    } else
        t_get_workers = t_get_flush_workers = t_accum_workers = t_accum_flush_workers = NULL;
    MPI_Gather(t_get_threads, num_threads, MPI_DOUBLE, t_get_workers, num_threads, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gather(t_get_flush_threads, num_threads, MPI_DOUBLE, t_get_flush_workers, num_threads, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gather(t_accum_threads, num_threads, MPI_DOUBLE, t_accum_workers, num_threads, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gather(t_accum_flush_threads, num_threads, MPI_DOUBLE, t_accum_flush_workers, num_threads, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    /* The reduce is causing an annoying problem: win_a, win_b, win_c become invalid on rank 0 after the reduce */
    //MPI_Reduce(threads_get_count, &tot_get_count, num_threads, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    //MPI_Reduce(threads_accum_count, &tot_accum_count, num_threads, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Gather(threads_get_count, num_threads, MPI_INT, workers_get_count, num_threads, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Gather(threads_accum_count, num_threads, MPI_INT, workers_accum_count, num_threads, MPI_INT, 0, MPI_COMM_WORLD);

    min_t_get = max_t_get = mean_t_get = min_t_get_flush = max_t_get_flush = mean_t_get_flush = 0;
    min_t_accum = max_t_accum = mean_t_accum = min_t_accum_flush = max_t_accum_flush = mean_t_accum_flush = 0;
    tot_get_count = tot_accum_count = 0;

    if (rank == 0) {
        int pi;
        double sum_t_get, sum_t_get_flush, sum_t_accum, sum_t_accum_flush;
        int nworkers_who_got, nworkers_who_accumed;

        nworkers_who_got = nworkers_who_accumed = 0;

        max_t_get = max_t_get_flush = max_t_accum = max_t_accum_flush = -1;
        min_t_get = min_t_get_flush = min_t_accum = min_t_accum_flush = 9999;
        sum_t_get = sum_t_get_flush = sum_t_accum = sum_t_accum_flush = 0;

        for (pi = 0; pi < num_workers; pi++) {
            if (t_get_workers[pi] > 0) {
                nworkers_who_got++;
                if (max_t_get < t_get_workers[pi])
                    max_t_get = t_get_workers[pi];
                if (min_t_get > t_get_workers[pi])
                    min_t_get = t_get_workers[pi];
                sum_t_get += t_get_workers[pi];
                /* If they got, they flushed too */
                if (max_t_get_flush < t_get_flush_workers[pi])
                    max_t_get_flush = t_get_flush_workers[pi];
                if (min_t_get_flush > t_get_flush_workers[pi])
                    min_t_get_flush = t_get_flush_workers[pi];
                sum_t_get_flush += t_get_flush_workers[pi];
            }

            if (t_accum_workers[pi] > 0) {
                nworkers_who_accumed++;
                if (max_t_accum < t_accum_workers[pi])
                    max_t_accum = t_accum_workers[pi];
                if (min_t_accum > t_accum_workers[pi])
                    min_t_accum = t_accum_workers[pi];
                sum_t_accum += t_accum_workers[pi];
                /* If they accumed, they flushed too */
                if (max_t_accum_flush < t_accum_flush_workers[pi])
                    max_t_accum_flush = t_accum_flush_workers[pi];
                if (min_t_accum_flush > t_accum_flush_workers[pi])
                    min_t_accum_flush = t_accum_flush_workers[pi];
                sum_t_accum_flush += t_accum_flush_workers[pi];
            }

            tot_get_count += workers_get_count[pi];
            tot_accum_count += workers_accum_count[pi];   
        }

        mean_t_get = sum_t_get / nworkers_who_got;
        mean_t_get_flush = sum_t_get_flush / nworkers_who_got;
        mean_t_accum = sum_t_accum / nworkers_who_accumed;
        mean_t_accum_flush = sum_t_accum_flush / nworkers_who_accumed;
    }
#else
    t2 = MPI_Wtime();
#endif

    for (thread_i = 0; thread_i < num_threads; thread_i++)
        MPI_Win_sync(wins_ab[thread_i]);    /* MEM_MODE: synchronize private and public window copies */
    MPI_Win_sync(win_c);    /* MEM_MODE: synchronize private and public window copies */
    if (rank == 0) {
        int mat_dim = tile_num * tile_dim;

#if SHOW_WORKLOAD_DIST
        int i;
        printf("Worker\tUnits\n");
        for (i = 0; i < num_workers; i++) {
            printf("%d\t%d\n", i, all_worker_counter[i]);
        }
        printf("\n");
#endif

#if FINE_TIME
        printf("mat_dim,tile_dim,work_units,nworkers,"
                "min_get_time,max_get_time,mean_get_time,min_accum_time,max_accum_time,mean_accum_time,"
                "min_get_flush_time,max_get_flush_time,mean_get_flush_time,min_accum_flush_time,max_accum_flush_time,mean_accum_flush_time\n");
        printf("%d,%d,%d,%d,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f\n", mat_dim, tile_dim, work_units, num_workers,
                min_t_get, max_t_get, mean_t_get, min_t_accum, max_t_accum, mean_t_accum,
                min_t_get_flush, max_t_get_flush, mean_t_get_flush, min_t_accum_flush, max_t_accum_flush, mean_t_accum_flush);
#else
        printf("mat_dim,tile_dim,work_units,nworkers,time\n");
        printf("%d,%d,%d,%d,%f\n", mat_dim, tile_dim, work_units, num_workers, t2 - t1);
#endif
    }

#if CHECK_FOR_ERRORS
    if (rank == 0) {
        /* Check matrices */
        int i, j, k;
        int tile_i, tile_j;
 
        size_t mat_dim = tile_num * tile_dim;
        double *mat_a = calloc(mat_dim * mat_dim, sizeof(double));
        double *mat_b = calloc(mat_dim * mat_dim, sizeof(double));
        double *mat_correct_c = calloc(mat_dim * mat_dim, sizeof(double));
        double *mat_c = calloc(mat_dim * mat_dim, sizeof(double));
        
        init_mat_according_to_map(mat_a, mat_dim); 
        init_mat_according_to_map(mat_b, mat_dim); 

        for (i = 0; i < mat_dim; i++) {
            for (j = 0; j < mat_dim; j++) {
               mat_c[i*mat_dim + j] = 0; 
               mat_correct_c[i*mat_dim + j] = 0; 
            }
        }

        for (i = 0; i < mat_dim; i++) {
            for (j = 0; j < mat_dim; j++) {
                for (k = 0; k < mat_dim; k++) {
                    mat_correct_c[i*mat_dim + j] += mat_a[i*mat_dim + k] * mat_b[k*mat_dim + j];
                }
            }
        }

        double *local_c = calloc(tile_dim * tile_dim, sizeof(double));
     
        for (i = 0; i < tile_num; i++) {
            for (j = 0; j < tile_num; j++) {
                int global_tile_c = tile_map[i*tile_num + j];
                
                if (global_tile_c != -1) {
                    int target_rank_c = target_rank_of_tile(global_tile_c, nprocs);
                    MPI_Aint target_offset_c = offset_of_tile(global_tile_c, nprocs, tile_dim); 
                    
                    MPI_Get(local_c, elements_in_tile, MPI_DOUBLE, target_rank_c, target_offset_c, elements_in_tile, MPI_DOUBLE, win_c);
                    MPI_Win_flush(target_rank_c, win_c);
#if DEBUG
                    //printf("Rank %d got tile %d with value %.1f from rank %d using offset %d\n", rank, global_tile_c, local_c[0], target_rank_c, target_offset_c); 
#endif
                    for (tile_i = 0; tile_i < tile_dim; tile_i++) {
                        for (tile_j = 0; tile_j < tile_dim; tile_j++) {
                           mat_c[i*tile_dim*mat_dim + j*tile_dim + tile_i*mat_dim + tile_j] = local_c[tile_i*tile_dim + tile_j]; 
                        }
                    }
                }
            }
        }

        /* Check for errors */
        int errors = 0;
#if DEBUG
        printf("Correct matrix:\n");
        for (i = 0; i < mat_dim; i++) {
            for (j = 0; j < mat_dim; j++) {
                printf("%.1f\t", mat_correct_c[i*mat_dim + j]); 
            }
            printf("\n");
        }
        printf("\n");
        printf("Computed matrix:\n");
        for (i = 0; i < mat_dim; i++) {
            for (j = 0; j < mat_dim; j++) {
                printf("%.1f\t", mat_c[i*mat_dim + j]); 
            }
            printf("\n");
        }
#endif
        for (i = 0; i < mat_dim; i++) {
            for (j = 0; j < mat_dim; j++) {
                if (mat_correct_c[i*mat_dim + j] != mat_c[i*mat_dim + j])
                    errors++;
            }
        }
        
        if (errors)
            fprintf(stderr, "Found %d errors\n", errors);
        if (errors == 0)
            fprintf(stderr, "Test passed!\n");
        
        free(local_c);
        free(mat_a);
        free(mat_b);
        free(mat_c);
        free(mat_correct_c);
    }
#endif

    for (thread_i = 0; thread_i < num_threads; thread_i++)
        MPI_Win_unlock_all(wins_ab[thread_i]);
    MPI_Win_unlock_all(win_c);
    MPI_Win_unlock_all(win_counter);

    MPI_Win_free(&win_counter);
    for (thread_i = 0; thread_i < num_threads; thread_i++)
        MPI_Win_free(&wins_ab[thread_i]);
    MPI_Win_free(&win_c);

    MPI_Info_free(&win_info);

#if FINE_TIME
    free(t_get_threads);
    free(t_get_flush_threads);
    free(t_accum_threads);
    free(t_accum_flush_threads);
    free(threads_get_count);
    free(threads_accum_count);
    free(workers_get_count);
    free(workers_accum_count);
    free(t_get_workers);
    free(t_get_flush_workers);
    free(t_accum_workers);
    free(t_accum_flush_workers);
#endif

#if SHOW_WORKLOAD_DIST
    free(threads_work_counter);
    free(all_worker_counter);
#endif

    free(tile_map);
    free(work_unit_table);

    free(wins_ab);

    MPI_Finalize();
    return 0;
}

#if COMPUTE
void dgemm(double *local_a, double *local_b, double *local_c, int tile_dim)
{
    int i, j, k;

    for (j = 0; j < tile_dim; j++) {
        for (i = 0; i < tile_dim; i++) {
            for (k = 0; k < tile_dim; k++)
                local_c[j + i * tile_dim] += local_a[k + i * tile_dim] * local_b[j + k * tile_dim];
        }
    }
}
#endif

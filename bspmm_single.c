/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */

#include "bspmm.h"

/*
 * Block sparse matrix multiplication using RMA operations, a global counter for workload
 * distribution, MPI_THREAD_SINGLE mode.
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
 * evenly distributed amongst the ranks. Each rank will locally accumulate C until
 * its next work unit corresponds to a different C tile.
 *
 * The distribution of work between the ranks of all the workers is dynamic:
 * each rank reads a counter to obtain its work id. The counter is updated
 * atomically each time it is read.
 */

#define OFI_WINDOW_HINTS 0
#define COMPUTE 1
#define FINE_TIME 1
#define WARMUP 1
#define CHECK_FOR_ERRORS 0
#define SHOW_WORKLOAD_DIST 0

#if COMPUTE
void dgemm(double *local_a, double *local_b, double *local_c, int tile_dim);
#endif

int main(int argc, char **argv)
{
    int rank, nprocs;
#if SHOW_WORKLOAD_DIST
    int my_work_counter, *all_worker_counter;
    int *alt_rank;
#endif
    int tile_dim, tile_num, *tile_map;
    size_t elements_in_tile, tile_size;
    size_t sub_mat_elements;
    int tot_non_zero_tiles, tot_tiles_per_rank;
    int p_dim, node_dim;
    int ppn;
    int *work_unit_table, work_units;
    double *sub_mat_a, *sub_mat_b, *sub_mat_c;
    MPI_Aint disp_a, disp_b, disp_c;
#if OFI_WINDOW_HINTS
    MPI_Info win_info;
#endif

    int work_id;
    int i, k, j;
    int prev_tile_c;
    double *target_tile;
    double *local_a, *local_b, *local_c;
    const int one = 1;
    
    double *win_mem;
    int *counter_win_mem;
    MPI_Win win, win_counter;

#if FINE_TIME
    double t_start;
    double t_get, t_accum;
    double t_get_flush, t_accum_flush;
    double t_per_get, t_per_accum;
    double t_per_get_flush, t_per_accum_flush;
    int get_counter, accum_counter;
    double *t_get_procs, *t_accum_procs;
    double *t_get_flush_procs, *t_accum_flush_procs;
    double min_t_get, max_t_get, mean_t_get;
    double min_t_accum, max_t_accum, mean_t_accum;
    double min_t_get_flush, max_t_get_flush, mean_t_get_flush;
    double min_t_accum_flush, max_t_accum_flush, mean_t_accum_flush;
    int tot_get_count, tot_accum_count;
#else
    double t1, t2;
#endif

    int in_node_p_dim;
    int node_i, node_j;
    int in_node_i, in_node_j;
    int rank_in_parray;
    MPI_Comm comm_world;

    /* initialize MPI environment */
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    /* argument checking and setting */
    if (setup(rank, nprocs, argc, argv, &tile_dim, &tile_num, &p_dim, &node_dim, &ppn)) {
        MPI_Finalize();
        exit(0);
    }
    elements_in_tile = tile_dim * tile_dim;
    tile_size = elements_in_tile * sizeof(double);

    in_node_p_dim = p_dim / node_dim;
    /* find my rank in the processor array from my rank in COMM_WORLD */
    node_i = rank / (ppn * node_dim);
    node_j = (rank / ppn) % node_dim;
    in_node_i = (rank / in_node_p_dim) % in_node_p_dim;
    in_node_j = rank % in_node_p_dim;
    rank_in_parray = (node_i * ppn * node_dim) + (in_node_i * p_dim) + (node_j * in_node_p_dim) + in_node_j;

    /* Change rank to match the logical ranks used in this application */
    MPI_Comm_split(MPI_COMM_WORLD, 0, rank_in_parray, &comm_world);
    MPI_Comm_rank(comm_world, &rank);

#if DEBUG
    if (rank == 0) {
        printf("tile_dim %d\n", tile_dim);
        printf("tile_num %d\n", tile_num);
        printf("p_dim %d\n", p_dim);
    }
#endif

    /* Create a map of non-sparse tiles in the whole matrix */
    tile_map = calloc(tile_num * tile_num, sizeof(int));
    init_tile_map(tile_map, tile_num, &tot_non_zero_tiles);
    /* For now, this map is the same for matrices A and B */
 
    /* The non-zero tiles are distributed amongst the ranks in a round-robin fashion */
    tot_tiles_per_rank = tot_non_zero_tiles / nprocs;
    if (tot_non_zero_tiles % nprocs) {
        /* Although only some of the ranks need extra tiles, we allocate the extra tiles
         * to all the ranks so that the displacement from the start of the window's memory
         * to the A (disp_a), B (disp_b) or C (disp_c) submatrix is the same for all ranks.*/ 
        tot_tiles_per_rank++;
    }
#if DEBUG
    printf("Rank %d: Non-zero tiles with me %d\n", rank, tot_tiles_per_rank);
#endif

    /* init work unit table */
    init_work_unit_table(tile_map, tile_map, tile_map, tile_num, &work_unit_table, &work_units);
#if DEBUG
    if (rank == 0) printf("work_units %d\n", work_units);
#endif

#if OFI_WINDOW_HINTS
    MPI_Info_create(&win_info);
    MPI_Info_set(win_info, "which_accumulate_ops", "sum");
    MPI_Info_set(win_info, "disable_shm_accumulate", "true");
#endif

    /* Non-zero tiles distributed evenly between the processes */
    sub_mat_elements = elements_in_tile * tot_tiles_per_rank;

    /* Allocate and create RMA windows for the tiles in A, B, and C */
    MPI_Win_allocate(3 * sub_mat_elements * sizeof(double), sizeof(double),
#if OFI_WINDOW_HINTS
                    win_info,
#else
                    MPI_INFO_NULL,
#endif
                    comm_world, &win_mem, &win);

    sub_mat_a = win_mem;
    sub_mat_b = sub_mat_a + sub_mat_elements;
    sub_mat_c = sub_mat_b + sub_mat_elements;

    MPI_Win_lock(MPI_LOCK_EXCLUSIVE, rank, MPI_MODE_NOCHECK, win);
    init_sub_mats(sub_mat_a, sub_mat_b, sub_mat_c, sub_mat_elements);
    MPI_Win_unlock(rank, win);

    /* Allocate RMA window for the counter that allows for load balancing */
    if (rank == 0) {
        MPI_Win_allocate(sizeof(int), sizeof(int),
#if OFI_WINDOW_HINTS
                win_info,
#else   
                MPI_INFO_NULL,
#endif   
                comm_world, &counter_win_mem, &win_counter);
        /* initialize global counter */
        MPI_Win_lock(MPI_LOCK_EXCLUSIVE, 0, MPI_MODE_NOCHECK, win_counter);
        *counter_win_mem = 0;
        MPI_Win_unlock(0, win_counter); /* MEM_MODE: update to my private window becomes
                                         * visible in public window */
    } else {
        MPI_Win_allocate(0, sizeof(int),
#if OFI_WIDOW_HINTS
                        win_info,       
#else
                        MPI_INFO_NULL,
#endif        
                        comm_world, &counter_win_mem, &win_counter);
    }

    disp_a = 0;
    disp_b = disp_a + sub_mat_elements;
    disp_c = disp_b + sub_mat_elements;

    MPI_Barrier(comm_world);

    MPI_Win_lock_all(MPI_MODE_NOCHECK, win);
    MPI_Win_lock_all(MPI_MODE_NOCHECK, win_counter);

    local_a = calloc(elements_in_tile, sizeof(double));
    local_b = calloc(elements_in_tile, sizeof(double));
    local_c = calloc(elements_in_tile, sizeof(double));

#if WARMUP
    /* Warmup */
    MPI_Barrier(comm_world);
    
    prev_tile_c = -1; 
    do {
        /* read and increment global counter atomically */
        MPI_Fetch_and_op(&one, &work_id, MPI_INT, 0, 0, MPI_SUM, win_counter);
        MPI_Win_flush(0, win_counter);
        if (work_id >= work_units)
            break;

        int global_tile_c = work_unit_table[work_id * 3 + 2];
        if (global_tile_c != prev_tile_c && prev_tile_c >= 0) {
            /* MPI_Accumulate locally accumulated C before proceeding */
            int target_rank_c = target_rank_of_tile(prev_tile_c, nprocs);
            MPI_Aint target_offset_c = offset_of_tile(prev_tile_c, nprocs, tile_dim);
            
            /* accumulate tile C (always use MPI since we need to ensure atomicity during accumulation) */
            MPI_Accumulate(local_c, elements_in_tile, MPI_DOUBLE, target_rank_c, disp_c + target_offset_c, elements_in_tile,
                           MPI_DOUBLE, MPI_SUM, win);
            MPI_Win_flush(target_rank_c, win);
            
            /* Reset the local C tile for local accumulation */
            memset(local_c, 0, tile_size);
        } 
        prev_tile_c = global_tile_c;

        /* calculate target rank from the work_id for A */
        int global_tile_a = work_unit_table[work_id * 3 + 0];
        int target_rank_a = target_rank_of_tile(global_tile_a, nprocs);
        MPI_Aint target_offset_a = offset_of_tile(global_tile_a, nprocs, tile_dim); 

        /* Obtain tile A */
        if (target_rank_a == rank) {
            /* Copy tile A from local memory */
            target_tile = &sub_mat_a[(int) target_offset_a];
            for (i = 0; i < tile_dim; i++)
                for (k = 0; k < tile_dim; k++)
                    local_a[i*tile_dim + k] = target_tile[i*tile_dim + k];
        } else {
            /* get tile A */
            MPI_Get(local_a, elements_in_tile, MPI_DOUBLE, target_rank_a, disp_a + target_offset_a, elements_in_tile, MPI_DOUBLE, win);
            MPI_Win_flush(target_rank_a, win);
        }

        /* calculate target rank from the work_id for B */
        int global_tile_b = work_unit_table[work_id * 3 + 1];
        int target_rank_b = target_rank_of_tile(global_tile_b, nprocs);
        MPI_Aint target_offset_b = offset_of_tile(global_tile_b, nprocs, tile_dim); 

        /* Obtain tile B */
        if (target_rank_b == rank) {
            /* Copy tile B from local memory */
            target_tile = &sub_mat_b[(int) target_offset_b];
            for (k = 0; k < tile_dim; k++)
                for (j = 0; j < tile_dim; j++)
                    local_b[k*tile_dim + j] = target_tile[k*tile_dim + j];
        } else {
            /* get tile B */
            MPI_Get(local_b, elements_in_tile, MPI_DOUBLE, target_rank_b, disp_b + target_offset_b, elements_in_tile, MPI_DOUBLE,
                    win);
            MPI_Win_flush(target_rank_b, win);
        }
        
        /* compute Cij += Aik * Bkj */
        dgemm(local_a, local_b, local_c, tile_dim);
    
    } while (work_id < work_units);

    if (prev_tile_c >= 0) {
        /* MPI_Accumulate locally accumulated C before finishing */
        int target_rank_c = target_rank_of_tile(prev_tile_c, nprocs);
        MPI_Aint target_offset_c = offset_of_tile(prev_tile_c, nprocs, tile_dim);
        
        /* accumulate tile C (always use MPI since we need to ensure atomicity during accumulation) */
        MPI_Accumulate(local_c, elements_in_tile, MPI_DOUBLE, target_rank_c, disp_c + target_offset_c, elements_in_tile,
                   MPI_DOUBLE, MPI_SUM, win);
        MPI_Win_flush(target_rank_c, win);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    
    MPI_Win_unlock_all(win);
    MPI_Win_unlock_all(win_counter);

    memset(local_c, 0, tile_size);
    
    MPI_Win_lock(MPI_LOCK_EXCLUSIVE, rank, MPI_MODE_NOCHECK, win);
    init_sub_mats(sub_mat_a, sub_mat_b, sub_mat_c, sub_mat_elements);
    MPI_Win_unlock(rank, win);

    if (rank == 0) {
        /* re-initialize global counter */
        MPI_Win_lock(MPI_LOCK_EXCLUSIVE, 0, MPI_MODE_NOCHECK, win_counter);
        *counter_win_mem = 0;
        MPI_Win_unlock(0, win_counter); 
    }
    
    MPI_Win_lock_all(MPI_MODE_NOCHECK, win);
    MPI_Win_lock_all(MPI_MODE_NOCHECK, win_counter);
#endif

    MPI_Barrier(MPI_COMM_WORLD);
    /* Benchmark */

    prev_tile_c = -1; 
#if SHOW_WORKLOAD_DIST
    my_work_counter = 0;
#endif

    MPI_Barrier(comm_world);
#if FINE_TIME
    t_get = t_accum = 0;
    t_get_flush = t_accum_flush = 0;
    get_counter = accum_counter = 0;
#else
    t1 = MPI_Wtime();
#endif
    do {
        /* read and increment global counter atomically */
        MPI_Fetch_and_op(&one, &work_id, MPI_INT, 0, 0, MPI_SUM, win_counter);
        MPI_Win_flush(0, win_counter);
#if DEBUG
        //printf("Rank %d\t%d\n", rank, work_id);
#endif
        if (work_id >= work_units)
            break;

#if SHOW_WORKLOAD_DIST
        my_work_counter++;
#endif

        int global_tile_c = work_unit_table[work_id * 3 + 2];
        if (global_tile_c != prev_tile_c && prev_tile_c >= 0) {
            /* MPI_Accumulate locally accumulated C before proceeding */
            int target_rank_c = target_rank_of_tile(prev_tile_c, nprocs);
            MPI_Aint target_offset_c = offset_of_tile(prev_tile_c, nprocs, tile_dim);
#if DEBUG
            double tile_sum = 0;
            int tile_i, tile_j;
            for (tile_i = 0; tile_i < tile_dim; tile_i++) {
                for (tile_j = 0; tile_j < tile_dim; tile_j++) {
                    tile_sum += local_c[tile_i*tile_dim + tile_j];
                }
            }
            printf("Rank %d accumulating tile %d with value %.1f on rank %d using offset %d\n", rank, prev_tile_c, tile_sum, target_rank_c, target_offset_c); 
#endif
            /* accumulate tile C (always use MPI since we need to ensure atomicity during accumulation) */
#if FINE_TIME
            accum_counter++;
            t_start = MPI_Wtime();
#endif
            MPI_Accumulate(local_c, elements_in_tile, MPI_DOUBLE, target_rank_c, disp_c + target_offset_c, elements_in_tile,
                           MPI_DOUBLE, MPI_SUM, win);
#if FINE_TIME
            t_accum += (MPI_Wtime() - t_start);
            t_start = MPI_Wtime();
#endif
            MPI_Win_flush(target_rank_c, win);
#if FINE_TIME
            t_accum_flush += (MPI_Wtime() - t_start);
#endif
            /* Reset the local C tile for local accumulation */
            memset(local_c, 0, tile_size);
        } 
        prev_tile_c = global_tile_c;

        /* calculate target rank from the work_id for A */
        int global_tile_a = work_unit_table[work_id * 3 + 0];
        int target_rank_a = target_rank_of_tile(global_tile_a, nprocs);
        MPI_Aint target_offset_a = offset_of_tile(global_tile_a, nprocs, tile_dim); 

        /* Obtain tile A */
        if (target_rank_a == rank) {
            /* Copy tile A from local memory */
            target_tile = &sub_mat_a[(int) target_offset_a];
            for (i = 0; i < tile_dim; i++)
                for (k = 0; k < tile_dim; k++)
                    local_a[i*tile_dim + k] = target_tile[i*tile_dim + k];
        } else {
#if DEBUG
            //printf("Rank %d trying to get A tile %d from rank %d\n", rank, global_tile_a, target_rank_a);
#endif
            /* get tile A */
#if FINE_TIME
            get_counter++;
            t_start = MPI_Wtime();
#endif
            MPI_Get(local_a, elements_in_tile, MPI_DOUBLE, target_rank_a, disp_a + target_offset_a, elements_in_tile, MPI_DOUBLE, win);
#if FINE_TIME
            t_get += (MPI_Wtime() - t_start);
            t_start = MPI_Wtime();
#endif
            MPI_Win_flush(target_rank_a, win);
#if FINE_TIME
            t_get_flush += (MPI_Wtime() - t_start);
#endif       
        }

        /* calculate target rank from the work_id for B */
        int global_tile_b = work_unit_table[work_id * 3 + 1];
        int target_rank_b = target_rank_of_tile(global_tile_b, nprocs);
        MPI_Aint target_offset_b = offset_of_tile(global_tile_b, nprocs, tile_dim); 

        /* Obtain tile B */
        if (target_rank_b == rank) {
            /* Copy tile B from local memory */
            target_tile = &sub_mat_b[(int) target_offset_b];
            for (k = 0; k < tile_dim; k++)
                for (j = 0; j < tile_dim; j++)
                    local_b[k*tile_dim + j] = target_tile[k*tile_dim + j];
        } else {
            /* get tile B */
#if DEBUG
            //printf("Rank %d trying to get B tile %d from rank %d\n", rank, global_tile_b, target_rank_b);
#endif
#if FINE_TIME
            get_counter++;
            t_start = MPI_Wtime();
#endif
            MPI_Get(local_b, elements_in_tile, MPI_DOUBLE, target_rank_b, disp_b + target_offset_b, elements_in_tile, MPI_DOUBLE,
                    win);
 #if FINE_TIME
            t_get += (MPI_Wtime() - t_start);
            t_start = MPI_Wtime();
#endif 
            MPI_Win_flush(target_rank_b, win);
#if FINE_TIME
            t_get_flush += (MPI_Wtime() - t_start);
#endif
        }
#if COMPUTE
        /* compute Cij += Aik * Bkj */
        dgemm(local_a, local_b, local_c, tile_dim);
#endif
    } while (work_id < work_units);

    if (prev_tile_c >= 0) {
        /* MPI_Accumulate locally accumulated C before finishing */
        int target_rank_c = target_rank_of_tile(prev_tile_c, nprocs);
        MPI_Aint target_offset_c = offset_of_tile(prev_tile_c, nprocs, tile_dim);
        
        /* accumulate tile C (always use MPI since we need to ensure atomicity during accumulation) */
#if DEBUG
        double tile_sum = 0;
        int tile_i, tile_j;
        for (tile_i = 0; tile_i < tile_dim; tile_i++) {
            for (tile_j = 0; tile_j < tile_dim; tile_j++) {
                tile_sum += local_c[tile_i*tile_dim + tile_j];
            }
        }
        printf("Rank %d accumulating tile %d with value %.1f on rank %d using offset %d\n", rank, prev_tile_c, tile_sum, target_rank_c, target_offset_c); 
#endif
#if FINE_TIME
        accum_counter++;
        t_start = MPI_Wtime();
#endif
        MPI_Accumulate(local_c, elements_in_tile, MPI_DOUBLE, target_rank_c, disp_c + target_offset_c, elements_in_tile,
                   MPI_DOUBLE, MPI_SUM, win);
#if FINE_TIME
        t_accum += (MPI_Wtime() - t_start);
        t_start = MPI_Wtime();
#endif
        MPI_Win_flush(target_rank_c, win);
#if FINE_TIME
        t_accum_flush += (MPI_Wtime() - t_start);
#endif
    }
 
    MPI_Barrier(MPI_COMM_WORLD);
    //printf("Rank %d: sub_mat_c[0] is %.1f\n", rank, sub_mat_c[0]);
    //printf("Rank %d: sub_mat_c[1] is %.1f\n", rank, sub_mat_c[1]);
    //printf("Rank %d done!\n", rank);

#if FINE_TIME
    if (get_counter > 0) { 
        t_per_get = t_get / get_counter;
        t_per_get_flush = t_get_flush / get_counter;
    } else
        t_per_get = t_per_get_flush = 0;

    if (accum_counter > 0) {
        t_per_accum = t_accum / accum_counter;
        t_per_accum_flush = t_accum_flush / accum_counter;
    } else
        t_per_accum = t_per_accum_flush = 0;

    if (rank == 0) {
        t_get_procs = calloc(nprocs, sizeof(double));
        t_accum_procs = calloc(nprocs, sizeof(double));
        t_get_flush_procs = calloc(nprocs, sizeof(double));
        t_accum_flush_procs = calloc(nprocs, sizeof(double));
        if (!t_get_procs || !t_accum_procs || !t_get_flush_procs || !t_accum_flush_procs) {
            fprintf(stderr, "Unable to allocate memory for t_get_procs, t_get_flush_procs, t_accum_flush_procs, or t_accum_procs\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    } else
        t_get_procs = t_accum_procs = t_get_flush_procs = t_accum_flush_procs = NULL;
    MPI_Gather(&t_per_get, 1, MPI_DOUBLE, t_get_procs, 1, MPI_DOUBLE, 0, comm_world);
    MPI_Gather(&t_per_get_flush, 1, MPI_DOUBLE, t_get_flush_procs, 1, MPI_DOUBLE, 0, comm_world);
    MPI_Gather(&t_per_accum, 1, MPI_DOUBLE, t_accum_procs, 1, MPI_DOUBLE, 0, comm_world);
    MPI_Gather(&t_per_accum_flush, 1, MPI_DOUBLE, t_accum_flush_procs, 1, MPI_DOUBLE, 0, comm_world);

    tot_get_count = tot_accum_count = 0;
    MPI_Reduce(&get_counter, &tot_get_count, 1, MPI_INT, MPI_SUM, 0, comm_world);
    MPI_Reduce(&accum_counter, &tot_accum_count, 1, MPI_INT, MPI_SUM, 0, comm_world);
    
    min_t_get = max_t_get = mean_t_get = 0;
    min_t_accum = max_t_accum = mean_t_accum = 0;
    min_t_get_flush = max_t_get_flush = mean_t_get_flush = 0;
    min_t_accum_flush = max_t_accum_flush = mean_t_accum_flush = 0;

    if (rank == 0) {
        int pi;
        double sum_t_get, sum_t_accum, sum_t_get_flush, sum_t_accum_flush;
        int nworkers_who_got, nworkers_who_accumed;

        nworkers_who_got = nworkers_who_accumed = 0;

        max_t_get = max_t_accum = max_t_get_flush = max_t_accum_flush = -1;
        min_t_get = min_t_accum = min_t_get_flush = min_t_accum_flush = 9999;
        sum_t_get = sum_t_accum = sum_t_get_flush = sum_t_accum_flush = 0;

        for (pi = 0; pi < nprocs; pi++) {
            if (t_get_procs[pi] > 0) {
                nworkers_who_got++;
                if (max_t_get < t_get_procs[pi])
                    max_t_get = t_get_procs[pi];
                if (min_t_get > t_get_procs[pi])
                    min_t_get = t_get_procs[pi];
                sum_t_get += t_get_procs[pi];
                /* Who got also flushed */
                if (max_t_get_flush < t_get_flush_procs[pi])
                    max_t_get_flush = t_get_flush_procs[pi];
                if (min_t_get_flush > t_get_flush_procs[pi])
                    min_t_get_flush = t_get_flush_procs[pi];
                sum_t_get_flush += t_get_flush_procs[pi];
            }
            
            if (t_accum_procs[pi] > 0) { 
                nworkers_who_accumed++;
                if (max_t_accum < t_accum_procs[pi])
                    max_t_accum = t_accum_procs[pi];
                if (min_t_accum > t_accum_procs[pi])
                    min_t_accum = t_accum_procs[pi];
                sum_t_accum += t_accum_procs[pi];
                /* Who accumed also flushed */
                if (max_t_accum_flush < t_accum_flush_procs[pi])
                    max_t_accum_flush = t_accum_flush_procs[pi];
                if (min_t_accum_flush > t_accum_flush_procs[pi])
                    min_t_accum_flush = t_accum_flush_procs[pi];
                sum_t_accum_flush += t_accum_flush_procs[pi];
            }
        }

        mean_t_get = sum_t_get / nworkers_who_got;
        mean_t_accum = sum_t_accum / nworkers_who_accumed;
        mean_t_get_flush = sum_t_get_flush / nworkers_who_got;
        mean_t_accum_flush = sum_t_accum_flush / nworkers_who_accumed;
    }
#else
    t2 = MPI_Wtime();
#endif
    
#if SHOW_WORKLOAD_DIST
    if (rank == 0) {
        all_worker_counter = calloc(nprocs, sizeof(int));
        alt_rank = calloc(nprocs, sizeof(int));
    } else {
        all_worker_counter = NULL;
        alt_rank = NULL;
    }

    MPI_Gather(&my_work_counter, 1, MPI_INT, all_worker_counter, 1, MPI_INT, 0, comm_world);
    MPI_Gather(&rank, 1, MPI_INT, alt_rank, 1, MPI_INT, 0, MPI_COMM_WORLD);
#endif

    MPI_Win_sync(win);    /* MEM_MODE: synchronize private and public window copies */
    if (rank == 0) {
        int mat_dim = tile_num * tile_dim;
#if SHOW_WORKLOAD_DIST
        printf("Worker\tReal-rank\tUnits\n");
        for (i = 0; i < nprocs; i++) {
            printf("%d\t%d\t%d\n", i, alt_rank[i], all_worker_counter[i]);
        }
        printf("\n");
#endif
#if FINE_TIME
        /* Each of the times reported are per operation */
        printf("mat_dim,tile_dim,work_units,nworkers,"
                "min_get_time,max_get_time,mean_get_time,"
                "min_accum_time,max_accum_time,mean_accum_time,"
                "min_get_flush_time,max_get_flush_time,mean_get_flush_time,"
                "min_accum_flush_time,max_accum_flush_time,mean_accum_flush_time\n");
        printf("%d,%d,%d,%d,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f\n", mat_dim, tile_dim, work_units, nprocs,
                min_t_get, max_t_get, mean_t_get,
                min_t_accum, max_t_accum, mean_t_accum,
                min_t_get_flush, max_t_get_flush, mean_t_get_flush,
                min_t_accum_flush, max_t_accum_flush, mean_t_accum_flush);
#else
        /* This time is the time for the whole kernel i.e. the observed time by the end user of the application */
        printf("mat_dim,tile_dim,work_units,nworkers,time\n");
        printf("%d,%d,%d,%d,%.9f\n", mat_dim, tile_dim, work_units, nprocs, t2 - t1);
#endif
    }

#if CHECK_FOR_ERRORS
    if (rank == 0) {
        /* Check matrices */
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
     
        int tile_i, tile_j;
        for (i = 0; i < tile_num; i++) {
            for (j = 0; j < tile_num; j++) {
                int global_tile_c = tile_map[i*tile_num + j];
                
                if (global_tile_c != -1) {
                    int target_rank_c = target_rank_of_tile(global_tile_c, nprocs);
                    MPI_Aint target_offset_c = offset_of_tile(global_tile_c, nprocs, tile_dim); 
                    
                    MPI_Get(local_c, elements_in_tile, MPI_DOUBLE, target_rank_c, disp_c + target_offset_c, elements_in_tile, MPI_DOUBLE, win);
                    MPI_Win_flush(target_rank_c, win);
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

        free(mat_a);
        free(mat_b);
        free(mat_correct_c);
        free(mat_c);
    }
#endif

    MPI_Win_unlock_all(win);
    MPI_Win_unlock_all(win_counter);

    MPI_Win_free(&win_counter);
    MPI_Win_free(&win);
    MPI_Comm_free(&comm_world);

#if OFI_WINDOW_HINTS
    MPI_Info_free(&win_info);
#endif    
 
    free(local_a);
    free(local_b);
    free(local_c);

#if FINE_TIME
    free(t_get_procs);
    free(t_accum_procs);
    free(t_get_flush_procs);
    free(t_accum_flush_procs);
#endif

    free(tile_map);
    free(work_unit_table);

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

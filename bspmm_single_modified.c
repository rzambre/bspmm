#include "bspmm.h"
#include <unistd.h>
#if USE_CBLAS
    #include <cblas.h>
#endif

#define FINE_TIME 1

#if FINE_TIME
#define START_FINE_TIME(is_warmup) ({                       \
        double t_start = 0.0;                               \
        if (!is_warmup) {                                   \
            t_start = MPI_Wtime();                          \
        }                                                   \
        t_start;                                            \
})
#else
#define START_FINE_TIME(...) ({0.0;})
#endif

#if FINE_TIME
#define GET_FINE_TIME(t_start, is_warmup) ({                \
        double temp_timer;                                  \
        if (!is_warmup) {                                   \
            temp_timer =  (MPI_Wtime() - t_start);          \
        }                                                   \
        temp_timer;                                         \
})
#else
#define GET_FINE_TIME(...) ({0.0;})
#endif

#if FINE_TIME
#define INCREMENT_COUNTER(counter, is_warmup) ({            \
    int temp_counter;                                       \
    if(!is_warmup) {                                        \
        counter ++;                                         \
        temp_counter =  counter;                            \
    }                                                       \
    temp_counter;                                           \
})
#else
#define INCREMENT_COUNTER(...) ({0;})
#endif
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
#define ACCUMULATE 1
#define WARMUP 1
#define CHECK_FOR_ERRORS 0
#define SHOW_WORKLOAD_DIST 0
#define DUMP_GET_TIMESTAMP 1

#define LOCAL_C_COUNT 8

static int rank, nprocs;
static int tile_dim, tile_num, p_dim, node_dim, ppn;
MPI_Comm comm_world_counter;

#if DUMP_GET_TIMESTAMP
    double START_TIME[64 * 1024];
    double END_TIME[64 * 1024];
    int timestamp_counter = 0;
#endif

#if SHOW_WORKLOAD_DIST
    static int my_work_counter, *all_worker_counter;
    static int *alt_rank;
#endif

static double t1 = 0.0, t2 = 0.0;
static double t_start = 0.0;
static double t_get = 0.0, t_local_get = 0.0, t_accum = 0.0, t_fetch_n_op = 0.0, t_comp = 0.0;
static double t_get_flush = 0.0, t_accum_flush = 0.0, t_fetch_n_op_flush = 0.0, t_global_accum_flush = 0.0;
static int get_counter = 0, local_get_counter = 0, accum_counter = 0, fetch_n_op_counter = 0, comp_counter = 0;

#if FINE_TIME
    static double min_t_get = 0.0, max_t_get = 0.0, mean_t_get = 0.0;
    static double min_t_local_get = 0.0, max_t_local_get = 0.0, mean_t_local_get = 0.0;
    static double min_t_accum = 0.0, max_t_accum = 0.0, mean_t_accum = 0.0;
    static double min_t_get_flush = 0.0, max_t_get_flush = 0.0, mean_t_get_flush = 0.0;
    static double min_t_accum_flush = 0.0, max_t_accum_flush = 0.0, mean_t_accum_flush = 0.0;
    static double min_t_fetch_n_op = 0.0, max_t_fetch_n_op = 0.0, mean_t_fetch_n_op = 0.0;
    static double min_t_fetch_n_op_flush = 0.0, max_t_fetch_n_op_flush = 0.0, mean_t_fetch_n_op_flush = 0.0;
    static double min_t_comp = 0.0, max_t_comp = 0.0, mean_t_comp = 0.0;
    static double min_t_global_accum_flush = 0.0, max_t_global_accum_flush = 0.0, mean_t_global_accum_flush = 0.0;
    static int tot_get_count = 0, tot_local_get_count = 0, tot_accum_count = 0, tot_fetch_n_op_count = 0, tot_comp_count = 0;
#endif

int cmp (const void * a, const void * b) {
    if (*(double*)a > *(double*)b)
        return 1;
    else if (*(double*)a < *(double*)b)
        return -1;
    else
        return 0; 
}

#if DUMP_GET_TIMESTAMP
void write_to_file(int rank) {
    int i, suffix_len;
    FILE *fp = NULL;
    char *suffix = NULL;
    char *prefix = "dump_";
    char *file_name = NULL;

    suffix_len = snprintf(NULL, 0, "%d", rank);
    suffix = calloc(1, (suffix_len + 1));
    file_name = calloc(1, (suffix_len+1+5));

    snprintf(suffix, (suffix_len + 1), "%d", rank);
    suffix[suffix_len] = '\0';
    strcat(file_name, prefix);
    strcat(file_name, suffix);
    printf("File Name: %s\n", file_name); fflush(stdout);
    fp = fopen(file_name, "w");
    if (NULL == fp) {
        printf("Error in file op !!");
        return;
    }
    for(i = 0; i < timestamp_counter; i++) {
        fprintf(fp, "%lf %lf\n", START_TIME[i], END_TIME[i]);
    }
    fclose(fp);
    free(file_name);
    free(suffix);
}
#endif

void calculate_timer(double *input_timer, int count, double **output_timer, double *min, double *max, double *median,
                        double *mean) {
    int i;
    double total_time;
    for (i = 0; i < count; i++) {
        (*output_timer)[i] =  input_timer[i] * 1.0e+6;
        total_time += (*output_timer)[i];
    }
    qsort(*output_timer, count, sizeof(double), cmp);
    *min = (*output_timer)[0];
    *max = (*output_timer)[count - 1];
    *median = (*output_timer)[count/2];
    *mean = (double)(total_time/count);

    return;
}

void print_timer(double *timer, int count, char *timer_name) {
    double min = 0.0, max = 0.0, median = 0.0, mean = 0.0;
    int i, temp_counter = 0;
    double *out_timer = NULL;
    out_timer = calloc(1, count * sizeof(*out_timer));

    calculate_timer(timer, count, &out_timer, &min, &max, &median, &mean);
    for (i = 0; i < count; i++) {
        if (out_timer[i] > 1.0) {
            temp_counter += 1;
        }
    }
    printf("[Rank %d]: %s - Count: %d >1us: %d Min: %lf Max: %lf Median: %lf Mean: %lf Total: %lf\n", rank, timer_name,
            count, temp_counter, min, max, median, mean, (mean * count)); fflush(stdout);
    free(out_timer);
}

#if FINE_TIME
void calculate_stats (double *inp_array, int size, double *min, double *max, double *mean) {
    int i;
    double temp_min, temp_max, temp_mean, sum;
    int nworkers_who_performed = 0;

    temp_min = 999999999999.0;
    temp_max = -1.0;
    temp_mean = 0.0;
    sum = 0.0;

    for (i = 0; i < size; i ++) {
        if (inp_array[i] > 0.0) {
            nworkers_who_performed ++;
            if (temp_max < inp_array[i]) {
                temp_max = inp_array[i];
            }
            if (temp_min > inp_array[i]) {
                temp_min = inp_array[i];
            }
            sum += inp_array[i];
        }
    }
    temp_mean = sum / nworkers_who_performed;

    (*min) = temp_min;
    (*max) = temp_max;
    (*mean) = temp_mean;
    
    return;
}

void calculate_fine_time(int mat_dim, int work_units, MPI_Comm comm_world) {
    double t_per_get, t_per_local_get, t_per_accum, t_per_fetch_n_op, t_per_comp;
    double t_per_get_flush, t_per_accum_flush, t_per_fetch_n_op_flush, t_per_global_accum_flush;
    double *t_get_procs, *t_local_get_procs, *t_accum_procs, *t_fetch_n_op_procs, *t_comp_procs;
    double *t_get_flush_procs, *t_accum_flush_procs, *t_fetch_n_op_flush_procs, *t_global_accum_flush_procs;

    if (get_counter > 0) { 
        t_per_get = t_get; // / get_counter;
        t_per_get_flush = t_get_flush; // / get_counter;
    } else {
        t_per_get = t_per_get_flush = 0;
    }

    if (local_get_counter > 0) {
        t_per_local_get = t_local_get; // / local_get_counter;
    } else {
        t_per_local_get = 0;
    }

    if (accum_counter > 0) {
        t_per_accum = t_accum; // / accum_counter;
        t_per_accum_flush = t_accum_flush; // / accum_counter;
        t_per_global_accum_flush = t_global_accum_flush;
    } else {
        t_per_accum = t_per_accum_flush = t_per_global_accum_flush = 0;
    }

    if (fetch_n_op_counter > 0) {
        t_per_fetch_n_op = t_fetch_n_op; // / fetch_n_op_counter;
        t_per_fetch_n_op_flush = t_fetch_n_op_flush; // / fetch_n_op_counter;
    } else {
        t_per_fetch_n_op = t_per_fetch_n_op_flush = 0;
    }

    if (comp_counter > 0) {
        t_per_comp = t_comp; // / comp_counter;
    } else {
        t_per_comp = 0;
    }

    if (rank == 0) {
        t_get_procs = calloc(nprocs, sizeof(double));
        t_local_get_procs = calloc(nprocs, sizeof(double));
        t_accum_procs = calloc(nprocs, sizeof(double));
        t_fetch_n_op_procs = calloc(nprocs, sizeof(double));
        t_comp_procs = calloc(nprocs, sizeof(double));
        t_get_flush_procs = calloc(nprocs, sizeof(double));
        t_accum_flush_procs = calloc(nprocs, sizeof(double));
        t_global_accum_flush_procs = calloc(nprocs, sizeof(double));
        t_fetch_n_op_flush_procs = calloc(nprocs, sizeof(double));

        if (!t_get_procs || !t_local_get_procs || !t_accum_procs || !t_fetch_n_op_procs || !t_comp_procs || !t_get_flush_procs ||
            !t_accum_flush_procs || !t_global_accum_flush_procs || !t_fetch_n_op_flush_procs) {
            fprintf(stderr, "Unable to allocate memory proc timers\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    } else {
        t_get_procs = t_local_get_procs = t_accum_procs = t_fetch_n_op_procs = t_comp_procs = t_get_flush_procs = t_accum_flush_procs =
        t_global_accum_flush_procs = t_fetch_n_op_flush_procs = NULL;
    }

    MPI_Gather(&t_per_get, 1, MPI_DOUBLE, t_get_procs, 1, MPI_DOUBLE, 0, comm_world);
    MPI_Gather(&t_per_local_get, 1, MPI_DOUBLE, t_local_get_procs, 1, MPI_DOUBLE, 0, comm_world);
    MPI_Gather(&t_per_get_flush, 1, MPI_DOUBLE, t_get_flush_procs, 1, MPI_DOUBLE, 0, comm_world);
    MPI_Gather(&t_per_accum, 1, MPI_DOUBLE, t_accum_procs, 1, MPI_DOUBLE, 0, comm_world);
    MPI_Gather(&t_per_accum_flush, 1, MPI_DOUBLE, t_accum_flush_procs, 1, MPI_DOUBLE, 0, comm_world);
    MPI_Gather(&t_per_global_accum_flush, 1, MPI_DOUBLE, t_global_accum_flush_procs, 1, MPI_DOUBLE, 0, comm_world);
    MPI_Gather(&t_per_fetch_n_op, 1, MPI_DOUBLE, t_fetch_n_op_procs, 1, MPI_DOUBLE, 0, comm_world);
    MPI_Gather(&t_per_fetch_n_op_flush, 1, MPI_DOUBLE, t_fetch_n_op_flush_procs, 1, MPI_DOUBLE, 0, comm_world);
    MPI_Gather(&t_per_comp, 1, MPI_DOUBLE, t_comp_procs, 1, MPI_DOUBLE, 0, comm_world);


    tot_get_count = tot_local_get_count = tot_accum_count = tot_fetch_n_op_count = tot_comp_count = 0;
    MPI_Reduce(&get_counter, &tot_get_count, 1, MPI_INT, MPI_SUM, 0, comm_world);
    MPI_Reduce(&local_get_counter, &tot_local_get_count, 1, MPI_INT, MPI_SUM, 0, comm_world);
    MPI_Reduce(&accum_counter, &tot_accum_count, 1, MPI_INT, MPI_SUM, 0, comm_world);
    MPI_Reduce(&fetch_n_op_counter, &tot_fetch_n_op_count, 1, MPI_INT, MPI_SUM, 0, comm_world);
    MPI_Reduce(&comp_counter, &tot_comp_count, 1, MPI_INT, MPI_SUM, 0, comm_world);
    
    min_t_get = max_t_get = mean_t_get = 0;
    min_t_local_get = max_t_local_get = mean_t_local_get = 0;
    min_t_accum = max_t_accum = mean_t_accum = 0;
    min_t_fetch_n_op = max_t_fetch_n_op = mean_t_fetch_n_op = 0;
    min_t_comp = max_t_comp = mean_t_comp = 0;
    min_t_get_flush = max_t_get_flush = mean_t_get_flush = 0;
    min_t_accum_flush = max_t_accum_flush = mean_t_accum_flush = 0;
    min_t_global_accum_flush = max_t_global_accum_flush = mean_t_global_accum_flush = 0;
    min_t_fetch_n_op_flush = max_t_fetch_n_op_flush = mean_t_fetch_n_op_flush = 0;
    
    if (rank == 0) {
        max_t_get = max_t_local_get = max_t_accum = max_t_get_flush = max_t_accum_flush = max_t_global_accum_flush = max_t_fetch_n_op =
        max_t_fetch_n_op_flush = max_t_comp = -1;
        min_t_get = min_t_local_get = min_t_accum = min_t_get_flush = min_t_accum_flush = min_t_global_accum_flush = min_t_fetch_n_op =
        min_t_fetch_n_op_flush = min_t_comp = 9999;

        calculate_stats(t_get_procs, nprocs, &min_t_get, &max_t_get, &mean_t_get);
        calculate_stats(t_local_get_procs, nprocs, &min_t_local_get, &max_t_local_get, &mean_t_local_get);
        calculate_stats(t_get_flush_procs, nprocs, &min_t_get_flush, &max_t_get_flush, &mean_t_get_flush);
        calculate_stats(t_accum_procs, nprocs, &min_t_accum, &max_t_accum, &mean_t_accum);
        calculate_stats(t_accum_flush_procs, nprocs, &min_t_accum_flush, &max_t_accum_flush, &mean_t_accum_flush);
        calculate_stats(t_global_accum_flush_procs, nprocs, &min_t_global_accum_flush, &max_t_global_accum_flush, &mean_t_global_accum_flush);
        calculate_stats(t_fetch_n_op_procs, nprocs, &min_t_fetch_n_op, &max_t_fetch_n_op, &mean_t_fetch_n_op);
        calculate_stats(t_fetch_n_op_flush_procs, nprocs, &min_t_fetch_n_op_flush, &max_t_fetch_n_op_flush, &mean_t_fetch_n_op_flush);
        calculate_stats(t_comp_procs, nprocs, &min_t_comp, &max_t_comp, &mean_t_comp);

    }

    free(t_get_procs);
    free(t_local_get_procs);
    free(t_get_flush_procs);
    free(t_accum_procs);
    free(t_accum_flush_procs);
    free(t_global_accum_flush_procs);
    free(t_fetch_n_op_procs);
    free(t_fetch_n_op_flush_procs);
    free(t_comp_procs);
}
#endif

void check_errors(int rank, int nprocs, int tile_num, int tile_dim, int *tile_map, size_t elements_in_tile,
                    double *local_c, MPI_Aint disp_c, MPI_Win win_c) {
    int i, j;
    
    MPI_Win_lock_all(MPI_MODE_NOCHECK, win_c);

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

        /* USE BLAS TO COMPUTE MAT_CORRECT_C for validation*/
#if USE_CBLAS
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, mat_dim, mat_dim, mat_dim, 1, mat_a, mat_dim,
                    mat_b, mat_dim, 1, mat_correct_c, mat_dim);
#else
        int k;
        for (i = 0; i < mat_dim; i++) {
            for (j = 0; j < mat_dim; j++) {
                for (k = 0; k < mat_dim; k++) {
                    mat_correct_c[i*mat_dim + j] += mat_a[i*mat_dim + k] * mat_b[k*mat_dim + j];
                }
            }
        }
#endif
        int tile_i, tile_j;
        for (i = 0; i < tile_num; i++) {
            for (j = 0; j < tile_num; j++) {
                int global_tile_c = tile_map[i*tile_num + j];
                
                if (global_tile_c != -1) {
                    int target_rank_c = target_rank_of_tile(global_tile_c, nprocs);
                    MPI_Aint target_offset_c = offset_of_tile(global_tile_c, nprocs, tile_dim); 
                    
                    MPI_Get(local_c, elements_in_tile, MPI_DOUBLE, target_rank_c, disp_c + target_offset_c,
                            elements_in_tile, MPI_DOUBLE, win_c);
                    MPI_Win_flush(target_rank_c, win_c);
#if DEBUG
                    printf("Rank %d got tile %d with value %.1f from rank %d using offset %ld\n", rank, global_tile_c,
                            local_c[0], target_rank_c, target_offset_c); 
#endif
                    for (tile_i = 0; tile_i < tile_dim; tile_i++) {
                        for (tile_j = 0; tile_j < tile_dim; tile_j++) {
                           mat_c[i*tile_dim*mat_dim + j*tile_dim + tile_i*mat_dim + tile_j] =
                            local_c[tile_i*tile_dim + tile_j]; 
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
    MPI_Win_unlock_all(win_c);

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

int invoke_get(int nprocs, int rank, size_t tile_dim, int *work_unit_table, int work_id, int work_unit_disp,
                double *sub_mat, double *local_buf, MPI_Aint disp, MPI_Win win, int is_warmup, int *target_rank){
    int i, j;
    int is_mpi_get = 0;
    double *target_tile = NULL;
    int elements_in_tile = tile_dim * tile_dim;
    int global_tile = work_unit_table[work_id * 3 + work_unit_disp];
    (*target_rank) = target_rank_of_tile(global_tile, nprocs);
    MPI_Aint target_offset = offset_of_tile(global_tile, nprocs, tile_dim);

    if ((*target_rank) == rank) {
        t_start = START_FINE_TIME(is_warmup);
        target_tile = &sub_mat[(int) target_offset];
        for (i = 0; i < tile_dim; i++) {
            for (j = 0; j < tile_dim; j++) {
                local_buf[i * tile_dim + j] = target_tile[i * tile_dim + j];
            }
        }
        t_local_get += GET_FINE_TIME(t_start, is_warmup);
        local_get_counter = INCREMENT_COUNTER(local_get_counter, is_warmup);
    } else {
        t_start = START_FINE_TIME(is_warmup);
        MPI_Get(local_buf, elements_in_tile, MPI_DOUBLE, (*target_rank), disp + target_offset, elements_in_tile,
                    MPI_DOUBLE, win);
        t_get += GET_FINE_TIME(t_start, is_warmup);
        get_counter = INCREMENT_COUNTER(get_counter, is_warmup);
        is_mpi_get = 1;
    }
    return is_mpi_get;
}

int bspmm_get(int nprocs, int rank, size_t tile_dim, int *work_unit_table, int work_id, double *sub_mat_a,
                double *sub_mat_b, double *local_a, double *local_b, MPI_Aint disp_a, MPI_Aint disp_b,
                MPI_Win win, int is_warmup, int *target_rank_a, int *target_rank_b) {
    int is_mpi_get_a = 0, is_mpi_get_b = 0, ret = 0;
#if DUMP_GET_TIMESTAMP
    START_TIME[timestamp_counter] = MPI_Wtime();
#endif
    is_mpi_get_a = invoke_get(nprocs, rank, tile_dim, work_unit_table, work_id, 0, sub_mat_a, local_a, disp_a, win,
                                is_warmup, target_rank_a);
    is_mpi_get_b = invoke_get(nprocs, rank, tile_dim, work_unit_table, work_id, 1, sub_mat_b, local_b, disp_b, win,
                                is_warmup, target_rank_b);
    if (is_mpi_get_a) {
        ret = 1;
    } else if (is_mpi_get_b) {
        ret = 2;
    } else if (is_mpi_get_a && is_mpi_get_b) {
        ret = 3;
    }
#if DUMP_GET_TIMESTAMP
    END_TIME[timestamp_counter] = MPI_Wtime();
    timestamp_counter ++;
#endif
    return ret;
}

void bspmm_get_flush(int is_mpi_get, int target_rank_a, int target_rank_b, MPI_Win win, int is_warmup) {
    t_start = START_FINE_TIME(is_warmup);
    switch (is_mpi_get) {
        case 1:
            MPI_Win_flush(target_rank_a, win);
            break;
        case 2: 
            MPI_Win_flush(target_rank_b, win);
            break;
        case 3:
            MPI_Win_flush(target_rank_a, win);
            MPI_Win_flush(target_rank_b, win);
            break;
        default:
            break;
    }
    t_get_flush += GET_FINE_TIME(t_start, is_warmup);
}

void bspmm_v2(size_t elements_in_tile, size_t tile_size, int *work_unit_table, int work_units, size_t sub_mat_elements,
            double * sub_mat_a, double *sub_mat_b, double *sub_mat_c, double *local_a, double *local_b, double *local_c,
            MPI_Aint disp_a, MPI_Aint disp_b, MPI_Aint disp_c, MPI_Win win, MPI_Win win_c,
            MPI_Win win_counter, int *counter_win_mem, MPI_Comm comm_world, int is_warmup) {
    
    int local_c_counter = 0;
    int local_c_idx = 0;
    double *base_local_a = local_a;
    double *base_local_b = local_b;
    double *base_local_c = local_c;
    double *next_local_a = NULL;
    double *next_local_b = NULL;
    int p, prev_tile_c,
        next_work_id, next_next_work_id, cur_buf_idx, next_buf_idx,
        cur_op, is_mpi_get, target_rank_a = -1, target_rank_b = -1, target_rank_c = -1;
    const int one = 1;
    const int two = 2;
    int accum_tracker[nprocs];

    for(p = 0; p < nprocs; p++) {
        accum_tracker[p] = 0;
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    if(!is_warmup) {
#if SHOW_WORKLOAD_DIST
        my_work_counter = 0;
#endif
        t1 = MPI_Wtime();
    }
    /*************** Initialization steps ***************/
    /* Reset it for next call */
    memset(local_c, 0, tile_size);
    
    /* initialize sub matrices */
    MPI_Win_lock(MPI_LOCK_EXCLUSIVE, rank, MPI_MODE_NOCHECK, win);
    MPI_Win_lock(MPI_LOCK_EXCLUSIVE, rank, MPI_MODE_NOCHECK, win_c);
    init_sub_mats(sub_mat_a, sub_mat_b, sub_mat_c, sub_mat_elements);
    MPI_Win_unlock(rank, win);
    MPI_Win_unlock(rank, win_c);

    if (rank == 0) {
        /* initialize global counter */
        MPI_Win_lock(MPI_LOCK_EXCLUSIVE, 0, MPI_MODE_NOCHECK, win_counter);
        *counter_win_mem = 0;
        MPI_Win_unlock(0, win_counter); /* MEM_MODE: update to my private window becomes
                                         * visible in public window */
    }

    /*************** BSPMM steps ***************/
    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Win_lock_all(MPI_MODE_NOCHECK, win);
    MPI_Win_lock_all(MPI_MODE_NOCHECK, win_c); // Do we need to lock win_c ??
    MPI_Win_lock_all(MPI_MODE_NOCHECK, win_counter);
    
    MPI_Barrier(comm_world);
    MPI_Barrier(comm_world_counter);
    prev_tile_c = -1;

    t_start = START_FINE_TIME(is_warmup);
    MPI_Fetch_and_op(&two, &next_work_id, MPI_INT, 0, 0, MPI_SUM, win_counter);
    t_fetch_n_op += GET_FINE_TIME(t_start, is_warmup);
    fetch_n_op_counter = INCREMENT_COUNTER(fetch_n_op_counter, is_warmup);

    t_start = START_FINE_TIME(is_warmup);
    MPI_Win_flush(0, win_counter);
    t_fetch_n_op_flush += GET_FINE_TIME(t_start, is_warmup);

    next_next_work_id = next_work_id + 1;
    fflush(stdout);

    /* bspmm should invoke both get A and get B - return 1 if one of the get is invoked */
    is_mpi_get = bspmm_get(nprocs, rank, tile_dim, work_unit_table, next_work_id, sub_mat_a, sub_mat_b, local_a, local_b, disp_a,
                disp_b, win, is_warmup, &target_rank_a, &target_rank_b);
    cur_buf_idx = 0;
    while(next_work_id < work_units) {
#if SHOW_WORKLOAD_DIST
        if (!is_warmup) {
            my_work_counter++;
        }
#endif
        cur_op = next_work_id;
        next_work_id = next_next_work_id;

        t_start = START_FINE_TIME(is_warmup);
        MPI_Win_flush(0, win_counter);
        t_fetch_n_op_flush += GET_FINE_TIME(t_start, is_warmup);
        
        bspmm_get_flush(is_mpi_get, target_rank_a, target_rank_b, win, is_warmup);

        /* Call accumulate here with */
#if ACCUMULATE
        int global_tile_c = work_unit_table[cur_op * 3 + 2];
        if (global_tile_c != prev_tile_c && prev_tile_c >= 0) {
            target_rank_c = target_rank_of_tile(prev_tile_c, nprocs);
            MPI_Aint target_offset_c = offset_of_tile(prev_tile_c, nprocs, tile_dim);
            
#if DEBUG
            if(!is_warmup) {
                double tile_sum = 0;
                int tile_i, tile_j;
                for (tile_i = 0; tile_i < tile_dim; tile_i++) {
                    for (tile_j = 0; tile_j < tile_dim; tile_j++) {
                        tile_sum += local_c[tile_i*tile_dim + tile_j];
                    }
                }
                printf("Rank %d accumulating tile %d with value %.1f on rank %d using offset %ld\n", rank, prev_tile_c, tile_sum, target_rank_c, target_offset_c);
            }
#endif
            t_start = START_FINE_TIME(is_warmup);
            MPI_Accumulate(local_c, elements_in_tile, MPI_DOUBLE, target_rank_c, disp_c + target_offset_c,
                            elements_in_tile, MPI_DOUBLE, MPI_SUM, win_c);
            t_accum += GET_FINE_TIME(t_start, is_warmup);
            accum_counter = INCREMENT_COUNTER(accum_counter, is_warmup);
            accum_tracker[target_rank_c] += 1;
            
            t_start = START_FINE_TIME(is_warmup);
            if (local_c_counter % LOCAL_C_COUNT == (LOCAL_C_COUNT - 1)) {
                MPI_Win_flush_local(target_rank_c, win_c);
            }
            t_accum_flush += GET_FINE_TIME(t_start, is_warmup);
            
            local_c_counter ++;
            local_c_idx = local_c_counter % LOCAL_C_COUNT;
            local_c = base_local_c + (local_c_idx * (elements_in_tile));

            /* Reset the local C tile for local accumulation */
            memset(local_c, 0, tile_size);
        }
        prev_tile_c = global_tile_c;
#endif

        local_a = base_local_a + (cur_buf_idx * (elements_in_tile));
        local_b = base_local_b + (cur_buf_idx * (elements_in_tile));
        next_buf_idx = (cur_buf_idx + 1) % 2;
        next_local_a = base_local_a + (next_buf_idx * (elements_in_tile));
        next_local_b = base_local_b + (next_buf_idx * (elements_in_tile));

        /* next work_id updated in last step. Don't invoke get and fetch_n_op in advance if not required*/
        if (next_work_id < work_units) {
            is_mpi_get = bspmm_get(nprocs, rank, tile_dim, work_unit_table, next_work_id, sub_mat_a, sub_mat_b,
                                    next_local_a, next_local_b, disp_a, disp_b, win, is_warmup, &target_rank_a, &target_rank_b);

            t_start = START_FINE_TIME(t_start);
            MPI_Fetch_and_op(&one, &next_next_work_id, MPI_INT, 0, 0, MPI_SUM, win_counter);
            t_fetch_n_op += GET_FINE_TIME(t_start, is_warmup);
            fetch_n_op_counter = INCREMENT_COUNTER(fetch_n_op_counter, is_warmup);
        }

#if COMPUTE
        t_start = START_FINE_TIME(t_start);
#if USE_CBLAS
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,tile_dim,tile_dim,tile_dim,1,local_a,tile_dim,
                    local_b,tile_dim,1,local_c,tile_dim);
#else
        /* compute Cij += Aik * Bkj */
        dgemm(local_a, local_b, local_c, tile_dim);
#endif
        t_comp += GET_FINE_TIME(t_start, is_warmup);
        comp_counter = INCREMENT_COUNTER(comp_counter, is_warmup);
#endif
        cur_buf_idx += next_buf_idx;
    }

#if ACCUMULATE
    if (prev_tile_c >= 0) {
        /* MPI_Accumulate locally accumulated C before finishing */
        int target_rank_c = target_rank_of_tile(prev_tile_c, nprocs);
        MPI_Aint target_offset_c = offset_of_tile(prev_tile_c, nprocs, tile_dim);
#if DEBUG
        if(!is_warmup) {
            double tile_sum = 0;
            int tile_i, tile_j;
            for (tile_i = 0; tile_i < tile_dim; tile_i++) {
                for (tile_j = 0; tile_j < tile_dim; tile_j++) {
                    tile_sum += local_c[tile_i*tile_dim + tile_j];
                }
            }
            printf("Rank %d accumulating tile %d with value %.1f on rank %d using offset %ld\n", rank, prev_tile_c, tile_sum, target_rank_c, target_offset_c);
        }
#endif
        t_start = START_FINE_TIME(is_warmup);
        MPI_Accumulate(local_c, elements_in_tile, MPI_DOUBLE, target_rank_c, disp_c + target_offset_c, elements_in_tile,
                       MPI_DOUBLE, MPI_SUM, win_c);
        t_accum += GET_FINE_TIME(t_start, is_warmup);
        accum_counter = INCREMENT_COUNTER(accum_counter, is_warmup);

        t_start = START_FINE_TIME(is_warmup);
        MPI_Win_flush_local(target_rank_c, win_c);
        t_accum_flush += GET_FINE_TIME(t_start, is_warmup);
    }

    t_start = START_FINE_TIME(is_warmup);
    MPI_Win_flush_all(win_c);
    t_global_accum_flush = GET_FINE_TIME(t_start, is_warmup);
#endif

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Win_sync(win);    /* MEM_MODE: synchronize private and public window copies */
    MPI_Win_sync(win_c);    /* MEM_MODE: synchronize private and public window copies */

    MPI_Win_unlock_all(win);
    MPI_Win_unlock_all(win_c);
    MPI_Win_unlock_all(win_counter);
    if(!is_warmup) {
        t2 = MPI_Wtime();
    }
}

void bspmm(size_t elements_in_tile, size_t tile_size, int *work_unit_table, int work_units, size_t sub_mat_elements,
            double * sub_mat_a, double *sub_mat_b, double *sub_mat_c, double *local_a, double *local_b, double *local_c,
            MPI_Aint disp_a, MPI_Aint disp_b, MPI_Aint disp_c, MPI_Win win, MPI_Win win_c,
            MPI_Win win_counter, int *counter_win_mem, MPI_Comm comm_world, int is_warmup) {
    int is_mpi_get, target_rank_a, target_rank_b, local_c_counter = 0, local_c_idx = 0;
    double *base_local_c = local_c;

    int p, work_id, prev_tile_c;
    const int one = 1;
    int accum_tracker[nprocs];

    for(p = 0; p < nprocs; p++) {
        accum_tracker[p] = 0;
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if (!is_warmup) {
#if SHOW_WORKLOAD_DIST
        my_work_counter = 0;
#endif

#if FINE_TIME
        t_get = t_local_get = t_accum = t_fetch_n_op = t_comp = 0.0;
        t_get_flush = t_accum_flush = t_global_accum_flush = t_fetch_n_op_flush = 0.0;
        get_counter = local_get_counter = accum_counter = fetch_n_op_counter = comp_counter = 0;
#endif
        if (rank == 0) {
            t1 = MPI_Wtime();
        }
    }


    /*************** Initialization steps ***************/
    /* Reset it for next call */
    memset(local_c, 0, tile_size);
    
    /* initialize sub matrices */
    MPI_Win_lock(MPI_LOCK_EXCLUSIVE, rank, MPI_MODE_NOCHECK, win);
    MPI_Win_lock(MPI_LOCK_EXCLUSIVE, rank, MPI_MODE_NOCHECK, win_c);
    init_sub_mats(sub_mat_a, sub_mat_b, sub_mat_c, sub_mat_elements);
    MPI_Win_unlock(rank, win);
    MPI_Win_unlock(rank, win_c);

    if (rank == 0) {
        /* initialize global counter */
        MPI_Win_lock(MPI_LOCK_EXCLUSIVE, 0, MPI_MODE_NOCHECK, win_counter);
        *counter_win_mem = 0;
        MPI_Win_unlock(0, win_counter); /* MEM_MODE: update to my private window becomes
                                         * visible in public window */
    }

    /*************** BSPMM steps ***************/
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Win_lock_all(MPI_MODE_NOCHECK, win);
    MPI_Win_lock_all(MPI_MODE_NOCHECK, win_c); // Do we need to lock win_c ??
    MPI_Win_lock_all(MPI_MODE_NOCHECK, win_counter);
    MPI_Barrier(comm_world);
    MPI_Barrier(comm_world_counter);

    prev_tile_c = -1; 

    do {
        t_start = START_FINE_TIME(is_warmup);
        /* read and increment global counter atomically */
        MPI_Fetch_and_op(&one, &work_id, MPI_INT, 0, 0, MPI_SUM, win_counter);
        t_fetch_n_op += GET_FINE_TIME(t_start, is_warmup);
        fetch_n_op_counter = INCREMENT_COUNTER(fetch_n_op_counter, is_warmup);

        t_start = START_FINE_TIME(is_warmup);
        MPI_Win_flush(0, win_counter);
        t_fetch_n_op_flush += GET_FINE_TIME(t_start, is_warmup);

        if (work_id >= work_units)
            break;

#if SHOW_WORKLOAD_DIST
        if (!is_warmup) {
            my_work_counter++;
        }
#endif
#if ACCUMULATE
        int global_tile_c = work_unit_table[work_id * 3 + 2];
        if (global_tile_c != prev_tile_c && prev_tile_c >= 0) {
            /* MPI_Accumulate locally accumulated C before proceeding */
            int target_rank_c = target_rank_of_tile(prev_tile_c, nprocs);
            MPI_Aint target_offset_c = offset_of_tile(prev_tile_c, nprocs, tile_dim);

            /* accumulate tile C (always use MPI since we need to ensure atomicity during accumulation) */
#if DEBUG
            if(!is_warmup) {
                double tile_sum = 0;
                int tile_i, tile_j;
                for (tile_i = 0; tile_i < tile_dim; tile_i++) {
                    for (tile_j = 0; tile_j < tile_dim; tile_j++) {
                        tile_sum += local_c[tile_i*tile_dim + tile_j];
                    }
                }
                printf("Rank %d accumulating tile %d with value %.1f on rank %d using offset %ld\n", rank, prev_tile_c, tile_sum, target_rank_c, target_offset_c);
            }
#endif
            t_start = START_FINE_TIME(is_warmup);
            MPI_Accumulate(local_c, elements_in_tile, MPI_DOUBLE, target_rank_c, disp_c + target_offset_c,
                            elements_in_tile, MPI_DOUBLE, MPI_SUM, win_c);
            t_accum += GET_FINE_TIME(t_start, is_warmup);
            accum_counter  = INCREMENT_COUNTER(accum_counter, is_warmup);

            accum_tracker[target_rank_c] += 1;

            t_start = START_FINE_TIME(is_warmup);
            if (local_c_counter % LOCAL_C_COUNT == (LOCAL_C_COUNT - 1)) {
                MPI_Win_flush_local(target_rank_c, win_c);
            }
            t_accum_flush += GET_FINE_TIME(t_start, is_warmup);

            local_c_counter ++;
            local_c_idx = local_c_counter % LOCAL_C_COUNT;
            local_c = base_local_c + (local_c_idx * (elements_in_tile));

            /* Reset the local C tile for local accumulation */
            memset(local_c, 0, tile_size);
        } 
        prev_tile_c = global_tile_c;
#endif

        /* Get A and B tiles*/
        is_mpi_get = bspmm_get(nprocs, rank, tile_dim, work_unit_table, work_id, sub_mat_a, sub_mat_b, local_a, local_b, disp_a,
                disp_b, win, is_warmup, &target_rank_a, &target_rank_b);
        
        bspmm_get_flush(is_mpi_get, target_rank_a, target_rank_b, win, is_warmup);

#if COMPUTE
        t_start = START_FINE_TIME(is_warmup);
#if USE_CBLAS
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,tile_dim,tile_dim,tile_dim,1,local_a,tile_dim,
                    local_b,tile_dim,1,local_c,tile_dim);
#else
        /* compute Cij += Aik * Bkj */
        dgemm(local_a, local_b, local_c, tile_dim);
#endif
        t_comp += GET_FINE_TIME(t_start, is_warmup);
        comp_counter  = INCREMENT_COUNTER(comp_counter, is_warmup);
#endif
    } while (work_id < work_units);

#if ACCUMULATE
    if (prev_tile_c >= 0) {
        /* MPI_Accumulate locally accumulated C before finishing */
        int target_rank_c = target_rank_of_tile(prev_tile_c, nprocs);
        MPI_Aint target_offset_c = offset_of_tile(prev_tile_c, nprocs, tile_dim);

#if DEBUG
        if(!is_warmup) {
            double tile_sum = 0;
            int tile_i, tile_j;
            for (tile_i = 0; tile_i < tile_dim; tile_i++) {
                for (tile_j = 0; tile_j < tile_dim; tile_j++) {
                    tile_sum += local_c[tile_i*tile_dim + tile_j];
                }
            }
            printf("Rank %d accumulating tile %d with value %.1f on rank %d using offset %ld\n", rank, prev_tile_c, tile_sum, target_rank_c, target_offset_c);
        }
#endif
        t_start = START_FINE_TIME(is_warmup);
        MPI_Accumulate(local_c, elements_in_tile, MPI_DOUBLE, target_rank_c, disp_c + target_offset_c, elements_in_tile,
                   MPI_DOUBLE, MPI_SUM, win_c);
        accum_tracker[target_rank_c] += 1;
        t_accum += GET_FINE_TIME(t_start, is_warmup);
        accum_counter = INCREMENT_COUNTER(accum_counter, is_warmup);

        START_FINE_TIME(is_warmup);
        MPI_Win_flush_local(target_rank_c, win_c);
        t_accum_flush += GET_FINE_TIME(t_start, is_warmup);
    }

    /* A final global flush to all ranks here */
    t_start = START_FINE_TIME(is_warmup);
    MPI_Win_flush_all(win_c);
    t_global_accum_flush += GET_FINE_TIME(t_start, is_warmup);
#endif

    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Win_sync(win);    /* MEM_MODE: synchronize private and public window copies */
    MPI_Win_sync(win_c);    /* MEM_MODE: synchronize private and public window copies */

    MPI_Win_unlock_all(win);
    MPI_Win_unlock_all(win_c);
    MPI_Win_unlock_all(win_counter);

    /* Reset local c*/
    local_c = base_local_c;
    if (rank == 0) {
        if (!is_warmup) {
            t2 = MPI_Wtime();
        }
    }
}

int main(int argc, char **argv) {
    size_t mat_dim;
    int *tile_map;
    size_t elements_in_tile, tile_size;
    size_t sub_mat_elements;
    int tot_non_zero_tiles, tot_tiles_per_rank;
    int *work_unit_table, work_units;
    double *sub_mat_a, *sub_mat_b, *sub_mat_c;
    MPI_Aint disp_a, disp_b, disp_c;

#if OFI_WINDOW_HINTS
    MPI_Info win_info;
#endif

    double *local_a, *local_b, *local_c;
    
    double *win_mem, *win_mem_c;
    int *counter_win_mem;
    MPI_Win win, win_c, win_counter;

    // int in_node_p_dim;
    // int node_i, node_j;
    // int in_node_i, in_node_j;
    // int rank_in_parray;
    // char bind_inp;
    MPI_Comm comm_world;

    /* initialize MPI environment */
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    /* Wait for input to help bind cpu */
    // if (0 == rank) {
    //     scanf("%c", &bind_inp);
    // }
    MPI_Barrier(MPI_COMM_WORLD);

    /* Setup step */
    if (setup(rank, nprocs, argc, argv, &tile_dim, &tile_num, &p_dim, &node_dim, &ppn)) {
        MPI_Finalize();
        exit(0);
    }

    mat_dim = tile_num * tile_dim;
    elements_in_tile = tile_dim * tile_dim;
    tile_size = elements_in_tile * sizeof(double);

    // in_node_p_dim = p_dim / node_dim;
    // /* find my rank in the processor array from my rank in COMM_WORLD */
    // node_i = rank / (ppn * node_dim);
    // node_j = (rank / ppn) % node_dim;
    // in_node_i = (rank / in_node_p_dim) % in_node_p_dim;
    // in_node_j = rank % in_node_p_dim;
    // rank_in_parray = (node_i * ppn * node_dim) + (in_node_i * p_dim) + (node_j * in_node_p_dim) + in_node_j;

    /* Change rank to match the logical ranks used in this application */
    // printf("Rank: %d rank_in_parray: %d\n", rank, rank_in_parray); fflush(stdout);
    MPI_Comm_split(MPI_COMM_WORLD, 0, rank, &comm_world);
    MPI_Comm_split(MPI_COMM_WORLD, 0, rank, &comm_world_counter);

    // MPI_Comm_split(MPI_COMM_WORLD, 0, rank_in_parray, &comm_world);

#if DEBUG
    if (rank == 0) {
        printf("tile_dim: %d tile_num: %d p_dim: %d\n", tile_dim, tile_num, p_dim);
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
    MPI_Win_allocate(2 * sub_mat_elements * sizeof(double), sizeof(double),
#if OFI_WINDOW_HINTS
                    win_info,
#else
                    MPI_INFO_NULL,
#endif
                    comm_world, &win_mem, &win);
    MPI_Win_allocate(sub_mat_elements * sizeof(double), sizeof(double),
#if OFI_WINDOW_HINTS
                    win_info,
#else
                    MPI_INFO_NULL,
#endif
                    comm_world, &win_mem_c, &win_c);

    sub_mat_a = win_mem;
    sub_mat_b = win_mem + sub_mat_elements;
    sub_mat_c = win_mem_c;

    /* Allocate RMA window for the counter that allows for load balancing */
    if (rank == 0) {
        MPI_Win_allocate(sizeof(int), sizeof(int),
#if OFI_WINDOW_HINTS
                win_info,
#else   
                MPI_INFO_NULL,
#endif   
                comm_world_counter, &counter_win_mem, &win_counter);
    } else {
        MPI_Win_allocate(0, sizeof(int),
#if OFI_WIDOW_HINTS
                        win_info,       
#else
                        MPI_INFO_NULL,
#endif        
                        comm_world_counter, &counter_win_mem, &win_counter);
    }

    /* Changed here as we are using different windows for each matrix */
    disp_a = 0;
    disp_b = 0;
    disp_c = 0;
    MPI_Barrier(comm_world_counter);
    MPI_Barrier(comm_world);

    local_a = calloc(elements_in_tile * 2, sizeof(double));
    local_b = calloc(elements_in_tile * 2, sizeof(double));
    local_c = calloc(elements_in_tile * LOCAL_C_COUNT, sizeof(double));

#if WARMUP
    bspmm_v2(elements_in_tile, tile_size, work_unit_table, work_units, sub_mat_elements,
            sub_mat_a, sub_mat_b, sub_mat_c, local_a, local_b, local_c, disp_a, disp_b, disp_c,
            win, win_c, win_counter, counter_win_mem, comm_world, 1);
#endif
    bspmm_v2(elements_in_tile, tile_size, work_unit_table, work_units, sub_mat_elements,
            sub_mat_a, sub_mat_b, sub_mat_c, local_a, local_b, local_c, disp_a, disp_b, disp_c,
            win, win_c, win_counter, counter_win_mem, comm_world, 0);

#if FINE_TIME
    calculate_fine_time(mat_dim, work_units, comm_world);
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

    if (rank == 0) {
#if SHOW_WORKLOAD_DIST
        printf("Worker\tReal-rank\tUnits\n");
        for (i = 0; i < nprocs; i++) {
            printf("%d\t%d\t%d\n", i, alt_rank[i], all_worker_counter[i]);
        }
        printf("\n");
#endif
#if FINE_TIME
        printf("%s, %s, %s, %s, %s, %s, %s, %s, %s, %s\n", "fetch_n_op", "fetch_n_op_flush", "get", "get_flush", \
                "local_get", "accum", "accum_flush", \
                "global_accum_flush", "comp", "total_time"); fflush(stdout);
        printf("%.4lf, %.4lf, %.4lf, %.4lf, %.4lf, %.4lf, %.4lf, %.4lf, %.4lf, %.4lf\n", (mean_t_fetch_n_op * 1.0e+6),
                (mean_t_fetch_n_op_flush * 1.0e+6), (mean_t_get * 1.0e+6),
                (mean_t_get_flush * 1.0e+6), (mean_t_local_get * 1.0e+6), (mean_t_accum * 1.0e+6),
                (mean_t_accum_flush * 1.0e+6), (mean_t_global_accum_flush * 1.0e+6), (mean_t_comp * 1.0e+6),
                (t2-t1)*1.0e+6); fflush(stdout);
#else
        /* This time is the time for the whole kernel i.e. the observed time by the end user of the application */
        printf("mat_dim\ttile_dim\twork_units\tnworkers\ttime(uSec)\n");
        printf("%ld\t%d\t%d\t%d\t%.9f\n", mat_dim, tile_dim, work_units, nprocs, (t2 - t1)*1.0e+6);
#endif
    }
    MPI_Barrier(MPI_COMM_WORLD);
#if DUMP_GET_TIMESTAMP
    /* Write the timestamps to file*/
    write_to_file(rank);
    MPI_Barrier(MPI_COMM_WORLD);
#endif

#if CHECK_FOR_ERRORS
    check_errors(rank, nprocs, tile_num, tile_dim, tile_map, elements_in_tile, local_c, disp_c, win_c);
#endif

    /** cleanup **/
    MPI_Win_free(&win_counter);
    MPI_Win_free(&win);
    MPI_Win_free(&win_c);
    MPI_Comm_free(&comm_world);

#if OFI_WINDOW_HINTS
    MPI_Info_free(&win_info);
#endif    
 
    free(local_a);
    free(local_b);
    free(local_c);

    free(tile_map);
    free(work_unit_table);

    MPI_Finalize();
    return 0;
}
#include <assert.h>
#include <limits.h>
#include <stdio.h>
#include <unistd.h>
#include <cstring>
#include <math.h>
#include <mpi.h>
#include <omp.h>

#define Root2 1.414213562373095

unsigned long long *res = NULL;

int ncpus = 0;
unsigned long long r = 0;
unsigned long long k = 0;
unsigned long long r_squr = 0;
unsigned long long r_half = 0;

unsigned long long calc_pixs_m(int rank){
    unsigned long long pixels = 0;
    for (unsigned long long x = (unsigned long long)rank; x < r_half; x+=(unsigned long long)ncpus) {
		unsigned long long y = ceil(sqrtl(r_squr - x*x)) - r_half;
        // printf("Rank %d x: %llu, y: %llu\n", rank, x, y);
		pixels += y;
		pixels %= k;
	}
    // printf("Rank %d pixels: %llu\n", rank, pixels);
    return pixels;
}

void task(int rk, int rk_t){
    res[rk] = calc_pixs_m(rk_t);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int mpi_rank, mpi_ranks, omp_threads;
    char hostname[HOST_NAME_MAX];

    assert(!gethostname(hostname, HOST_NAME_MAX));
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_ranks);

    r = atoll(argv[1]);
	k = atoll(argv[2]);
    r_squr = r * r;
    r_half = ceil((double)r / (double)Root2);

    unsigned long long pixels = 0;
    unsigned long long pixels_sum = 0;
    res = (unsigned long long*)malloc(sizeof(unsigned long long) * 100);
    memset(res, 0, sizeof(unsigned long long) * 100);

#pragma omp parallel
    {
        omp_threads = omp_get_num_threads();
        ncpus = mpi_ranks * omp_threads;

        int omp_thread = omp_get_thread_num();
        int rank_t = mpi_rank * omp_threads + omp_thread;
        // printf("RANK_T %d: rank %2d/%2d, thread %2d/%2d, NCPUS %d\n", rank_t, mpi_rank, mpi_ranks,
        //        omp_thread, omp_threads, ncpus);
        
        task(omp_thread, rank_t);
    }

    for(int i = 0; i < omp_threads; i++){
        // printf("Rank %d Thread %d Res: %llu\n", mpi_rank, i, res[i]);
        pixels += res[i];
		pixels %= k;
    }
    pixels = ((pixels * 2) % k);

    // printf("Rank %d Pixels %llu, Sum of Pixels: %llu\n", mpi_rank, pixels, pixels_sum);

    MPI_Reduce(&pixels, &pixels_sum, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    // printf("Rank %d, pixel_sum: %llu\n", mpi_rank, pixels_sum);
    
    free(res);

    if(mpi_rank == 0){
        pixels_sum = ((pixels_sum % k) + ((r_half * r_half) % k) % k);
        printf("%llu\n", (4 * pixels_sum) % k);
    }

    MPI_Finalize();
}

#include <assert.h>
#include <stdio.h>
#include <math.h>
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

void task(int rk){
    res[rk] = calc_pixs_m(rk);
}

int main(int argc, char** argv) {
	if (argc != 3) {
		fprintf(stderr, "must provide exactly 2 arguments!\n");
		return 1;
	}

	r = atoll(argv[1]);
	k = atoll(argv[2]);
    r_squr = r * r;
    r_half = ceil((double)r / (double)Root2);

	unsigned long long pixels = 0;
    res = (unsigned long long*)malloc(sizeof(unsigned long long) * 100);

    #pragma omp parallel
    {
        ncpus = omp_get_num_threads();
        int rank = omp_get_thread_num();
        task(rank);
    }

	for(int i = 0; i < ncpus; i++){
        pixels += res[i];
		pixels %= k;
    }
	pixels = ((pixels * 2) % k) + ((r_half * r_half) % k);
    pixels %= k;
    free(res);
    
	printf("%llu\n", (4 * pixels) % k);
}

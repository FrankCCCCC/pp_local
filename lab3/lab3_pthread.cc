#include <assert.h>
#include <stdio.h>
#include <math.h>
#include <pthread.h>

#define Root2 1.414213562373095

unsigned long long *res = NULL;
int *ranks = NULL;

int ncpus = 0;
unsigned long long r = 0;
unsigned long long k = 0;
unsigned long long r_squr = 0;
unsigned long long r_half = 0;
// unsigned long long r_res_start = 0;
// unsigned long long seg_len_s = 0;
// unsigned long long seg_len_e = 0;

// unsigned long long get_seg_len_s(unsigned long long r){
//     return r;
// }

// unsigned long long get_seg_len_e(unsigned long long r){
//     return ceil(sqrtl(r_squr - ((r - 1) * (r - 1))));
// }

// unsigned long long calc_pixs_l(int rank){
//     unsigned long long pixels = 0;
//     // unsigned long long c = 0;
//     unsigned long long rank_u = (unsigned long long)rank;
//     for (unsigned long long x = seg_len_e + rank_u; x <= seg_len_s; x+=(unsigned long long)ncpus) {
//         unsigned long long y = 0;
//         // x is target segment length
//         // Calculate start point of thickness
//         unsigned long long thick_s = ceil(sqrtl(r_squr - (x * x)));
//         // Calculate end point of thickness
//         unsigned long long thick_e = ceil(sqrtl(r_squr - ((x - 1) * (x - 1))));
//         // Calculate the area of the segment whose size is x
//         y = (((thick_e - thick_s) % k) * (x % k)) % k;

// 		pixels += y;
// 		pixels %= k;
//         // c++;
// 	}

//     // printf("Rank %d C: %llu, seg_len_s: %llu, seg_len_e: %llu\n", rank, c, seg_len_s, seg_len_e);
//     return pixels;
// }

// unsigned long long calc_pixs_shortcut(){
//     return (seg_len_e * r) % k;
// }

// unsigned long long calc_pixs_r(int rank){
//     unsigned long long pixels = 0;
//     for (unsigned long long x = r_res_start + rank; x < r; x+=(unsigned long long)ncpus) {
// 		unsigned long long y = r - ceil(sqrtl(r_squr - x*x));
//         // printf("Rank %d y: %llu\n", rank, y);
// 		pixels += y;
// 		pixels %= k;
// 	}
//     // printf("Rank %d pixels: %llu\n", rank, pixels);
//     return pixels;
// }

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

// unsigned long long calc_pixs(int rank){
//     unsigned long long pixels = 0;
//     for (unsigned long long x = (unsigned long long)rank; x < r; x+=(unsigned long long)ncpus) {
// 		unsigned long long y = ceil(sqrtl(r_squr - x*x));
// 		pixels += y;
// 		pixels %= k;
// 	}
//     return pixels;
// }

// unsigned long long calc_pixs(int rank){
    
// }

void* task(void* rk){
    int rank = *((int*)rk);
    res[rank] = calc_pixs_m(rank);

    pthread_exit(NULL);
}

int main(int argc, char** argv) {
	if (argc != 3) {
		fprintf(stderr, "must provide exactly 2 arguments!\n");
		return 1;
	}
    cpu_set_t cpuset;
	sched_getaffinity(0, sizeof(cpuset), &cpuset);
	ncpus = CPU_COUNT(&cpuset);

	r = atoll(argv[1]);
	k = atoll(argv[2]);
    r_squr = r * r;
    // r_res_start = ceil(sqrtl(r_squr - (r - 1) * (r - 1)));
    r_half = ceil((double)r / (double)Root2);

    // seg_len_s = get_seg_len_s(r);
    // seg_len_e = get_seg_len_e(r);
	unsigned long long pixels = 0;
    res = (unsigned long long*)malloc(sizeof(unsigned long long) * ncpus);
    ranks = (int*)malloc(sizeof(int) * ncpus);

    pthread_t threads[ncpus];

    for(int i = 0; i < ncpus; i++){
        ranks[i] = i;
        int rc = pthread_create(&(threads[i]), NULL, task, (void*)&(ranks[i]));
        if (rc) {
            printf("ERROR; return code from pthread_create() is %d\n", rc);
            exit(-1);
        }
    }


    for(int i = 0; i < ncpus; i++){
        pthread_join(threads[i], NULL);
    }

	for(int i = 0; i < ncpus; i++){
        pixels += res[i];
		pixels %= k;
    }
    pixels = ((pixels * 2) % k) + ((r_half * r_half) % k);
    pixels %= k;
    free(res);
    free(ranks);
	printf("%llu\n", (4 * pixels) % k);

    pthread_exit(NULL);
}

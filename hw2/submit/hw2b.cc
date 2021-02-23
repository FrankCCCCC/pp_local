#include <stdio.h>
#include <stdlib.h>
#include <cstring>
#include <png.h>
#include <assert.h>
#include <emmintrin.h>
#include <immintrin.h>
#include <mpi.h>
#include <omp.h>
#include <pthread.h>

// #include <time.h>

#define MASTER_RANK 0
#define WORKING_TAG 1
#define DONE_TAG 0

// Hyperparameters
int PROC_CHUNK_SIZE = 512;
int PROC_CHUNK_SIZE_DEC = 0;

// Global Variables of Mandlebrot Set Calculation
int cpu_num = 0;
int iters = 0;
double left = 0;
double right = 0;
double lower = 0;
double upper = 0;
int width = 0;
int height = 0;
int area = 0;
int *image = NULL;
double *x0s = NULL;
double *y0s = NULL;

// int *checks = NULL;

// Global Variables of Writing PNG 
int row_size = 0;
png_bytep raw_img = NULL;

// Global Variables for Master
int ps_p = 0;
int not_recv_task = 0;
pthread_mutex_t g_lock;

typedef struct{
    int thread_id;
    int threads_num;
    pthread_mutex_t * lock;
}MasterThreadArg;

// Global Variables for MPI
int rank = 0;
int comm_size = 0;

// Vectorize Constant
const int vec_scale = 1;
const int vec_gap = 1 << vec_scale;
const double zero_vec[2] = {0};
const double one_vec[2] = {1};
const double two_vec[2] = {2, 2};
const double four_vec[2] = {4, 4};
double iters_vec[2] = {0};
double lowers_vec[2] = {0};
double lefts_vec[2] = {0};
double x_pix_ratio_vec[2] = {0};
double y_pix_ratio_vec[2] = {0};

double x_pix_ratio = 0;
double y_pix_ratio = 0;

const __m128d zerov = _mm_loadu_pd(zero_vec);
const __m128d onev = _mm_loadu_pd(one_vec);
const __m128d twov = _mm_loadu_pd(two_vec);
const __m128d fourv = _mm_loadu_pd(four_vec);
const __m128d fullv = _mm_or_pd(onev, onev);
__m128d itersv = _mm_loadu_pd(zero_vec);
__m128d lowersv = _mm_loadu_pd(zero_vec);
__m128d leftsv = _mm_loadu_pd(zero_vec);
__m128d x_pix_ratiosv = _mm_loadu_pd(zero_vec);
__m128d y_pix_ratiosv = _mm_loadu_pd(zero_vec);

void set_chunk_size(){
    int upper = (area / (cpu_num * comm_size * cpu_num * comm_size));
    PROC_CHUNK_SIZE = ((upper * 2) >> 1) << 1;
    if(PROC_CHUNK_SIZE <= 0){PROC_CHUNK_SIZE = 2;}
    // PROC_CHUNK_SIZE_DEC = (PROC_CHUNK_SIZE * PROC_CHUNK_SIZE) / (2 * area);
    // printf("PROC_CHUNK_SIZE %d\n", PROC_CHUNK_SIZE);
}

void assign_iters_vec(int iters){
    // Iters
    iters_vec[0] = (double)iters; iters_vec[1] = (double)iters;
    itersv = _mm_loadu_pd(iters_vec);

    // lower
    lowers_vec[0] = lowers_vec[1] = lower;
    lowersv = _mm_loadu_pd(lowers_vec);

    // left
    lefts_vec[0] = lefts_vec[1] = left;
    leftsv = _mm_loadu_pd(lefts_vec);

    // X, Y Pixels Ratio
    x_pix_ratio = ((right - left) / width);
    y_pix_ratio = ((upper - lower) / height);
    x_pix_ratio_vec[0] = x_pix_ratio_vec[1] = x_pix_ratio;
    y_pix_ratio_vec[0] = y_pix_ratio_vec[1] = y_pix_ratio;
    x_pix_ratiosv = _mm_loadu_pd(x_pix_ratio_vec);
    y_pix_ratiosv = _mm_loadu_pd(y_pix_ratio_vec);
}

void assign_x0s_y0s_sig(int ps){
    const double is_vec[2] = {(double)(ps % width), (double)((ps + 1) % width)};
    const double js_vec[2] = {(double)(ps / width), (double)((ps + 1) / width)};

    __m128d isv = _mm_loadu_pd(is_vec);
    __m128d jsv = _mm_loadu_pd(js_vec);

    _mm_store_pd(&(x0s[ps]), _mm_add_pd(_mm_mul_pd(isv, x_pix_ratiosv), leftsv));
    _mm_store_pd(&(y0s[ps]), _mm_add_pd(_mm_mul_pd(jsv, y_pix_ratiosv), lowersv));
}

void png_write_sig(int x, int y){
    int p = image[(height - 1 - y) * width + x];
    png_bytep color = &(raw_img[row_size * y]) + x * 3;
    if (p != iters) {
        if (p & 16) {
            color[0] = 240;
            color[1] = color[2] = p % 16 * 16;
        } else {
            color[0] = p % 16 * 16;
        }
    }
}

void image_to_png_omp(int ps, int size){
    // const int omp_chunk_size = ((size / (cpu_num * cpu_num)) << 1);
    #pragma omp parallel num_threads(cpu_num)
    {
        #pragma omp for schedule(guided) nowait
        for(int i = ps; i < ps + size; i++){
            int x = i % width;
            int y = (height - 1 - (i / width));
            png_write_sig(x, y);
        }
    }
}

void write_png(const char* filename, int iters, int width, int height, const int* buffer) {
    FILE* fp = fopen(filename, "wb");
    assert(fp);
    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    assert(png_ptr);
    png_infop info_ptr = png_create_info_struct(png_ptr);
    assert(info_ptr);
    png_init_io(png_ptr, fp);
    png_set_IHDR(png_ptr, info_ptr, width, height, 8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_set_filter(png_ptr, 0, PNG_NO_FILTERS);
    png_write_info(png_ptr, info_ptr);
    png_set_compression_level(png_ptr, 1);
    
    for (int y = 0; y < height; ++y) {
        png_write_row(png_ptr, &(raw_img[row_size * y]));
    }
    png_write_end(png_ptr, NULL);
    png_destroy_write_struct(&png_ptr, &info_ptr);
    fclose(fp);
}

void core_cal_sse2_sig(double *x0, double *y0, int *image){
    __m128d repeatsv = _mm_loadu_pd(zero_vec);
    __m128d xv = _mm_loadu_pd(zero_vec);
    __m128d yv = _mm_loadu_pd(zero_vec);
    __m128d length_squaredv = _mm_loadu_pd(zero_vec);

    const __m128d x0v = _mm_loadu_pd(x0);
    const __m128d y0v = _mm_loadu_pd(y0);

    while(1){
        __m128d compv = _mm_and_pd(_mm_cmplt_pd(repeatsv, itersv), _mm_cmplt_pd(length_squaredv, fourv));
        compv = (__m128d)_mm_slli_epi64((__m128i)compv, 54);
        compv = (__m128d)_mm_srli_epi64((__m128i)compv, 2);

        unsigned long int comp_vec[2] = {0, 0};
        _mm_store_pd((double*)comp_vec, compv);
        // printf("Compv0: %lu %lu, Compv: %lu %lu\n", comp_vec0[0], comp_vec0[1], comp_vec[0], comp_vec[1]);
        if((comp_vec[0] == 0) && (comp_vec[1] == 0)){break;}
        
        __m128d tempv = zerov;
        tempv = _mm_add_pd(_mm_sub_pd(_mm_mul_pd(xv, xv), _mm_mul_pd(yv, yv)), x0v) ;
        yv = _mm_add_pd(_mm_mul_pd(twov, _mm_mul_pd(xv, yv)), y0v);
        xv = tempv;
        length_squaredv = _mm_add_pd(_mm_mul_pd(xv, xv), _mm_mul_pd(yv, yv));

        repeatsv = _mm_add_pd(repeatsv, compv);
    }   
    double image_temp[2] = {0, 0};
    _mm_store_pd(image_temp, repeatsv);
    image[0] = (int)(image_temp[0]);
    image[1] = (int)(image_temp[1]);

    // printf("Iters %d, (%d, %d) <- (%lf, %lf)\n", count, image[i], image[i + 1], image_temp[0], image_temp[1]);
}

void core_cal(double x0, double y0, int *image){
    int repeats = 0;
    double x = 0;
    double y = 0;
    double length_squared = 0;
    while (repeats < iters && length_squared < 4) {
        double temp = x * x - y * y + x0;
        y = 2 * x * y + y0;
        x = temp;
        length_squared = x * x + y * y;
        ++repeats;
    }

    *image = repeats;
}

void core_cal_sse2_omp(int ps, int size){
    const int size_vec = (size >> vec_scale) << vec_scale;
    const int max_loop_vec = ps + size_vec;
    const int max_loop = ps + size;
    const int omp_chunk_size = ((size / (cpu_num * cpu_num)) << 1);

    #pragma omp parallel num_threads(cpu_num)
    {
        #pragma omp for schedule(guided) nowait
        for(int i = ps; i < max_loop_vec; i+=vec_gap){
            assign_x0s_y0s_sig(i);
            core_cal_sse2_sig(&(x0s[i]), &(y0s[i]), &(image[i]));
        }
    }

    for(int i = max_loop_vec; i < max_loop; i++){
        core_cal(x0s[i], y0s[i], &(image[i]));
    }

    image_to_png_omp(ps, size);
}

int assign_task(int *);

void master();

void slave();

int main(int argc, char** argv) {
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

    // Get CPU number
    cpu_set_t cpu_set;
    sched_getaffinity(0, sizeof(cpu_set), &cpu_set);
    cpu_num = CPU_COUNT(&cpu_set);

	/* argument parsing */
    assert(argc == 9);
    const char* filename = argv[1];
    iters = strtol(argv[2], 0, 10);
    left = strtod(argv[3], 0);
    right = strtod(argv[4], 0);
    lower = strtod(argv[5], 0);
    upper = strtod(argv[6], 0);
    width = strtol(argv[7], 0, 10);
    height = strtol(argv[8], 0, 10);
    area = width * height;
    assign_iters_vec(iters);
    set_chunk_size();

    /* allocate memory for image */
    image = (int*)malloc(area * sizeof(int));
    memset(image, 0, area * sizeof(int));
    assert(image);
    // checks = (int*)malloc(area * sizeof(int));

    // allocate for png
    row_size = 3 * width * sizeof(png_byte);
    raw_img = (png_bytep)malloc(row_size * height);

    // Allocate for x0, y0
    x0s = (double*)malloc(sizeof(double) * area);
    y0s = (double*)malloc(sizeof(double) * area);

    /* mandelbrot set */
	if(rank == MASTER_RANK){
		master();

        /* draw and cleanup */
        write_png(filename, iters, width, height, image);
	}else{
		slave();
	}

    MPI_Finalize();
}

// void  update_PROC_CHUNK_SIZE(){
//     PROC_CHUNK_SIZE -= PROC_CHUNK_SIZE_DEC;
//     if(PROC_CHUNK_SIZE <= 1){PROC_CHUNK_SIZE = 2;}
// }

int assign_task(int *ps_size){
	ps_size[0] = ps_p;
	if(ps_p >= area){return 0;}

	if(ps_p + PROC_CHUNK_SIZE >= area){
		ps_size[1] = area - ps_p;
		ps_p = area;
	}else{
		ps_size[1] = PROC_CHUNK_SIZE;
		ps_p += PROC_CHUNK_SIZE;
	}
    // update_PROC_CHUNK_SIZE();
	return 1;
}

void *master_thread_task_comm(void *arg){
    MasterThreadArg *mt_arg = (MasterThreadArg*)arg;
    int ps_p = 0;
    int not_recv_task = 0;
    int dum_p = 0;
	MPI_Status status;
    MPI_Request req;

    // Send the first task
    pthread_mutex_lock(mt_arg->lock);
    for(int tkq = 0; tkq < 3; tkq++){
        for(int i = 1; i < comm_size; i++){
            int ps_size[2] = {0};
            int is_get_task = assign_task(ps_size);        
            if(!is_get_task){break;}
            MPI_Isend(ps_size, 2, MPI_INT, i, WORKING_TAG, MPI_COMM_WORLD, &req);
            not_recv_task++;
        }
    }
    pthread_mutex_unlock(mt_arg->lock);

	// Receive results and dispatch new task
	for(;;){
        int ps_size[2] = {0};
        pthread_mutex_lock(mt_arg->lock);
		int is_get_task = assign_task(ps_size);
        pthread_mutex_unlock(mt_arg->lock);

        if(!is_get_task){break;}

		MPI_Recv(&dum_p, 1, MPI_INT, MPI_ANY_SOURCE, WORKING_TAG, MPI_COMM_WORLD, &status);
		MPI_Isend(ps_size, 2, MPI_INT, status.MPI_SOURCE, WORKING_TAG, MPI_COMM_WORLD, &req);
	}

    // Recieve the remaining tasks
    for(;not_recv_task > 0;){
		MPI_Recv(&dum_p, 1, MPI_INT, MPI_ANY_SOURCE, WORKING_TAG, MPI_COMM_WORLD, &status);
        not_recv_task--;
	}

	// Terminate
	for(int i = 1; i < comm_size; i++){
		int ps = 0;
		MPI_Isend(&ps, 1, MPI_INT, i, DONE_TAG, MPI_COMM_WORLD, &req);
	}

    pthread_exit(NULL);
}

void master(){
    pthread_t* comm_thread = (pthread_t*)malloc(sizeof(pthread_t));
    pthread_mutex_t *lock = (pthread_mutex_t*)malloc(sizeof(pthread_mutex_t));
    MasterThreadArg *arg = (MasterThreadArg*)malloc(sizeof(MasterThreadArg));
    pthread_mutex_init(lock, NULL);
    arg->lock = lock;
    arg->thread_id = 0;
    arg->threads_num = 1;

    pthread_create(comm_thread, NULL, master_thread_task_comm, (void*)(arg));
    
    for(;;){
        int ps_size[2] = {0};
        pthread_mutex_lock(arg->lock);
        int is_get_task = assign_task(ps_size);
        pthread_mutex_unlock(arg->lock);

        if(!is_get_task){break;}
        core_cal_sse2_omp(ps_size[0], ps_size[1]);
    }

    pthread_join(*comm_thread, NULL);

	// Reduce
    MPI_Reduce(MPI_IN_PLACE, (unsigned int*)raw_img, row_size * height / sizeof(unsigned int), MPI_UNSIGNED, MPI_BOR, MASTER_RANK, MPI_COMM_WORLD);
}

void slave(){
	int ps_size[2] = {0};
	int dum_p = 0;
	MPI_Status status;
    MPI_Request req;
    
	for(;;){
		MPI_Recv(ps_size, 2, MPI_INT, MASTER_RANK, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

		if(status.MPI_TAG == DONE_TAG){
			break;
		}

		core_cal_sse2_omp(ps_size[0], ps_size[1]);

		MPI_Isend(&dum_p, 1, MPI_INT, MASTER_RANK, WORKING_TAG, MPI_COMM_WORLD, &req);
	}

	// Reduce
    MPI_Reduce((unsigned int*)raw_img, (unsigned int*)raw_img, row_size * height / sizeof(unsigned int), MPI_UNSIGNED, MPI_BOR, MASTER_RANK, MPI_COMM_WORLD);
}
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#define PNG_NO_SETJMP
#include <sched.h>
#include <assert.h>
#include <png.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <emmintrin.h>
#include <immintrin.h>
#include <time.h>
#include <pthread.h>

// Hyperparameter 
int CHUNK_SIZE = 128;

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

// Global Variables of Writing PNG 
int row_size = 0;
png_bytep raw_img = NULL;

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
    int squ = cpu_num * cpu_num;
    if(squ % 2 || squ <= 1){
        CHUNK_SIZE = squ + 1;
    }else{
        CHUNK_SIZE = squ;
    }

    // CHUNK_SIZE = area / (cpu_num * cpu_num);
    // if(CHUNK_SIZE <= 1){CHUNK_SIZE = 2;}
    // else{CHUNK_SIZE = CHUNK_SIZE >> 2 << 1;}
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

void image_to_png(int ps, int size){
    for(int i = ps; i < ps + size; i++){
        int x = i % width;
        int y = (height - 1 - (i / width));
        png_write_sig(x, y);
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

void core_cal_sse2(int ps, int size){
    const int size_vec = (size >> vec_scale) << vec_scale;
    const int max_loop_vec = ps + size_vec;
    const int max_loop = ps + size;

    for(int i = ps; i < max_loop_vec; i+=vec_gap){
        assign_x0s_y0s_sig(i);
        core_cal_sse2_sig(&(x0s[i]), &(y0s[i]), &(image[i]));
    }

    for(int i = max_loop_vec; i < max_loop; i++){
        core_cal(x0s[i], y0s[i], &(image[i]));
    }

    image_to_png(ps, size);
}

// Task Queue
typedef struct Task{
    int ps;
    int size;
}Task;
typedef struct Task_Queue{
    int size;
    int head;
    Task *queue;
}Task_Queue;
Task_Queue *create_queue(int size){
    Task_Queue *q = (Task_Queue*)malloc(sizeof(Task_Queue)); 
    q->queue = (Task*)malloc(sizeof(Task) * size);
    q->head = 0;
    q->size = size;
    return q;
}
int is_empty(Task_Queue *q){return q->head == 0;}
int is_full(Task_Queue *q){return q->head < q->size;}
int size(Task_Queue *q){return q->size;}
int push(Task t, Task_Queue *q){
    if(is_full(q)){
        q->queue[q->head++] = t;
        return 1;
    }else{
        return 0;
    }
}
Task *pop(Task_Queue *q){
    if(is_empty(q)){
        return NULL;    
    }else{
        return &(q->queue[--q->head]);
    }
}


typedef struct{
    pthread_t *threads;
    Task_Queue *queue;
    pthread_mutex_t *lock;
    int threads_num;
    // int is_submit_done;
    int is_finish;
}ThreadPool;

typedef struct{
    ThreadPool *pool;
    int thread_id;
}WorkerArg;

ThreadPool *create_thread_pool(int, Task_Queue*);
int get_num_tasks(ThreadPool*);
int is_task_queue_empty(ThreadPool*);
void *worker(void*);
void start_pool(ThreadPool*);
void set_finish(ThreadPool *);
void end_pool(ThreadPool*);

void make_task(int ps, int size, Task_Queue *queue){
    Task t;
    t.ps = ps;
    t.size = size;
    // printf("Pushed Task: ps %d, size %d\n", t.ps, t.size);
    push(t, queue);
}

void thread_task_func(Task *t){
    // printf("Task PS: %d, Size: %d\n", t->ps, t->size);
    core_cal_sse2(t->ps, t->size);
}


int main(int argc, char** argv) {
    // printf("HI, hw2 is running\n");
    /* detect how many CPUs are available */
    cpu_set_t cpu_set;
    sched_getaffinity(0, sizeof(cpu_set), &cpu_set);
    cpu_num = CPU_COUNT(&cpu_set);
    // printf("%d cpus available\n", cpu_num);

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

    /* allocate memory for image */
    image = (int*)malloc(area * sizeof(int));
    // memset(image, 0, area * sizeof(int));
    assert(image);

    // allocate for png
    row_size = 3 * width * sizeof(png_byte);
    raw_img = (png_bytep)malloc(row_size * height);

    // Allocate for x0, y0
    x0s = (double*)malloc(sizeof(double) * area);
    y0s = (double*)malloc(sizeof(double) * area);

    // Set CHUNK_SIZE
    set_chunk_size();

    // Submit Tasks
    Task_Queue *tq = create_queue(area / CHUNK_SIZE + 1);
    int idx = 0;
    for(idx = 0; idx + CHUNK_SIZE < area; idx+=CHUNK_SIZE){
        make_task(idx, CHUNK_SIZE, tq);
    }
    make_task(idx, area - idx, tq);

    ThreadPool *pool = create_thread_pool(cpu_num, tq);
    start_pool(pool);
    set_finish(pool);
    for(;(!pool->is_finish) || (!is_task_queue_empty(pool));){
        int is_has_task = 0;
        Task *task = NULL;

        pthread_mutex_lock(pool->lock);
        if(!is_task_queue_empty(pool)){
            task = pop(pool->queue);
            is_has_task = 1;
        }
        pthread_mutex_unlock(pool->lock);
        if(is_has_task){
            thread_task_func(task);
        }
    }
    end_pool(pool);

    /* draw and cleanup */
    write_png(filename, iters, width, height, image);
}

ThreadPool *create_thread_pool(int threads_num, Task_Queue *queue){
    ThreadPool *pool = (ThreadPool *)malloc(sizeof(ThreadPool));
    pool->threads = NULL;
    pool->queue = queue;
    pool->lock = (pthread_mutex_t*)malloc(sizeof(pthread_mutex_t));
    pthread_mutex_init(pool->lock, NULL);
    pool->threads_num = threads_num;
    pool->is_finish = 0;

    return pool;
}

int get_threads_num(ThreadPool* pool){
    return pool->threads_num;
}
int get_num_tasks(ThreadPool* pool){
    return size(pool->queue);
}
int is_task_queue_empty(ThreadPool* pool){
    return is_empty(pool->queue);
}

void *worker(void *worker_arg_v){
    WorkerArg *worker_arg = (WorkerArg*)worker_arg_v;
    ThreadPool *pool = worker_arg->pool;
    int thread_id = worker_arg->thread_id;
    // printf("Thread %d Created(Self: %d)\n", thread_id, pthread_self());
    Task *task = NULL;

    for(;(!pool->is_finish) || (!is_task_queue_empty(pool));){
        int is_has_task = 0;

        pthread_mutex_lock(pool->lock);
        if(!is_task_queue_empty(pool)){
            // printf("Thread %d Getting Task\n", thread_id);
            task = pop(pool->queue);
            is_has_task = 1;
        }
        pthread_mutex_unlock(pool->lock);
        if(is_has_task){
            // printf("Thread %d Exec Task\n", thread_id);
            thread_task_func(task);
        }
    }

    pthread_exit(NULL);
}

// Create and start the threads
void start_pool(ThreadPool* pool){
    pool->threads = (pthread_t*)malloc(sizeof(pthread_t) * get_threads_num(pool));
    WorkerArg *worker_args = (WorkerArg*)malloc(sizeof(WorkerArg) * get_threads_num(pool));
    for(int i = 0; i < get_threads_num(pool); i++){
        worker_args[i].pool = pool;
        worker_args[i].thread_id = i;
        pthread_create(&(pool->threads[i]), NULL, worker, (void*)(&(worker_args[i])));
    }
}

void set_finish(ThreadPool *pool){
    pool->is_finish = 1;
}

// Join the threads
void end_pool(ThreadPool* pool){
    pool->is_finish = 1;
    for(int i = 0; i < get_threads_num(pool); i++){
        pthread_join(pool->threads[i], NULL);
    }
}
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <pthread.h>
#include <omp.h>
#include <emmintrin.h>

#define DISINF 6000*1000+1000
#define FULL 2147483647
#define DISZSELF 0
#define SIZEOFINT sizeof(int)
#define VECGAP 4
#define VECSCALE 2
#define STOPVAL -1

// int vec_counter = 0, non_vec_counter = 0;

int cpu_num = 0;
int vertex_num = 0, edge_num = 0, graph_size = 0;
int is_residual = 0;
int *buf = NULL;
int *graph = NULL;
int *block_deps = NULL;
int num_blocks = 0, block_size = 0, block_num_squr = 0, block_num_cubic = 0;
int block_assign_step = 0;

const int zero_vec[VECGAP] = {0};
const int one_vec[VECGAP] = {1, 1, 1, 1};
const unsigned int full_vec[VECGAP] = {FULL, FULL, FULL, FULL};
const __m128i zero_v = _mm_loadu_si128((const __m128i*)zero_vec);
const __m128i one_v = _mm_loadu_si128((const __m128i*)one_vec);
const __m128i full_v = _mm_loadu_si128((const __m128i*)full_vec);

void show_mat(int *g, int n){
    for(int i = 0; i < n; i++){
        for(int j = 0; j < n; j++){
            printf("%d\t", g[i * n + j]);
        }
        printf("\n");
    }
}

void show_m128i(__m128i *m){
    printf("(%d\t%d\t%d\t%d)\n", ((int*)m)[0], ((int*)m)[1], ((int*)m)[2], ((int*)m)[3]);
}
int get_graph_row(int idx){return idx / vertex_num;}
int get_graph_col(int idx){return idx % vertex_num;}
int get_graph_idx(int row, int col){return row * vertex_num + col;}
int* get_graph_addr(int row, int col){return &(graph[row*vertex_num + col]);}
int get_graph(int row, int col){return graph[row*vertex_num + col];}
void set_graph(int row, int col, int val){graph[row*vertex_num + col] = val;}

typedef struct{
    int i;
    int j;
    int k;
}BlockDim;
void init_block();
BlockDim get_BlockDim(int, int, int);
BlockDim get_block_pos(int, int, int);
BlockDim get_block_size(int, int, int);
// void init_block_deps();
// int check_block_dep(int, int, int);

void graph_malloc();
void omp_buf2graph(int *);

void relax_v(int*, int*, int*);
void relax(int, int, int*);
void relax_block(int, int, int);
void block_floyd_warshall();

// typedef struct{
//     pthread_t *threads;
//     pthread_mutex_t *lock;
//     pthread_mutex_t *sync_lock;
//     pthread_cond_t *sync_cond;
//     BlockDim *task_queue;
//     int sync_counter;
//     int threads_num;
//     // int is_submit_done;
//     int is_finish;
// }ThreadPool;

// typedef struct{
//     ThreadPool *pool;
//     int thread_id;
// }WorkerArg;

// ThreadPool *create_thread_pool(int);
// void init_task_queue(ThreadPool*);
// void sync_threads(ThreadPool*);
// void get_task();
// void *worker(void*);
// void start_pool(ThreadPool*);
// void set_finish(ThreadPool *);
// void end_pool(ThreadPool*);

int main(int argc, char** argv) {
    cpu_set_t cpu_set;
    sched_getaffinity(0, sizeof(cpu_set), &cpu_set);
    cpu_num = CPU_COUNT(&cpu_set);
    // printf("%d cpus available\n", cpu_num);

    assert(argc == 3);
    FILE *f_r = NULL, *f_w = NULL;
    f_r = fopen(argv[1], "r");
    f_w = fopen(argv[2], "w");
    assert(f_r != NULL);
    assert(f_w != NULL);

    fread(&vertex_num, SIZEOFINT, 1, f_r);
    fread(&edge_num, SIZEOFINT, 1, f_r);
    graph_size = vertex_num * vertex_num;
    buf = (int*)malloc(edge_num * SIZEOFINT * 3);
    fread(buf, SIZEOFINT, edge_num * 3, f_r);
    graph_malloc();
    init_block();
    // init_block_deps();
    
    // printf("Vertex: %d Edge: %d\n", vertex_num, edge_num);
    omp_buf2graph(buf);

    // ThreadPool *pool = create_thread_pool(cpu_num);
    // init_task_queue(pool);
    // start_pool(pool);
    // end_pool(pool);
    block_floyd_warshall();

    fwrite(graph, SIZEOFINT, graph_size, f_w);
    return 0;
}

void graph_malloc(){
    graph = (int*)malloc(graph_size * SIZEOFINT);
    memset(graph, DISZSELF, graph_size * SIZEOFINT);
}

void omp_buf2graph(int *buf){
    const int EDGE0REMARK = -1; 
    #pragma omp for schedule(guided)
    for(int i = omp_get_thread_num()*3; i < edge_num*3; i+=(omp_get_num_threads()*3)){
        // printf("Func: Edge %d - SRC: %d DST: %d WEIGHT: %d\n", i, buf[i], buf[i + 1], buf[i + 2]);
        if(buf[i + 2] == 0){
            set_graph(buf[i], buf[i + 1], EDGE0REMARK);
        }else{
            set_graph(buf[i], buf[i + 1], buf[i + 2]);
        }
    }

    #pragma omp for schedule(guided)
    for(int idx = omp_get_thread_num(); idx < graph_size; idx+=omp_get_num_threads()){
        int i = idx / vertex_num, j = idx % vertex_num;
        if(get_graph(i, j) == 0 && i != j){
            set_graph(i, j, DISINF);
        }else if(get_graph(i, j) == EDGE0REMARK){
            set_graph(i, j, 0);
        }
    }
}

void init_block(){
    // Set Block Size
    block_size = 64;
    // block_size = (((int)ceil(vertex_num / sqrt(cpu_num))) >> VECSCALE) << VECSCALE;
    if(block_size > vertex_num){block_size = vertex_num;}
    else if(block_size < VECGAP){block_size = VECGAP;}
    // printf("Block Size: %d\n", block_size);
    is_residual = vertex_num % block_size > 0;
    
    // Set Number of blocks
    num_blocks = vertex_num / block_size + is_residual;

    block_num_squr = num_blocks * num_blocks;
    block_num_cubic = num_blocks * block_num_squr;
}
BlockDim get_BlockDim(int i, int j, int k){
    BlockDim b = {i, j, k};
    return b;
}
BlockDim get_block_pos(int b_i, int b_j, int b_k){
    BlockDim bd;

    bd.i = block_size * b_i;
    bd.j = block_size * b_j;
    bd.k = block_size * b_k;
    return bd;
}

BlockDim get_block_size(int b_i, int b_j, int b_k){
    BlockDim bd;
    const int quo = vertex_num / block_size;
    if(b_i < quo){bd.i = block_size;}
    else if(b_i == num_blocks - 1){bd.i = vertex_num % block_size;}
    else{bd.i = 0;}
    
    if(b_j < quo){bd.j = block_size;}
    else if(b_j == num_blocks - 1){bd.j = vertex_num % block_size;}
    else{bd.j = 0;}

    if(b_k < quo){bd.k = block_size;}
    else if(b_k == num_blocks - 1){bd.k = vertex_num % block_size;}
    else{bd.k = 0;}
    return bd;
}

// Relax with intermediate sequence k, from sequence i to j
int relax_v(int *aij, int aik, int *akj){
    __m128i aij_v = _mm_loadu_si128((const __m128i*)aij);
    // printf("aij_v:\n");
    const int aik_vec[VECGAP] = {aik, aik, aik, aik};
    __m128i aik_v = _mm_loadu_si128((const __m128i*)aik_vec);
    // printf("aik_v:\n");
    __m128i akj_v = _mm_loadu_si128((const __m128i*)akj);
    // printf("akj_v:\n");

    __m128i sum_v = _mm_add_epi32(aik_v, akj_v);
    // printf("sum_v:\n");
    __m128i compare_gt_v = _mm_cmpgt_epi32(aij_v, sum_v);
    // printf("compare_gt_v:\n");
    __m128i compare_let_v = _mm_xor_si128(compare_gt_v, full_v);
    // printf("compare_let_v:\n");

    __m128i compgt_sum = _mm_and_si128(compare_gt_v, sum_v);
    // printf("compgt_sum:\n");
    __m128i complet_aij = _mm_and_si128(compare_let_v, aij_v);
    // printf("complet_aij:\n");
    __m128i res_v = _mm_or_si128(_mm_and_si128(compare_gt_v, sum_v), _mm_and_si128(compare_let_v, aij_v));
    // printf("res_v:\n");

    _mm_storeu_si128((__m128i*)aij, res_v);
    // printf("AIJ: %d %d %d %d\n", aij[0], aij[1], aij[2], aij[3]);

    return ((int*)(&compare_gt_v))[0] || ((int*)(&compare_gt_v))[1] || ((int*)(&compare_gt_v))[2] || ((int*)(&compare_gt_v))[3];
}
// Relax with intermediate node k, from node i to j
int relax_s(int *aij, int aik, int akj){
    if((*aij) > aik + akj){
        (*aij) = aik + akj;
        return 1;
    }
    return 0;
}
// Relax the node from A(i,j) to A(i,j+size), includes node which j+size > vertex_num
void relax(int idx, int ak, int size){
    int ai = idx / vertex_num, aj = idx % vertex_num;
    int i = ai, j = aj, remain_size = size;
    for(i = ai; i < vertex_num; i++){
        if(remain_size <= 0){return;}
        int truncated_size = j + remain_size > vertex_num? vertex_num - j : remain_size;
        int vec_size = (truncated_size >> VECSCALE) << VECSCALE;
        int vec_end = j + vec_size, single_end = j + truncated_size;
        remain_size -= truncated_size;
        
        // Relax with Vectorization speed up
        for(; j < vec_end; j+=VECGAP){
            relax_v(get_graph_addr(i, j), get_graph(i, ak), get_graph_addr(ak, j));
        }
        // Single relax
        for(; j < single_end; j++){
            relax_s(get_graph_addr(i, j), get_graph(i, ak), get_graph(ak, j));
        }
        j = 0;
    }
}
// b_i, b_j, b_k are the index of the block on the dimension i, j, k
void relax_block(int b_i, int b_j, int b_k){
    BlockDim bidx = get_block_pos(b_i, b_j, b_k);
    BlockDim bdim = get_block_size(b_i, b_j, b_k);
    // printf("B(%d %d %d), IDX(%d %d %d) DIM(%d %d %d)\n", b_i, b_j, b_k, bidx.i, bidx.j, bidx.k, bdim.i, bdim.j, bdim.k);
    for(int k = bidx.k; k < bidx.k + bdim.k; k++){
        for(int i = bidx.i; i < bidx.i + bdim.i; i++){
            relax(get_graph_idx(i, bidx.j), k, bdim.j);
        }
    }
}
// Without Vectorization, b_i, b_j, b_k are the index of the block on the dimension i, j, k
void relax_block_s(int b_i, int b_j, int b_k){
    BlockDim bidx = get_block_pos(b_i, b_j, b_k);
    BlockDim bdim = get_block_size(b_i, b_j, b_k);
    // printf("B(%d %d %d), IDX(%d %d %d) DIM(%d %d %d)\n", b_i, b_j, b_k, bidx.i, bidx.j, bidx.k, bdim.i, bdim.j, bdim.k);
    // printf("Thread %d B(%d %d %d), IDX(%d %d %d) DIM(%d %d %d)\n", omp_get_num_threads(), b_i, b_j, b_k, bidx.i, bidx.j, bidx.k, bdim.i, bdim.j, bdim.k);
    for(int k = bidx.k; k < bidx.k + bdim.k; k++){
        for(int i = bidx.i; i < bidx.i + bdim.i; i++){
            for(int j = bidx.j; j < bidx.j + bdim.j; j++){
                relax_s(get_graph_addr(i, j), get_graph(i, k), get_graph(k, j));
            }
        }
    }
}

void block_floyd_warshall(){
    for(int k = 0; k < num_blocks; k++){
        relax_block(k, k, k);

        #pragma omp parallel for schedule(static)
        for(int j = 0; j < num_blocks; j++){
            if(j == k){continue;}
            relax_block(k, j, k);
        }
        #pragma omp parallel for schedule(static) 
        for(int i = 0; i < num_blocks; i++){
            if(i == k){continue;}
            relax_block(i, k, k);
        }
        #pragma omp parallel for schedule(static) collapse(2)
        for(int i = 0; i < num_blocks; i++){
            for(int j = 0; j < num_blocks; j++){
                if(i == k || j == k){continue;}
                relax_block(i, j, k);
            }
        }
    }
}

// void init_block_deps(){
//     block_deps = (int*)malloc(SIZEOFINT * block_num_squr);
//     memset(block_deps, 0, block_num_squr * SIZEOFINT);
//     // for(int i = 1; i < block_num_squr; i++){
//     //     block_deps[i] = -1;
//     // }
// }
// // Get the block dependency of block(b_i, b_j)
// int get_block_dep(int b_i, int b_j){
//     return block_deps[b_i * num_blocks + b_j];
// }
// // Increase block dependency(k) by 1
// int set_block_dep(int b_i, int b_j){
//     return ++block_deps[b_i * num_blocks + b_j];
// }
// int show_block_dep(int b_i, int b_j, int b_k){
//     if(b_i == b_j && b_j == b_k){
//         printf("Dep of B(%d %d %d) is B(%d %d): %d, res: %d\n",  b_i, b_j, b_k, b_i, b_j, get_block_dep(b_i, b_j), get_block_dep(b_i, b_j) == b_k);
//         return get_block_dep(b_i, b_j) == b_k;
//     }else if(b_i == b_k || b_j == b_k){
//         printf("Dep of B(%d %d %d) is B(%d %d): %d, res: %d\n",  b_i, b_j, b_k, b_k, b_k, get_block_dep(b_k, b_k), get_block_dep(b_k, b_k) > b_k);
//         return get_block_dep(b_k, b_k) > b_k;
//     }else{
//         return get_block_dep(b_i, b_k) > b_k && get_block_dep(b_k, b_j) > b_k;
//     }
// }
// // Check block(b_i, b_j) dependency(k) is satisfy(equal) to required dependency b_k
// int check_block_dep(int b_i, int b_j, int b_k){
//     if(b_i == b_j && b_j == b_k){
//         return get_block_dep(b_i, b_j) == b_k;
//     }else if(b_i == b_k || b_j == b_k){
//         return get_block_dep(b_k, b_k) > b_k;
//     }else{
//         return get_block_dep(b_i, b_k) > b_k && get_block_dep(b_k, b_j) > b_k;
//     }
// }

// ThreadPool *create_thread_pool(int threads_num){
//     ThreadPool *pool = (ThreadPool *)malloc(sizeof(ThreadPool));
//     pool->threads = NULL;
//     pool->lock = (pthread_mutex_t*)malloc(sizeof(pthread_mutex_t));
//     pool->sync_lock = (pthread_mutex_t*)malloc(sizeof(pthread_mutex_t));
//     pool->sync_cond = (pthread_cond_t*)malloc(sizeof(pthread_cond_t));
//     pool->sync_counter = 0;
//     pool->task_queue = NULL;
//     pthread_mutex_init(pool->lock, NULL);
//     pthread_mutex_init(pool->sync_lock, NULL);
//     pthread_cond_init(pool->sync_cond, NULL);
//     pool->threads_num = threads_num;
//     pool->is_finish = 0;

//     return pool;
// }
// void init_task_queue(ThreadPool *pool){
//     pool->task_queue = (BlockDim*)malloc(sizeof(BlockDim) * block_num_cubic);
//     int task_counter = 0;
//     for(int k = 0; k < num_blocks; k++){
//         pool->task_queue[task_counter++] = get_BlockDim(k, k, k);

//         for(int j = 0; j < num_blocks; j++){
//             // printf("A %d\n", j);
//             if(j == k){continue;}
//             pool->task_queue[task_counter++] = get_BlockDim(k, j, k);
//             // printf("A %d Done\n", j);
//         }
//         // printf("A FINISH\n");
//         for(int i = 0; i < num_blocks; i++){
//             if(i == k){continue;}
//             // printf("B %d\n", i);
//             // relax_block(i, k, k);
//             pool->task_queue[task_counter++] = get_BlockDim(i, k, k);
//             // printf("B %d Done\n", i);
//         }
//         // printf("B FINISH\n");
//         for(int i = 0; i < num_blocks; i++){
//             for(int j = 0; j < num_blocks; j++){
//                 if(i == k || j == k){continue;}
//                 // printf("C %d:%d\n", i, j);
//                 // relax_block(i, j, k);
//                 pool->task_queue[task_counter++] = get_BlockDim(i, j, k);
//                 // printf("C %d:%d Done\n", i, j);
//             }
//         }
//     }
// }
// int get_threads_num(ThreadPool* pool){
//     return pool->threads_num;
// }
// void sync_threads(ThreadPool* pool){
//     pthread_mutex_lock(pool->sync_lock);
//     pool->sync_counter++;
//     if(pool->sync_counter < pool->threads_num){
//         pthread_cond_wait(pool->sync_cond, pool->sync_lock);
//     }else{
//         pthread_cond_signal(pool->sync_cond);
//     }
//     pthread_mutex_unlock(pool->sync_lock);
// }
// void get_task(BlockDim *task, ThreadPool *pool){
//     static int counter = 0;

//     pthread_mutex_lock(pool->lock);
//     if(block_assign_step < block_num_cubic){
//         task->i = pool->task_queue[block_assign_step].i;
//         task->j = pool->task_queue[block_assign_step].j;
//         task->k = pool->task_queue[block_assign_step].k;
//         // counter = (counter + 1) % block_num_squr;
//         block_assign_step++;
//     }else{
//         task->i = STOPVAL;
//         task->j = STOPVAL;
//         task->k = STOPVAL;
//     }
//     pthread_mutex_unlock(pool->lock);
// }

// void *worker(void *arg){
//     int thread_id = ((WorkerArg*)arg)->thread_id;
//     ThreadPool *pool = ((WorkerArg*)arg)->pool;
//     // printf("Created Thread %d\n", thread_id);

//     // buf2graph(buf, graph, thread_id, edge_num, pool->threads_num);
//     BlockDim task_block;
//     int counter = 0;
//     for(;;){
//         get_task(&task_block, pool);
//         // printf("Thread %d Got B(%d %d %d)\n", thread_id, task_block.i, task_block.j, task_block.k);
//         if(task_block.i == STOPVAL && task_block.j == STOPVAL && task_block.k == STOPVAL){break;}
//         while(true){
//             if(check_block_dep(task_block.i, task_block.j, task_block.k)){
//                 // show_block_dep(task_block.i, task_block.j, task_block.k);
//                 break;
//             }else{
//                 counter++;
//                 // show_block_dep(task_block.i, task_block.j, task_block.k);
//             }
//         }
//         // printf("Thread %d Got B(%d %d %d) Passed Dep: %d\n", thread_id, task_block.i, task_block.j, task_block.k, get_block_dep(task_block.i, task_block.j));
//         relax_block(task_block.i, task_block.j, task_block.k);
//         // printf("Thread %d Done B(%d %d %d)\n", thread_id, task_block.i, task_block.j, task_block.k);
//         set_block_dep(task_block.i, task_block.j);
//     }

//     pthread_exit(NULL);
// }

// void start_pool(ThreadPool* pool){
//     pool->threads = (pthread_t*)malloc(sizeof(pthread_t) * get_threads_num(pool));
//     WorkerArg *worker_args = (WorkerArg*)malloc(sizeof(WorkerArg) * get_threads_num(pool));
//     printf("Creating %d Threads\n", get_threads_num(pool));
//     for(int i = 0; i < get_threads_num(pool); i++){
//         worker_args[i].pool = pool;
//         worker_args[i].thread_id = i;
//         pthread_create(&(pool->threads[i]), NULL, worker, (void*)(&(worker_args[i])));
//     }
// }

// void set_finish(ThreadPool *pool){
//     pool->is_finish = 1;
// }

// // Join the threads
// void end_pool(ThreadPool* pool){
//     pool->is_finish = 1;
//     for(int i = 0; i < get_threads_num(pool); i++){
//         pthread_join(pool->threads[i], NULL);
//     }
// }
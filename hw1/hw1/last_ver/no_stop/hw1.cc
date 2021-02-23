#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <boost/sort/spreadsort/spreadsort.hpp>

// void show_arr(float *a, int n){
//     for(int i = 0; i < n; i++){
//         printf("%f ", a[i]);
//     }
//     printf("\n");
// }

// int rank: Current process rank
// int c_size: COMM_WORLD size
// int n: length of sequence
int count_b_size(int rank, int c_size, int n){
    // n / c_size
    int q = n / c_size;
    int r = n % c_size;
    if(r == 0){
        // Divisiable
        return q;
    }else{
        // handle residual
        if(rank < r){return q + 1;}
        else{return q;}
    }
}

// int rank: Current process rank
// int c_size: COMM_WORLD size
// int n: length of sequence
int count_b_start(int rank, int c_size, int n){
    // n / c_size
    int q = n / c_size;
    int r = n % c_size;
    if(r == 0){
        // Divisiable
        return q * rank;
    }else{
        // handle residual
        if(rank < r){return q * rank + rank;}
        else{return q * rank + r;}
    }
}

// Merge and Reallocate
// float *buf: The buf that current process has
// int b_size: the size of buf
// float *buf_t: The buf that receive from other process
// int b_t_size: The size of buf_t
// int in_pair_id: The id of element in a pair, the left one is 0 and the right one is 1
void merge_realc(float *buf_m, float *buf, int b_size, float *buf_t, int b_t_size, int in_pair_id, int *is_change){
    int b_m_size = b_size;
    // float *buf_m = (float*)malloc(sizeof(float) * b_m_size);
    int i = 0, b_i = 0, t_i = 0;

    if(!in_pair_id){
        // If in_pair_id == 0, redirect the array to smaller segment.
        for(i = 0, b_i = 0, t_i = 0; i < b_m_size && b_i < b_size && t_i < b_t_size;){
            if(buf_t[t_i] >= buf[b_i]){buf_m[i++] = buf[b_i++];}
            else{buf_m[i++] = buf_t[t_i++]; *is_change = 1;}
        }

        while(b_i < b_size && i < b_m_size){buf_m[i++] = buf[b_i++];}

        while(t_i < b_t_size && i < b_m_size){buf_m[i++] = buf_t[t_i++]; *is_change = 1;}
    }else{
        // If in_pair_id == 1, redirect the array to larger segment.
        for(i = b_m_size - 1, b_i = b_size - 1, t_i = b_t_size - 1; i >= 0 && b_i >= 0 && t_i >= 0;){
            if(buf[b_i] >= buf_t[t_i]){buf_m[i--] = buf[b_i--];}
            else{buf_m[i--] = buf_t[t_i--]; *is_change = 1;}
        }
        while(b_i >= 0 && i >= 0){buf_m[i--] = buf[b_i--];}

        while(t_i >= 0 && i >= 0){buf_m[i--] = buf_t[t_i--]; *is_change = 1;}
    }

    if(b_size == 1){*is_change = 1;}

    // return buf_m;
}

// int rank: Current process rank
// int c_size: COMM_WORLD size
// float *buf: array of the sequence for corresponding  segment
// int b_size: buf size
// int n: length of sequence
float* OESort(int rank, int c_size, float *buf, int b_size, int n){
    boost::sort::spreadsort::spreadsort(buf, &(buf[b_size]));
    // printf("rank %d has array[%d]: %f %f\n", rank, b_size, buf[0], buf[1]);
    // show_arr(buf, b_size);
    int is_all_change = 1;
    int b_t_real_size = count_b_size(0, c_size, n);
    float *buf_t = (float*)malloc(sizeof(float) * b_t_real_size);
    float *buf_m = (float*)malloc(sizeof(float) * b_size);

    for(int phase = 0; phase <= c_size; phase++){
        // Check whether the target rank exist or not
        int target_rank = -1, in_pair_id = 0;
        int is_change = 0;
        if(rank % 2 == phase % 2){
            // Even Rank in Even Phase & Odd Rank in Odd Phase
            target_rank = (rank + 1) < c_size? (rank + 1) : -1;
            // Left one in a pair
            in_pair_id = 0;
        }else{
            // Odd Rank in Even Phase & Even Rank in Odd Phase
            target_rank = (rank - 1) >= 0? (rank - 1) : -1;
            // Right one in a pair
            in_pair_id = 1;
        }
        if(target_rank == -1){
            // MPI_Allreduce(&is_change, &is_all_change, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD);
            continue;
        }

        // Phase Sorting
        MPI_Request req_s, req_r;
        MPI_Isend(buf, b_size, MPI_FLOAT, target_rank, in_pair_id, MPI_COMM_WORLD, &req_s);
        int b_t_size = count_b_size(target_rank, c_size, n);
        // memset(buf_t, '0', sizeof(float) * b_t_real_size);
        // memset(buf_m, '0', sizeof(float) * b_size);

        MPI_Irecv(buf_t, b_t_size, MPI_FLOAT, target_rank, (!in_pair_id), MPI_COMM_WORLD, &req_r);
        MPI_Wait(&req_r, MPI_STATUS_IGNORE);
        // printf("Phase %d rank %d(as id %d, TR %d) has reccived[%d]: %f %f\n", 
        //         phase, rank, in_pair_id, target_rank, b_t_size, buf_t[0], buf_t[1]);

        merge_realc(buf_m, buf, b_size, buf_t, b_t_size, in_pair_id, &is_change);
        // float *buf_m = merge_realc(buf, b_size, buf_t, b_t_size, in_pair_id, &is_change);
        // MPI_Allreduce(&is_change, &is_all_change, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD);
        // float *buf_d = buf;
        // free(buf_d);
        // buf = buf_m;
        // for(int i = 0; i < b_size; i++){
        //     buf[i] = buf_m[i];
        // }
        memcpy(buf, buf_m, sizeof(float) * b_size);
    }
    free(buf_t);
    free(buf_m);

    return buf;
}

int main(int argc, char** argv) {
	MPI_Init(&argc,&argv);
	int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

    int n = atoi(argv[1]);
    int data_start = count_b_start(rank, size, n), 
        data_size = count_b_size(rank, size, n);
    float *data = (float*)malloc(sizeof(float) * n);
	
	MPI_File in_f, out_f;
	MPI_File_open(MPI_COMM_WORLD, argv[2], MPI_MODE_RDONLY, MPI_INFO_NULL, &in_f);
    MPI_File_open(MPI_COMM_WORLD, argv[3], MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &out_f);
	MPI_File_read_at(in_f, sizeof(float) * data_start, data, data_size, MPI_FLOAT, MPI_STATUS_IGNORE);

    data = OESort(rank, size, data, data_size, n);
    // printf("Rank %d Out %f\n", rank, data[0]);
    // show_arr(data, data_size);

    // printf("c_size %d, data_start %d, data_size %d, n %d\n", size, data_start, data_size, n);
    // show_arr(data, data_size);
	MPI_File_write_at(out_f, sizeof(float) * data_start, data, data_size, MPI_FLOAT, MPI_STATUS_IGNORE);
    MPI_File_close(&in_f);
    MPI_File_close(&out_f);
	MPI_Finalize();
}
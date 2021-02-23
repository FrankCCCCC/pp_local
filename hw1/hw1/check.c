// #include <cstdio>
#include <stdio.h>
#include <stdlib.h>

void show_seg(float *a, float *b){
    for(float *i = a; i < b; i++){
        printf("%.3e ", *i);
    }
}

int main(int argc, char** argv) {
    int n = 0, f_size = 0;
    int seg_len = 5;
    FILE *ptr;

    ptr = fopen(argv[1],"rb");  // r for read, b for binary

    fseek(ptr, 0L, SEEK_END);
    f_size = ftell(ptr);
    n = f_size / sizeof(float);
    rewind(ptr);

    float *buffer = (float*)malloc(sizeof(float) * n);
    fread(buffer, sizeof(float), n, ptr); // read 10 bytes to our buffer

    int is_wrong = 0;
    printf("File: %s with size %d bytes(%d floats)\n", argv[1], f_size, n);
    printf("The first elements ");
    show_seg(buffer, &(buffer[5]));
    printf("\n");
    
    for(int i = 0; i < n - 1; i++){
        if(buffer[i] > buffer[i + 1]){
            printf("Wrong at %d bytes (%d floats): ", i * sizeof(float), i);
            show_seg(&(buffer[i - seg_len >= 0? i - seg_len : 0]), &(buffer[i + seg_len < n? i + seg_len : n - 1]));
            printf("\n");

            is_wrong = 1;
        }
    }

    if(!is_wrong){printf("All Correct\n");}

    // for(int i = 0; i < n; i++){
    //     printf("%f ", buffer[i]);
    // }
    // printf("\n");
}
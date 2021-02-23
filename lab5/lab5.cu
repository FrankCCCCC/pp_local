  
#include <iostream>
#include <cstdlib>
#include <cassert>
#include <zlib.h>
#include <png.h>

#include <cuda_runtime.h>
#include <cuda.h>
#include <memory>
#include <vector_types.h>

#define MASK_N 2
#define MASK_X 5
#define MASK_Y 5
#define SCALE 8
#define TH_DIM 32

const dim3 thread_dim(TH_DIM, TH_DIM);
const int block_num = 5000;

/* Hint 7 */
// this variable is used by device
__constant__ int mask[MASK_N][MASK_X][MASK_Y] = { 
    {{ -1, -4, -6, -4, -1},
     { -2, -8,-12, -8, -2},
     {  0,  0,  0,  0,  0}, 
     {  2,  8, 12,  8,  2}, 
     {  1,  4,  6,  4,  1}},
    {{ -1, -2,  0,  2,  1}, 
     { -4, -8,  0,  8,  4}, 
     { -6,-12,  0, 12,  6}, 
     { -4, -8,  0,  8,  4}, 
     { -1, -2,  0,  2,  1}} 
};

int read_png(const char* filename, unsigned char** image, unsigned* height, 
             unsigned* width, unsigned* channels) {

    unsigned char sig[8];
    FILE* infile;
    infile = fopen(filename, "rb");

    fread(sig, 1, 8, infile);
    if (!png_check_sig(sig, 8))
        return 1;   /* bad signature */

    png_structp png_ptr;
    png_infop info_ptr;

    png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png_ptr)
        return 4;   /* out of memory */
  
    info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr) {
        png_destroy_read_struct(&png_ptr, NULL, NULL);
        return 4;   /* out of memory */
    }

    png_init_io(png_ptr, infile);
    png_set_sig_bytes(png_ptr, 8);
    png_read_info(png_ptr, info_ptr);
    int bit_depth, color_type;
    png_get_IHDR(png_ptr, info_ptr, width, height, &bit_depth, &color_type, NULL, NULL, NULL);

    png_uint_32  i, rowbytes;
    png_bytep  row_pointers[*height];
    png_read_update_info(png_ptr, info_ptr);
    rowbytes = png_get_rowbytes(png_ptr, info_ptr);
    *channels = (int) png_get_channels(png_ptr, info_ptr);

    if ((*image = (unsigned char *) malloc(rowbytes * *height)) == NULL) {
        png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
        return 3;
    }

    for (i = 0;  i < *height;  ++i)
        row_pointers[i] = *image + i * rowbytes;
    png_read_image(png_ptr, row_pointers);
    png_read_end(png_ptr, NULL);
    return 0;
}

void write_png(const char* filename, png_bytep image, const unsigned height, const unsigned width, 
               const unsigned channels) {
    FILE* fp = fopen(filename, "wb");
    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    png_infop info_ptr = png_create_info_struct(png_ptr);
    png_init_io(png_ptr, fp);
    png_set_IHDR(png_ptr, info_ptr, width, height, 8,
                 PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_set_filter(png_ptr, 0, PNG_NO_FILTERS);
    png_write_info(png_ptr, info_ptr);
    png_set_compression_level(png_ptr, 1);

    png_bytep row_ptr[height];
    for (int i = 0; i < height; ++ i) {
        row_ptr[i] = image + i * width * channels * sizeof(unsigned char);
    }
    png_write_image(png_ptr, row_ptr);
    png_write_end(png_ptr, NULL);
    png_destroy_write_struct(&png_ptr, &info_ptr);
    fclose(fp);
}

// __device__ int get_mask(int *mask, int n, int x, int y){
//     return mask[n * MASK_X * MASK_Y + x * MASK_Y + y];
// }

/* Hint 5 */
// this function is called by host and executed by device
// extern __shared__ int sm_s[];
__global__ void sobel(unsigned char* s, unsigned char* t, unsigned height, unsigned width, unsigned channels) {
    int  x, y, i, v, u;
    int  R, G, B;
    float val[MASK_N*3] = {0.0};

    // const int x_divisor = blockDim.x - 1;
    // const int x_width_div = width / x_divisor;
    // const int x_width_mod = width % x_divisor;
    // const int x_start = threadIdx.x * x_width_div;
    // const int x_width = threadIdx.x < blockDim.x - 1? x_start + x_width_div : x_start + x_width_mod;
    // const int x_gap = 1;

    const int x_start = threadIdx.x;
    const int x_width = width;
    const int x_gap = blockDim.x;

    const int y_start = blockIdx.x * blockDim.y + threadIdx.y;
    const int y_height = height;
    const int y_gap = gridDim.x * blockDim.y;
    
    const int adjustX = (MASK_X % 2) ? 1 : 0;
    const int adjustY = (MASK_Y % 2) ? 1 : 0;
    const int xBound = MASK_X / 2;
    const int yBound = MASK_Y / 2;

    const int kernel_width = 2 * xBound + adjustX + TH_DIM - 1;

    // __shared__ int sm_mask[MASK_N][MASK_X][MASK_Y];
    __shared__ unsigned char sm_s[40000];
    // printf("BLock %d, Thread (%d, %d) Created, Conv Box (%d : %d, %d : %d), Kernel Width: %d\n", blockIdx.x, threadIdx.x, threadIdx.y, \
    //         -xBound, xBound + adjustX + blockDim.x - 1, -yBound,  yBound + adjustY + blockDim.y - 1, kernel_width);

    // for(int i = 0; i < MASK_N; i++){
    //     for(int x = threadIdx.x; x < MASK_X; x+=blockDim.x){
    //         for(int y = threadIdx.y; y < MASK_Y; y+=blockDim.y){
    //             sm_mask[i][x][y] = mask[i][x][y];
    //         }
    //     }
    // }
    // __syncthreads();

    char mask[MASK_N][MASK_X][MASK_Y] = { 
        {{ -1, -4, -6, -4, -1},
         { -2, -8,-12, -8, -2},
         {  0,  0,  0,  0,  0}, 
         {  2,  8, 12,  8,  2}, 
         {  1,  4,  6,  4,  1}},
        {{ -1, -2,  0,  2,  1}, 
         { -4, -8,  0,  8,  4}, 
         { -6,-12,  0, 12,  6}, 
         { -4, -8,  0,  8,  4}, 
         { -1, -2,  0,  2,  1}} 
    };

    /* Hint 6 */
    // parallel job by blockIdx, blockDim, threadIdx 
    for (y = y_start; y < y_height; y+=y_gap) {
        for (x = x_start; x < x_width; x+=x_gap) {

            for (v = -yBound; v < yBound + adjustY; v+=2) {
                for (u = -xBound; u < xBound + adjustX; u+=2) {
                    if ((x + u) >= 0 && (x + u) < width && y + v >= 0 && y + v < height) {
                        int base = channels * (kernel_width * (v + yBound + threadIdx.y) + (u + xBound + threadIdx.x));
                        sm_s[base + 2] = s[channels * (width * (y+v) + (x+u)) + 2];
                        sm_s[base + 1] = s[channels * (width * (y+v) + (x+u)) + 1];
                        sm_s[base + 0] = s[channels * (width * (y+v) + (x+u)) + 0];
                    }
                }
            }
            __syncthreads();

            for (i = 0; i < MASK_N; ++i) {
                val[i*3+2] = 0.0;
                val[i*3+1] = 0.0;
                val[i*3] = 0.0;

                for (v = -yBound; v < yBound + adjustY; v++) {
                    for (u = -xBound; u < xBound + adjustX; u++) {
                        if ((x + u) >= 0 && (x + u) < width && y + v >= 0 && y + v < height) {
                            int base = channels * (kernel_width * (v + yBound + threadIdx.y) + (u + xBound + threadIdx.x));

                            R = sm_s[base + 2];
                            G = sm_s[base + 1];
                            B = sm_s[base + 0];

                            // R = s[channels * (width * (y+v) + (x+u)) + 2];
                            // G = s[channels * (width * (y+v) + (x+u)) + 1];
                            // B = s[channels * (width * (y+v) + (x+u)) + 0];
                            
                            val[i*3+2] += R * mask[i][u + xBound][v + yBound];
                            val[i*3+1] += G * mask[i][u + xBound][v + yBound];
                            val[i*3+0] += B * mask[i][u + xBound][v + yBound];

                            // printf("B(%d (%d %d)): RGB(%d %d %d) | sm_s(%d %d %d)\n", blockIdx.x, threadIdx.x, threadIdx.y, R, G, B, sm_s[sm_base_idx + 2], sm_s[sm_base_idx + 1], sm_s[sm_base_idx + 0]);
                        }    
                    }
                }
            }

            float totalR = 0.0;
            float totalG = 0.0;
            float totalB = 0.0;
            for (i = 0; i < MASK_N; ++i) {
                totalR += val[i * 3 + 2] * val[i * 3 + 2];
                totalG += val[i * 3 + 1] * val[i * 3 + 1];
                totalB += val[i * 3 + 0] * val[i * 3 + 0];
            }

            totalR = sqrt(totalR) / SCALE;
            totalG = sqrt(totalG) / SCALE;
            totalB = sqrt(totalB) / SCALE;
            const unsigned char cR = (totalR > 255.0) ? 255 : totalR;
            const unsigned char cG = (totalG > 255.0) ? 255 : totalG;
            const unsigned char cB = (totalB > 255.0) ? 255 : totalB;
            t[channels * (width * y + x) + 2] = cR;
            t[channels * (width * y + x) + 1] = cG;
            t[channels * (width * y + x) + 0] = cB;
        }
    }
    // memcpy(t, s, (height * width * channels * sizeof(unsigned char)));
    // t[idx_x] = idx_x;
}

int main(int argc, char** argv) {

    assert(argc == 3);
    unsigned height, width, channels;
    unsigned char* host_s = NULL;
    read_png(argv[1], &host_s, &height, &width, &channels);
    unsigned char* host_t = (unsigned char*) malloc(height * width * channels * sizeof(unsigned char));

    printf("Channel: %d\n", channels);
    
    /* Hint 1 */
    // cudaMalloc(...) for device src and device dst
    unsigned char *cuda_mem_s = NULL, *cuda_mem_t = NULL;
    cudaMalloc((void **)&cuda_mem_s, (height * width * channels * sizeof(unsigned char)));
    cudaMalloc((void **)&cuda_mem_t, (height * width * channels * sizeof(unsigned char)));

    /* Hint 2 */
    // cudaMemcpy(...) copy source image to device (filter matrix if necessary)
    cudaMemcpy(cuda_mem_s, host_s, (height * width * channels * sizeof(unsigned char)), cudaMemcpyHostToDevice);

    // for(int i = 0; i < 10; i++){
    //     printf("Before-S: %d, T:%d\n", host_s[i], host_t[i]);
    // }

    /* Hint 3 */
    // acclerate this function
    sobel<<<block_num, thread_dim>>>(cuda_mem_s, cuda_mem_t, height, width, channels);
    // sobel(host_s, host_t, height, width, channels);
    
    /* Hint 4 */
    // cudaMemcpy(...) copy result image to host 
    cudaMemcpy(host_t, cuda_mem_t, (height * width * channels * sizeof(unsigned char)), cudaMemcpyDeviceToHost);

    // for(int i = 0; i < 10; i++){
    //     printf("After-S: %d, T:%d\n", host_s[i], host_t[i]);
    // }

    write_png(argv[2], host_t, height, width, channels);

    return 0;
}
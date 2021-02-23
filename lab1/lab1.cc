#include <assert.h>
#include <stdio.h>
#include <math.h>
#include <mpi.h>

int main(int argc, char** argv) {
	if (argc != 3) {
		fprintf(stderr, "must provide exactly 2 arguments!\n");
		return 1;
	}
	// Set up MPI
	int size, rank;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Radius & Mod
	unsigned long long r = atoll(argv[1]);
	unsigned long long k = atoll(argv[2]);

    // Parallel
    unsigned long long pixels = 0;
    
    for(unsigned long long x = (unsigned long long)rank; x < r; x += (unsigned long long)size){
        unsigned long long row_pixels = (unsigned long long)ceil(sqrtl(r * r - x * x));
        pixels = (row_pixels + pixels) % k;
    }
    // printf("Proc %d: %llu\n", rank, pixels);

    if(rank == 0){
        unsigned long long *pixels_from_procs = (unsigned long long*)malloc(sizeof(unsigned long long) * size);
        
        for(int i = 1; i < size; i++){
            // MPI_Status status;
            MPI_Recv(&(pixels_from_procs[i]), 1, MPI_UNSIGNED_LONG_LONG, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            // printf("Received from %d: %llu pixels\n", i, pixels_from_procs[i]);
        }
        
        for(int i = 1; i < size; i++){
            pixels = (pixels + pixels_from_procs[i]) % k;
        }
        pixels = (4 * pixels) % k;

        // printf("PIXELS: %llu\n", pixels);
        printf("%llu\n", pixels);
    }else{
        // tag 0, send to rank 0
        MPI_Send(&pixels, 1, MPI_UNSIGNED_LONG_LONG, 0, 0, MPI_COMM_WORLD);
        // printf("Proc %d Sended\n", rank);
    }

    MPI_Finalize();
}

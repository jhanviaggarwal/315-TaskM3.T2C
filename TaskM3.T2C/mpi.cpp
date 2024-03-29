#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define N 1000 // Matrix size

// Function to initialize matrices A and B
void initialize_matrices(int *A, int *B, int size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            A[i * size + j] = rand() % 10; // Initialize A with random values
            B[i * size + j] = rand() % 10; // Initialize B with random values
        }
    }
}

// Function to perform matrix multiplication
void matrix_mult(int *A, int *B, int *C, int size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            C[i * size + j] = 0;
            for (int k = 0; k < size; k++) {
                C[i * size + j] += A[i * size + k] * B[k * size + j];
            }
        }
    }
}

int main(int argc, char *argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int *A, *B, *C;
    A = (int *)malloc(N * N * sizeof(int));
    B = (int *)malloc(N * N * sizeof(int));
    C = (int *)malloc(N * N * sizeof(int));

    // Initialize matrices A and B
    initialize_matrices(A, B, N);

    // Divide work among processes
    int chunk_size = N / size;
    int start = rank * chunk_size;
    int end = (rank + 1) * chunk_size;
    if (rank == size - 1) {
        end = N; // Last process takes care of remaining rows
    }

    // Each process performs matrix multiplication for its assigned rows
    matrix_mult(A + start * N, B, C + start * N, end - start);

    // Gather the result from all processes to process 0
    MPI_Gather(C + start * N, chunk_size * N, MPI_INT, C, chunk_size * N, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        // Print the resulting matrix C (optional)
        printf("Resulting matrix C:\n");
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                printf("%d ", C[i * N + j]);
            }
            printf("\n");
        }
    }

    free(A);
    free(B);
    free(C);

    MPI_Finalize();
    return 0;
}
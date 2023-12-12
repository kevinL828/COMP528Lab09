#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>


//task to be performed by GPU. Don't forget to add global
__global__ void vectorArithmetic(float *z, const float *x, const float *y, float A, int num) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num) {
        z[i] = A * x[i] + y[i];
    }
}

int main(void) {
    //declaration of host variables
    const int num = 50;
    float *z, *x, *y;
    float *d_z, *d_x, *d_y; // Device pointers
    float A = 34;

    //initialising host variables
    z = (float*) malloc(num * sizeof(float));
    x = (float*) malloc(num * sizeof(float));
    y = (float*) malloc(num * sizeof(float));

    for(int i = 0; i < num; i++) {
        x[i] = i;
        y[i] = 7 * i;
    }

    // Allocate device memory
    cudaMalloc((void**)&d_z, num * sizeof(float));
    cudaMalloc((void**)&d_x, num * sizeof(float));
    cudaMalloc((void**)&d_y, num * sizeof(float));

    // Copy data from host to device
    cudaMemcpy(d_x, x, num * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, num * sizeof(float), cudaMemcpyHostToDevice);

    // Launch the kernel
    int blockSize = 256;
    int gridSize = (num + blockSize - 1) / blockSize;
    vectorArithmetic<<<gridSize, blockSize>>>(d_z, d_x, d_y, A, num);

    // Copy result back to host
    cudaMemcpy(z, d_z, num * sizeof(float), cudaMemcpyDeviceToHost);

    //postprocessing: output to terminal
    for(int j = 0; j < num; j++) {
        printf("%f ", z[j]);
    }
    printf("\n");

    // Free device memory
    cudaFree(d_z);
    cudaFree(d_x);
    cudaFree(d_y);

    // Free host memory
    free(z);
    free(x);
    free(y);

    return 0;
}

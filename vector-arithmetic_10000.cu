#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__global__ void vectorArithmetic(float *z, const float *x, const float *y, float A, int num) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;  // Total number of threads
    for (int i = index; i < num; i += stride) {
        z[i] = A * x[i] + y[i];
    }
}

int main(void) {
    const int num = 1000000;  // Increase num to 1,000,000
    float *z, *x, *y;
    float *d_z, *d_x, *d_y;
    float A = 34;

    z = (float*) malloc(num * sizeof(float));
    x = (float*) malloc(num * sizeof(float));
    y = (float*) malloc(num * sizeof(float));

    for(int i = 0; i < num; i++) {
        x[i] = i;
        y[i] = 7 * i;
    }

    cudaMalloc((void**)&d_z, num * sizeof(float));
    cudaMalloc((void**)&d_x, num * sizeof(float));
    cudaMalloc((void**)&d_y, num * sizeof(float));

    cudaMemcpy(d_x, x, num * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, num * sizeof(float), cudaMemcpyHostToDevice);

    // Timing the kernel execution
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Experiment with these values
    int blockSize = 256;
    int gridSize = (num + blockSize - 1) / blockSize;
    vectorArithmetic<<<gridSize, blockSize>>>(d_z, d_x, d_y, A, num);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(z, d_z, num * sizeof(float), cudaMemcpyDeviceToHost);

    // Output timing result
    printf("Time taken: %f ms\n", milliseconds);

    cudaFree(d_z);
    cudaFree(d_x);
    cudaFree(d_y);

    free(z);
    free(x);
    free(y);

    return 0;
}

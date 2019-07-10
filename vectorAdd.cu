#include <stdio.h>
# include "cuda_runtime.h"
# include "cuda_profiler_api.h"


__global__
void add(int n, float *x, float *y, float *z)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < n) z[i] = x[i] + y[i];
  //if (i< n) z[i]++;
}

int main() {
    int N = 1<<15;
    float *x, *y, *z, *d_x, *d_y, *d_z;
/*
    x = (float*)malloc(N*sizeof(float));
    y = (float*)malloc(N*sizeof(float));
    z = (float*)malloc(N*sizeof(float));
*/
/*
    cudaMalloc(&d_x, N*sizeof(float));
    cudaMalloc(&d_y, N*sizeof(float));
    cudaMalloc(&d_z, N*sizeof(float));
*/
    cudaHostAlloc((void **)&x,  N*sizeof(float), cudaHostAllocMapped );
    cudaHostAlloc((void **)&y,  N*sizeof(float), cudaHostAllocMapped );
    cudaHostAlloc((void **)&z,  N*sizeof(float), cudaHostAllocMapped );
    for (int i = 0; i < N; i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }
/*
    cudaMemcpy(d_x, x, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_z, z, N*sizeof(float), cudaMemcpyHostToDevice);
*/

    cudaHostGetDevicePointer((void **)&d_x, x, 0);
    cudaHostGetDevicePointer((void **)&d_y, y, 0);
    cudaHostGetDevicePointer((void **)&d_z, z, 0);

    //cudaDeviceReset();
    cudaDeviceSynchronize();
    cudaProfilerStart();
    add<<<(N+255)/256, 256>>>(N, d_x, d_y, d_z);
    cudaProfilerStop();
    cudaDeviceSynchronize();

    float sum = 0.0;
    for (int i=0; i<N; i++) {
        sum = z[i] + sum;
    }

    printf("Sum = %f\n", sum);
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_z);
    cudaFreeHost(x);
    cudaFreeHost(y);
    cudaFreeHost(z);
}

# include <stdio.h>
# include <stdint.h>

# include "cuda_runtime.h"
# define ENTRIES 256

//compile nvcc *.cu -o test

__global__ void global_latency (unsigned int *my_array, int array_length, int iterations,  unsigned int *duration, unsigned int *index);
void parametric_measure_global (int N, int iterations, int stride);
void measure_global (unsigned int, unsigned int, unsigned int);

int main (int argc, char **argv) {

    if (argc < 4) {
        printf ("Usage: ./<executable>   from_mb   to_mb   stride_size_kb\n");
        return 0;
    }

    /* Array size in mega bytes 1st argument. */
    unsigned int from_mb = (unsigned int) atof (argv[1]);

    /* Array size in mega bytes 1st argument. */
    unsigned int to_mb = (unsigned int) atof (argv[2]);

    /* Stride size in kilo bytes 1st argument. */
    unsigned int stride_size_kb = (unsigned int) atof (argv[3]);

    cudaSetDevice (0);
    measure_global (from_mb, to_mb, stride_size_kb);
    cudaDeviceReset ();
    return 0;
}


void measure_global (unsigned int from_mb, unsigned int to_mb, unsigned int stride_size_kb) {

    int N, iterations, stride; 
    //stride in element
    iterations = 1;
    
    stride = stride_size_kb * 1024 / sizeof (unsigned int); //some stride
    //1. The L1 TLB has 16 entries. Test with N_min=28 *1024*256, N_max>32*1024*256
    //2. The L2 TLB has 65 entries. Test with N_min=128*1024*256, N_max=160*1024*256
    for (N = from_mb * 1024 * 256; N <= to_mb * 1024 * 256; N += stride) {
        printf ("\n=====%3.1f MB array, warm TLB, read 256 element====\n", sizeof (unsigned int) * (float) N / 1024 / 1024);
        printf ("Stride = %d element, %d MB\n", stride, stride * sizeof (unsigned int) / 1024 / 1024);
        parametric_measure_global (N, iterations, stride);
        printf ("===============================================\n\n");
    }
}


void parametric_measure_global (int N, int iterations, int stride) {
    cudaDeviceReset ();

    cudaError_t error_id;
    
    int i;
    unsigned int *h_a;
    /* allocate arrays on CPU */
    h_a = (unsigned int *) malloc (sizeof (unsigned int) * (N + 2));
    unsigned int *d_a;
    /* allocate arrays on GPU */
    error_id = cudaMalloc ((void **) &d_a, sizeof (unsigned int) * (N + 2));
    if (error_id != cudaSuccess) {
        printf ("Error 1.0 is %s\n", cudaGetErrorString (error_id));
    }

    /* initialize array elements on CPU with pointers into d_a. */
    for (i = 0; i < N; i++) {
        //original:
        h_a[i] = (i + stride) % N;    
    }

    h_a[N] = 0;
    h_a[N+1] = 0;
    /* copy array elements from CPU to GPU */
    error_id = cudaMemcpy (d_a, h_a, sizeof(unsigned int) * N, cudaMemcpyHostToDevice);
    if (error_id != cudaSuccess) {
        printf("Error 1.1 is %s\n", cudaGetErrorString (error_id));
    }

    unsigned int *h_index = (unsigned int *) malloc (sizeof (unsigned int) * ENTRIES);
    unsigned int *h_timeinfo = (unsigned int *) malloc (sizeof (unsigned int) * ENTRIES);

    unsigned int *duration;
    error_id = cudaMalloc ((void **) &duration, sizeof (unsigned int) * ENTRIES);
    if (error_id != cudaSuccess) {
        printf("Error 1.2 is %s\n", cudaGetErrorString (error_id));
    }

    unsigned int *d_index;
    error_id = cudaMalloc ((void **) &d_index, sizeof (unsigned int) * ENTRIES);
    if (error_id != cudaSuccess) {
        printf ("Error 1.3 is %s\n", cudaGetErrorString (error_id));
    }

    cudaDeviceSynchronize ();
    /* launch kernel*/
    dim3 Db = dim3 (1);
    dim3 Dg = dim3 (1, 1, 1);

    global_latency <<<Dg, Db>>>(d_a, N, iterations, duration, d_index);

    cudaDeviceSynchronize ();

    error_id = cudaGetLastError ();
    if (error_id != cudaSuccess) {
        printf ("Error kernel is %s\n", cudaGetErrorString (error_id));
    }

    /* copy results from GPU to CPU */
    cudaDeviceSynchronize ();

    error_id = cudaMemcpy((void *) h_timeinfo, (void *) duration, sizeof (unsigned int) * ENTRIES, cudaMemcpyDeviceToHost);
    if (error_id != cudaSuccess) {
        printf ("Error 2.0 is %s\n", cudaGetErrorString(error_id));
    }

    error_id = cudaMemcpy((void *) h_index, (void *) d_index, sizeof (unsigned int) * ENTRIES, cudaMemcpyDeviceToHost);
    if (error_id != cudaSuccess) {
        printf ("Error 2.1 is %s\n", cudaGetErrorString (error_id));
    }

    cudaDeviceSynchronize ();

    for (i = 0; i < ENTRIES - 1; i++)
        printf ("%d\t %d\n", h_index[i], h_timeinfo[i + 1]);

    /* free memory on GPU */
    cudaFree (d_a);
    cudaFree (d_index);
    cudaFree (duration);

    /*free memory on CPU */
    free (h_a);
    free (h_index);
    free (h_timeinfo);
    
    cudaDeviceReset();
}

static __device__ __inline__ uint64_t global_timer () {
    uint64_t timer;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(timer));
    return timer;
}

__global__ void global_latency (unsigned int *my_array, int array_length, int iterations, unsigned int *duration, unsigned int *index) {

    unsigned int start_time, end_time;
    unsigned int j = 0, sum = 0; 

    __shared__ unsigned int s_tvalue[ENTRIES];
    __shared__ unsigned int s_index[ENTRIES];

    int k;
    for(k = 0; k < ENTRIES; k++) {
        s_index[k] = 0;
        s_tvalue[k] = 0;
    }

    //first round, warm the TLB
    for (k = 0; k < iterations * ENTRIES; k++) 
        j = my_array[j];

    //second round, begin timestamp
    j = (j != 0) ? 0 : j;
    for (k = 0; k < iterations * ENTRIES; k++) {
        start_time = clock64 ();

        j = my_array[j];
        s_index[k] = j;
        sum += j;
        end_time = clock64 ();

        s_tvalue[k] = (end_time - start_time);
    }

    my_array[array_length] = sum;
    my_array[array_length + 1] = my_array[j];

    for (k = 0; k < 256; k++) {
        index[k] = s_index[k];
        duration[k] = s_tvalue[k];
    }
}

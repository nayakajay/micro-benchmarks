# include <stdio.h>
# include <stdint.h>

# include "cuda_runtime.h"
# include "cuda_profiler_api.h"

#define ITERATIONS 2
#define DEBUG 1
#define MAX_SHARED_E 2048

#include <time.h>
#include <sys/time.h>
#define USECPSEC 1000000ULL

unsigned long long dtime_usec (unsigned long long start) {

  timeval tv;
  gettimeofday (&tv, 0);
  return ((tv.tv_sec * USECPSEC) + tv.tv_usec) - start;
}

//compile nvcc *.cu -o test

__global__ void init_device_array (unsigned int *, unsigned int, unsigned long long);
__global__ void global_latency (unsigned int *, unsigned long long , unsigned int, unsigned int *, unsigned long *);
unsigned int parametric_measure_global (unsigned long long , int, unsigned int);
void measure_global (unsigned int, unsigned int, unsigned int);

unsigned long long KB_TO_B (unsigned int x) {
    return (x * 1024LLU);
}
unsigned long long MB_TO_B (unsigned int x)  {
    return (x * 1024LLU * 1024LLU);
}

int main (int argc, char **argv) {

    if (argc < 5) {
        printf ("Usage: ./<executable>   from_mb   to_mb   stride_size_kb   device_no (0-1)\n");
        return 0;
    }

    /* Array size in mega bytes 1st argument. */
    unsigned int from_mb = (unsigned int) atof (argv[1]);

    /* Array size in mega bytes 2nd argument. */
    unsigned int to_mb = (unsigned int) atof (argv[2]);

    /* Stride size in kilo bytes 3rd argument. */
    unsigned int stride_size_kb = (unsigned int) atof (argv[3]);
    /* Set device number if there are many! */
    int device = atoi (argv[4]);

    cudaSetDevice (device);

    measure_global (from_mb, to_mb, stride_size_kb);

    cudaDeviceReset ();

    return 0;
}

void measure_global (unsigned int from_mb, unsigned int to_mb, unsigned int stride_size_kb) {

    unsigned int stride, stride_b, entries;
    unsigned long long N, array_size_b, elements;

    // stride in element, change here if needed!
    stride_b = KB_TO_B (stride_size_kb);
    // if argument is non zero take that, else 1 element stride
    stride_b = (stride_b > 0) ? stride_b : sizeof (unsigned int);
    stride = stride_b / sizeof (unsigned int);

    unsigned int d_time = 0;
    unsigned long long difft;
    for (int i = from_mb; i <= to_mb; i += 2) {
        // chage below for different array size creation!
        array_size_b = MB_TO_B (i);
        elements = array_size_b / sizeof (unsigned int);
        entries = array_size_b / stride_b;
        N = elements;

        /* Why 2 * MAX_SHARED_E, well there are 2 shared arrays in the kernel! */
        if (ITERATIONS * entries > (2 * MAX_SHARED_E)) {
            printf ("Hey, shared_memory is limited, either reduce ITERATIONS or increase stride!!");
            return;
        }

        printf ("\n=====%llu MB array, warm TLB, read %d elements====\n", array_size_b / MB_TO_B(1), entries);
        printf ("Iterations = %d, Stride = %u element, %u KB\n", ITERATIONS, stride, stride_size_kb);
        difft = dtime_usec (0);

        d_time = parametric_measure_global (N, stride, entries);

        difft = dtime_usec (difft);
        printf ("(H) Kernel Duration: %lf\n", (difft / ITERATIONS) / (float) USECPSEC);
        printf ("(D) Kernel Duration: %u\n", d_time);

        printf("\n===============================================\n\n");
    }
}

unsigned int parametric_measure_global (unsigned long long N, int stride, unsigned int entries) {

    cudaDeviceReset ();
    cudaError_t error_id;
    int i;

    unsigned int *h_a;
    /* allocate arrays on CPU, pinned host memory */
    cudaHostAlloc ((void **) &h_a, sizeof(unsigned int) * (N + 2), cudaHostAllocMapped);

    /* initialize array elements on CPU with pointers into d_a. */
    for (i = 0; i < N; i++) {        
        // Original:    
        h_a[i] = (i + stride) % N;
    }

    h_a[N] = 0;
    h_a[N + 1] = 0;

    unsigned int *d_a;
    /* allocate arrays on GPU */
    error_id = cudaHostGetDevicePointer ((void **) &d_a, h_a, 0);
    if (error_id != cudaSuccess) {
        printf ("Error 1.0 is %s\n", cudaGetErrorString (error_id));
    }

    unsigned int data_entries = ITERATIONS * entries;

    unsigned int *duration;
    unsigned int *h_duration;
    cudaHostAlloc ((void **) &h_duration, sizeof(unsigned int) * data_entries, cudaHostAllocMapped);
    error_id = cudaHostGetDevicePointer ((void **) &duration, h_duration, 0);;
    if (error_id != cudaSuccess) {
        printf ("Error 1.1 is %s\n", cudaGetErrorString (error_id));
    }

    unsigned long *address;
    unsigned long *h_address;
    cudaHostAlloc ((void **) &h_address, sizeof(unsigned int) * data_entries, cudaHostAllocMapped);
    error_id = cudaHostGetDevicePointer ((void **) &address, h_address, 0);;
    if (error_id != cudaSuccess) {
        printf ("Error 1.1 is %s\n", cudaGetErrorString (error_id));
    }

    cudaDeviceSynchronize ();
    /* launch kernel*/
    dim3 Db = dim3 (1);
    dim3 Dg = dim3 (1, 1, 1);

    init_device_array <<<Dg, Db>>> (d_a, N, entries);

    cudaProfilerStart ();
    global_latency <<<Dg, Db>>> (d_a, N, entries, duration, address);
    cudaProfilerStop ();

    cudaDeviceSynchronize ();

    unsigned int result = h_a[N + 1];
    cudaDeviceSynchronize ();

    int d_sum = 0, a_time;
    if (DEBUG) {
        printf ("\nAddress\t\tTime\n");
        for (int j = 0; j < ITERATIONS; j++) {
            printf ("==Iteration: %d ================================\n\n", j);
            for (int i = 0; i < entries; i++) {
                int idx = j * entries + i;
                a_time = duration[idx] / ITERATIONS;
                d_sum += a_time;
                printf ("%p : %u\n", (void *) address[i], a_time);
            }
            printf ("===============================================\n\n");
        }
    }

    /* free memory on CPU */
    cudaFreeHost (h_a);
    cudaFreeHost (h_address);
    cudaFreeHost (h_duration);
    
    cudaDeviceReset ();
    return d_sum;
}

__global__ void init_device_array (unsigned int *my_array, unsigned int entries, unsigned long long length) {

    unsigned int k = 0, j, sum = 0;
    // first round, warm the TLB
    for (k = 0; k < entries; k++) {
        j = my_array[j];
        sum += j;
    }

    my_array[length] = sum;
}

__global__ void global_latency (unsigned int *my_array, unsigned long long array_length,
                                unsigned int entries, unsigned int *d_time, unsigned long *addr) {

    unsigned int start_time, end_time, pos_sum = 0, duration = 0;
    unsigned int j = 0, i, k, idx;

	// __shared__ unsigned int s_tvalue[MAX_SHARED_E];
	// __shared__ unsigned int s_index[MAX_SHARED_E];

    j = (j != 0) ? 0 : j;
    // second round, begin timestamp
    for (i = 0; i < ITERATIONS; i++) {
        for (k = 0; k < entries; k++) {

            idx = i * entries + k;
            addr[idx] = (unsigned long) &my_array[j];
            start_time = clock64 ();
            j = my_array[j];
            pos_sum += j;
            end_time = clock64 ();

            duration += (end_time - start_time);
            d_time[idx] = (end_time - start_time);

        }
    }

    // copy from shared to global array
    /*for (i = 0; i < ITERATIONS; i++) {
        for (k = 0; k < entries; k++) {

            idx = i * entries + k;
            addr[idx] = s_index[idx];
            d_time[idx] = s_tvalue[idx];

        }
    }*/

    my_array[array_length] = pos_sum;
    my_array[array_length + 1] = duration;
}

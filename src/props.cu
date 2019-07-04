# include <stdio.h>
# include <stdint.h>

# include "cuda_runtime.h"

int main (int argc, char **argv) {

    int dev_no = (int) atoi (argv[1]);

    cudaDeviceProp props;
    cudaGetDeviceProperties (&props, dev_no);
    printf ("l2CacheSize: %d\n", props.l2CacheSize);

    printf ("concurrentKernels: %d\n", props.concurrentKernels);

    size_t maxL2Fetch = 10;
    cudaDeviceGetLimit (&maxL2Fetch, cudaLimitMaxL2FetchGranularity);
    printf ("cudaLimitMaxL2FetchGranularity: %lu\n", maxL2Fetch);

    printf ("clockRate: %d\n", props.clockRate);

    printf ("Compute Capability: %d\n", props.major);

    printf ("Minor: %d\n", props.minor);

    return 0;
}

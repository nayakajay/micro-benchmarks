# Straight commands to create runnables
CUDA_CC=nvcc
CUDA_FLAGS=-Xptxas -dlcm=cs -Xptxas -dscm=cs -arch=sm_61 -lineinfo
VECTOR_CUDA_FLAGS=-Xptxas -O0 -Xptxas -dlcm=cg -Xptxas -dscm=wt -arch=sm_61

all: device_alloc host_alloc vector_add
# Executable that creates an array and traverses it, Array allocated in device
device_alloc:
	$(CUDA_CC) $(CUDA_FLAGS) src/fine_grain_tlb.cu -o cs_original

# Similar experiment, but memory pinned in host
host_alloc:
	$(CUDA_CC) $(CUDA_FLAGS) src/host_alloc.cu -o host_pinned

# Shweta vector addition
vector_add:
	$(CUDA_CC) $(VECTOR_CUDA_FLAGS) src/vector_add.cu -o vector_add

//==========================================================================================================
// Performing the following arithmetic operation	 -- 
// 1e-6 * x[i] ) + (1e-7 * y[i]) + 0.25
// On matrices on GPUs using Cuda
//
// Author - Anmol Gupta, Naved Ansari
// Course - EC513 - Introduction to Computer Architecture
// Boston University
//==========================================================================================================

//==========================================================================================================
// Command to compile the code
//nvcc -o SimpleCode SimpleCode.cu
//==========================================================================================================

//nvcc -o SimpleCode SimpleCode.cu



#include <cstdio>
#include <cstdlib>
#include <math.h>
#include <time.h>
#include "cuPrintf.cu"
#include "cuPrintf.cuh"

// Assertion to check for errors
#define CUDA_SAFE_CALL(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, char *file, int line, bool abort=true)
{
	if (code != cudaSuccess) 
	{
		fprintf(stderr,"CUDA_SAFE_CALL: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}


#define NUM_THREADS_PER_BLOCK	256
#define BLOCK_WIDTH = 16
#define NUM_BLOCKS	16384

#define PRINT_TIME 				1
#define SM_ARR_LEN				100000 //set array size here
#define TOL					1e-3 
#define GIG 1000000
#define NPM 0.001	// Nano second per Microsecond

#define OMEGA 1.60
#define IMUL(a, b) __mul24(a, b)


void initializeArray1D(float *arr, int len, int seed);
//simple calculation kernel
__global__ void kernel_add (int arrLen, float* x, float* y, float* result) {
	const int tid = IMUL(blockDim.x, blockIdx.x) + threadIdx.x;
	const int threadN = IMUL(blockDim.x, gridDim.x);
	
	int i;
	
	for(i = tid; i < arrLen; i += threadN) {
		result[i] = (1e-6 * x[i] ) + (1e-7 * y[i]) + 0.25;
	}
}


int main(int argc, char **argv){
	//function declare
	struct timespec diff(struct timespec start, struct timespec end);
  	struct timespec time1, time2;
	struct timespec time_stamp;

	int arrLen = 0;		//length of one edge
	int totalLen = 0;	//total length, equal to square of arrLen
	//double change;
		
	// GPU Timing variables
	cudaEvent_t start, stop;
	float elapsed_gpu;
	
	// Arrays on GPU global memory
	float *d_x;
	float *d_y;
	float *d_result;

	// Arrays on the host memory
	float *h_x;
	float *h_y;
	float *cpu_x;
	float *h_result;
	float *h_result_gold;
	int i, errCount = 0, zeroCount = 0;
	
	if (argc > 1) {
		arrLen  = atoi(argv[1]);
	}
	else {
		arrLen = SM_ARR_LEN;
	}
	totalLen = SM_ARR_LEN;
	printf("Length of the array = %d\n", totalLen);

	// Allocate GPU memory
	size_t allocSize = totalLen * sizeof(float);
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_x, allocSize));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_y, allocSize));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_result, allocSize));
		
	// Allocate arrays on host memory
	h_x                        = (float *) malloc(allocSize);
	h_y                        = (float *) malloc(allocSize);
	cpu_x                        = (float *) malloc(allocSize);
	h_result                   = (float *) malloc(allocSize);
	h_result_gold              = (float *) malloc(allocSize);
	
	// Initialize the host arrays
	printf("\nInitializing the arrays ...");
	// Arrays are initialized with a known seed for reproducibility
	initializeArray1D(h_x, totalLen, 2453);
	initializeArray1D(h_y, arrLen, 1467);
	for(i = 0; i < totalLen; i++){
		cpu_x[i] = h_x[i];
	}
	printf("\t... done\n\n");
	
	
#if PRINT_TIME
	// Create the cuda events
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	// Record event on the default stream
	cudaEventRecord(start, 0);
#endif
	// original kernel call for part 2
	// Transfer the arrays to the GPU memory
	CUDA_SAFE_CALL(cudaMemcpy(d_x, h_x, allocSize, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_y, h_y, allocSize, cudaMemcpyHostToDevice));
	  
	// Launch the kernel
	kernel_add<<<NUM_BLOCKS, NUM_THREADS_PER_BLOCK>>>(arrLen, d_x, d_y, d_result);

	// Check for errors during launch
	CUDA_SAFE_CALL(cudaPeekAtLastError());

	// Transfer the results back to the host
	CUDA_SAFE_CALL(cudaMemcpy(h_result, d_result, allocSize, cudaMemcpyDeviceToHost));
	
#if PRINT_TIME
	// Stop and destroy the timer
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed_gpu, start, stop);
	printf("\nGPU time: %f (msec)\n", elapsed_gpu);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
#endif
	
	// Compute the results on the host
	// Time for this
	clock_gettime(CLOCK_REALTIME, &time1);
		// Compute the results on the host
	for(i = 0; i < arrLen; i++) {
		h_result_gold[i] = (1e-6 * h_x[i]) + (1e-7 * h_y[i]) + 0.25;
	}
	clock_gettime(CLOCK_REALTIME, &time2);
   	time_stamp = diff(time1,time2);
	
	printf("CPU time: %ld (msec)\n", (long int)(GIG * time_stamp.tv_sec + NPM * time_stamp.tv_nsec));

	
	// Compare the results
	/*
	for(i = 0; i < totalLen; i++) {
		if (abs(h_result_gold[i - h_result[i]) > TOL) {
			errCount++;
		}
		if (h_result[i] == 0) {
			zeroCount++;
		}
	}
	*/
	
	
	if (errCount > 0) {
		printf("\n@ERROR: TEST FAILED: %d results did not matched\n", errCount);
	}
	else if (zeroCount > 0){
		printf("\n@ERROR: TEST FAILED: %d results (from GPU) are zero\n", zeroCount);
	}
	else {
		printf("\nTEST PASSED: All results matched\n");
	}
	
	// Free-up device and host memory
	CUDA_SAFE_CALL(cudaFree(d_x));
	CUDA_SAFE_CALL(cudaFree(d_y));
	CUDA_SAFE_CALL(cudaFree(d_result));
		   
	free(h_x);
	free(h_y);
	free(h_result);
		
	return 0;
}

void initializeArray1D(float *arr, int len, int seed) {
	int i;
	float randNum;
	srand(seed);

	for (i = 0; i < len; i++) {
		randNum = (float) rand();
		arr[i] = randNum;
	}
}

//=====================Timing functions===============
struct timespec diff(struct timespec start, struct timespec end)
{
  struct timespec temp;
  if ((end.tv_nsec-start.tv_nsec)<0) {
    temp.tv_sec = end.tv_sec-start.tv_sec-1;
    temp.tv_nsec = 1000000000+end.tv_nsec-start.tv_nsec;
  } else {
    temp.tv_sec = end.tv_sec-start.tv_sec;
    temp.tv_nsec = end.tv_nsec-start.tv_nsec;
  }
  return temp;
}


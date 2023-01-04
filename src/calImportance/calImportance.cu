#include <stdio.h>
#include <stdint.h>
#include "../library.h"

#define CHECK(call)\
{\
    const cudaError_t error = call;\
    if (error != cudaSuccess)\
    {\
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);\
        fprintf(stderr, "code: %d, reason: %s\n", error,\
                cudaGetErrorString(error));\
        exit(EXIT_FAILURE);\
    }\
}

/**
 * @param useDevice = false (default)
 * @param blockSize=dim3(1) (default)
 */
void callImportance(uint8_t *in1, uint8_t *in2, int nRows, int nCols, uint8_t *out, bool useDevice, dim3 blockSize) {
	GpuTimer timer;
	timer.Start();
	if (useDevice == false) {
    addMatrix_host(in1, in2, nRows, nCols, out);
	}
	else { // Use device
    // Allocate device memories
    uint8_t * d_in1, * d_in2, * d_out;
    size_t nBytes = nRows * nCols * sizeof(uint8_t);
    CHECK(cudaMalloc(&d_in1, nBytes));
    CHECK(cudaMalloc(&d_in2, nBytes));
    CHECK(cudaMalloc(&d_out, nBytes));

    // Copy data to device memories
    CHECK(cudaMemcpy(d_in1, in1, nBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_in2, in2, nBytes, cudaMemcpyHostToDevice));

    // Set grid size and call kernel
    dim3 gridSize((nCols - 1) / blockSize.x + 1, (nRows - 1) / blockSize.y + 1);
    addMatrix_kernel<<<gridSize, blockSize>>>(d_in1, d_in2, nRows, nCols, d_out);

    // Copy result from device memory
    CHECK(cudaMemcpy(out, d_out, nBytes, cudaMemcpyDeviceToHost));

    // Free device memories
    CHECK(cudaFree(d_in1));
    CHECK(cudaFree(d_in2));
    CHECK(cudaFree(d_out));
	}
	timer.Stop();
	float time = timer.Elapsed();
	printf("Processing time (%s): %f ms\n\n", useDevice == true? "use device" : "use host", time);
}
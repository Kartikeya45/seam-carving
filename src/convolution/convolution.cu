#include <stdio.h>
#include <stdint.h>
#include "../library.h"

__constant__ float dc_filter[9];

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
 * @param blockSize = dim3(1, 1) (default)
 * @param kernelType = 1 (default)
 * 
 * kernel 1: use GMEM
 * kernel 2: apply Shared memmory
 * kernel 3: apply Shared memmory, Constant memmory
 */
void convolution(uint8_t * inPixels, int width, int height, float * filter, int filterWidth, 
        uint8_t * outPixels, bool useDevice, dim3 blockSize, int kernelType) {

	if (useDevice == false) {
        detectEdges_host(inPixels, width, height, filter, filterWidth, outPixels);
	}
	else { // Use device
		GpuTimer timer;
		
		printf("\nKernel %i, ", kernelType);
		// Allocate device memories
		uint8_t * d_inPixels, * d_outPixels;
		float * d_filter;
		size_t pixelsSize = width * height * sizeof(uint8_t);
		size_t filterSize = filterWidth * filterWidth * sizeof(float);
		CHECK(cudaMalloc(&d_inPixels, pixelsSize));
		CHECK(cudaMalloc(&d_outPixels, pixelsSize));

		if (kernelType == 1 || kernelType == 2) {
			CHECK(cudaMalloc(&d_filter, filterSize));
		}

		// Copy data to device memories
		CHECK(cudaMemcpy(d_inPixels, inPixels, pixelsSize, cudaMemcpyHostToDevice));
		if (kernelType == 1 || kernelType == 2) {
			CHECK(cudaMemcpy(d_filter, filter, filterSize, cudaMemcpyHostToDevice));
		} else {
			// Copy data from "filter" (on host) to "dc_filter" (on CMEM of device)
            cudaMemcpyToSymbol(dc_filter,filter,filterSize);
		}

		// Call kernel
		dim3 gridSize((width-1)/blockSize.x + 1, (height-1)/blockSize.y + 1);
		printf("block size %ix%i, grid size %ix%i\n", blockSize.x, blockSize.y, gridSize.x, gridSize.y);
        size_t share_size = (blockSize.x + filterWidth - 1) * (blockSize.y + filterWidth - 1) * sizeof(uint8_t);

		timer.Start();
		if (kernelType == 1) {
            detectEdges_kernel1<<<gridSize, blockSize>>>(d_inPixels, width, height, d_filter, filterWidth, d_outPixels);
		} 
        else if (kernelType == 2) {
            detectEdges_kernel2<<<gridSize, blockSize, share_size>>>(d_inPixels, width, height, d_filter, filterWidth, d_outPixels);
		}
		else {
            detectEdges_kernel3<<<gridSize, blockSize, share_size>>>(d_inPixels, width, height, filterWidth, d_outPixels);
		}
		timer.Stop();
		float time = timer.Elapsed();
		printf("Kernel time: %f ms\n", time);
		cudaDeviceSynchronize();
		CHECK(cudaGetLastError());

		// Copy result from device memory
		CHECK(cudaMemcpy(outPixels, d_outPixels, pixelsSize, cudaMemcpyDeviceToHost));

		// Free device memories
		CHECK(cudaFree(d_inPixels));
		CHECK(cudaFree(d_outPixels));
		if (kernelType == 1 || kernelType == 2) {
			CHECK(cudaFree(d_filter));
		}
	}
}
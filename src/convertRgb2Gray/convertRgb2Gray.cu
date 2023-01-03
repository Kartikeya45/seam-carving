#include <stdio.h>
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
 * gray = 0.299*red + 0.587*green + 0.114*blue  
 */
void convertRgb2Gray(uint8_t * inPixels, int width, int height, uint8_t * outPixels, bool useDevice, dim3 blockSize) {
	GpuTimer timer;//
	timer.Start();//
	if (useDevice == false)
    convertRgb2Gray_host(inPixels, width, height, outPixels);
	
	else { // use device
		// Allocate device memories
    uint8_t * d_inPixels;
    uint8_t * d_outPixels;
    size_t nBytes = width * height * sizeof(int);
    CHECK(cudaMalloc(&d_inPixels, nBytes));
    CHECK(cudaMalloc(&d_outPixels, nBytes));

		// Copy data to device memories
    CHECK(cudaMemcpy(d_inPixels, inPixels, nBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_outPixels, outPixels, nBytes, cudaMemcpyHostToDevice));

		// Set grid size and call kernel
    dim3 gridSize((width - 1) / blockSize.x + 1, (height - 1) / blockSize.y + 1);
    convertRgb2GrayKernel<<<gridSize, blockSize>>>(d_inPixels, width, height, d_outPixels);

    // Check kernel error
		cudaError_t errSync  = cudaGetLastError();
		cudaError_t errAsync = cudaDeviceSynchronize();
		if (errSync != cudaSuccess) 
		  printf("Sync kernel error: %s\n", cudaGetErrorString(errSync));
		if (errAsync != cudaSuccess)
		  printf("Async kernel error: %s\n", cudaGetErrorString(errAsync));

		// Copy result from device memories
    CHECK(cudaMemcpy(outPixels, d_outPixels, nBytes, cudaMemcpyDeviceToHost));

		// Free device memories
    CHECK(cudaFree(d_inPixels));
    CHECK(cudaFree(d_outPixels));
	}
	timer.Stop();//
	float time = timer.Elapsed();//
	printf("Processing time (%s): %f ms\n\n", useDevice == true? "use device" : "use host", time);
}